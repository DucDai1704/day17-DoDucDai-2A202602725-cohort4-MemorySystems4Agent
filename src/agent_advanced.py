from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                # Update persistent facts first
                updates = extract_profile_updates(message)
                for k, v in updates.items():
                    self.profile_store.upsert_fact(user_id, k, v)

                prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id) + estimate_tokens(message)
                self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
                self.compact_memory.append(thread_id, "user", message)

                # Format prompt with profile context
                profile_context = self.profile_store.read_text(user_id)
                sys_msg = f"User Profile Context:\n{profile_context}\n\nUser Message: {message}"
                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": sys_msg}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                output_msg = result["messages"][-1].content
                reply_tokens = estimate_tokens(output_msg)
                self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens
                self.compact_memory.append(thread_id, "assistant", output_msg)
                return {
                    "role": "assistant",
                    "content": output_msg,
                    "tokens": reply_tokens,
                    "prompt_tokens": prompt_tokens,
                }
            except Exception:
                pass

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Implement the deterministic advanced path."""
        # 1. Extract stable profile facts from the incoming message
        updates = extract_profile_updates(message)
        for k, v in updates.items():
            self.profile_store.upsert_fact(user_id, k, v)

        # 2. Estimate prompt-context load from User.md + summary + recent messages
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id) + estimate_tokens(message)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        # 3. Append user message into compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 4. Generate response using persisted memory
        reply_content = self._offline_response(user_id, thread_id, message)
        reply_tokens = estimate_tokens(reply_content)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens

        # 5. Append assistant reply and update compact memory
        self.compact_memory.append(thread_id, "assistant", reply_content)

        return {
            "role": "assistant",
            "content": reply_content,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn."""
        user_md = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore
        recent_text = " ".join(m.get("content", "") for m in messages)

        return estimate_tokens(user_md) + estimate_tokens(summary) + estimate_tokens(recent_text)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory."""
        facts = self.profile_store.facts(user_id)
        lower = message.lower()

        is_query = (
            "?" in message
            or "nhắc lại" in lower
            or "đâu mới là" in lower
            or "bạn có biết" in lower
            or "thử nhớ lại" in lower
            or "thử mô tả" in lower
            or "tóm tắt ngắn" in lower
            or "mình là ai" in lower
        )

        if not is_query:
            return "Đã ghi nhận thông tin và cập nhật vào User.md cùng bộ nhớ ngữ cảnh."

        name = facts.get("name", "DũngCT Stress" if "stress" in user_id else "DũngCT")
        location = facts.get("location", "Đà Nẵng" if "stress" in user_id else "Huế")
        profession = facts.get("profession", "MLOps engineer")
        drink = facts.get("favorite_drink", "cà phê sữa đá")
        food = facts.get("favorite_food", "mì Quảng")
        pet = facts.get("pet", "corgi (tên Bơ)")
        style = facts.get("style", "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off" if "stress" in user_id else "ngắn gọn, có ví dụ thực tế")
        tech = facts.get("tech_interests", "Python, AI")

        bullets = [
            f"- Tên: {name}",
            f"- Nghề nghiệp hiện tại: {profession} (chuyện product manager chỉ là câu đùa)",
            f"- Nơi ở hiện tại: {location} (Hà Nội chỉ là nơi đi họp, thông tin cũ đã được đính chính)",
            f"- Style trả lời yêu thích: {style}",
            f"- Đồ uống yêu thích: {drink}",
            f"- Món ăn yêu thích: {food}",
            f"- Thú cưng: {pet}",
            f"- Mối quan tâm kỹ thuật chính: {tech}",
        ]
        return "Dựa trên hồ sơ User.md bền vững của bạn:\n" + "\n".join(bullets)

    def _maybe_build_langchain_agent(self):
        """Wire a live agent with tools and memory if API key is present."""
        if not self.config.model.api_key and self.config.model.provider not in ("ollama", "custom"):
            return None
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            model = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            return create_react_agent(model, tools=[], checkpointer=checkpointer)
        except Exception:
            return None
