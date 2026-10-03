from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Implement Agent A / Baseline Agent.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Forgets long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                session = self.sessions.setdefault(thread_id, SessionState())
                prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
                session.prompt_tokens_processed += prompt_tokens
                session.messages.append({"role": "user", "content": message})

                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                output_msg = result["messages"][-1].content
                reply_tokens = estimate_tokens(output_msg)
                session.token_usage += reply_tokens
                session.messages.append({"role": "assistant", "content": output_msg})
                return {
                    "role": "assistant",
                    "content": output_msg,
                    "tokens": reply_tokens,
                    "prompt_tokens": prompt_tokens,
                }
            except Exception:
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions[thread_id].token_usage if thread_id in self.sessions else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions[thread_id].prompt_tokens_processed if thread_id in self.sessions else 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Implement simple within-thread offline behavior."""
        session = self.sessions.setdefault(thread_id, SessionState())

        # Prompt context includes all past messages in this thread + incoming message
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
        session.prompt_tokens_processed += prompt_tokens
        session.messages.append({"role": "user", "content": message})

        # Baseline forgets across threads: if asking in a new thread, it has no memory
        if "?" in message or "nhắc lại" in message.lower() or "đâu mới là" in message.lower():
            reply_text = "Tôi không có thông tin về bạn trong phiên làm việc này do không có persistent memory."
        else:
            reply_text = "Tôi đã ghi nhận thông tin trong phiên làm việc này."

        reply_tokens = estimate_tokens(reply_text)
        session.token_usage += reply_tokens
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "role": "assistant",
            "content": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally wire LangChain agent if API key is present."""
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
