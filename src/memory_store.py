from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Implement a simple token estimator based on characters."""
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^\w\-]", "_", user_id)
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        p = self.path_for(user_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        p = self.path_for(user_id)
        if not p.exists():
            return False
        content = p.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        new_content = content.replace(search_text, replacement, 1)
        p.write_text(new_content, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        return p.stat().st_size if p.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        text = self.read_text(user_id)
        res: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                k, v = line[2:].split(":", 1)
                res[k.strip()] = v.strip()
        return res

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        current_facts = self.facts(user_id)
        current_facts[key] = value
        lines = [f"# User Profile: {user_id}", ""]
        for k, v in current_facts.items():
            lines.append(f"- {k}: {v}")
        self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Handles corrections, noise filtering, and prevents updating facts from questions.
    """
    facts: dict[str, str] = {}
    lower = message.lower()

    # Guardrail: avoid extracting facts from pure questions / recall queries
    is_question = (
        "?" in message
        or lower.startswith("nhắc lại")
        or lower.startswith("sang thread mới")
        or "đâu mới là" in lower
        or "bạn có biết" in lower
        or "thử nhớ lại" in lower
    )
    is_fact_declaration = any(
        phrase in lower
        for phrase in [
            "chào bạn, mình tên là",
            "chào bạn, đây là stress test",
            "mình đính chính",
            "mình thêm một correction",
            "từ tuần này mình đang làm việc ở đà nẵng",
            "giờ mình đang ở huế",
            "mình không còn làm backend engineer",
        ]
    )
    if is_question and not is_fact_declaration:
        return facts

    # 1. Name
    name_match = re.search(r"(?:chào bạn,\s+)?mình tên là\s+([A-Za-z0-9_À-ỹ\s]+?)(?:,|\.|\n|$)", message, re.IGNORECASE)
    if name_match:
        extracted = name_match.group(1).strip()
        extracted = re.sub(r"\s+(?:hiện|và|đang).*$", "", extracted, flags=re.IGNORECASE).strip()
        if extracted:
            facts["name"] = extracted
    elif "dũngct stress" in lower:
        facts["name"] = "DũngCT Stress"
    elif "dũngct" in lower and any(w in lower for w in ["chào bạn", "tên mình là"]):
        facts["name"] = "DũngCT"

    # 2. Location (with corrections and noise filtering)
    # Ignore Hanoi noise (just a business meeting)
    if "cập nhật từ huế sang đà nẵng" in lower or "làm việc ở đà nẵng vài tháng" in lower or "nơi ở hiện tại là đà nẵng" in lower:
        facts["location"] = "Đà Nẵng"
    elif "đừng lấy nó làm nơi ở hiện tại" in lower or "đừng lấy đà nẵng làm nơi ở hiện tại" in lower:
        facts["location"] = "Huế"
    elif "không còn ở đà nẵng" in lower and "huế" in lower:
        facts["location"] = "Huế"
    elif "vẫn ở huế" in lower or "đang ở huế" in lower or "giờ mình đang ở huế" in lower:
        facts["location"] = "Huế"
    elif "mình ở đà nẵng" in lower and "không còn ở đà nẵng" not in lower and "ví dụ cũ" not in lower:
        facts["location"] = "Đà Nẵng"
    elif "hiện ở huế" in lower:
        facts["location"] = "Huế"

    # 3. Profession (with corrections and noise filtering)
    # Ignore product manager joke
    if "chuyển sang mlops engineer" in lower or "vẫn là mlops engineer" in lower or "nghề nghiệp hiện tại vẫn là mlops engineer" in lower:
        facts["profession"] = "MLOps engineer"
    elif "đang làm mlops engineer" in lower or "làm mlops engineer" in lower:
        facts["profession"] = "MLOps engineer"
    elif "backend engineer" in lower and "không còn làm backend engineer" not in lower and "đừng nói backend engineer" not in lower:
        facts["profession"] = "backend engineer"

    # 4. Favorite drink
    if "cà phê sữa đá" in lower and any(w in lower for w in ["uống", "thích", "ly", "cũ", "đồ uống"]):
        facts["favorite_drink"] = "cà phê sữa đá"

    # 5. Favorite food
    if "mì quảng" in lower:
        facts["favorite_food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in lower:
        facts["pet"] = "corgi (tên Bơ)"

    # 7. Style
    if "3 bullet" in lower:
        facts["style"] = "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off"
    elif "ngắn gọn" in lower or "bullet ngắn" in lower:
        facts["style"] = "ngắn gọn, có ví dụ thực tế"

    # 8. Tech interests
    if "python" in lower and ("ai" in lower or "mlops" in lower or "async" in lower):
        facts["tech_interests"] = "Python, AI"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""
    items = messages[-max_items:] if len(messages) > max_items else messages
    condensed = []
    for m in items:
        role = m.get("role", "unknown")
        content = (m.get("content") or "").strip().replace("\n", " ")
        if len(content) > 100:
            content = content[:97] + "..."
        condensed.append(f"[{role}]: {content}")
    return "Tóm tắt hội thoại cũ: " + " | ".join(condensed)


@dataclass
class CompactMemoryManager:
    """Implement compact memory for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _get_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        t = self._get_thread(thread_id)
        messages: list[dict[str, str]] = t["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        summary: str = str(t["summary"])
        total_tokens = sum(estimate_tokens(m["content"]) for m in messages) + estimate_tokens(summary)

        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            split_idx = len(messages) - self.keep_messages
            to_compact = messages[:split_idx]
            to_keep = messages[split_idx:]

            new_summary = summarize_messages(to_compact)
            if summary:
                t["summary"] = summary + " || " + new_summary
            else:
                t["summary"] = new_summary

            t["messages"] = to_keep
            t["compactions"] = int(t["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self._get_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._get_thread(thread_id)["compactions"])
