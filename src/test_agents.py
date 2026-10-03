from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests."""
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=80,  # nhỏ để compact kích hoạt sớm
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    cfg = make_config(tmp_path)
    store = UserProfileStore(cfg.state_dir / "profiles")
    user_id = "test_user"

    # Write
    store.write_text(user_id, "# User Profile\n- name: Alice\n- location: Hanoi\n")
    assert "name: Alice" in store.read_text(user_id)
    assert store.file_size(user_id) > 0

    # Edit
    success = store.edit_text(user_id, "location: Hanoi", "location: Da Nang")
    assert success is True
    content = store.read_text(user_id)
    assert "location: Da Nang" in content
    assert "location: Hanoi" not in content

    # Non-existent replacement returns False
    assert store.edit_text(user_id, "location: Saigon", "location: Hue") is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    cfg = make_config(tmp_path)
    agent = AdvancedAgent(cfg, force_offline=True)
    thread_id = "test_thread"

    # Message with enough length to exceed compact_threshold_tokens (80)
    msg = (
        "Đây là một lượt trao đổi có nội dung tương đối dài để kiểm tra việc bộ nhớ ngắn hạn "
        "của advanced agent có tự động kích hoạt compact memory hay không khi số token tích lũy vượt ngưỡng quy định."
    )
    for i in range(5):
        agent.reply("user1", thread_id, f"{msg} (Lượt {i})")

    assert agent.compaction_count(thread_id) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    # Session 1: introduce facts
    msg = "Chào bạn, mình tên là DũngCT. Mình ở Đà Nẵng và thích cà phê sữa đá."
    baseline.reply("dungct", "thread_1", msg)
    advanced.reply("dungct", "thread_1", msg)

    # Session 2: recall question in a NEW thread
    q = "Mình tên gì và đồ uống yêu thích là gì?"
    ans_base = baseline.reply("dungct", "thread_2", q)["content"]
    ans_adv = advanced.reply("dungct", "thread_2", q)["content"]

    # Baseline should forget
    assert "DũngCT" not in ans_base
    assert "cà phê sữa đá" not in ans_base

    # Advanced should remember
    assert "DũngCT" in ans_adv
    assert "cà phê sữa đá" in ans_adv


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)
    thread_id = "long_thread"

    long_msg = "Đây là một đoạn văn bản dài nhằm tạo áp lực tải trọng ngữ cảnh lên prompt context của các agent trong quá trình trao đổi qua nhiều lượt. " * 8
    for _ in range(8):
        baseline.reply("user1", thread_id, long_msg)
        advanced.reply("user1", thread_id, long_msg)

    # Baseline accumulates all previous turns without compaction,
    # while Advanced compacts older messages into a summary.
    assert advanced.prompt_token_usage(thread_id) < baseline.prompt_token_usage(thread_id)
