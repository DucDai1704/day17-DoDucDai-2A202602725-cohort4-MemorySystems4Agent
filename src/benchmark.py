import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None  # type: ignore


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return recall ratio (0.0 - 1.0) based on how many expected facts appear in answer."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    hits = sum(1 for item in expected if item.lower() in ans_lower)
    return hits / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Compute lightweight quality score combining recall, structure, and length."""
    rec = recall_points(answer, expected)
    structured_bonus = 0.15 if ("- " in answer or "\n" in answer) else 0.05
    length_bonus = 0.15 if len(answer.strip()) >= 30 else 0.0
    return min(1.0, round(rec * 0.7 + structured_bonus + length_bonus, 2))


def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over conversations."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    initial_file_sizes: dict[str, int] = {}
    final_file_sizes: dict[str, int] = {}
    total_compactions = 0

    for conv in conversations:
        user_id = conv["user_id"]
        conv_id = conv["id"]
        thread_id = f"thread_{conv_id}"

        if user_id not in initial_file_sizes:
            initial_file_sizes[user_id] = getattr(agent, "memory_file_size", lambda u: 0)(user_id)

        # 1. Feed all turns to the agent
        for turn in conv.get("turns", []):
            res = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += res.get("tokens", 0)
            total_prompt_tokens += res.get("prompt_tokens", 0)

        total_compactions += agent.compaction_count(thread_id)

        # 2. Ask recall questions in a FRESH thread (measuring cross-session recall)
        for idx, q_data in enumerate(conv.get("recall_questions", [])):
            recall_thread_id = f"recall_{conv_id}_{idx}"
            question = q_data["question"]
            expected = q_data["expected_contains"]

            res = agent.reply(user_id, recall_thread_id, question)
            ans = res.get("content", "")
            total_agent_tokens += res.get("tokens", 0)
            total_prompt_tokens += res.get("prompt_tokens", 0)

            recall_scores.append(recall_points(ans, expected))
            quality_scores.append(heuristic_quality(ans, expected))

        final_file_sizes[user_id] = getattr(agent, "memory_file_size", lambda u: 0)(user_id)

    avg_recall = (sum(recall_scores) / len(recall_scores)) if recall_scores else 0.0
    avg_quality = (sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0
    memory_growth = sum(final_file_sizes.get(u, 0) - initial_file_sizes.get(u, 0) for u in initial_file_sizes)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall * 100, 1),
        response_quality=round(avg_quality * 100, 1),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print tabulated markdown output."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall (%)",
        "Response quality (%)",
        "Memory growth (bytes)",
        "Compactions",
    ]
    data = [
        [
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score}%",
            f"{r.response_quality}%",
            f"{r.memory_growth_bytes:,}",
            r.compactions,
        ]
        for r in rows
    ]
    if tabulate:
        return tabulate(data, headers=headers, tablefmt="github")

    # Fallback to simple markdown table
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = ["| " + " | ".join(str(c) for c in row) + " |" for row in data]
    return "\n".join([header_line, sep_line] + data_lines)


def run_benchmark_suite(suite_name: str, data_path: Path, config: LabConfig) -> list[BenchmarkRow]:
    """Run one benchmark suite comparing Baseline and Advanced agents."""
    conversations = load_conversations(data_path)

    # Clean profiles directory before running
    profiles_dir = config.state_dir / "profiles"
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline Agent
    baseline_agent = BaselineAgent(config, force_offline=True)
    baseline_row = run_agent_benchmark("Baseline Agent", baseline_agent, conversations, config)

    # Clean profiles before Advanced Agent run
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # 2. Advanced Agent
    advanced_agent = AdvancedAgent(config, force_offline=True)
    advanced_row = run_agent_benchmark("Advanced Agent", advanced_agent, conversations, config)

    rows = [baseline_row, advanced_row]
    print(f"\n### {suite_name}")
    print(format_rows(rows))
    return rows


def main() -> None:
    """Run both benchmark suites: Standard and Long-Context Stress."""
    config = load_config(Path(__file__).resolve().parent.parent)

    print("================================================================================")
    print("           BENCHMARKING MEMORY SYSTEMS FOR AI AGENT (DAY 17)")
    print("================================================================================")

    # Suite 1: Standard Benchmark
    standard_data = config.data_dir / "conversations.json"
    run_benchmark_suite("Standard Benchmark (data/conversations.json)", standard_data, config)

    # Suite 2: Long-Context Stress Benchmark
    stress_data = config.data_dir / "advanced_long_context.json"
    run_benchmark_suite("Long-Context Stress Benchmark (data/advanced_long_context.json)", stress_data, config)

    print("\n================================================================================")
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
