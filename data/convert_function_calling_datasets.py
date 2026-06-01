from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def make_prompt(query: str, tools: list[dict[str, Any]]) -> str:
    tool_docs = []
    for tool in tools:
        name = str(tool.get("name", "")).strip()
        description = str(tool.get("description", "")).strip()
        parameters = tool.get("parameters", {})
        tool_docs.append(
            "\n".join(
                [
                    f"Name: {name}",
                    f"Description: {description}",
                    f"Parameters: {pretty_json(parameters)}",
                    "Output: Successful response.",
                    " - Format: application/json",
                    " - Structure: Object",
                ]
            )
        )

    return "\n".join(
        [
            "Your task is to answer the user's question using available tools.",
            "You have access to the following tools:",
            "\n".join(tool_docs),
            "",
            "Use the following format:",
            "Thought: you should always think about what to do",
            "Action: the action to take, should be one of the function names.",
            "Action Input: the input to the action, must be in JSON format. All action inputs must be grounded in the user request or conversation context.",
            "",
            "Begin!",
            f"Question: {query}",
        ]
    )


def calls_to_ground_truth(calls: list[dict[str, Any]]) -> str:
    gt_calls = []
    for call in calls:
        gt_calls.append(
            {
                "Action": str(call["name"]).strip(),
                "Action_Input": pretty_json(call.get("arguments", {})),
            }
        )
    return pretty_json(gt_calls)


def make_verl_row(
    *,
    data_source: str,
    prompt: str,
    ground_truth: str,
    source_dataset: str,
    source_id: str,
    source_group: str,
    split: str,
    num_tools: int,
    num_calls: int,
) -> dict[str, Any]:
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "tooluse",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": source_id,
            "source_dataset": source_dataset,
            "source_id": source_id,
            "source_group": source_group,
            "num_tools": num_tools,
            "num_calls": num_calls,
        },
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dumps(row) + "\n")


def split_rows(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    val_size = int(round(len(rows) * val_ratio))
    if val_ratio > 0 and len(rows) > 1:
        val_size = max(1, val_size)
    val_indices = set(indices[:val_size])

    train_rows = []
    val_rows = []
    for idx, row in enumerate(rows):
        out = dict(row)
        out["extra_info"] = dict(row["extra_info"])
        if idx in val_indices:
            out["extra_info"]["split"] = "test"
            val_rows.append(out)
        else:
            out["extra_info"]["split"] = "train"
            train_rows.append(out)
    return train_rows, val_rows


def split_rows_by_group(
    rows: list[dict[str, Any]], val_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    groups = sorted({row["extra_info"]["source_group"] for row in rows})
    random.Random(seed).shuffle(groups)
    val_size = int(round(len(groups) * val_ratio))
    if val_ratio > 0 and len(groups) > 1:
        val_size = max(1, val_size)
    val_groups = set(groups[:val_size])

    train_rows = []
    val_rows = []
    for row in rows:
        out = dict(row)
        out["extra_info"] = dict(row["extra_info"])
        if row["extra_info"]["source_group"] in val_groups:
            out["extra_info"]["split"] = "test"
            val_rows.append(out)
        else:
            out["extra_info"]["split"] = "train"
            train_rows.append(out)
    return train_rows, val_rows


def update_stats(stats: Counter, prefix: str, reason: str) -> None:
    stats[f"{prefix}/{reason}"] += 1


def validate_calls(calls: Any, source_id: str) -> list[dict[str, Any]]:
    if not isinstance(calls, list):
        raise ValueError(f"{source_id}: answers must be a list")
    if len(calls) == 0:
        raise ValueError(f"{source_id}: answers must be non-empty")

    normalized = []
    for call_idx, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError(f"{source_id}: answer {call_idx} must be an object")
        name = str(call.get("name", "")).strip()
        if not name:
            raise ValueError(f"{source_id}: answer {call_idx} missing name")
        arguments = call.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError(f"{source_id}: answer {call_idx} arguments must be an object")
        normalized.append({"name": name, "arguments": arguments})
    return normalized


def validate_tools(tools: Any, source_id: str) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        raise ValueError(f"{source_id}: tools must be a list")
    if len(tools) == 0:
        raise ValueError(f"{source_id}: tools must be non-empty")
    for tool_idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"{source_id}: tool {tool_idx} must be an object")
        if not str(tool.get("name", "")).strip():
            raise ValueError(f"{source_id}: tool {tool_idx} missing name")
    return tools


def print_counter(title: str, stats: Counter) -> None:
    print(f"{title}:")
    for key in sorted(stats):
        print(f"  {key}={stats[key]}")


def lint_rows(rows: list[dict[str, Any]], dataset_name: str) -> None:
    stats = Counter()
    prompt_lengths = []
    num_calls = []
    bad_examples = []

    for row in rows:
        prompt = row["prompt"][0]["content"]
        prompt_lengths.append(len(prompt))
        try:
            gt_calls = json.loads(row["reward_model"]["ground_truth"])
        except Exception:
            update_stats(stats, "lint", "bad_ground_truth_json")
            continue
        if not isinstance(gt_calls, list) or len(gt_calls) == 0:
            update_stats(stats, "lint", "empty_ground_truth")
            continue

        num_calls.append(len(gt_calls))
        for call in gt_calls:
            action = call.get("Action")
            action_input = call.get("Action_Input")
            if f"Name: {action}" not in prompt:
                update_stats(stats, "lint", "action_not_in_prompt")
                if len(bad_examples) < 3:
                    bad_examples.append(action)
            try:
                parsed_input = json.loads(action_input) if isinstance(action_input, str) else action_input
            except Exception:
                update_stats(stats, "lint", "bad_action_input_json")
                continue
            if not isinstance(parsed_input, dict):
                update_stats(stats, "lint", "action_input_not_dict")

    if prompt_lengths:
        sorted_lengths = sorted(prompt_lengths)
        stats["lint/prompt_chars_min"] = sorted_lengths[0]
        stats["lint/prompt_chars_p50"] = sorted_lengths[len(sorted_lengths) // 2]
        stats["lint/prompt_chars_p95"] = sorted_lengths[int(0.95 * (len(sorted_lengths) - 1))]
        stats["lint/prompt_chars_max"] = sorted_lengths[-1]
    if num_calls:
        sorted_calls = sorted(num_calls)
        stats["lint/num_calls_min"] = sorted_calls[0]
        stats["lint/num_calls_p50"] = sorted_calls[len(sorted_calls) // 2]
        stats["lint/num_calls_max"] = sorted_calls[-1]

    print_counter(f"{dataset_name} lint", stats)
    if bad_examples:
        print(f"{dataset_name} lint/action_not_in_prompt_examples={bad_examples}")


def load_json_value(value: Any, field: str, source_id: str) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    if value is None:
        raise ValueError(f"{source_id}: missing {field}")
    return value


def convert_apigen(root: Path, val_ratio: float, seed: int) -> None:
    source_path = root / "xlam_function_calling_60k.json"
    with source_path.open(encoding="utf-8") as f:
        raw_rows = json.load(f)

    rows = []
    stats = Counter()
    for raw in raw_rows:
        source_id = f"apigen-{raw.get('id')}"
        try:
            if "query" not in raw or not str(raw.get("query", "")).strip():
                raise ValueError(f"{source_id}: missing query")
            tools = load_json_value(raw.get("tools"), "tools", source_id)
            calls = load_json_value(raw.get("answers"), "answers", source_id)
            tools = validate_tools(tools, source_id)
            calls = validate_calls(calls, source_id)
            prompt = make_prompt(str(raw["query"]), tools)
            ground_truth = calls_to_ground_truth(calls)
        except json.JSONDecodeError:
            update_stats(stats, "skip", "json_parse_error")
            continue
        except ValueError as exc:
            update_stats(stats, "skip", str(exc).split(": ", 1)[-1].replace(" ", "_"))
            continue
        except Exception:
            update_stats(stats, "skip", "unexpected_error")
            continue

        rows.append(
            make_verl_row(
                data_source="tooluse_apigen",
                prompt=prompt,
                ground_truth=ground_truth,
                source_dataset="apigen",
                source_id=source_id,
                source_group=source_id,
                split="train",
                num_tools=len(tools),
                num_calls=len(calls),
            )
        )
        update_stats(stats, "keep", "rows")

    train_rows, val_rows = split_rows(rows, val_ratio=val_ratio, seed=seed)
    lint_rows(rows, "apigen")
    write_jsonl(train_rows, root / "train.json")
    write_jsonl(val_rows, root / "test.json")
    print(f"apigen: converted={len(rows)} train={len(train_rows)} test={len(val_rows)} skipped={sum(v for k, v in stats.items() if k.startswith('skip/'))}")
    print_counter("apigen stats", stats)


def find_matching(text: str, start: int, open_ch: str, close_ch: str) -> int | None:
    depth = 0
    in_string = False
    quote = ""
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue
        if ch in {"'", '"'}:
            in_string = True
            quote = ch
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return idx
    return None


def extract_toolace_tools(system: str) -> list[dict[str, Any]] | None:
    marker = "Here is a list of functions in JSON format that you can invoke:"
    marker_idx = system.find(marker)
    search_start = marker_idx + len(marker) if marker_idx >= 0 else 0
    start = system.find("[", search_start)
    if start < 0:
        return None
    end = find_matching(system, start, "[", "]")
    if end is None:
        return None
    tools = json.loads(system[start : end + 1])
    return tools if isinstance(tools, list) else None


def split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts = []
    start = 0
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    in_string = False
    quote = ""
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue
        if ch in {"'", '"'}:
            in_string = True
            quote = ch
        elif ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
        elif (
            ch == delimiter
            and paren_depth == 0
            and bracket_depth == 0
            and brace_depth == 0
        ):
            part = text[start:idx].strip()
            if part:
                parts.append(part)
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def parse_value(value: str) -> Any:
    stripped = value.strip()
    if stripped in {"true", "false", "null"}:
        return {"true": True, "false": False, "null": None}[stripped]
    try:
        return ast.literal_eval(stripped)
    except Exception:
        try:
            return json.loads(stripped)
        except Exception:
            return stripped


def parse_toolace_calls(text: str) -> list[dict[str, Any]] | None:
    stripped = text.strip()
    if not stripped.startswith("["):
        return None
    end = find_matching(stripped, 0, "[", "]")
    if end is None or stripped[end + 1 :].strip():
        return None
    inner = stripped[1:end].strip()
    if not inner:
        return []

    calls = []
    for call_text in split_top_level(inner):
        open_idx = call_text.find("(")
        if open_idx < 0:
            return None
        close_idx = find_matching(call_text, open_idx, "(", ")")
        if close_idx is None or call_text[close_idx + 1 :].strip():
            return None
        name = call_text[:open_idx].strip()
        if not name:
            return None
        args_text = call_text[open_idx + 1 : close_idx].strip()
        arguments = {}
        if args_text:
            for arg_idx, arg_text in enumerate(split_top_level(args_text)):
                if "=" in arg_text:
                    key, value = arg_text.split("=", 1)
                    key = key.strip()
                else:
                    key = f"arg{arg_idx}"
                    value = arg_text
                if not key:
                    return None
                arguments[key] = parse_value(value)
        calls.append({"name": name, "arguments": arguments})
    return calls


def render_toolace_context(conversations: list[dict[str, str]]) -> str:
    if len(conversations) == 1 and conversations[0].get("from") == "user":
        return str(conversations[0].get("value", ""))

    chunks = ["Conversation so far:"]
    role_names = {"user": "User", "assistant": "Assistant", "tool": "Tool"}
    for turn in conversations:
        role = role_names.get(turn.get("from"), str(turn.get("from", "unknown")).title())
        chunks.append(f"{role}: {turn.get('value', '')}")
    chunks.append("Respond with the next tool call for the latest user request.")
    return "\n".join(chunks)


def convert_toolace(root: Path, val_ratio: float, seed: int) -> None:
    source_path = root / "data.json"
    with source_path.open(encoding="utf-8") as f:
        raw_rows = json.load(f)

    rows = []
    stats = Counter()
    for item_idx, raw in enumerate(raw_rows):
        source_prefix = f"toolace-{item_idx}"
        try:
            tools = extract_toolace_tools(raw["system"])
            if not tools:
                update_stats(stats, "skip_item", "missing_tools")
                continue
            tools = validate_tools(tools, source_prefix)
        except Exception:
            update_stats(stats, "skip_item", "tool_parse_error")
            continue

        conversations = raw.get("conversations", [])
        if not isinstance(conversations, list):
            update_stats(stats, "skip_item", "bad_conversations")
            continue

        emitted_for_item = 0
        for turn_idx, turn in enumerate(conversations):
            if turn.get("from") != "assistant":
                continue
            update_stats(stats, "turn", "assistant_total")
            calls = parse_toolace_calls(str(turn.get("value", "")))
            if not calls:
                if str(turn.get("value", "")).strip().startswith("["):
                    update_stats(stats, "turn", "parser_failed_toollike")
                continue
            update_stats(stats, "turn", "assistant_call_total")
            context = conversations[:turn_idx]
            if not context or context[-1].get("from") != "user":
                prev_role = "none" if not context else str(context[-1].get("from", "unknown"))
                update_stats(stats, "skip_turn", f"prev_{prev_role}")
                continue
            update_stats(stats, "turn", "kept_prev_user")
            source_id = f"{source_prefix}-turn{turn_idx}"
            prompt = make_prompt(render_toolace_context(context), tools)
            ground_truth = calls_to_ground_truth(calls)
            rows.append(
                make_verl_row(
                    data_source="tooluse_toolace",
                    prompt=prompt,
                    ground_truth=ground_truth,
                    source_dataset="toolace",
                    source_id=source_id,
                    source_group=source_prefix,
                    split="train",
                    num_tools=len(tools),
                    num_calls=len(calls),
                )
            )
            emitted_for_item += 1
        if emitted_for_item == 0:
            update_stats(stats, "skip_item", "no_kept_turns")

    train_rows, val_rows = split_rows_by_group(rows, val_ratio=val_ratio, seed=seed)
    train_groups = {row["extra_info"]["source_group"] for row in train_rows}
    val_groups = {row["extra_info"]["source_group"] for row in val_rows}
    overlap = train_groups & val_groups
    if overlap:
        raise ValueError(f"ToolACE group split leaked {len(overlap)} groups")
    lint_rows(rows, "toolace")
    write_jsonl(train_rows, root / "train.json")
    write_jsonl(val_rows, root / "test.json")
    print(
        "toolace: "
        f"converted={len(rows)} train={len(train_rows)} test={len(val_rows)} "
        f"train_groups={len(train_groups)} test_groups={len(val_groups)}"
    )
    print_counter("toolace stats", stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert APIGen and ToolACE into verl RLHF JSON schema.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "apigen", "toolace"],
        help="Datasets to convert.",
    )
    parser.add_argument("--base-dir", type=Path, default=Path("datasets"), help="Directory containing dataset folders.")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Deterministic validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic train/test split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = {"apigen", "toolace"} if "all" in args.datasets else set(args.datasets)
    if "apigen" in selected:
        convert_apigen(args.base_dir / "apigen", val_ratio=args.val_ratio, seed=args.seed)
    if "toolace" in selected:
        convert_toolace(args.base_dir / "toolace", val_ratio=args.val_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
