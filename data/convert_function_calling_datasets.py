from __future__ import annotations

import argparse
import ast
import json
import random
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
            "Action Input: the input to the action, must be in JSON format. All of the action input must be realistic and from the user.",
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
    skipped = 0
    for raw in raw_rows:
        source_id = f"apigen-{raw.get('id')}"
        try:
            tools = load_json_value(raw.get("tools"), "tools", source_id)
            calls = load_json_value(raw.get("answers"), "answers", source_id)
            if not isinstance(tools, list) or not isinstance(calls, list) or len(calls) == 0:
                skipped += 1
                continue
            prompt = make_prompt(str(raw["query"]), tools)
            ground_truth = calls_to_ground_truth(calls)
        except Exception:
            skipped += 1
            continue

        rows.append(
            make_verl_row(
                data_source="tooluse_apigen",
                prompt=prompt,
                ground_truth=ground_truth,
                source_dataset="apigen",
                source_id=source_id,
                split="train",
                num_tools=len(tools),
                num_calls=len(calls),
            )
        )

    train_rows, val_rows = split_rows(rows, val_ratio=val_ratio, seed=seed)
    write_jsonl(train_rows, root / "train.json")
    write_jsonl(val_rows, root / "test.json")
    print(f"apigen: converted={len(rows)} train={len(train_rows)} test={len(val_rows)} skipped={skipped}")


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
    skipped_items = 0
    skipped_turns = 0
    for item_idx, raw in enumerate(raw_rows):
        source_prefix = f"toolace-{item_idx}"
        try:
            tools = extract_toolace_tools(raw["system"])
            if not tools:
                skipped_items += 1
                continue
        except Exception:
            skipped_items += 1
            continue

        conversations = raw.get("conversations", [])
        if not isinstance(conversations, list):
            skipped_items += 1
            continue

        emitted_for_item = 0
        for turn_idx, turn in enumerate(conversations):
            if turn.get("from") != "assistant":
                continue
            calls = parse_toolace_calls(str(turn.get("value", "")))
            if not calls:
                continue
            context = conversations[:turn_idx]
            if not context or context[-1].get("from") != "user":
                skipped_turns += 1
                continue
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
                    split="train",
                    num_tools=len(tools),
                    num_calls=len(calls),
                )
            )
            emitted_for_item += 1
        if emitted_for_item == 0:
            skipped_turns += 1

    train_rows, val_rows = split_rows(rows, val_ratio=val_ratio, seed=seed)
    write_jsonl(train_rows, root / "train.json")
    write_jsonl(val_rows, root / "test.json")
    print(
        "toolace: "
        f"converted={len(rows)} train={len(train_rows)} test={len(val_rows)} "
        f"skipped_items={skipped_items} skipped_turns={skipped_turns}"
    )


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
