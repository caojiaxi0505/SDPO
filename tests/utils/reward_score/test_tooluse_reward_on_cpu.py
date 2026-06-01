import importlib.util
from pathlib import Path


def load_tooluse_reward():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "verl" / "utils" / "reward_score" / "feedback" / "tooluse.py"
    spec = importlib.util.spec_from_file_location("tooluse_reward", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tooluse_reward_accepts_non_word_action_names():
    reward = load_tooluse_reward()
    ground_truth = (
        '[{"Action":"math_toolkit.sum-of multiples","Action_Input":"{\\"limit\\":10,\\"base\\":3}"}]'
    )
    solution = (
        "Thought: use the function.\n"
        "Action: math_toolkit.sum-of multiples\n"
        'Action Input: {"limit":10,"base":3}'
    )

    result = reward.compute_score(solution, ground_truth)

    assert result["score"] == 1.0
    assert result["incorrect_format"] == 0


def test_tooluse_reward_preserves_duplicate_call_pairing():
    reward = load_tooluse_reward()
    ground_truth = (
        '[{"Action":"foo","Action_Input":"{\\"x\\":1}"},'
        '{"Action":"foo","Action_Input":"{\\"x\\":2}"}]'
    )
    wrong_solution = (
        'Action: foo\nAction Input: {"x":2}\n'
        'Action: foo\nAction Input: {"x":2}'
    )
    correct_solution = (
        'Action: foo\nAction Input: {"x":1}\n'
        'Action: foo\nAction Input: {"x":2}'
    )

    assert reward.compute_score(wrong_solution, ground_truth)["score"] == 0.0
    assert reward.compute_score(correct_solution, ground_truth)["score"] == 1.0


def test_tooluse_reward_handles_nested_json_inputs():
    reward = load_tooluse_reward()
    ground_truth = (
        '[{"Action":"nested_tool","Action_Input":"{\\"payload\\":{\\"a\\":1,\\"b\\":[2,3]}}"}]'
    )
    solution = 'Action: nested_tool\nAction Input: {"payload":{"a":1,"b":[2,3]}}'

    assert reward.compute_score(solution, ground_truth)["score"] == 1.0
