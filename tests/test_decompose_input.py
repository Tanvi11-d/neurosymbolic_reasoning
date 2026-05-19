"""Sub-process 1 DECOMPOSE — DecomposeInput and normalize_decompose_result only."""

import pytest

from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import DecomposeInput, normalize_decompose_result


def test_decompose_requires_callable():
    with pytest.raises(ValueError, match="llm_client.*decompose"):
        DecomposeInput()("", {}, {})


def test_decompose_passes_args_to_hook():
    seen: dict = {}

    def hook(iota, gamma, reg):
        seen["iota"] = iota
        seen["gamma"] = gamma
        seen["reg"] = reg
        return {"turns": [], "config": {}, "tools": {}}

    out = DecomposeInput(decompose=hook)("hello", {"a": 1}, {"t": 2})
    assert seen == {"iota": "hello", "gamma": {"a": 1}, "reg": {"t": 2}}
    assert out == {"turns": [], "config": {}, "tools": {}}


def test_decompose_preserves_unit_tasks():
    t = UnitTask("t1", "x", ("p",))

    def hook(_i, _g, _r):
        return {"turns": [t], "config": {"k": 1}, "tools": {"z": 3}}

    out = DecomposeInput(decompose=hook)("", {}, {})
    assert out["turns"] == [t]
    assert out["config"] == {"k": 1}
    assert out["tools"] == {"z": 3}


def test_decompose_coerces_dict_turns():
    def hook(_i, _g, _r):
        return {
            "turns": [
                {"id": "a", "text": "one", "depends_on": ["x"]},
                {"id": "b", "depends_on": ()},
            ],
            "config": {},
        }

    out = DecomposeInput(decompose=hook)("", {}, {})
    assert len(out["turns"]) == 2
    assert out["turns"][0] == UnitTask("a", "one", ("x",))
    assert out["turns"][1] == UnitTask("b", "", ())
    assert out["tools"] == {}


def test_decompose_optional_reasoning_string():
    def hook(_i, _g, _r):
        return {
            "turns": [],
            "config": {},
            "tools": {},
            "reasoning": "split into steps",
        }

    out = DecomposeInput(decompose=hook)("", {}, {})
    assert out["reasoning"] == "split into steps"


def test_decompose_drops_non_string_reasoning():
    def hook(_i, _g, _r):
        return {"turns": [], "config": {}, "tools": {}, "reasoning": 99}

    out = DecomposeInput(decompose=hook)("", {}, {})
    assert "reasoning" not in out


def test_normalize_not_dict():
    with pytest.raises(TypeError, match="dict"):
        normalize_decompose_result("bad")


def test_normalize_turns_not_list():
    with pytest.raises(TypeError, match="turns"):
        normalize_decompose_result({"turns": "nope", "config": {}, "tools": {}})


def test_normalize_turn_dict_bad_id():
    with pytest.raises(ValueError, match="id"):
        normalize_decompose_result({"turns": [{"id": ""}], "config": {}, "tools": {}})


def test_normalize_duplicate_task_ids():
    with pytest.raises(ValueError, match="duplicate"):
        normalize_decompose_result({
            "turns": [
                UnitTask("same"),
                UnitTask("same"),
            ],
            "config": {},
            "tools": {},
        })


def test_normalize_config_must_be_dict():
    with pytest.raises(TypeError, match="config"):
        normalize_decompose_result({"turns": [], "config": [], "tools": {}})


def test_normalize_tools_must_be_dict():
    with pytest.raises(TypeError, match="tools"):
        normalize_decompose_result({"turns": [], "config": {}, "tools":1})


def test_normalize_depends_on_wrong_type():
    with pytest.raises(TypeError, match="depends_on"):
        normalize_decompose_result({
            "turns": [{"id": "a", "depends_on": "bad"}],
            "config": {},
            "tools": {},
        })


def test_initialize_called_once():
    d = DecomposeInput(decompose=lambda *_: {"turns": [], "config": {}, "tools": {}})
    assert not d._initialized
    d("", {}, {})
    assert d._initialized
