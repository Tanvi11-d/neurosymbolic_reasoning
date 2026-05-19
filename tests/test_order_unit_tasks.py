"""GET-ORDER (OrderUnitTasks): DAG topo, cycle detection, deterministic sibling order."""

import pytest

from nre.library.deterministic import topological_levels
from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import OrderUnitTasks


def test_chain_linear_order():
    t1 = UnitTask("t1", depends_on=())
    t2 = UnitTask("t2", depends_on=("t1",))
    t3 = UnitTask("t3", depends_on=("t2",))
    out = OrderUnitTasks().forward([t3, t1, t2])  # shuffled input
    assert out["has_cycle"] is False
    assert out["ordered"] == ["t1", "t2", "t3"]
    assert out.get("used_llm_sibling_order") is False


def test_parallel_roots_sorted_by_id():
    """Independent tasks: topo uses sorted queue → lexicographic id order."""
    tb = UnitTask("tb", depends_on=())
    ta = UnitTask("ta", depends_on=())
    out = OrderUnitTasks().forward([tb, ta])
    assert out["has_cycle"] is False
    assert out["ordered"] == ["ta", "tb"]


def test_fan_out_after_common_parent():
    """t2 and t3 both depend on t1; siblings ordered by id."""
    t1 = UnitTask("t1", depends_on=())
    t3 = UnitTask("t3", depends_on=("t1",))
    t2 = UnitTask("t2", depends_on=("t1",))
    out = OrderUnitTasks().forward([t3, t1, t2])
    assert out["has_cycle"] is False
    assert out["ordered"] == ["t1", "t2", "t3"]


def test_cycle_detected():
    t1 = UnitTask("t1", depends_on=("t2",))
    t2 = UnitTask("t2", depends_on=("t1",))
    out = OrderUnitTasks().forward([t1, t2])
    assert out["has_cycle"] is True
    assert out["ordered"] == []


def test_topological_levels_fan_out():
    graph = {"t1": ["t2", "t3"], "t2": [], "t3": []}
    assert topological_levels(graph) == [["t1"], ["t2", "t3"]]


def test_topological_levels_cycle_raises():
    graph = {"a": ["b"], "b": ["a"]}
    with pytest.raises(RuntimeError, match="cycle"):
        topological_levels(graph)
