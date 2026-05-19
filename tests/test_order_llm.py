"""GET-ORDER LLM path: sibling batches, permutation validation, fallback."""

from unittest.mock import MagicMock

from nre.schemas import LLMSiblingOrderOutput
from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import OrderUnitTasks


def test_order_unit_tasks_llm_reorders_parallel_siblings():
    t2 = UnitTask("t2", text="Prefer run second in narrative", depends_on=())
    t1 = UnitTask("t1", text="Prefer run first", depends_on=())
    client = MagicMock()
    client.chat.return_value = LLMSiblingOrderOutput(
        ordered_task_ids=["t2", "t1"],
        reasoning="user asked for B before A",
    )
    out = OrderUnitTasks(llm_client=client).forward([t2, t1])
    assert out["has_cycle"] is False
    assert out["ordered"] == ["t2", "t1"]
    assert out["used_llm_sibling_order"] is True
    client.chat.assert_called_once()
    call_kw = client.chat.call_args.kwargs
    assert call_kw["response_model"] is LLMSiblingOrderOutput


def test_order_unit_tasks_invalid_llm_perm_falls_back_sorted():
    t1 = UnitTask("t1", depends_on=())
    t2 = UnitTask("t2", depends_on=())
    client = MagicMock()
    client.chat.return_value = LLMSiblingOrderOutput(ordered_task_ids=["t2"])
    out = OrderUnitTasks(llm_client=client).forward([t1, t2])
    assert out["ordered"] == ["t1", "t2"]
    assert out["used_llm_sibling_order"] is False


def test_order_unit_tasks_llm_skipped_for_single_node_batch():
    t1 = UnitTask("t1", depends_on=())
    t2 = UnitTask("t2", depends_on=("t1",))
    client = MagicMock()
    out = OrderUnitTasks(llm_client=client).forward([t1, t2])
    assert out["ordered"] == ["t1", "t2"]
    client.chat.assert_not_called()
    assert out["used_llm_sibling_order"] is False


def test_order_unit_tasks_llm_two_waves_second_batch_only():
    """First wave one node; second wave two parallel — one LLM call."""
    t1 = UnitTask("t1", depends_on=())
    tb = UnitTask("tb", depends_on=("t1",))
    ta = UnitTask("ta", depends_on=("t1",))
    client = MagicMock()

    def chat_side_effect(_messages, *, response_model=None, **kwargs):
        return LLMSiblingOrderOutput(ordered_task_ids=["tb", "ta"])

    client.chat.side_effect = chat_side_effect
    out = OrderUnitTasks(llm_client=client).forward([tb, t1, ta])
    assert out["ordered"] == ["t1", "tb", "ta"]
    assert out["used_llm_sibling_order"] is True
    assert client.chat.call_count == 1
