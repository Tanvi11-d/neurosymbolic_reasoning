"""Layout: :mod:`nre.schemas`, :mod:`nre.llm` (transport), kernel LLM runners in :mod:`nre.library.hybrid` (re-exported from :mod:`nre.primitives.hybrid`)."""

from nre import schemas as canonical_schemas
from nre.utils import tool_registry


def test_schema_submodules_align_with_barrel():
    from nre.schemas import check as check_s
    from nre.schemas import decompose as decompose_s
    from nre.schemas import match as match_s
    from nre.schemas import order as order_s
    from nre.schemas import task as task_s

    assert task_s.LLMTaskItem is canonical_schemas.LLMTaskItem
    assert decompose_s.LLMDecomposeOutput is canonical_schemas.LLMDecomposeOutput
    assert check_s.LLMCheckRemediationOutput is canonical_schemas.LLMCheckRemediationOutput
    assert order_s.LLMSiblingOrderOutput is canonical_schemas.LLMSiblingOrderOutput
    assert match_s.LLMMatchFuncOutput is canonical_schemas.LLMMatchFuncOutput


def test_hybrid_defines_kernel_llm_runners():
    from nre.primitives.hybrid import (
        run_llm_decompose,
        run_llm_fill_missing_params,
        run_llm_match_func,
        run_llm_prereq_remediation,
        run_llm_sibling_order_batch,
    )

    assert callable(run_llm_decompose)
    assert callable(run_llm_prereq_remediation)
    assert callable(run_llm_sibling_order_batch)
    assert callable(run_llm_match_func)
    assert callable(run_llm_fill_missing_params)


def test_llm_package_is_transport_only():
    import nre.llm as llm

    assert hasattr(llm, "OpenRouterClient")
    assert hasattr(llm, "OpenRouterSettings")
    assert not hasattr(llm, "run_llm_decompose")


def test_tool_registry_helpers_public():
    assert callable(tool_registry.tool_registry_to_schemas)
    assert callable(tool_registry.format_tool_schemas)
    assert callable(tool_registry.tools_registry_metadata)


def test_llm_task_bridge_in_primitives():
    from nre.primitives import llm_task_item_to_unit_task
    from nre.primitives.llm_bridge import llm_task_item_to_unit_task as direct

    assert llm_task_item_to_unit_task is direct
