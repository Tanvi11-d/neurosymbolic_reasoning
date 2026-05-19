"""Stable import path for hybrid kernel exports.

:class:`HybridPrimitive` is defined in :mod:`nre.primitives.base`.
Concrete kernel stages and ``run_llm_*`` helpers live in :mod:`nre.library.hybrid`.
"""

from __future__ import annotations

from nre.library.hybrid import (
    CHECK_REMEDIATION_SYSTEM_PROMPT,
    DECOMPOSE_SYSTEM_PROMPT,
    MATCH_FILL_SYSTEM_PROMPT,
    MATCH_FUNC_SYSTEM_PROMPT,
    ORDER_SIBLINGS_SYSTEM_PROMPT,
    CheckPrerequisites,
    DecomposeInput,
    MatchFuncsAndParams,
    OrderUnitTasks,
    PolicyConstraints,
    build_check_remediation_user_prompt,
    build_decompose_user_prompt,
    build_match_func_user_prompt,
    build_match_fill_user_prompt,
    build_sibling_order_user_prompt,
    normalize_decompose_result,
    run_llm_decompose,
    run_llm_fill_missing_params,
    run_llm_match_func,
    run_llm_prereq_remediation,
    run_llm_sibling_order_batch,
    validate_sibling_permutation,
)
from nre.primitives.base import HybridPrimitive

__all__ = [
    "HybridPrimitive",
    "normalize_decompose_result",
    "DECOMPOSE_SYSTEM_PROMPT",
    "build_decompose_user_prompt",
    "run_llm_decompose",
    "CHECK_REMEDIATION_SYSTEM_PROMPT",
    "build_check_remediation_user_prompt",
    "run_llm_prereq_remediation",
    "ORDER_SIBLINGS_SYSTEM_PROMPT",
    "build_sibling_order_user_prompt",
    "run_llm_sibling_order_batch",
    "validate_sibling_permutation",
    "MATCH_FUNC_SYSTEM_PROMPT",
    "MATCH_FILL_SYSTEM_PROMPT",
    "build_match_func_user_prompt",
    "run_llm_match_func",
    "build_match_fill_user_prompt",
    "run_llm_fill_missing_params",
    "OrderUnitTasks",
    "CheckPrerequisites",
    "DecomposeInput",
    "MatchFuncsAndParams",
    "PolicyConstraints",
]
