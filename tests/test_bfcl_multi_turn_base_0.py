from nre.examples.agentic_reasoner import build_turn_engine
from nre.primitives.deterministic import InitScratchpad, UnitTask
from nre.primitives.hybrid import (
    CheckPrerequisites,
    DecomposeInput,
    MatchFuncsAndParams,
    OrderUnitTasks,
)

IOTA_1 = (
    "Create a directory called 'archive' in workspace and move report.txt into it."
)
IOTA_2 = "Search the moved report for any mention of 'revenue'."


def _bfcl_turn1_decompose(iota: str, _gamma: dict, _tools: dict) -> dict:
    assert iota == IOTA_1
    return {
        "turns": [
            UnitTask("t1", "nav", ()),
            UnitTask("t2", "mkdir", ("t1",)),
            UnitTask("t3", "mv", ("t2",)),
        ],
        "config": {
            "cwd": "/",
            "cwd_is_workspace": False,
            "archive_exists": False,
        },
        "tools": {},
    }


def _bfcl_prereqs(task: UnitTask) -> list[str]:
    if task.id == "t1":
        return []
    if task.id == "t2":
        return ["cwd_is_workspace"]
    if task.id == "t3":
        return ["cwd_is_workspace", "archive_exists"]
    return []


def _bfcl_satisfied(_key: str, val: object, _s: dict) -> bool:
    return bool(val)


def _bfcl_match_turn1(task: UnitTask, _tools: dict) -> str | None:
    return {"t1": "cd", "t2": "mkdir", "t3": "mv"}.get(task.id)


def _make_turn1_tools(s: dict, log: list[dict]) -> dict:
    def cd(folder: str):
        log.append({"name": "cd", "kwargs": {"folder": folder}})
        s["cwd"] = "/workspace"
        s["cwd_is_workspace"] = True
        return {}

    def mkdir(dir_name: str):
        log.append({"name": "mkdir", "kwargs": {"dir_name": dir_name}})
        s["archive_exists"] = True
        return {}

    def mv(source: str, destination: str):
        log.append({"name": "mv", "kwargs": {"source": source, "destination": destination}})
        return {}

    return {
        "cd": {"params": ["folder"], "execute": cd},
        "mkdir": {"params": ["dir_name"], "execute": mkdir},
        "mv": {"params": ["source", "destination"], "execute": mv},
    }


def test_bfcl_turn1_golden_sequence():
    state: dict = {}
    log: list[dict] = []
    tools = _make_turn1_tools(state, log)
    tau = {
        "t1": {"folder": "workspace"},
        "t2": {"dir_name": "archive"},
        "t3": {"source": "report.txt", "destination": "archive"},
    }

    engine = build_turn_engine(
        decompose_input=DecomposeInput(decompose=_bfcl_turn1_decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(
            get_prereqs=_bfcl_prereqs,
            satisfied=_bfcl_satisfied,
        ),
        match_funcs=MatchFuncsAndParams(match_func=_bfcl_match_turn1),
    )
    out = engine.run(
        "agent_turn",
        iota=IOTA_1,
        gamma={},
        tools=tools,
        scratchpad_carry=state,
        turn_context_by_task_id=tau,
    )

    assert out["cycle"] is False
    assert out["ordered_task_ids"] == ["t1", "t2", "t3"]
    assert all(s.get("status") == "success" for s in out["steps"])
    assert state["cwd"] == "/workspace"
    assert state["cwd_is_workspace"] is True
    assert state["archive_exists"] is True

    assert log == [
        {"name": "cd", "kwargs": {"folder": "workspace"}},
        {"name": "mkdir", "kwargs": {"dir_name": "archive"}},
        {"name": "mv", "kwargs": {"source": "report.txt", "destination": "archive"}},
    ]


def _bfcl_turn2_decompose(iota: str, _gamma: dict, _tools: dict) -> dict:
    assert iota == IOTA_2
    return {
        "turns": [
            UnitTask("u1", "cd archive", ()),
            UnitTask("u2", "grep", ("u1",)),
        ],
        "config": {},
        "tools": {},
    }


def _bfcl_prereqs_turn2(task: UnitTask) -> list[str]:
    return []


def _bfcl_match_turn2(task: UnitTask, _tools: dict) -> str | None:
    return {"u1": "cd", "u2": "grep"}.get(task.id)


def _make_turn2_tools(s: dict, log: list[dict]) -> dict:
    def cd(folder: str):
        log.append({"name": "cd", "kwargs": {"folder": folder}})
        s["cwd"] = "/workspace/archive"
        return {}

    def grep(file_name: str, pattern: str):
        log.append({
            "name": "grep",
            "kwargs": {"file_name": file_name, "pattern": pattern},
        })
        return {"hits": 1}

    return {
        "cd": {"params": ["folder"], "execute": cd},
        "grep": {"params": ["file_name", "pattern"], "execute": grep},
    }


def test_bfcl_turn2_golden_sequence():
    state = {
        "cwd": "/workspace",
        "cwd_is_workspace": True,
        "archive_exists": True,
    }
    log: list[dict] = []
    tools = _make_turn2_tools(state, log)
    tau = {
        "u1": {"folder": "archive"},
        "u2": {"file_name": "report.txt", "pattern": "revenue"},
    }

    engine = build_turn_engine(
        decompose_input=DecomposeInput(decompose=_bfcl_turn2_decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(
            get_prereqs=_bfcl_prereqs_turn2,
            satisfied=_bfcl_satisfied,
        ),
        match_funcs=MatchFuncsAndParams(match_func=_bfcl_match_turn2),
    )
    out = engine.run(
        "agent_turn",
        iota=IOTA_2,
        gamma={},
        tools=tools,
        scratchpad_carry=state,
        turn_context_by_task_id=tau,
    )

    assert out["cycle"] is False
    assert out["ordered_task_ids"] == ["u1", "u2"]
    assert all(s.get("status") == "success" for s in out["steps"])

    assert log == [
        {"name": "cd", "kwargs": {"folder": "archive"}},
        {
            "name": "grep",
            "kwargs": {"file_name": "report.txt", "pattern": "revenue"},
        },
    ]


def test_bfcl_turn2_chained_from_turn1_state():
    state: dict = {}
    log1: list[dict] = []
    tools1 = _make_turn1_tools(state, log1)
    engine1 = build_turn_engine(
        decompose_input=DecomposeInput(decompose=_bfcl_turn1_decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(
            get_prereqs=_bfcl_prereqs,
            satisfied=_bfcl_satisfied,
        ),
        match_funcs=MatchFuncsAndParams(match_func=_bfcl_match_turn1),
    )
    engine1.run(
        "agent_turn",
        iota=IOTA_1,
        gamma={},
        tools=tools1,
        scratchpad_carry=state,
        turn_context_by_task_id={
            "t1": {"folder": "workspace"},
            "t2": {"dir_name": "archive"},
            "t3": {"source": "report.txt", "destination": "archive"},
        },
    )

    log2: list[dict] = []
    tools2 = _make_turn2_tools(state, log2)
    engine2 = build_turn_engine(
        decompose_input=DecomposeInput(decompose=_bfcl_turn2_decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(
            get_prereqs=_bfcl_prereqs_turn2,
            satisfied=_bfcl_satisfied,
        ),
        match_funcs=MatchFuncsAndParams(match_func=_bfcl_match_turn2),
    )
    engine2.run(
        "agent_turn",
        iota=IOTA_2,
        gamma={},
        tools=tools2,
        scratchpad_carry=state,
        turn_context_by_task_id={
            "u1": {"folder": "archive"},
            "u2": {"file_name": "report.txt", "pattern": "revenue"},
        },
    )

    assert log1[0]["name"] == "cd"
    assert log2[0]["kwargs"] == {"folder": "archive"}
