"""P3 — γ → S deep-flatten inside ``InitScratchpad`` (spec Sub-process 1 ``Extract(γ)``).

The spec has DECOMPOSE emit ``config ← Extract(γ)`` which then seeds ``S`` in
INIT-SCRATCHPAD. In the shipping implementation ``DECOMPOSE`` (LLM) almost never
returns non-empty ``config``, so ``S = ∅`` for every task — the LOOKUP priority
chain has no scratchpad to read from.

This test suite pins the behavior we want: ``InitScratchpad`` accepts an
optional ``gamma`` argument and deep-walks it into dot-path keys. Those keys
join ``config`` in the returned seed dict, with ``config`` winning on
collisions (spec: ``config`` is the DECOMPOSE-authoritative output).

INV-S is preserved: we only add keys.
"""

from __future__ import annotations

from nre.primitives.deterministic import InitScratchpad


# ── Deep flatten: dict ───────────────────────────────────────────────


def test_gamma_flat_dict_becomes_dot_keys():
    g = {"user": "alice", "count": 3}
    seed = InitScratchpad()({}, gamma=g)
    assert seed["user"] == "alice"
    assert seed["count"] == 3


def test_gamma_nested_dict_becomes_dot_paths():
    g = {"vehicle": {"fuelLevel": 10.5, "engineState": "stopped"}}
    seed = InitScratchpad()({}, gamma=g)
    assert seed["vehicle.fuelLevel"] == 10.5
    assert seed["vehicle.engineState"] == "stopped"


def test_gamma_deep_nested_preserved():
    g = {"fs": {"root": {"data": {"project": {"file.txt": "content"}}}}}
    seed = InitScratchpad()({}, gamma=g)
    assert seed["fs.root.data.project.file.txt"] == "content"


# ── Deep flatten: lists ──────────────────────────────────────────────


def test_gamma_list_of_scalars_indexed():
    g = {"tags": ["a", "b", "c"]}
    seed = InitScratchpad()({}, gamma=g)
    assert seed["tags[0]"] == "a"
    assert seed["tags[1]"] == "b"
    assert seed["tags[2]"] == "c"


def test_gamma_list_of_dicts():
    g = {"tweets": [{"id": 0, "text": "hi"}, {"id": 1, "text": "bye"}]}
    seed = InitScratchpad()({}, gamma=g)
    assert seed["tweets[0].id"] == 0
    assert seed["tweets[0].text"] == "hi"
    assert seed["tweets[1].id"] == 1
    assert seed["tweets[1].text"] == "bye"


# ── config wins on collision ─────────────────────────────────────────


def test_config_overrides_gamma_on_key_collision():
    """Spec: DECOMPOSE's ``config`` is the authoritative seed — γ fills gaps only."""
    g = {"user": "from_gamma"}
    c = {"user": "from_config"}
    seed = InitScratchpad()(c, gamma=g)
    assert seed["user"] == "from_config"


def test_gamma_fills_keys_config_does_not_supply():
    g = {"a": 1, "b": 2}
    c = {"c": 3}
    seed = InitScratchpad()(c, gamma=g)
    assert seed == {"a": 1, "b": 2, "c": 3}


# ── Backward compatibility ───────────────────────────────────────────


def test_no_gamma_arg_equals_old_behavior():
    """Existing callers pass only ``config``; behavior must be unchanged."""
    seed = InitScratchpad()({"k": "v"})
    assert seed == {"k": "v"}


def test_gamma_none_equals_no_gamma():
    seed = InitScratchpad()({"k": "v"}, gamma=None)
    assert seed == {"k": "v"}


def test_empty_gamma_equals_no_gamma():
    seed = InitScratchpad()({"k": "v"}, gamma={})
    assert seed == {"k": "v"}


# ── Edge cases ───────────────────────────────────────────────────────


def test_gamma_with_none_values_preserved():
    """``None`` in γ is a valid value (distinct from absent). Must survive flatten."""
    g = {"cfg": {"nullable": None, "set": "v"}}
    seed = InitScratchpad()({}, gamma=g)
    assert "cfg.nullable" in seed
    assert seed["cfg.nullable"] is None
    assert seed["cfg.set"] == "v"


def test_gamma_empty_dict_inner_not_emitted():
    """Empty inner dicts create no keys (no placeholder)."""
    g = {"outer": {}}
    seed = InitScratchpad()({}, gamma=g)
    assert seed == {}


def test_gamma_empty_list_inner_not_emitted():
    g = {"tags": []}
    seed = InitScratchpad()({}, gamma=g)
    assert seed == {}


def test_bfcl_style_filesystem_gamma():
    """Realistic fragment from the failing BFCL cases — GorillaFileSystem tree."""
    g = {
        "GorillaFileSystem": {
            "root": {
                "data": {
                    "type": "directory",
                    "contents": {
                        "project": {
                            "type": "directory",
                            "contents": {
                                "analysis_report.csv": {
                                    "type": "file",
                                    "content": "Data analysis results...",
                                },
                            },
                        },
                    },
                },
            },
        }
    }
    seed = InitScratchpad()({}, gamma=g)
    assert (
        seed[
            "GorillaFileSystem.root.data.contents.project.contents.analysis_report.csv.content"
        ]
        == "Data analysis results..."
    )
    assert (
        seed[
            "GorillaFileSystem.root.data.contents.project.contents.analysis_report.csv.type"
        ]
        == "file"
    )


def test_bfcl_style_vehicle_gamma():
    """VehicleControlAPI fragment from case 58/92."""
    g = {
        "VehicleControlAPI": {
            "fuelLevel": 0.0,
            "engineState": "stopped",
            "doorStatus": {"driver": "unlocked", "passenger": "unlocked"},
            "destination": "San Francisco",
        }
    }
    seed = InitScratchpad()({}, gamma=g)
    assert seed["VehicleControlAPI.fuelLevel"] == 0.0
    assert seed["VehicleControlAPI.engineState"] == "stopped"
    assert seed["VehicleControlAPI.doorStatus.driver"] == "unlocked"
    assert seed["VehicleControlAPI.destination"] == "San Francisco"
