"""Normalize tool registries (Ω) into prompt-friendly schema text — shared by kernel LLM prompts."""

from __future__ import annotations

from typing import Any


def _merge_param_enums_from_entry(entry: dict[str, Any]) -> dict[str, list[Any]]:
    """Collect ``param_name -> allowed values`` from ``param_enums`` and param dicts."""
    merged: dict[str, list[Any]] = {}
    raw = entry.get("param_enums")
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, (list, tuple)) and v:
                merged[k] = list(v)
    params = entry.get("params") or entry.get("parameters") or []
    if isinstance(params, list):
        for p in params:
            if isinstance(p, dict):
                pname = p.get("name")
                en = p.get("enum")
                if pname and isinstance(en, (list, tuple)) and en:
                    merged[str(pname)] = list(en)
    return merged


def canonicalize_enum_value(value: Any, allowed: list[Any]) -> Any:
    """If *value* is a string and *allowed* has exactly one case-insensitive match, return it."""
    if value in allowed:
        return value
    if not isinstance(value, str):
        return value
    str_items = [a for a in allowed if isinstance(a, str)]
    if not str_items:
        return value
    matches = [a for a in str_items if a.lower() == value.lower()]
    if len(matches) == 1:
        return matches[0]
    return value


def canonicalize_bindings_with_registry(
    bindings: dict[str, Any],
    registry_entry: dict[str, Any],
) -> dict[str, Any]:
    """Map string params onto declared ``enum`` / ``param_enums`` (e.g. ``buy`` → ``Buy``)."""
    if not isinstance(bindings, dict) or not isinstance(registry_entry, dict):
        return dict(bindings)
    out = dict(bindings)
    for pname, allowed in _merge_param_enums_from_entry(registry_entry).items():
        if pname in out:
            out[pname] = canonicalize_enum_value(out[pname], allowed)
    return out


def tools_registry_metadata(tools_registry: dict[str, Any]) -> dict[str, Any]:
    """Strip callables and heavy objects; keep description + param names for LLM prompts."""
    out: dict[str, Any] = {}
    for name in sorted(tools_registry.keys()):
        entry = tools_registry[name]
        if not isinstance(entry, dict):
            continue
        params = entry.get("params") or entry.get("parameters") or []
        if not isinstance(params, list):
            params = []
        row: dict[str, Any] = {
            "description": str(entry.get("description", "") or ""),
            "params": [p if isinstance(p, str) else p.get("name", "") for p in params],
        }
        pe = _merge_param_enums_from_entry(entry)
        if pe:
            row["param_enums"] = pe
        out[name] = row
    return out


def tool_registry_to_schemas(tools_registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert registry dicts into structured schema rows (name, description, parameters[*])."""
    schemas: list[dict[str, Any]] = []
    for name in sorted(tools_registry.keys()):
        entry = tools_registry[name]
        if not isinstance(entry, dict):
            continue
        params = entry.get("params") or entry.get("parameters") or []
        if not isinstance(params, list):
            params = []
        param_objs: list[dict[str, Any]] = []
        for p in params:
            if isinstance(p, str):
                param_objs.append({"name": p, "type": "Any", "required": True})
            elif isinstance(p, dict) and "name" in p:
                param_objs.append(dict(p))
        pe_all = _merge_param_enums_from_entry(entry)
        pdescs = entry.get("param_descriptions") or {}
        for obj in param_objs:
            pname = obj.get("name")
            if pname and pname in pe_all:
                obj["enum"] = pe_all[pname]
            if pname and isinstance(pdescs, dict) and pname in pdescs:
                obj["description"] = pdescs[pname]
        row: dict[str, Any] = {
            "name": name,
            "description": entry.get("description", ""),
            "parameters": param_objs,
        }
        resp_props = entry.get("response_properties")
        if isinstance(resp_props, dict) and resp_props:
            row["response_properties"] = resp_props
        schemas.append(row)
    return schemas


def format_tool_schemas(tool_schemas: list[dict[str, Any]]) -> str:
    """Human-readable bullet list: ``name(p*: type) — description``."""
    lines: list[str] = []
    for schema in tool_schemas:
        name = schema["name"]
        desc = schema.get("description", "")
        parts: list[str] = []
        param_desc_notes: list[str] = []
        for p in schema.get("parameters", []):
            pname = p["name"]
            ptype = p.get("type", "Any")
            req = "*" if p.get("required", True) else ""
            seg = f"{pname}{req}: {ptype}"
            enum_vals = p.get("enum")
            if isinstance(enum_vals, (list, tuple)) and enum_vals:
                shown = enum_vals[:40]
                ev = ", ".join(
                    repr(x) if isinstance(x, str) else str(x) for x in shown
                )
                if len(enum_vals) > 40:
                    ev += ", …"
                seg += f" — allowed: {ev}"
            parts.append(seg)
            pdesc = p.get("description")
            if isinstance(pdesc, str) and pdesc.strip():
                param_desc_notes.append(f"    - {pname}: {pdesc.strip()}")
        sig = f"{name}({', '.join(parts)})"
        lines.append(f"- {sig} — {desc}" if desc else f"- {sig}")
        if param_desc_notes:
            lines.append("  parameters:")
            lines.extend(param_desc_notes)
        resp_props = schema.get("response_properties")
        if isinstance(resp_props, dict) and resp_props:
            field_parts: list[str] = []
            for field, spec in resp_props.items():
                if isinstance(spec, dict):
                    fdesc = spec.get("description")
                    if isinstance(fdesc, str) and fdesc.strip():
                        field_parts.append(f"{field} ({fdesc.strip()})")
                        continue
                field_parts.append(str(field))
            lines.append(f"  returns: {', '.join(field_parts)}")
    return "\n".join(lines)
