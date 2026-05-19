"""``Primitive.log`` / ``printf`` and request-scoped :func:`primitive_log_session`."""

from __future__ import annotations

from nre.primitives import InitScratchpad, format_primitive_log_for_api, primitive_log_session


def test_primitive_log_buffered_inside_session_only() -> None:
    p = InitScratchpad()
    p.log.info("hello %s", "world")
    # No session: no buffer (get returns None implicitly via append no-op)
    from nre.primitives.log_buffer import get_primitive_log_buffer

    assert get_primitive_log_buffer() is None

    with primitive_log_session() as buf:
        p.log.debug("dbg")
        p.printf("a", "b", end="")
        assert len(buf) == 2
        assert buf[0]["level"] == "DEBUG"
        assert "dbg" in buf[0]["data"]["message"]
        assert buf[0]["ts_iso"].endswith("Z")
        assert buf[1]["level"] == "INFO"
        assert "a b" in buf[1]["data"]["message"]

        shaped = format_primitive_log_for_api(buf)
        assert shaped["schema_version"] == 1
        assert len(shaped["events"]) == 2
        assert shaped["events"] == buf
        names = {g["primitive"] for g in shaped["by_primitive"]}
        assert names == {"InitScratchpad"}

    assert get_primitive_log_buffer() is None
