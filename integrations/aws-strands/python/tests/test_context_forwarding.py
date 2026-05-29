"""RunAgentInput.context must reach per-thread Strands agent state AND the model.

Two channels:
  1. ``strands_agent.state["agui_context"]`` — for tools that read context at
     runtime (mirrors the LangGraph integration).
  2. The outgoing user message sent to the LLM — appended as a readable
     "Context:" block so the model sees catalog schemas, usage guidelines, etc.

Covers:
  - ``_format_agui_context_text`` formatting helper (unit tests)
  - Context injected into user message on the legacy path (replay=False)
  - Context injected into native history on the replay path (replay=True)
  - Multimodal messages (list content) are not modified
  - Empty context → nothing injected
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from strands import Agent
from strands.agent.state import AgentState
from strands.tools.registry import ToolRegistry

from ag_ui.core import (
    Context,
    ImageInputContent,
    InputContentUrlSource,
    RunAgentInput,
    UserMessage,
)

try:
    from strands.types.json_dict import JSONSerializableDict  # strands <2.0
except ImportError:
    try:
        from strands.types import JSONSerializableDict  # strands >=2.0 (reorganized)
    except ImportError:
        class JSONSerializableDict(dict):  # type: ignore[no-redef]
            def set(self, key, value): self[key] = value  # noqa: E704

from ag_ui_strands.agent import StrandsAgent, _format_agui_context_text
from ag_ui_strands.config import StrandsAgentConfig


def _mock_model():
    m = MagicMock()
    m.stateful = False
    return m


class _CapturingCore:
    """Stand-in for StrandsAgentCore that records state writes and stream_async args."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.tool_registry = ToolRegistry()
        self.state = AgentState()
        self.messages = []           # Replay path sets this before stream_async(None)
        self.stream_async_msg = None  # Records what stream_async received

    async def stream_async(self, msg):
        self.stream_async_msg = msg
        if False:
            yield


def _run_input(context, thread_id="t-ctx"):
    return RunAgentInput(
        thread_id=thread_id,
        run_id="r1",
        state={},
        messages=[UserMessage(id="u1", content="hello")],
        tools=[],
        context=context,
        forwarded_props={},
    )


def _run_input_multimodal(context, thread_id="t-ctx-mm"):
    img = ImageInputContent(
        source=InputContentUrlSource(type="url", value="https://example.com/img.png")
    )
    return RunAgentInput(
        thread_id=thread_id,
        run_id="r1",
        state={},
        messages=[UserMessage(id="u1", content=[img])],
        tools=[],
        context=context,
        forwarded_props={},
    )


async def _drive_one_event(ag: StrandsAgent, run_input: RunAgentInput) -> _CapturingCore:
    async for _ in ag.run(run_input):
        break
    return ag._agents_by_thread[run_input.thread_id]


async def _run_to_completion(ag: StrandsAgent, run_input: RunAgentInput) -> _CapturingCore:
    """Consume all events so context injection (post-RunStarted) actually runs."""
    async for _ in ag.run(run_input):
        pass
    return ag._agents_by_thread[run_input.thread_id]


# ---------------------------------------------------------------------------
# Unit tests for _format_agui_context_text
# ---------------------------------------------------------------------------

class TestFormatAguiContextText:
    def test_empty_list_returns_empty_string(self):
        assert _format_agui_context_text([]) == ""

    def test_none_returns_empty_string(self):
        assert _format_agui_context_text(None) == ""

    def test_single_entry_with_description_and_value(self):
        ctx = [Context(description="catalog", value='{"items":[]}')]
        assert _format_agui_context_text(ctx) == 'Context:\ncatalog: {"items":[]}'

    def test_description_only(self):
        ctx = [Context(description="use render_a2ui to render UI", value="")]
        assert _format_agui_context_text(ctx) == "Context:\nuse render_a2ui to render UI"

    def test_value_only(self):
        ctx = [Context(description="", value="some schema")]
        assert _format_agui_context_text(ctx) == "Context:\nsome schema"

    def test_both_blank_entry_skipped(self):
        ctx = [Context(description="", value="")]
        assert _format_agui_context_text(ctx) == ""

    def test_blank_entries_among_valid_entries_are_skipped(self):
        ctx = [
            Context(description="", value=""),
            Context(description="key", value="val"),
        ]
        assert _format_agui_context_text(ctx) == "Context:\nkey: val"

    def test_multiple_entries(self):
        ctx = [
            Context(description="schema", value='{"x":1}'),
            Context(description="guidelines", value="be helpful"),
        ]
        result = _format_agui_context_text(ctx)
        assert result == 'Context:\nschema: {"x":1}\nguidelines: be helpful'

    def test_dict_entries_supported(self):
        ctx = [{"description": "foo", "value": "bar"}]
        assert _format_agui_context_text(ctx) == "Context:\nfoo: bar"

    def test_dict_missing_fields_handled(self):
        ctx = [{"description": "only-desc"}]
        assert _format_agui_context_text(ctx) == "Context:\nonly-desc"


# ---------------------------------------------------------------------------
# State forwarding (pre-existing behaviour, now using extended _CapturingCore)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_forwarded_to_agent_state():
    template = Agent(model=_mock_model())
    ag = StrandsAgent(template, name="test")

    ctx = [
        Context(description="catalog", value='{"items":["a","b"]}'),
        Context(description="user_id", value="u-42"),
    ]

    with patch("ag_ui_strands.agent.StrandsAgentCore", _CapturingCore):
        instance = await _drive_one_event(ag, _run_input(ctx))

    stored = instance.state.get("agui_context")
    assert stored == [
        {"description": "catalog", "value": '{"items":["a","b"]}'},
        {"description": "user_id", "value": "u-42"},
    ], f"expected context forwarded to state, got {stored!r}"


@pytest.mark.asyncio
async def test_empty_context_writes_empty_list():
    template = Agent(model=_mock_model())
    ag = StrandsAgent(template, name="test")

    with patch("ag_ui_strands.agent.StrandsAgentCore", _CapturingCore):
        instance = await _drive_one_event(ag, _run_input([]))

    assert instance.state.get("agui_context") == []


# ---------------------------------------------------------------------------
# Message injection — legacy path (replay_history_into_strands=False)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_appended_to_user_message_legacy_path():
    """On the legacy path context text is appended to the stream_async message arg."""
    template = Agent(model=_mock_model())
    config = StrandsAgentConfig(replay_history_into_strands=False)
    ag = StrandsAgent(template, name="test", config=config)

    ctx = [Context(description="catalog", value="my-catalog")]

    with patch("ag_ui_strands.agent.StrandsAgentCore", _CapturingCore):
        instance = await _run_to_completion(ag, _run_input(ctx, thread_id="t-legacy"))

    msg = instance.stream_async_msg
    assert isinstance(msg, str), f"expected str, got {type(msg)}"
    assert "Context:" in msg
    assert "catalog: my-catalog" in msg


@pytest.mark.asyncio
async def test_no_context_injection_when_empty_legacy_path():
    """Empty context → stream_async receives the raw user message without a Context block."""
    template = Agent(model=_mock_model())
    config = StrandsAgentConfig(replay_history_into_strands=False)
    ag = StrandsAgent(template, name="test", config=config)

    with patch("ag_ui_strands.agent.StrandsAgentCore", _CapturingCore):
        instance = await _run_to_completion(ag, _run_input([], thread_id="t-legacy-empty"))

    assert "Context:" not in (instance.stream_async_msg or "")


@pytest.mark.asyncio
async def test_multimodal_message_not_modified_legacy_path():
    """Multimodal (list-content) user messages are passed through without a Context block."""
    template = Agent(model=_mock_model())
    config = StrandsAgentConfig(replay_history_into_strands=False)
    ag = StrandsAgent(template, name="test", config=config)

    ctx = [Context(description="schema", value="rich-schema")]

    # convert_agui_content_to_strands returns a list for media content;
    # context injection must be skipped in this case.
    fake_blocks = [{"image": {"source": {"url": "https://example.com/img.png"}}}]
    with patch("ag_ui_strands.agent.StrandsAgentCore", _CapturingCore), \
         patch("ag_ui_strands.agent.convert_agui_content_to_strands", return_value=fake_blocks):
        instance = await _run_to_completion(
            ag, _run_input_multimodal(ctx, thread_id="t-legacy-mm")
        )

    # user_message is a list — context must NOT be appended
    assert isinstance(instance.stream_async_msg, list), (
        f"expected list for multimodal, got {type(instance.stream_async_msg)}"
    )
    # No "Context:" text was injected into any block
    for block in instance.stream_async_msg:
        text = block.get("text", "") if isinstance(block, dict) else ""
        assert "Context:" not in text


# ---------------------------------------------------------------------------
# Message injection — replay path (replay_history_into_strands=True, default)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_appended_to_last_user_message_replay_path():
    """On the replay path the context block is appended to the last user text message."""
    template = Agent(model=_mock_model())
    ag = StrandsAgent(template, name="test")  # default: replay=True

    ctx = [Context(description="guidelines", value="always use render_a2ui")]

    with patch("ag_ui_strands.agent.StrandsAgentCore", _CapturingCore):
        instance = await _run_to_completion(ag, _run_input(ctx, thread_id="t-replay"))

    assert instance.stream_async_msg is None, "replay path must call stream_async(None)"
    assert instance.messages, "native history should be non-empty"

    last_user = next(
        (m for m in reversed(instance.messages) if m.get("role") == "user"), None
    )
    assert last_user is not None
    content = last_user.get("content", [])
    assert content and "text" in content[0]
    text = content[0]["text"]
    assert "Context:" in text, f"expected 'Context:' in message text, got: {text!r}"
    assert "guidelines: always use render_a2ui" in text


@pytest.mark.asyncio
async def test_no_context_injection_when_empty_replay_path():
    """Empty context → native history user message is not modified."""
    template = Agent(model=_mock_model())
    ag = StrandsAgent(template, name="test")

    with patch("ag_ui_strands.agent.StrandsAgentCore", _CapturingCore):
        instance = await _run_to_completion(ag, _run_input([], thread_id="t-replay-empty"))

    last_user = next(
        (m for m in reversed(instance.messages) if m.get("role") == "user"), None
    )
    if last_user:
        content = last_user.get("content", [])
        text = content[0].get("text", "") if content else ""
        assert "Context:" not in text


@pytest.mark.asyncio
async def test_multimodal_message_not_modified_replay_path():
    """Replay path skips context injection for user messages with non-text content blocks."""
    template = Agent(model=_mock_model())
    ag = StrandsAgent(template, name="test")

    ctx = [Context(description="schema", value="rich-schema")]

    fake_blocks = [{"image": {"source": {"url": "https://example.com/img.png"}}}]
    with patch("ag_ui_strands.agent.StrandsAgentCore", _CapturingCore), \
         patch("ag_ui_strands.agent.convert_agui_content_to_strands", return_value=fake_blocks):
        instance = await _run_to_completion(
            ag, _run_input_multimodal(ctx, thread_id="t-replay-mm")
        )

    last_user = next(
        (m for m in reversed(instance.messages) if m.get("role") == "user"), None
    )
    assert last_user is not None
    content = last_user.get("content", [])
    assert content
    # Media block has no "text" key — no context should have been injected
    assert "text" not in content[0], (
        f"expected image block without 'text', got: {content[0]}"
    )
