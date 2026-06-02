# MLB Live Agent — MLB_Live_Agent.py
# Streamlit web UI + Anthropic agentic loop for MLB pitching analysis.
#
# EXECUTION
#   export ANTHROPIC_API_KEY=sk-ant-...
#   streamlit run MLB_Live_Agent.py
#
# ARCHITECTURE
#   This file is the Streamlit entry point. All data tools, ChromaDB helpers,
#   Plotly chart builders, tool schemas, and the system prompt live in
#   agent_tools.py. The agent loop follows the same pattern as fe2c.py:
#   send → parse tool_use blocks → dispatch → collect tool_results → repeat
#   until stop_reason == "end_turn" or MAX_ITERATIONS is reached.
#   Prompt caching is applied to the system prompt and last tool schema to
#   reduce API cost across multi-step queries.
#
# PYTHON VERSION
#   Requires Python 3.13.x. See agent_tools.py for details.

from __future__ import annotations

import json
import sys
from datetime import date
from typing import TYPE_CHECKING

import streamlit as st

if sys.version_info >= (3, 14):
    st.error(
        "Python 3.14+ detected. chromadb and sentence-transformers require "
        "Python 3.13.x. Switch interpreters and restart Streamlit: "
        "VS Code → Command Palette → 'Python: Select Interpreter' → /usr/local/bin/python3.13"
    )
    st.stop()

if TYPE_CHECKING:
    import anthropic  # type: ignore[import-untyped]

from agent_tools import (
    CLAUDE_MODEL,
    MAX_ITERATIONS,
    MLB_TEAM_IDS,
    SYSTEM_PROMPT,
    TOOL_SCHEMAS,
    TOOLS,
    _get_chroma_collection,
    plot_il_timeline,
    plot_pitch_heatmap,
    plot_pitch_type_distribution,
    plot_regression_results,
    plot_velocity_by_pitch_type,
    plot_velocity_trend,
    plot_3d_pitch_movement,
)

# Page config

st.set_page_config(
    page_title="MLB Live Scout",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cached ChromaDB accessor
# st.cache_resource persists the collection object for the lifetime of the
# Streamlit server process — the sentence-transformer model (~90MB) downloads
# only once per server session rather than once per page load.
@st.cache_resource(show_spinner="Loading pitcher index...")
def _get_cached_chroma_collection():
    return _get_chroma_collection()


# Session state initialisation
# display_messages: plain text turns shown in the chat UI
# api_messages:     full Anthropic API format (includes tool_use / tool_result blocks)
# charts:           list of (title, go.Figure) tuples rendered above the chat

for key, default in [
    ("display_messages", []),
    ("api_messages",     []),
    ("charts",           []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# Dispatch and chart rendering
def _dispatch_tool(tool_name: str, tool_input: dict) -> tuple[str, str | None, dict | None]:
    """
    Call the named tool and parse its JSON result.
    Returns (text_for_claude, chart_type, chart_data).
    chart_type / chart_data are None when the tool produces no chart.
    """
    if tool_name not in TOOLS:
        return f"ERROR: Unknown tool '{tool_name}'.", None, None

    tool_fn, _ = TOOLS[tool_name]
    raw = tool_fn(tool_input)

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return str(raw), None, None

    text       = parsed.get("text", str(raw))
    chart_type = parsed.get("chart_type")
    chart_data = parsed.get("chart_data")
    return text, chart_type, chart_data


def _build_charts(chart_type: str, chart_data: dict) -> list[tuple[str, object]]:
    """Convert chart_data from a tool result into (title, go.Figure) pairs."""
    figs: list[tuple[str, object]] = []

    if chart_type == "pitcher_statcast":
        name = chart_data.get("pitcher_name", "Pitcher")
        if chart_data.get("pitch_mix"):
            figs.append((
                f"{name} — Pitch Mix",
                plot_pitch_type_distribution(chart_data["pitch_mix"], name),
            ))
        if chart_data.get("velo_by_type"):
            figs.append((
                f"{name} — Avg Velocity",
                plot_velocity_by_pitch_type(chart_data["velo_by_type"], name),
            ))

    elif chart_type == "velocity_trend":
        name   = chart_data.get("pitcher_name", "Pitcher")
        season = chart_data.get("season", "")
        if chart_data.get("records"):
            figs.append((
                f"{name} — Velocity Trend {season}",
                plot_velocity_trend(chart_data["records"], name, season),
            ))

    elif chart_type == "pitch_heatmap":
        name  = chart_data.get("pitcher_name", "Pitcher")
        pt    = chart_data.get("pitch_type", "")
        px_   = chart_data.get("plate_x", [])
        pz_   = chart_data.get("plate_z", [])
        if px_ and pz_:
            figs.append((
                f"{name} — {pt} Location Heatmap",
                plot_pitch_heatmap(px_, pz_, pt, name),
            ))
        pfx_x = chart_data.get("pfx_x", [])
        pfx_z = chart_data.get("pfx_z", [])
        if pfx_x and pfx_z:
            figs.append((
                f"{name} — {pt} 3D Movement",
                plot_3d_pitch_movement(
                    pfx_x, pfx_z,
                    chart_data.get("speeds", []),
                    chart_data.get("descriptions", []),
                    pt, name,
                ),
            ))

    elif chart_type == "il_timeline":
        team   = chart_data.get("team", "")
        season = chart_data.get("season", "")
        stints = chart_data.get("stints", [])
        if stints:
            figs.append((
                f"{team} IL Timeline {season}",
                plot_il_timeline(stints, team, season),
            ))

    elif chart_type == "regression":
        preds  = chart_data.get("predictions", [])
        target = chart_data.get("target", "")
        fi     = chart_data.get("feature_importance", [])
        r2     = chart_data.get("r2", 0.0)
        if preds and fi:
            figs.append((
                f"Regression: {target}  (R²={r2:.3f})",
                plot_regression_results(preds, target, fi, r2),
            ))

    return figs


# Agent loop 
# Build messages, call Claude, parse
# tool_use blocks, dispatch, collect tool_results, append both turns, repeat
# until end_turn or MAX_ITERATIONS. Prompt caching reduces API cost on the
# static system prompt and tool schemas across iterations.

def run_agent(question: str, sidebar_ctx: dict) -> str:
    import anthropic as _anthropic  # type: ignore[import-untyped]

    client = _anthropic.Anthropic()

    # Cache the system prompt and last tool schema, both are static across
    # all iterations of this question, so cache hits avoid re-encoding them.
    cached_system = [
        {"type": "text", "text": SYSTEM_PROMPT.format(**sidebar_ctx),
         "cache_control": {"type": "ephemeral"}}
    ]
    cached_tools = TOOL_SCHEMAS[:-1] + [
        {**TOOL_SCHEMAS[-1], "cache_control": {"type": "ephemeral"}}
    ]

    # Build message history: all prior API turns + this new question
    messages = list(st.session_state.api_messages)
    messages.append({"role": "user", "content": question})

    for iteration in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=cached_system,
            tools=cached_tools,
            messages=messages,
        )

        text_blocks = [b for b in response.content if b.type == "text"]
        tool_blocks = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "end_turn" or not tool_blocks:
            final = " ".join(b.text for b in text_blocks).strip()
            # Persist the complete exchange to API history for follow-up questions
            st.session_state.api_messages.append({"role": "user", "content": question})
            st.session_state.api_messages.append(
                {"role": "assistant", "content": response.content}
            )
            return final or "Analysis complete. See charts above."

        # Dispatch each tool call and collect results
        tool_results = []
        for tool_call in tool_blocks:
            text_result, chart_type, chart_data = _dispatch_tool(
                tool_call.name, tool_call.input
            )
            if chart_type and chart_data:
                st.session_state.charts.extend(_build_charts(chart_type, chart_data))

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tool_call.id,
                "content":     text_result,
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user",      "content": tool_results})

    return "Analysis reached maximum depth. See charts and partial results above."


# Sidebar 

team_keys = sorted(MLB_TEAM_IDS.keys())

with st.sidebar:
    st.title("⚾ MLB Live Scout")
    st.caption("Powered by Baseball Savant + Claude")
    st.divider()

    my_team  = st.selectbox("My Team",       team_keys, index=team_keys.index("SF"))
    opponent = st.selectbox("Opponent Team",  team_keys, index=team_keys.index("LAD"))
    season   = st.number_input("Season", min_value=2020, max_value=2026, value=2025, step=1)

    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start Date", value=date(int(season), 3, 1))
    with col_b:
        end_date   = st.date_input("End Date",   value=date(int(season), 11, 1))

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.display_messages = []
        st.session_state.api_messages     = []
        st.rerun()
    if st.button("Clear Charts", use_container_width=True):
        st.session_state.charts = []
        st.rerun()

    st.divider()
    st.caption("ChromaDB Index")
    try:
        col   = _get_cached_chroma_collection()
        count = col.count() if col is not None else 0
        st.metric("Pitchers Indexed", count)
    except Exception:
        st.caption("Not yet initialized")

    st.divider()
    with st.expander("Example queries"):
        st.markdown(
            "- What is Logan Webb's pitch mix this season?\n"
            "- Show me the Dodgers' injury history in 2025\n"
            "- Build an ERA regression from K%, BB%, and barrel rate\n"
            "- Show Webb's slider location heatmap\n"
            "- Find pitchers with declining velocity\n"
            "- What's SF's upcoming schedule?"
        )


# Main area 
st.header("⚾ MLB Live Scout")
st.caption(
    f"Analyzing **{opponent}** pitchers for the **{my_team}** offense — "
    f"{start_date} to {end_date}"
)

# Chart grid — 2-column layout, rendered above the chat history
if st.session_state.charts:
    st.subheader("Analysis Charts")
    pairs = st.session_state.charts
    for i in range(0, len(pairs), 2):
        cols = st.columns(2)
        for j, (title, fig) in enumerate(pairs[i : i + 2]):
            with cols[j]:
                st.caption(title)
                st.plotly_chart(fig, use_container_width=True)
    st.divider()

# Chat history
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input: sidebar context is injected into the system prompt on each call
sidebar_ctx = {
    "my_team":    my_team,
    "opponent":   opponent,
    "season":     int(season),
    "start_date": str(start_date),
    "end_date":   str(end_date),
    "today":      str(date.today()),
}

if prompt := st.chat_input(f"Ask about {opponent} pitchers, injuries, tendencies..."):
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pulling live data and analyzing..."):
            answer = run_agent(prompt, sidebar_ctx)
        st.markdown(answer)

    st.session_state.display_messages.append({"role": "assistant", "content": answer})
    st.rerun()
