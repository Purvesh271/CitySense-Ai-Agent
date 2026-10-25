import os
import uuid
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from tavily import TavilyClient

load_dotenv()

st.set_page_config(
    page_title="CitySense - AI-Powered City Intelligence Agent",
    page_icon="🌆",
    layout="wide",
)

# ---------- external CSS (style.css next to this file) ----------
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


# =========================
# Weather Tool
# =========================
@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN&appid={api_key}&units=metric"
    )
    try:
        data = requests.get(url, timeout=10).json()
    except requests.RequestException as e:
        return f"Error: could not reach weather service ({e})"

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    feels = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    desc = data["weather"][0]["description"]
    return (
        f"Weather in {city}: {desc}, {temp}°C "
        f"(feels like {feels}°C), humidity {humidity}%"
    )


# =========================
# News Tool
# =========================
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3,
    )
    results = response.get("results", [])
    if not results:
        return f"No news found for {city}"

    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        news_list.append(f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}...")
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# =========================
# LLM + Agent
# =========================
@st.cache_resource
def get_agent():
    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0,
        max_tokens=800,
        reasoning_effort="none",
    )
    return create_agent(
        llm,
        tools=[get_weather, get_news],
        system_prompt=(
            "You are a helpful city assistant. When you share news, include "
            "the links. Keep answers friendly and concise."
        ),
        # Replaces the input()-based approval: the agent pauses and the UI
        # shows Approve / Deny buttons instead.
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"get_weather": True, "get_news": True}
            )
        ],
        checkpointer=InMemorySaver(),
    )


# =========================
# UI helpers / state
# =========================
TOOL_META = {
    "get_weather": ("🌤️", "Weather"),
    "get_news": ("📰", "News"),
}
CITIES = ["Bhopal", "Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata"]

ss = st.session_state
ss.setdefault("messages", [])        # {"role", "content", "tools"}
ss.setdefault("thread_id", str(uuid.uuid4()))
ss.setdefault("pending", None)       # action requests awaiting approval
ss.setdefault("queued", None)        # prompt from sidebar buttons
ss.setdefault("tool_calls", 0)
ss.setdefault("auto_approve", False)

agent = get_agent()


def config():
    return {"configurable": {"thread_id": ss.thread_id}}


def text_of(content):
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return content or ""


def process(result):
    """Auto-approve, pause for approval, or store the final answer."""
    while result.get("__interrupt__"):
        reqs = result["__interrupt__"][0].value["action_requests"]
        if ss.auto_approve:
            ss.tool_calls += len(reqs)
            result = agent.invoke(
                Command(resume={"decisions": [{"type": "approve"} for _ in reqs]}),
                config=config(),
            )
        else:
            ss.pending = reqs
            return

    ss.pending = None
    used = []
    for m in reversed(result["messages"]):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage) and m.name and m.name not in used:
            used.append(m.name)
    ss.messages.append(
        {
            "role": "assistant",
            "content": text_of(result["messages"][-1].content),
            "tools": list(reversed(used)),
        }
    )


def send(prompt):
    ss.messages.append({"role": "user", "content": prompt, "tools": []})
    with st.spinner("Thinking…"):
        process(
            agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config(),
            )
        )


def decide(approve: bool):
    reqs = ss.pending
    if approve:
        ss.tool_calls += len(reqs)
        decisions = [{"type": "approve"} for _ in reqs]
    else:
        decisions = [
            {"type": "reject", "message": "Tool call denied by user."} for _ in reqs
        ]
    with st.spinner("Working…"):
        process(agent.invoke(Command(resume={"decisions": decisions}), config=config()))


def reset():
    ss.messages, ss.pending, ss.queued = [], None, None
    ss.tool_calls = 0
    ss.thread_id = str(uuid.uuid4())


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown(
        '<div class="brand">CitySense</div>'
        '<div class="brand-sub">AI-Powered City Intelligence Agent</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-title">Quick look</div>', unsafe_allow_html=True)
    city = st.selectbox("City", CITIES, label_visibility="collapsed")
    c1, c2 = st.columns(2)
    if c1.button("🌤️ Weather", use_container_width=True, disabled=bool(ss.pending)):
        ss.queued = f"What's the weather in {city}?"
    if c2.button("📰 News", use_container_width=True, disabled=bool(ss.pending)):
        ss.queued = f"Give me the latest news in {city}."
    if st.button("✨ Full briefing", use_container_width=True, disabled=bool(ss.pending)):
        ss.queued = f"Give me the weather and latest news for {city}."

    st.markdown('<div class="side-title">Safety</div>', unsafe_allow_html=True)
    ss.auto_approve = st.toggle(
        "Auto-approve tool calls",
        value=ss.auto_approve,
        help="Off: you approve every tool call. On: tools run without asking.",
    )

    st.markdown('<div class="side-title">Session</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.markdown(
        f'<div class="stat"><b>{len(ss.messages)}</b><span>messages</span></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="stat"><b>{ss.tool_calls}</b><span>tool calls</span></div>',
        unsafe_allow_html=True,
    )
    st.button("Clear chat", on_click=reset, use_container_width=True)

    missing = [
        k for k in ("GROQ_API_KEY", "OPENWEATHER_API_KEY", "TAVILY_API_KEY")
        if not os.getenv(k)
    ]
    if missing:
        st.warning("Missing in .env: " + ", ".join(missing))

# =========================
# Hero
# =========================
st.markdown(
    """
    <div class="hero">
      <div class="hero-glow"></div>
      <div class="hero-text">
        <h1>CitySense</h1>
        <p><b>AI-Powered City Intelligence Agent</b></p>
        <p>Live weather and headlines for any city. Every tool call waits for your approval.</p>
      </div>
      <svg class="skyline" viewBox="0 0 600 80" preserveAspectRatio="none" aria-hidden="true">
        <path d="M0 80V52h30V34h26v18h20V20h24v32h28V40h22V26h30v26h26V44h24V30h28v22h30V38h24V16h22v36h28V46h26V32h24v20h26V42h28V80z"/>
      </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Chat history
# =========================
if not ss.messages:
    st.markdown(
        '<div class="empty">Pick a city in the sidebar, or type below: '
        "<i>“Weather in Indore and any news there?”</i></div>",
        unsafe_allow_html=True,
    )

for m in ss.messages:
    with st.chat_message(m["role"], avatar="🧭" if m["role"] == "assistant" else "🙂"):
        if m["tools"]:
            chips = "".join(
                f'<span class="chip">{TOOL_META.get(t, ("🔧", t))[0]} '
                f'{TOOL_META.get(t, ("🔧", t))[1]}</span>'
                for t in m["tools"]
            )
            st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
        st.markdown(m["content"])

# =========================
# Approval card
# =========================
if ss.pending:
    st.markdown(
        '<div class="approval"><div class="approval-title">Approval needed</div>',
        unsafe_allow_html=True,
    )
    for r in ss.pending:
        icon, _ = TOOL_META.get(r["name"], ("🔧", r["name"]))
        args = ", ".join(f"{k}={v!r}" for k, v in r["args"].items())
        st.markdown(
            f'<div class="call">{icon} <b>{r["name"]}</b><code>{args}</code></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    a, d, _ = st.columns([1, 1, 3])
    if a.button("Approve", type="primary", use_container_width=True):
        decide(True)
        st.rerun()
    if d.button("Deny", use_container_width=True):
        decide(False)
        st.rerun()

# =========================
# Input
# =========================
typed = st.chat_input("Ask about a city…", disabled=bool(ss.pending))
prompt = typed or ss.queued
ss.queued = None
if prompt:
    send(prompt)
    st.rerun()