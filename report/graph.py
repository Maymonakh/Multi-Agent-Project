"""The Report Agent as a LangGraph StateGraph (same tools, same builder as report/agent.py).

    START -> agent -+-> (LLM called tools) -> tools -> agent ...
                    +-> (finished / no tool calls / too many steps) -> render -> END
"""
import json
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from report.agent import MAX_STEPS, SYSTEM, describe_content
from report.builder import ReportBuilder
from report.tools import make_tools


class ReportState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    steps: int
    log: list
    path: str


def build_report_graph(builder: ReportBuilder, llm, output_path: str, max_steps: int = MAX_STEPS):
    tools = make_tools(builder)
    by_name = {t.name: t for t in tools}
    model = llm.bind_tools(tools)

    def agent(state: ReportState):
        reply = model.invoke(state["messages"])
        return {"messages": [reply], "steps": state.get("steps", 0) + 1}

    def run_tools(state: ReportState):
        messages, log = [], list(state.get("log", []))
        for call in state["messages"][-1].tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                result = f"Error: there is no tool called {call['name']}."
            else:
                try:
                    result = str(tool.invoke(call["args"]))
                except Exception as error:
                    result = f"Error: {error}"
            log.append(f"{call['name']}({json.dumps(call['args'], ensure_ascii=False)[:120]}) -> {result[:100]}")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
        return {"messages": messages, "log": log}

    def route(state: ReportState):
        last = state["messages"][-1]
        if builder.finished or state.get("steps", 0) >= max_steps or not getattr(last, "tool_calls", None):
            return "render"
        return "tools"

    def render(state: ReportState):
        return {"path": builder.render(output_path)}

    graph = StateGraph(ReportState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("render", render)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "render": "render"})
    graph.add_edge("tools", "agent")
    graph.add_edge("render", END)
    return graph.compile()


def run_report_graph(builder: ReportBuilder, style_request: str, llm, output_path: str) -> tuple[str, list]:
    """Returns (pdf_path, log). Same result as run_report_agent + builder.render."""
    graph = build_report_graph(builder, llm, output_path)
    task = (f"Style request from the user:\n{style_request or '(none: choose a good professional design)'}"
            f"\n\nContent you can use:\n{describe_content(builder)}")
    final = graph.invoke({"messages": [SystemMessage(content=SYSTEM), HumanMessage(content=task)]})
    return final["path"], final.get("log", [])
