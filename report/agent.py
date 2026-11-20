"""Report Agent built with LangGraph.

        START -> designer --(has tool calls?)--> tools -> designer ... -> END
                    |                                            
                    +--(no tool calls / finish_report / step limit)--> END

- designer : the LLM decides the next design / content tool calls.
- tools    : runs those calls on the ReportBuilder (fonts, colors, headings, tables...).
The graph loops until the LLM calls finish_report, stops calling tools, or hits MAX_STEPS.
"""
import json
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from report.builder import ReportBuilder
from report.design import COLOR_KEYS, ROLES
from report.fonts import available_fonts
from report.tools import make_tools

MAX_STEPS = 25

SYSTEM = """You are the Report Designer Agent. You create a professional PDF report by calling tools.

How to work:
1. Read the style request. Use the design tools (apply_preset, set_page, set_font, set_colors, set_spacing,
   set_body_alignment, set_header_footer, set_cover_page) to match it. Choose fonts, sizes and colors yourself
   when the request is vague. Make good, readable design choices.
2. Build the content with the content tools, in a logical order (for example: question, key finding callout,
   answer, analysis, extra sections).
3. add_answer is REQUIRED. It inserts the original answer unchanged. Never retype the answer yourself.
4. Use only the facts and numbers in the content you were given. Never invent anything.
5. If a tool returns "Error", read the message and call it again correctly.
6. Finish with finish_report.
Call several tools in each step when you can."""


class ReportState(TypedDict):
    messages: Annotated[list, add_messages]   # the conversation between the LLM and the tools
    steps: int                                # how many times the designer ran
    log: list[str]                            # human-readable tool log (shown in the UI)


def describe_content(b: ReportBuilder) -> str:
    a = b.analysis
    info = {
        "question": b.question,
        "answer_preview": b.answer[:1200],
        "analysis_available": {
            "summary": bool(a.get("reasoning")),
            "calculations": len(a.get("calculations") or {}),
            "comparison": bool((a.get("comparison") or {}).get("values_by_document")
                               or (a.get("comparison") or {}).get("similarities")),
            "tables": len(a.get("tables") or []),
        },
        "extra_sections": [e.get("heading", "") for e in b.extras],
        "available_fonts": list(available_fonts()),
        "text_roles": ROLES,
        "color_names": COLOR_KEYS,
    }
    return json.dumps(info, ensure_ascii=False, indent=1)


def build_report_graph(builder: ReportBuilder, llm, max_steps: int = MAX_STEPS):
    """Compile the LangGraph that lets the LLM drive `builder` through its tools."""
    tools = make_tools(builder)
    by_name = {t.name: t for t in tools}
    model = llm.bind_tools(tools)

    def designer(state: ReportState) -> dict:
        reply = model.invoke(state["messages"])
        return {"messages": [reply], "steps": state["steps"] + 1}

    def run_tools(state: ReportState) -> dict:
        results, log = [], []
        for call in state["messages"][-1].tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                result = f"Error: there is no tool called {call['name']}."
            else:
                try:
                    result = str(tool.invoke(call["args"]))
                except Exception as error:          # bad arguments from the model: let it retry
                    result = f"Error: {error}"
            log.append(f"{call['name']}({json.dumps(call['args'], ensure_ascii=False)[:120]}) -> {result[:100]}")
            results.append(ToolMessage(content=result, tool_call_id=call["id"]))
        return {"messages": results, "log": state["log"] + log}

    def route(state: ReportState) -> str:
        """After the designer: run tools, or stop."""
        if builder.finished or state["steps"] >= max_steps:
            return END
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    def after_tools(state: ReportState) -> str:
        """After the tools: stop if finish_report was called, otherwise let the designer continue."""
        return END if builder.finished else "designer"

    graph = StateGraph(ReportState)
    graph.add_node("designer", designer)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "designer")
    graph.add_conditional_edges("designer", route, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", after_tools, {"designer": "designer", END: END})
    return graph.compile()


def run_report_agent(builder: ReportBuilder, style_request: str, llm, max_steps: int = MAX_STEPS) -> list[str]:
    """Run the LangGraph agent. Returns the tool log (shown in the UI)."""
    app = build_report_graph(builder, llm, max_steps)
    start = {
        "messages": [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=f"Style request from the user:\n{style_request or '(none: choose a good professional design)'}"
                                 f"\n\nContent you can use:\n{describe_content(builder)}"),
        ],
        "steps": 0,
        "log": [],
    }
    final = app.invoke(start, config={"recursion_limit": max_steps * 2 + 5})
    return final["log"]
