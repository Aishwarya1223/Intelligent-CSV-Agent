# langgraph_integration.py
import sys, os; 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
from typing import Any, Dict

from agents.data_analyst_agent import DataAnalystAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.report_agent import ReportAgent

try:
    from langgraph.graph import Graph, Node, State
except Exception:
    Graph = None
    Node = None
    State = dict

analyst = DataAnalystAgent()
evaluator = EvaluatorAgent()
reporter = ReportAgent(data_analyst=analyst, evaluator=evaluator, reuse_evaluator_assistant=True)

def data_analyst_node(state: Dict[str, Any]):
    csv_path = state.get("path")
    n_insights = state.get("n_insights", 3)
    tone = state.get("tone", "concise and actionable")
    # Run analysis (sync wrapper)
    insights = analyst.analyze(csv_path, n_insights=n_insights, tone=tone)
    visuals = {}
    try:
        visuals = analyst.visual_summary(csv_path)
    except Exception as e:
        visuals = {"error": str(e)}
    state["insights"] = insights
    state["visuals"] = visuals
    return state

def evaluator_node(state: Dict[str, Any]):
    insights = state.get("insights", "")
    evaluation = evaluator.evaluate(insights)
    state["evaluation"] = evaluation
    # compute overall score safely
    overall = 0.0
    try:
        overall = float(evaluation.get("average_scores", {}).get("overall", 0.0))
    except Exception:
        overall = 0.0
    state["overall_score"] = overall
    return state

def fix_node(state: Dict[str, Any]):
   
    threshold = state.get("fix_threshold", 4.0)
    if state.get("overall_score", 0.0) < threshold:
        # change tone and rerun
        improved_tone = state.get("improved_tone", "clear and professional")
        state["retries"] = state.get("retries", 0) + 1
        insights = analyst.analyze(state["path"], n_insights=state.get("n_insights", 3), tone=improved_tone)
        state.setdefault("improved_insights", []).append(insights)
        # re-evaluate
        state["evaluation"] = evaluator.evaluate(insights)
        try:
            state["overall_score"] = float(state["evaluation"].get("average_scores", {}).get("overall", 0.0))
        except Exception:
            state["overall_score"] = 0.0
    return state

def report_node(state: Dict[str, Any]):
    # Generate final report (LLM executive summary + html)
    path = state["path"]
    n = state.get("n_insights", 3)
    # Using reporter to create and save report; return path to md/html
    report_path = reporter.generate_report(
        path,
        n_insights=n,
        out_dir=state.get("out_dir", "reports"),
        use_llm_summary=state.get("use_llm_summary", True),
        polish_report=state.get("polish_report", False),
        save_html=state.get("save_html", True),
    )
    state["report_path"] = report_path
    return state

def build_and_run_graph(csv_path: str, n_insights: int = 3, out_dir: str = "reports"):
    # Initial state
    state = {
        "path": csv_path,
        "n_insights": n_insights,
        "tone": "concise and actionable",
        "fixed_tone": "clear and professional",
        "fix_threshold": 4.0,
        "retries": 0,
        "out_dir": out_dir,
        "use_llm_summary": True,
        "polish_report": True,
        "save_html": True,
    }

    if Graph is not None:
        g = Graph(name="csv_analyst_pipeline")
        # register nodes (API names differ by version — consult docs)
        g.add_node(Node(name="data_analyst", fn=data_analyst_node))
        g.add_node(Node(name="evaluator", fn=evaluator_node))
        g.add_node(Node(name="fix", fn=fix_node))
        g.add_node(Node(name="report", fn=report_node))
        # example edges: data_analyst -> evaluator -> fix (conditional) -> report
        g.add_edge("data_analyst", "evaluator")
        g.add_edge("evaluator", "fix")
        g.add_edge("fix", "report")
        # run graph with initial state (API could be g.run(state) or similar)
        res_state = g.run(state)  # adapt to your LangGraph runtime call
    else:
        # fallback sequential run (same semantics as the graph)
        state = data_analyst_node(state)
        state = evaluator_node(state)
        state = fix_node(state)
        state = report_node(state)
        res_state = state

    return res_state

if __name__ == "__main__":
    out = build_and_run_graph("./data/sample.csv", n_insights=4, out_dir="reports")
    print("Pipeline finished. Report:", out.get("report_path"))
