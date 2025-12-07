# langgraph_integration.py
import sys, os;
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import pathlib
import time
from typing import Any, Dict
from agents.data_analyst_agent import DataAnalystAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.report_agent import ReportAgent
import numpy as np
try:
    from langgraph.graph import Graph, Node, State
except Exception:
    Graph = None
    Node = None
    State = dict

analyst = DataAnalystAgent()
evaluator = EvaluatorAgent()

reporter = ReportAgent(data_analyst=analyst, evaluator=evaluator, reuse_evaluator_assistant=True)


def get_dataset_folder(csv_path: str, base_dir: str = "outputs") -> str:
    """
    Create and return dataset-specific folder structure:
      outputs/<dataset_name>/
        plots/
          histograms/
          top_correlations/
        reports/
    """
    name = pathlib.Path(csv_path).stem
    dataset_folder = os.path.join(base_dir, name)
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, "plots", "histograms"), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, "plots", "top_correlations"), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, "reports"), exist_ok=True)
    return dataset_folder


def _call_visual_summary_safe(analyst_obj, csv_path: str, out_dir: str):
    """
    Call visual_summary with out_dir if supported, else call without and note a warning.
    """
    try:
        return analyst_obj.visual_summary(csv_path, out_dir=out_dir)
    except TypeError:

        try:
            result = analyst_obj.visual_summary(csv_path)
            return result
        except Exception as e:
            raise


def _call_generate_report_safe(reporter_obj, path: str, n_insights: int, out_dir: str, use_llm_summary: bool, polish_report: bool, save_html: bool):
    """
    Call generate_report with out_dir if supported, else call and try to move result into out_dir.
    """
    try:
        return reporter_obj.generate_report(
            path,
            n_insights=n_insights,
            out_dir=out_dir,
            use_llm_summary=use_llm_summary,
            polish_report=polish_report,
            save_html=save_html,
        )
    except TypeError:
        report_path = reporter_obj.generate_report(
            path,
            n_insights=n_insights,
            use_llm_summary=use_llm_summary,
            polish_report=polish_report,
            save_html=save_html,
        )
        try:
            dst_folder = os.path.join(out_dir, "reports")
            os.makedirs(dst_folder, exist_ok=True)
            basename = os.path.basename(report_path)
            dst = os.path.join(dst_folder, basename)
            if os.path.exists(report_path):
                os.replace(report_path, dst)
                return dst
        except Exception:
            pass
        return report_path


def data_analyst_node(state: Dict[str, Any]):
    csv_path = state.get("path")
    n_insights = state.get("n_insights", 3)
    tone = state.get("tone", "concise and actionable")
    out_dir = state.get("out_dir")

    # Run analysis (sync wrapper)
    insights = None
    visuals = {}
    try:
        insights = analyst.analyze(csv_path, n_insights=n_insights, tone=tone)
    except Exception as e:
        state.setdefault("errors", []).append({"node": "data_analyst.analyze", "error": str(e)})
        insights = ""

    try:
        visuals = _call_visual_summary_safe(analyst, csv_path, out_dir)
    except Exception as e:
        visuals = {"error": str(e)}
        state.setdefault("errors", []).append({"node": "data_analyst.visuals", "error": str(e)})

    state["insights"] = insights
    state["visuals"] = visuals
    return state


def _normalize_evaluator_raw(raw):
    """
    Accepts whatever evaluator.evaluate returned and normalizes into a dict:
    { "per_model_scores": ..., "failures": ..., ... } or {"raw": raw}
    """
    if isinstance(raw, tuple) and len(raw) == 2:
        return {"per_model_scores": raw[0] or {}, "failures": raw[1] or {}}
    if isinstance(raw, dict):
        return raw.copy()
    return {"raw": raw}


# Put this near the other helper functions in langgraph_integration.py

def _compute_overall_from_evaluation(evaluation: dict) -> float:
    """Try several fallbacks to compute a single overall score (0.0..5.0)."""
    if not isinstance(evaluation, dict):
        return 0.0

    # 1) explicit average_scores.overall
    try:
        v = evaluation.get("average_scores", {}).get("overall")
        if v is not None:
            return float(v)
    except Exception:
        pass

    # 2) ensemble -> mean of ensemble matrix
    try:
        ens = evaluation.get("ensemble")
        if isinstance(ens, dict):
            matrix = ens.get("ensemble")
            if matrix is not None:
                import numpy as _np
                # coerce to array and take mean of all values
                arr = _np.asarray(matrix, dtype=float)
                if arr.size:
                    return float(_np.nanmean(arr))
    except Exception:
        pass

    # 3) per_model_scores -> mean of model means
    try:
        pms = evaluation.get("per_model_scores") or {}
        import numpy as _np
        vals = []
        if isinstance(pms, dict):
            for arr in pms.values():
                try:
                    a = _np.asarray(arr, dtype=float)
                    if a.size:
                        vals.append(float(_np.nanmean(a)))
                except Exception:
                    pass
        if vals:
            return float(sum(vals) / len(vals))
    except Exception:
        pass

    return 0.0


# Put this near the other helper functions in langgraph_integration.py

def _compute_overall_from_evaluation(evaluation: dict) -> float:
    """Try several fallbacks to compute a single overall score (0.0..5.0)."""
    if not isinstance(evaluation, dict):
        return 0.0

    # 1) explicit average_scores.overall
    try:
        v = evaluation.get("average_scores", {}).get("overall")
        if v is not None:
            return float(v)
    except Exception:
        pass

    # 2) ensemble -> mean of ensemble matrix
    try:
        ens = evaluation.get("ensemble")
        if isinstance(ens, dict):
            matrix = ens.get("ensemble")
            if matrix is not None:
                import numpy as _np
                # coerce to array and take mean of all values
                arr = _np.asarray(matrix, dtype=float)
                if arr.size:
                    return float(_np.nanmean(arr))
    except Exception:
        pass

    # 3) per_model_scores -> mean of model means
    try:
        pms = evaluation.get("per_model_scores") or {}
        import numpy as _np
        vals = []
        if isinstance(pms, dict):
            for arr in pms.values():
                try:
                    a = _np.asarray(arr, dtype=float)
                    if a.size:
                        vals.append(float(_np.nanmean(a)))
                except Exception:
                    pass
        if vals:
            return float(sum(vals) / len(vals))
    except Exception:
        pass

    return 0.0


def _normalize_models_for_evaluator(models):
    """
    Lightweight normalizer used by evaluator_node.
    Accepts list of (name,kwargs), strings, or dicts. Coerces kwargs['model'] to str.
    """
    norm = []
    for i, m in enumerate(models):
        if isinstance(m, (list, tuple)) and len(m) == 2:
            name = str(m[0])
            kw = dict(m[1] or {})
            kw["model"] = str(kw.get("model", name))
            norm.append((name, kw))
            continue
        if isinstance(m, str):
            norm.append((m, {"model": m}))
            continue
        if isinstance(m, dict):
            name = str(m.get("name") or m.get("model"))
            kw = dict(m)
            kw["model"] = str(kw.get("model", name))
            norm.append((name, kw))
            continue
        raise ValueError(f"models[{i}] unsupported type {type(m).__name__}")
    return norm


def evaluator_node(state: Dict[str, Any]):
    """
    Run evaluator on state['insights'] and populate state['evaluation'] and state['overall_score'].
    This function normalizes any return shape from evaluator.evaluate and records failures.
    """
    insights = state.get("insights", "")
    ensemble_method = state.get("ensemble_method", "mean")
    weights = state.get("weights")

    # short-circuit: no insights -> nothing to evaluate
    if not insights:
        state.setdefault("errors", []).append({"node": "evaluator", "error": "no_insights_to_evaluate"})
        state["evaluation"] = {}
        state["overall_score"] = float(state.get("overall_score", 0.0) or 0.0)
        return state

    # choose models: explicit in state else evaluator.default_models
    models = state.get("models") or getattr(evaluator, "default_models", None)
    if not models:
        # set a safe default with string model names
        evaluator.default_models = [
            ("gpt4o-mini", {"model": "gpt-4o-mini", "temperature": 0.0})
        ]
        models = evaluator.default_models

    # defensive normalize before calling evaluator.evaluate (prevents model=1 cases)
    try:
        norm_models = _normalize_models_for_evaluator(models)
    except Exception as e:
        state.setdefault("errors", []).append({"node": "evaluator.normalize_models", "error": str(e)})
        state["evaluation"] = {}
        state["overall_score"] = float(state.get("overall_score", 0.0) or 0.0)
        return state

    # call evaluator - keep errors captured and recorded
    try:
        raw = evaluator.evaluate(models=norm_models, insights=insights,
                                 ensemble_method=ensemble_method, weights=weights)
    except Exception as e:
        # evaluator.evaluate may raise (e.g., network or parse failures bubbled up)
        state.setdefault("errors", []).append({"node": "evaluator.evaluate", "error": str(e)})
        state["evaluation"] = {}
        state["overall_score"] = float(state.get("overall_score", 0.0) or 0.0)
        return state

    # normalize raw return into dict (use your existing normalizer if present)
    try:
        evaluation = _normalize_evaluator_raw(raw)
    except Exception:
        evaluation = {"raw": raw}

    # collect per-model failures (if any) and convert parse failures into a failures dict
    failures = {}
    per_model_scores: Dict[str, Any] = {}
    raw_map = evaluation.get("per_model_raw") or evaluation.get("per_model_outputs") or {}

    # prefer evaluator._scores_from_output if available
    scores_from_output = getattr(evaluator, "_scores_from_output", None)

    for model_name, raw_out in (raw_map.items() if isinstance(raw_map, dict) else []):
        # raw_out can be dict {"status":.., "scores":.., "raw":..} or other shapes
        if isinstance(raw_out, dict):
            status = raw_out.get("status")
            if status != "ok":
                # record failure (include error/raw for debugging)
                failures[model_name] = {"status": status, "error": raw_out.get("error"), "raw": raw_out.get("raw")}
                continue
            # attempt to extract numeric array
            arr = None
            try:
                if scores_from_output:
                    arr = scores_from_output(raw_out)
                else:
                    # fallback: look for 'scores' key
                    arr = np.asarray(raw_out.get("scores", []), dtype=float)
                    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] != 3:
                        arr = None
            except Exception:
                arr = None

            if arr is None:
                failures[model_name] = {"status": "parse_failed", "error": "could_not_parse_scores", "raw": raw_out}
            else:
                per_model_scores[model_name] = arr
        else:
            # unexpected raw_out shape -> mark as failure
            failures[model_name] = {"status": "unknown_shape", "raw": raw_out}

    # If evaluation returned per_model_scores already (some evaluators may), merge them
    if not per_model_scores and isinstance(evaluation.get("per_model_scores"), dict):
        # ensure numpy arrays for consistency
        for k, v in evaluation["per_model_scores"].items():
            try:
                per_model_scores[k] = np.asarray(v, dtype=float)
            except Exception:
                failures[k] = {"status": "parse_failed", "error": "coerce_failed", "raw": v}

    # Build canonical evaluation dict to store in state
    canonical_eval: Dict[str, Any] = {}
    canonical_eval.update({k: v for k, v in evaluation.items() if k not in ("per_model_raw", "per_model_scores")})
    canonical_eval["per_model_raw"] = raw_map
    canonical_eval["per_model_scores"] = per_model_scores
    if failures:
        canonical_eval.setdefault("failures", {}).update(failures)

    # compute overall_score from canonical_eval
    overall = _compute_overall_from_evaluation(canonical_eval)
    # if overall is zero and we had readable per_model_scores, compute fallback mean
    if overall == 0.0 and per_model_scores:
        try:
            import numpy as _np
            vals = []
            for a in per_model_scores.values():
                try:
                    vals.append(float(_np.nanmean(a)))
                except Exception:
                    pass
            if vals:
                overall = float(sum(vals) / len(vals))
        except Exception:
            pass

    # surface failures into state.errors for easier debugging/logging
    if failures:
        for m, info in failures.items():
            state.setdefault("errors", []).append({"node": "evaluator.model_failure", "model": m, "info": info})

    state["evaluation"] = canonical_eval
    state["overall_score"] = float(overall or state.get("overall_score", 0.0) or 0.0)
    return state


def fix_node(state: Dict[str, Any]):
    """
    If overall_score < threshold and retries remain:
    - re-run analyst.analyze with improved tone
    - evaluate insights via evaluator_node (reuses normalization)
    All errors are recorded into state['errors'] and previous overall_score is preserved on failure.
    """
    threshold = state.get("fix_threshold", 4.0)
    max_retries = state.get("max_retries", 2)
    if state.get("overall_score", 0.0) >= threshold or state.get("retries", 0) >= max_retries:
        return state

    state["retries"] = state.get("retries", 0) + 1
    tone = state.get("improved_tone", "clear and professional")
    prev_overall = float(state.get("overall_score", 0.0) or 0.0)

    try:
        # re-generate improved insights (use sync wrapper)
        improved = analyst.analyze(state["path"], n_insights=state.get("n_insights", 3), tone=tone)
        state.setdefault("improved_insights", []).append(improved)
        state["insights"] = improved  # let evaluator_node read it

        # delegate evaluation (single canonical place)
        state = evaluator_node(state)

        # ensure we didn't accidentally drop overall_score
        state["overall_score"] = float(state.get("overall_score", prev_overall) or prev_overall)
    except Exception as e:
        # record and preserve previous overall score
        state.setdefault("errors", []).append({"node": "fix", "error": str(e)})
        state["overall_score"] = prev_overall

    return state


def report_node(state: Dict[str, Any]):
    path = state["path"]
    n = state.get("n_insights", 3)
    out_dir = state.get("out_dir", "outputs")
    try:
        report_path = _call_generate_report_safe(
            reporter, path,
            n_insights=n,
            out_dir=out_dir,
            use_llm_summary=state.get("use_llm_summary", True),
            polish_report=state.get("polish_report", False),
            save_html=state.get("save_html", True),
        )
        state["report_path"] = report_path
    except Exception as e:
        state.setdefault("errors", []).append({"node": "report.generate_report", "error": str(e)})
        state["report_path"] = state.get("report_path", None)
    return state


def build_and_run_graph(csv_path: str, n_insights: int = 3, out_dir: str = "outputs"):
    """
    Runs the pipeline and ensures outputs are saved under outputs/<dataset_name>/...
    """
    dataset_folder = get_dataset_folder(csv_path, base_dir=out_dir)
    state = {
        "path": csv_path,
        "n_insights": n_insights,
        "tone": "concise and actionable",
        "improved_tone": "clear and professional",
        "fix_threshold": 4.0,
        "retries": 0,
        "max_retries": 2,
        "out_dir": dataset_folder,
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
        try:
            res_state = g.run(state)
        except Exception as e:
            state.setdefault("errors", []).append({"node": "graph.run", "error": str(e)})
            pass
        else:
            return res_state

    # Fallback sequential run (safe, per-node try/except)
    # data analyst
    try:
        state = data_analyst_node(state)
    except Exception as e:
        state.setdefault("errors", []).append({"node": "data_analyst", "error": str(e)})
        return state

    try:
        state = evaluator_node(state)
    except Exception as e:
        state.setdefault("errors", []).append({"node": "evaluator", "error": str(e)})
        # ensure downstream keys exist
        state.setdefault("evaluation", {})
        state["overall_score"] = float(state.get("evaluation", {}).get("average_scores", {}).get("overall", 0.0))

    try:
        state = fix_node(state)
    except Exception as e:
        state.setdefault("errors", []).append({"node": "fix", "error": str(e)})

    # report
    try:
        state = report_node(state)
    except Exception as e:
        state.setdefault("errors", []).append({"node": "report", "error": str(e)})

    return state


if __name__ == "__main__":
    out = build_and_run_graph("./data/loan_approval.csv", n_insights=4, out_dir="outputs")
    print("Pipeline finished. Report:", out.get("report_path"))