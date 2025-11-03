# coordinator_agent.py
import logging
import traceback
import datetime
from pathlib import Path
from typing import Optional, Dict, Any
try:
    from data_analyst_agent import DataAnalystAgent
except Exception as e:
    raise ImportError("DataAnalystAgent Does not exists") from e

try:
    from evaluator_agent import EvaluatorAgent
except Exception as e:
    raise ImportError("EvaluatorAgent Does not exists") from e

try:
    from report_agent import ReportAgent
except Exception as e:
    raise ImportError("ReportAgent Does not exists") from e

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


class CoordinatorAgent:
    """
    Coordinator that runs the full pipeline:
      - DataAnalystAgent: produces insights and visuals
      - EvaluatorAgent: scores and suggests rewrites
      - ReportAgent: makes a final report (MD/HTML)
    Features:
      - auto_fix: if overall score < fix_threshold, re-run analysis with alternate tone
      - max_retries: maximum self-improvement attempts
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        auto_fix: bool = True,
        fix_threshold: float = 4.0,
        max_retries: int = 1,
    ):
        self.model = model
        self.auto_fix = auto_fix
        self.fix_threshold = float(fix_threshold)
        self.max_retries = int(max_retries)

        self.analyst = DataAnalystAgent(model=self.model)
        self.evaluator = EvaluatorAgent(model=self.model)
        self.reporter = ReportAgent(data_analyst=self.analyst, evaluator=self.evaluator, model=self.model, reuse_evaluator_assistant=True)

    def run_full_pipeline(
        self,
        path: str,
        n_insights: int = 3,
        tone: str = "concise and actionable",
        out_dir: str = "reports",
        use_llm_summary: bool = True,
        polish_report: bool = False,
        save_html: bool = True,
    ) -> Dict[str, Any]:
        
        metadata: Dict[str, Any] = {
            "path": path,
            "timestamp": datetime.datetime.now().isoformat(sep=" ", timespec="seconds"),
            "retries": 0,
            "report_path": None,
            "overall_score": None,
        }

        try:
            logger.info("Starting pipeline for %s", path)

            # Step 1: initial analysis
            logger.info("Running DataAnalystAgent.analyze (n_insights=%s)", n_insights)
            insights_text = self.analyst.analyze(path, n_insights=n_insights, tone=tone)
            metadata["insights"] = insights_text

            # Step 2: evaluation
            logger.info("Evaluating insights with EvaluatorAgent")
            evaluation = self.evaluator.evaluate(insights_text)
            metadata["evaluation"] = evaluation

            overall = None
            try:
                overall = float(evaluation.get("average_scores", {}).get("overall", 0.0))
            except Exception:
                overall = 0.0
            metadata["overall_score"] = overall

            # Step 3: self-improvement loop (attempt retries if score below threshold)
            retries = 0
            while self.auto_fix and retries < self.max_retries and overall < self.fix_threshold:
                retries += 1
                metadata["retries"] = retries
                logger.info("Overall score %.2f below threshold %.2f. Attempting retry %d/%d", overall, self.fix_threshold, retries, self.max_retries)
                # change tone to be more "clear and professional" on retry
                try:
                    improved_tone = "clear and professional"
                    insights_text = self.analyst.analyze(path, n_insights=n_insights, tone=improved_tone)
                    metadata.setdefault("improved_insights", []).append(insights_text)
                    evaluation = self.evaluator.evaluate(insights_text)
                    metadata.setdefault("improved_evaluations", []).append(evaluation)
                    overall = float(evaluation.get("average_scores", {}).get("overall", 0.0))
                    metadata["overall_score"] = overall
                except Exception as e:
                    logger.exception("Retry failed: %s", e)
                    break

            # Step 4: visuals + report generation
            logger.info("Generating report (use_llm_summary=%s,polish=%s,html=%s)", use_llm_summary, polish_report, save_html)
            report_path = self.reporter.generate_report(
                path=path,
                n_insights=n_insights,
                out_dir=out_dir,
                use_llm_summary=use_llm_summary,
                polish_report=polish_report,
                save_html=save_html,
            )
            metadata["report_path"] = report_path

            logger.info("Pipeline finished successfully. Report saved to %s", report_path)
            return metadata

        except Exception as e:
            logger.error("Pipeline failed: %s", e)
            metadata["error"] = str(e)
            metadata["traceback"] = traceback.format_exc()
            return metadata
