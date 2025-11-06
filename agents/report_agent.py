import sys, os; 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os,json,datetime,asyncio,re
from pathlib import Path
from typing import List, Tuple, Dict, Any,Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import markdown as md_lib
import nest_asyncio
try:
    from agents.data_analyst_agent import DataAnalystAgent
except Exception as e:
    raise ImportError("Failed to import DataAnalystAgent. Ensure all dependencies are installed.") from e

try:
    from agents.evaluator_agent import EvaluatorAgent
except Exception as e:
    raise ImportError("Failed to import EvaluatorAgent. Ensure all dependencies are installed.") from e
from dotenv import load_dotenv
load_dotenv()


os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

class ReportAgent:
    """
    Report agent that composes analyst output + evaluator feedback + visuals
    and optionally uses an AssistantAgent to (a) generate an executive summary
    and (b) polish the final markdown.
    """

    def __init__(
        self,
        data_analyst: Optional[DataAnalystAgent] = None,
        evaluator: Optional[EvaluatorAgent] = None,
        model: str = "gpt-4o-mini",
        reuse_evaluator_assistant: bool = True,
    ):
        # create or reuse analyst
        self.analyst = data_analyst or DataAnalystAgent(model=model)
        # create or reuse evaluator
        self.evaluator = evaluator or EvaluatorAgent(model=model)

        # Decide which AssistantAgent this ReportAgent uses:
        # - if reuse_evaluator_assistant True: use evaluator.assistant (reuse same LLM instance)
        # - otherwise create a fresh AssistantAgent (useful for different system prompts / rate limits)
        if reuse_evaluator_assistant and hasattr(self.evaluator, "assistant"):
            self.assistant = self.evaluator.assistant
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("Please set OPENAI_API_KEY for ReportAgent.")
            model_client = OpenAIChatCompletionClient(model=model, api_key=api_key)
            self.assistant = AssistantAgent(
                name="report_agent",
                model_client=model_client,
                description="Agent that generates human-ready reports from insights and visuals."
            )
    @staticmethod
    def _extract_text(result) -> str:
        return result.messages[-1].content

    async def _generate_exec_summary_async(self, insights_text: str, evaluation: Dict[str, Any]) -> str:
        """
        Ask the assistant to write a short executive summary (4-6 sentences).
        Uses result.messages[-1].content extraction.
        """
        prompt = (
            "You are an experienced data analyst writing an executive summary for business stakeholders.\n\n"
            "Key findings:\n"
            f"{insights_text}\n\n"
            "Evaluation (JSON):\n"
            f"{json.dumps(evaluation, indent=2)}\n\n"
            "Write a clear, concise executive summary (4-6 sentences) suitable for a non-technical manager. "
            "Make one short actionable recommendation at the end."
        )
        result = await self.assistant.run(task=prompt)
        return self._extract_text(result)

    async def _polish_markdown_async(self, raw_md: str) -> str:
        """
        Optional: ask the assistant to rewrite/polish the given markdown to be more reader-friendly.
        """
        prompt = (
            "You are a technical writer. Improve the readability and tone of the following markdown report "
            "for a business audience. Keep all sections and images, but make language clearer and concise.\n\n"
            f"REPORT_MARKDOWN:\n{raw_md}\n\n"
            "Return the improved markdown only."
        )
        result = await self.assistant.run(task=prompt)
        return self._extract_text(result)

    # Sync wrappers for convenience
    def _generate_exec_summary(self, insights_text: str, evaluation: Dict[str, Any]) -> str:
        coro = self._generate_exec_summary_async(insights_text, evaluation)
        try:
            return asyncio.run(coro)
        except RuntimeError:
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)

    def _polish_markdown(self, raw_md: str) -> str:
        coro = self._polish_markdown_async(raw_md)
        try:
            return asyncio.run(coro)
        except RuntimeError:
            
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
    def _render_html(self, markdown_text: str) -> str:
        """
        Convert Markdown to HTML.
        Prefer Python-Markdown if installed, else fallback to simple regex replacements.
        """
        try:
            # Using the official markdown library for best results
            html = md_lib.markdown(markdown_text, extensions=["tables", "fenced_code"])
        except Exception:
            # fallback minimal converter
            html = markdown_text
            html = re.sub(r'^# (.*)$', r'<h1>\1</h1>', html, flags=re.M)
            html = re.sub(r'^## (.*)$', r'<h2>\1</h2>', html, flags=re.M)
            html = re.sub(r'^### (.*)$', r'<h3>\1</h3>', html, flags=re.M)
            html = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img alt="\1" src="\2" style="max-width:100%;height:auto;">', html)
            html = html.replace("\n", "<br/>\n")

        # Wrap nicely for viewing
        wrapped = f"""
        <html>
        <head>
            <meta charset='utf-8'>
            <title>CSV Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                pre {{ background: #f4f4f4; padding: 10px; border-radius: 8px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ccc; padding: 6px; text-align: left; }}
                img {{ display: block; margin: 10px auto; max-width: 90%; }}
                h1, h2, h3 {{ color: #333; }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        return wrapped
    def _ensure_outdir(self, out_dir: str) -> Path:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def generate_report(
        self,
        path: str,
        n_insights: int = 3,
        out_dir: str = "reports",
        use_llm_summary: bool = True,
        polish_report: bool = False,
        save_html: bool = False,
    ) -> str:
        """
        Full synchronous pipeline:
          1) get insights (DataAnalystAgent)
          2) generate visuals (DataAnalystAgent.visual_summary)
          3) evaluate insights (EvaluatorAgent)
          4) get LLM executive summary (optional)
          5) build markdown, optionally polish via assistant
        Returns path to saved markdown (or polished markdown if polish_report=True).
        """
        out_path = self._ensure_outdir(out_dir)
        tstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"report_{Path(path).stem}_{tstamp}"
        md_path = out_path / f"{base}.md"

        # 1) insights
        insights_text = self.analyst.analyze(path, n_insights=n_insights)

        # 2) visuals
        visuals = {}
        try:
            visuals = self.analyst.visual_summary(path)
        except Exception as e:
            visuals = {"error": str(e)}

        # 3) evaluation
        try:
            evaluation = self.evaluator.evaluate(insights_text)
        except Exception as e:
            evaluation = {"error": str(e)}

        # 4) LLM executive summary (optional)
        exec_summary = None
        if use_llm_summary:
            try:
                exec_summary = self._generate_exec_summary(insights_text, evaluation)
            except Exception as e:
                exec_summary = f"Executive summary generation failed: {e}"

        # 5) render markdown (simple, reuse your previous rendering pattern)
        md = []
        md.append(f"# CSV Analysis Report: {Path(path).name}\n")
        md.append(f"*Generated: {datetime.datetime.now().isoformat(sep=' ', timespec='seconds')}*\n")
        md.append(f"**Data file**: `{path}`\n\n")

        if exec_summary:
            md.append("## Executive Summary (LLM-generated)\n")
            md.append(exec_summary + "\n\n")
        else:
            lines = [l.strip() for l in insights_text.strip().splitlines() if l.strip()]
            if lines:
                md.append("## Executive Summary\n")
                md.append("\n".join(lines[:3]) + "\n\n")

        md.append("## Full Insights\n")
        md.append("```\n" + insights_text.strip() + "\n```\n\n")

        md.append("## Evaluator Results\n")
        md.append("```\n" + json.dumps(evaluation, indent=2) + "\n```\n\n")

        md.append("## Visuals\n")
        vis = visuals.get("visuals") if isinstance(visuals, dict) else visuals
        if vis:
            for k, v in vis.items():
                if isinstance(v, list):
                    for fp in v:
                        md.append(f"![{Path(fp).name}]({fp})\n\n")
                elif isinstance(v, str):
                    md.append(f"![{Path(v).name}]({v})\n\n")
        else:
            md.append("_No visuals generated._\n")

        raw_md = "\n".join(md)

        # 6) optional polishing via LLM
        final_md = raw_md
        if polish_report:
            try:
                final_md = self._polish_markdown(raw_md)
            except Exception as e:
                # fallback to raw markdown if polishing fails
                final_md = raw_md + f"\n\n<!-- polishing_failed: {e} -->"

        # save
        md_path.write_text(final_md, encoding="utf-8")

        # optionally also save HTML
        if save_html:
            try:
                html = self._render_html(final_md)
                html_path = out_path / f"{base}.html"
                html_path.write_text(html, encoding="utf-8")
            except Exception as e:
                # do not fail whole pipeline because HTML rendering failed; include debug comment
                md_path.write_text(final_md + f"\n\n<!-- html_save_failed: {e} -->", encoding="utf-8")

        return str(md_path)