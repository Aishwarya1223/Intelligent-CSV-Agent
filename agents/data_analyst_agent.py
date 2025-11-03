# assistant_agent.py

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
import asyncio
import json
from typing import Dict,List,Tuple,Any,Optional
from autogen import FunctionTool
from tools.csv_tools import *
import os
from prompts import PromptTemplate,INSIGHT_TEMPLATE
from tools.viz_tools import *
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
load_dotenv()

csv_tools=[preview_csv_tool,
       summarize_numeric_tool,
       compute_correlations_tool,
       plot_column_hist_tool,
       generate_insights_tool,
       ]

class DataAnalystAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize AssistantAgent automatically."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set OPENAI_API_KEY in your environment.")

        # Create model + assistant
        model_client = OpenAIChatCompletionClient(model=model, api_key=api_key)
        all_tools = csv_tools + viz_tools

        self.assistant = AssistantAgent(
            name="data_analyst_agent",
            model_client=model_client,
            description="An intelligent CSV analyst capable of data exploration and visualization.",
            tools=all_tools
        )

        self.template = INSIGHT_TEMPLATE

    async def analyze_async(
        self, path: str, n_insights: int = 3, tone: str = "concise and actionable"
    ) -> str:
        """Analyze CSV using local tools + LLM insight generation."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        summary = summarize_numeric(path)
        corrs = compute_correlations(path, top_k=10)

        compact = {
            c: {k: v for k, v in s.items() if k in ["mean", "std", "min", "max"]}
            for c, s in summary["numeric_summary"].items()
        }

        summary_json = json.dumps({"numeric_summary": compact}, separators=(",", ":"))
        corr_json = json.dumps(
            [{"c1": c1, "c2": c2, "corr": round(val, 3)} for c1, c2, val in corrs],
            separators=(",", ":")
        )

        prompt = self.template.format(
            n_insights=n_insights,
            tone=tone,
            summary_json=summary_json,
            correlations_json=corr_json
        )

        result = await self.assistant.run(task=prompt)
        return result.messages[-1].content

    def analyze(self, path: str, n_insights: int = 3, tone: str = "concise and actionable") -> str:
        """Sync wrapper for analyze_async."""
        coro = self.analyze_async(path, n_insights, tone)
        try:
            return asyncio.run(coro)
        except RuntimeError:
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)

    def preview(self, path: str, n: int = 5) -> Dict[str, Any]:
        return preview_csv(path, n)

    def summarize(self, path: str) -> Dict[str, Any]:
        return summarize_numeric(path)

    def correlations(self, path: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        return compute_correlations(path, top_k)

    def plot_histogram(self, path: str, column: str, out_path: str = "hist.png") -> str:
        return plot_column_hist(path, column, out_path)

    def plot_all_histograms(self, path: str) -> List[str]:
        return plot_numeric_histograms(path)

    def plot_heatmap(self, path: str) -> str:
        return plot_correlation_heatmap(path)

    def plot_scatter_matrix(self, path: str) -> str:
        return plot_scatter_matrix(path)

    def plot_top_correlations(self, path: str, top_k: int = 5) -> List[str]:
        return plot_top_correlations(path, top_k)

    def visual_summary(self, path: str) -> Dict[str, Any]:
        return generate_visual_summary(path)