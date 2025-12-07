# data_analyst_agent.py
import sys, os; 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
import asyncio
import json
from typing import Dict,List,Tuple,Any,Optional
from autogen_core.tools import FunctionTool
from tools.csv_tools import *
import os
from agents.prompts import INSIGHT_TEMPLATE

from tools.viz_tools import *
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import pathlib
import traceback
import httpx
import os

# path to your merged cert bundle (update to the actual absolute path)
os.environ["SSL_CERT_FILE"] = r"D:\Code files\Intelligent-csv-analyst\merged_cacert.pem"
os.environ["REQUESTS_CA_BUNDLE"] = r"D:\Code files\Intelligent-csv-analyst\merged_cacert.pem"


from dotenv import load_dotenv
load_dotenv()

csv_tools=[preview_csv_tool,
       summarize_numeric_tool,
       compute_correlations_tool,
       plot_column_hist_tool,
       generate_insights_tool,
       ]

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

def get_dataset_folder_for_path(csv_path: str, base_dir: str = "outputs") -> str:
    name = pathlib.Path(csv_path).stem
    folder = os.path.join(base_dir, name)
    os.makedirs(folder, exist_ok=True)
    return folder

class DataAnalystAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize AssistantAgent automatically."""

        # create an insecure HTTPX client (verify=False)
        transport = httpx.AsyncHTTPTransport(verify=False)

        model_client = OpenAIChatCompletionClient(
            model=model,
            client=httpx.AsyncClient(transport=transport, timeout=30.0)
        )

        all_tools = csv_tools + viz_tools

        self.assistant = AssistantAgent(
            name="data_analyst_agent",
            model_client=model_client,
            description="An intelligent CSV analyst capable of data exploration and visualization.",
            tools=all_tools
        )

        self.template = INSIGHT_TEMPLATE

    async def analyze_async(
    self, path: str, n_insights: int = 3, tone: str = "concise and actionable", out_dir: Optional[str] = None
) -> str:
        """Analyze CSV using local tools + LLM insight generation. Returns insight text (or raises)."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        # determine a sensible out_dir if caller didn't supply
        if out_dir is None:
            out_dir = get_dataset_folder_for_path(path)

        # numeric summary & correlations
        summary = summarize_numeric(path)
        corrs = compute_correlations(path, 10)

        # clean numeric values using pandas utilities (no hard-coded symbols)
        compact = {}
        for col, stats in summary.get("numeric_summary", {}).items():
            s = pd.to_numeric(pd.Series({k: stats.get(k) for k in ("mean","std","min","max")}), errors="coerce")
            clean = s.replace([np.inf, -np.inf], np.nan).round(3).where(~s.isna(), None)
            compact[col] = clean.to_dict()

        # clean correlations
        corr_items = []
        for item in corrs:
            c1 = item.get("c1") if isinstance(item, dict) else item[0]
            c2 = item.get("c2") if isinstance(item, dict) else item[1]
            raw = item.get("corr") if isinstance(item, dict) else item[2]
            corr_items.append({"c1": c1, "c2": c2, "corr": float(raw) if np.isfinite(raw) else None})

        summary_json = json.dumps({"numeric_summary": compact}, separators=(",", ":"), ensure_ascii=False)
        corr_json = json.dumps(corr_items, separators=(",", ":"), ensure_ascii=False)

        prompt = self.template.format(
            n_insights=n_insights,
            tone=tone,
            summary_json=summary_json,
            correlations_json=corr_json
        )

        # run LLM — catch network/connection errors and return structured error so coordinator can log it
        try:
            result = await self.assistant.run(task=prompt)
            # defensive extraction
            content = getattr(result.messages[-1], "content", None) or str(result)
            return content
        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(
                f"Connection error while calling assistant.run: {e}\n\nTraceback:\n{tb}"
            )


    def analyze(self, path: str, n_insights: int = 3, tone: str = "concise and actionable") -> str:
        """Sync wrapper that runs analyze_async robustly in any environment."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        coro = self.analyze_async(path, n_insights, tone)

        # Case A: no loop or loop not running -> create/use a fresh loop
        if loop is None or not loop.is_running():
            new_loop = loop or asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(coro)
            finally:
                try:
                    new_loop.close()
                except Exception:
                    pass
                try:
                    # unset only if we set it
                    asyncio.set_event_loop(None)
                except Exception:
                    pass
        import nest_asyncio
        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)



    def preview(self, path: str, n: int = 5) -> Dict[str, Any]:
        return preview_csv(path, n)

    def summarize(self, path: str) -> Dict[str, Any]:
        return summarize_numeric(path)

    def correlations(self, path: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        return compute_correlations(path, top_k)

    def plot_histogram(self, path: str, column: str, out_dir: Optional[str] = None, out_path: str = None) -> str:
        out_dir = out_dir or get_dataset_folder_for_path(path)
        # call tool which accepts out_dir in your viz_tools implementation
        return plot_column_hist(path, column, out_dir=out_dir, subfolder="plots/histograms", filename=out_path)

    def plot_all_histograms(self, path: str, out_dir: Optional[str] = None) -> List[str]:
        out_dir = out_dir or get_dataset_folder_for_path(path)
        return plot_numeric_histograms(path, out_dir=out_dir)

    def plot_heatmap(self, path: str, out_dir: Optional[str] = None) -> str:
        out_dir = out_dir or get_dataset_folder_for_path(path)
        return plot_correlation_heatmap(path, out_dir=out_dir)

    def plot_scatter_matrix(self, path: str, out_dir: Optional[str] = None) -> str:
        out_dir = out_dir or get_dataset_folder_for_path(path)
        return plot_scatter_matrix(path, out_dir=out_dir)

    def plot_top_correlations(self, path: str, top_k: int = 5, out_dir: Optional[str] = None) -> List[str]:
        out_dir = out_dir or get_dataset_folder_for_path(path)
        return plot_top_correlations(path, top_k=top_k, out_dir=out_dir)

    def visual_summary(self, path: str, out_dir: Optional[str] = None) -> Dict[str, Any]:
        out_dir = out_dir or get_dataset_folder_for_path(path)
        return generate_visual_summary(path, out_dir=out_dir)
 
