# viz_tools.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Optional, Dict, Any
from autogen_core.tools import FunctionTool
from tools.csv_tools import load_csv_safely, compute_correlations

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _plots_base(out_dir: str) -> str:
    """Return base plots folder inside dataset out_dir."""
    base = os.path.join(out_dir, "plots")
    _ensure_dir(base)
    return base

# inside plot_numeric_histograms(...)
def plot_numeric_histograms(path: str, out_dir: str = "outputs", bins: int = 30) -> List[str]:
    df = load_csv_safely(path)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return []
    hist_dir = os.path.join(out_dir, "plots", "histograms")
    os.makedirs(hist_dir, exist_ok=True)
    saved_files = []
    for col in numeric_cols:
        out_path = os.path.join(hist_dir, f"{col}_hist.png")
        plt.figure(figsize=(5, 4))
        plt.hist(df[col].dropna(), bins=bins)
        plt.title(f"Histogram of {col}")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        saved_files.append(out_path)
    return saved_files



def plot_correlation_heatmap(path: str, out_dir: str = "outputs") -> str:
    """
    Save correlation heatmap to <out_dir>/plots/correlation_heatmap.png
    """
    try:
        df = load_csv_safely(path)
    except Exception as e:
        print(f"[plot_correlation_heatmap] failed to load CSV: {e}")
        return ""

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        print("[plot_correlation_heatmap] no numeric columns found.")
        return ""

    corr = numeric_df.corr(numeric_only=True)
    out_base = _plots_base(out_dir)
    out_path = os.path.join(out_base, "correlation_heatmap.png")
    _ensure_dir(os.path.dirname(out_path))

    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, cmap="coolwarm", annot=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path
    except Exception as e:
        print(f"[plot_correlation_heatmap] plotting failed: {e}")
        return ""


def plot_scatter_matrix(path: str, out_dir: str = "outputs", sample_size: int = 500) -> str:
    """
    Save a scatter matrix (pairplot) to <out_dir>/plots/scatter_matrix.png
    Downsamples large datasets for performance.
    """
    try:
        df = load_csv_safely(path)
    except Exception as e:
        print(f"[plot_scatter_matrix] failed to load CSV: {e}")
        return ""

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        print("[plot_scatter_matrix] no numeric columns found.")
        return ""

    if len(numeric_df) > sample_size:
        numeric_df = numeric_df.sample(sample_size, random_state=42)

    out_base = _plots_base(out_dir)
    out_path = os.path.join(out_base, "scatter_matrix.png")
    _ensure_dir(os.path.dirname(out_path))

    try:
        sns.pairplot(numeric_df)
        plt.suptitle("Scatter Matrix", y=1.02)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path
    except Exception as e:
        print(f"[plot_scatter_matrix] plotting failed: {e}")
        return ""


def plot_top_correlations(
    path: str, top_k: int = 5, out_dir: str = "outputs"
) -> List[str]:
    """
    For each of the top_k strongest correlation pairs, create a scatter plot.
    Saved to <out_dir>/plots/top_correlations/
    Returns list of saved plot paths.
    """
    try:
        df = load_csv_safely(path)
    except Exception as e:
        print(f"[plot_top_correlations] failed to load CSV: {e}")
        return []

    # compute_correlations returns list of (c1, c2, corr) or dicts depending on your csv_tools
    try:
        corr_pairs = compute_correlations(path, top_k=top_k)
    except Exception as e:
        print(f"[plot_top_correlations] compute_correlations failed: {e}")
        corr_pairs = []

    out_base = _plots_base(out_dir)
    out_top = os.path.join(out_base, "top_correlations")
    _ensure_dir(out_top)

    saved = []
    # handle both tuple and dict formats
    for item in corr_pairs:
        if isinstance(item, dict):
            c1 = item.get("c1")
            c2 = item.get("c2")
            val = item.get("corr", 0.0)
        else:
            c1, c2, val = item

        if c1 not in df.columns or c2 not in df.columns:
            continue
        try:
            plt.figure(figsize=(5, 4))
            plt.scatter(df[c1], df[c2], alpha=0.6)
            plt.title(f"{c1} vs {c2} (corr={val:.2f})")
            plt.xlabel(c1)
            plt.ylabel(c2)
            out_path = os.path.join(out_top, f"{c1}_vs_{c2}.png")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            saved.append(out_path)
        except Exception as e:
            print(f"[plot_top_correlations] failed for {c1} vs {c2}: {e}")
    return saved


def generate_visual_summary(path: str, out_dir: str = "outputs") -> Dict[str, Any]:
    """
    Generate all standard visualizations and return metadata dictionary.
    Saves into <out_dir>/plots/...
    """
    visuals = {}
    visuals["histograms"] = plot_numeric_histograms(path, out_dir=out_dir)
    visuals["correlation_heatmap"] = plot_correlation_heatmap(path, out_dir=out_dir)
    visuals["scatter_matrix"] = plot_scatter_matrix(path, out_dir=out_dir)
    visuals["top_correlation_plots"] = plot_top_correlations(path, out_dir=out_dir)
    return {"visuals": visuals, "status": "success"}


# FunctionTool wrappers (unchanged signatures but now accept out_dir)
try:
    plot_histograms_tool = FunctionTool(
        plot_numeric_histograms, description="Plots histograms for all numeric columns in a CSV."
    )
    plot_correlation_heatmap_tool = FunctionTool(
        plot_correlation_heatmap, description="Generates a correlation heatmap for numeric columns."
    )
    plot_scatter_matrix_tool = FunctionTool(
        plot_scatter_matrix, description="Plots a scatter matrix for numeric columns."
    )
    plot_top_correlations_tool = FunctionTool(
        plot_top_correlations, description="Creates scatter plots for top correlated column pairs."
    )
    generate_visual_summary_tool = FunctionTool(
        generate_visual_summary, description="Generates all standard visualizations and returns file paths."
    )

    viz_tools = [
        plot_histograms_tool,
        plot_correlation_heatmap_tool,
        plot_scatter_matrix_tool,
        plot_top_correlations_tool,
        generate_visual_summary_tool,
    ]
except Exception:
    viz_tools = []