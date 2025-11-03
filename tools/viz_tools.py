# viz_tools.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
from autogen_core import FunctionTool
from csv_tools import load_csv_safely

def plot_numeric_histograms(path: str, out_dir: str = "plots/histograms", bins: int = 30) -> List[str]:
    """
    Plot histograms for all numeric columns and save them to files.
    Returns a list of saved file paths.
    """
    df = load_csv_safely(path)
    os.makedirs(out_dir, exist_ok=True)
    saved_files = []

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        plt.figure(figsize=(5, 4))
        plt.hist(df[col].dropna(), bins=bins, color="skyblue", edgecolor="black")
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        out_path = os.path.join(out_dir, f"{col}_hist.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        saved_files.append(out_path)
    return saved_files


def plot_correlation_heatmap(path: str, out_path: str = "plots/correlation_heatmap.png") -> str:
   
    df = load_csv_safely(path)
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr(numeric_only=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_scatter_matrix(path: str, out_path: str = "plots/scatter_matrix.png", sample_size: int = 500) -> str:
    """
    Plot a scatter matrix (pairplot) for numeric columns.
    Downsamples large datasets for performance.
    """
    df = load_csv_safely(path)
    numeric_df = df.select_dtypes(include="number")

    if len(numeric_df) > sample_size:
        numeric_df = numeric_df.sample(sample_size, random_state=42)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sns.pairplot(numeric_df)
    plt.suptitle("Scatter Matrix", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_top_correlations(
    path: str, top_k: int = 5, out_dir: str = "plots/top_correlations"
) -> List[str]:
    """
    For each of the top_k strongest correlation pairs, create a scatter plot.
    Returns list of saved plot paths.
    """
    from tools.csv_tools import compute_correlations  # safe local import

    df = load_csv_safely(path)
    corr_pairs = compute_correlations(path, top_k=top_k)
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    for c1, c2, val in corr_pairs:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        plt.figure(figsize=(5, 4))
        plt.scatter(df[c1], df[c2], alpha=0.6, color="teal")
        plt.title(f"{c1} vs {c2} (corr={val:.2f})")
        plt.xlabel(c1)
        plt.ylabel(c2)
        out_path = os.path.join(out_dir, f"{c1}_vs_{c2}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        saved.append(out_path)
    return saved

def generate_visual_summary(path: str) -> Dict[str, Any]:
    """
    Generate all standard visualizations and return metadata dictionary.
    Useful for ReportAgent integration.
    """
    plots = {
        "histograms": plot_numeric_histograms(path),
        "correlation_heatmap": plot_correlation_heatmap(path),
        "scatter_matrix": plot_scatter_matrix(path),
        "top_correlation_plots": plot_top_correlations(path),
    }
    return {"visuals": plots, "status": "success"}

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
