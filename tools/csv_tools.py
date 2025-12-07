# tools/csv_tools.py
# Leaner, safer CSV utilities for quick analysis

import os
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from autogen_core.tools import FunctionTool

MAX_BYTES = 50 * 1024 * 1024

def load_csv_safely(path: str, max_rows: int = 100_000) -> pd.DataFrame:
    """Load CSV robustly: sniff delimiter, try utf-8/latin1, sample if large, coerce numeric-like cols."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    size = os.path.getsize(path)
    if size > MAX_BYTES:
        raise ValueError(f"File too large (> {MAX_BYTES} bytes).")

    with open(path, "rb") as fb:
        sample_bytes = fb.read(16_384)
    for enc in ("utf-8", "latin1"):
        try:
            sample = sample_bytes.decode(enc, errors="replace")
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample)
                sep = dialect.delimiter
            except Exception:
                sep = None
            try:
                if sep:
                    df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                else:
                    df = pd.read_csv(path, encoding=enc, low_memory=False)
                break
            except Exception:
                continue
        except Exception:
            continue
    else:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    for col in df.columns:
        if df[col].dtype == "object":
            sample_vals = df[col].dropna().astype(str).head(50).tolist()
            numeric_like = 0
            for v in sample_vals:
                vv = v.replace(",", "").replace("$", "").replace("€", "").replace("₹", "").strip()
                try:
                    float(vv)
                    numeric_like += 1
                except Exception:
                    pass
            if len(sample_vals) > 0 and numeric_like / len(sample_vals) > 0.6:
                df[col] = df[col].astype(str).str.replace(r"[,\$\€\₹\s]", "", regex=True)
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
def preview_csv(path: str, n_rows: int = 5) -> Dict[str, Any]:
    df = load_csv_safely(path)
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "preview": df.head(n_rows).to_dict(orient="records"),
    }

def summarize_numeric(path: str, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    df = load_csv_safely(path)
    num = df.select_dtypes(include="number")
    if columns:
        missing_cols = [c for c in columns if c not in num.columns]
        if missing_cols:
            raise KeyError(f"Requested numeric columns not found: {missing_cols}")
        num = num[columns]
    desc = num.describe(percentiles=[0.25, 0.5, 0.75]).T
    return {"numeric_summary": desc.to_dict(orient="index")}

def missing_value_summary(path: str) -> Dict[str, Any]:
    df = load_csv_safely(path)
    total = len(df)
    missing = df.isna().sum().to_dict()
    return {
        "missing_counts": missing,
        "missing_percentages": {k: (v / total) * 100 for k, v in missing.items()},
    }


def compute_correlations(path: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Return top_k absolute correlations among numeric columns as list of dicts:
    [{"c1":"colA","c2":"colB","corr":0.87}, ...]
    """
    df = load_csv_safely(path)
    corr = df.corr(numeric_only=True)
    pairs: List[Tuple[str, str, float]] = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    pairs.sort(key=lambda x: abs(x[2]) if x[2] is not None else 0.0, reverse=True)
    return [{"c1": a, "c2": b, "corr": float(np.nan_to_num(v, nan=0.0))} for a, b, v in pairs[:top_k]]


def plot_column_hist(path: str,column: str,out_dir: str = "outputs",subfolder: str = "plots/histograms",filename: Optional[str] = None,bins: int = 30,) -> str:
    df = load_csv_safely(path)
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    folder = os.path.join(out_dir, subfolder)
    os.makedirs(folder, exist_ok=True)
    filename = filename or f"{column}_hist.png"
    out_path = os.path.join(folder, filename)

    plt.figure(figsize=(6, 4))
    plt.hist(df[column].dropna(), bins=bins)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def generate_insights(summary: Dict[str, Any], correlations: List[Dict[str, Any]]) -> str:
    """
    Build short plain-English insights.
    `summary` should be the output of summarize_numeric().
    `correlations` should be output of compute_correlations().
    """
    num_cols = list(summary.get("numeric_summary", {}).keys())
    lines = [f"Analyzed {len(num_cols)} numeric columns."]
    for col, stats in summary.get("numeric_summary", {}).items():
        mean = stats.get("mean")
        if mean is None or np.isnan(mean):
            lines.append(f"- {col}: no numeric data")
        else:
            lines.append(f"- {col}: avg={mean:.2f}, std={stats.get('std', float('nan')):.2f}")
    if correlations:
        lines.append("\nTop correlations:")
        for c in correlations:
            lines.append(f"  - {c['c1']} vs {c['c2']}: corr={c['corr']:.2f}")
    return "\n".join(lines)


preview_csv_tool = FunctionTool(preview_csv, description="Preview CSV (rows, columns, sample).")
summarize_numeric_tool = FunctionTool(summarize_numeric, description="Summary stats for numeric columns.")
compute_correlations_tool = FunctionTool(compute_correlations, description="Top numeric correlations (dicts).")
plot_column_hist_tool = FunctionTool(
    plot_column_hist,
    description="Save a histogram for a numeric column. Accepts out_dir to place file under dataset outputs."
)
generate_insights_tool = FunctionTool(generate_insights, description="Generate plain-English insights.")