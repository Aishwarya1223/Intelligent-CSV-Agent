# tools/csv_tools.py
# Auto-generated placeholder
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from typing import Dict, Any, Optional,List,Tuple
from autogen_core.tools import FunctionTool

def load_csv_safely(path: str,max_rows: int = 100000)->pd.DataFrame:
    
    if not os.path.exists(path):
        raise ValueError("File not found!")
    file_size=os.path.getsize(path)
    if file_size > 50 *1024 *1024:
        raise ValueError("File is too large for analysis (limit=50MB)")
    
    df=pd.read_csv(path)
    if len(df) > max_rows:
        df=df.sample(max_rows,random_state=42)
    return df

def preview_csv(path: str, n_rows: int=5) -> Dict[str,Any]:
    df=load_csv_safely(path)
    preview=df.head(n_rows).to_dict(orient='records')
    return{
        'num_rows':len(df),
        'num_columns': len(df.columns),
        "columns":df.columns.tolist(),
        "preview":preview,
    }
def summarize_numeric(path:str,
                      columns: Optional[List[str]]=None)->Dict[str,Any]:
    df=load_csv_safely(path)
    num_df=df.select_dtypes(include='number')
    if columns:
        num_df=num_df[columns]
    desc=num_df.describe(percentiles=[0.25,0.5,0.75]).T
    summary=desc.to_dict(orient='index')
    return {"numeric_summary":summary}

def missing_value_summary(path:str)->Dict[str,Any]:
    df=load_csv_safely(path)
    missing=df.isna().sum().to_dict()
    total=len(df)
    return{
        "missing_counts":missing,
        "missing_percentages":{k:(v/total) * 100 for k, v in missing.items()}
    }

def compute_correlations(path: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
    """
    Compute top_k absolute correlations among numeric columns.
    Returns a list of (col1, col2, correlation).
    """
    df = load_csv_safely(path)
    corr = df.corr(numeric_only=True)
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:top_k]
def plot_column_hist(path: str, column: str, out_path: str = "histogram.png", bins: int = 30) -> str:
    """
    Plot a histogram for a numeric column and save to file.
    """
    df = load_csv_safely(path)
    if column not in df.columns:
        raise ValueError(f"Column {column} not found.")
    plt.figure(figsize=(6, 4))
    plt.hist(df[column].dropna(), bins=bins, color="skyblue", edgecolor="black")
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

def generate_insights(summary: Dict[str, Any], correlations: List[Dict[str, Any]]) -> str:
    """
    Generate plain-English insights from numeric summaries and correlations.
    correlations: list of dicts like {"c1": "colA", "c2": "colB", "corr": 0.87}
    """
    lines = []
    lines.append(f"Analyzed {len(summary.get('numeric_summary', {}))} numeric columns.")
    for col, stats in summary.get("numeric_summary", {}).items():
        mean = stats.get("mean")
        if mean is None:
            lines.append(f"- {col}: no numeric data")
        else:
            lines.append(f"- {col}: avg={mean:.2f}")
    if correlations:
        lines.append("\nStrongest correlations:")
        for item in correlations:
            c1 = item.get("c1")
            c2 = item.get("c2")
            val = item.get("corr")
            try:
                valf = float(val)
                lines.append(f"  - {c1} vs {c2}: corr={valf:.2f}")
            except Exception:
                lines.append(f"  - {c1} vs {c2}: corr={val}")
    return "\n".join(lines)



preview_csv_tool=FunctionTool(preview_csv,
                              description="Gives a preview of the csv file")
summarize_numeric_tool=FunctionTool(summarize_numeric,
                                    description="Gives a summary of numerical columns in the csv")

compute_correlations_tool=FunctionTool(compute_correlations,
                                  description='Computes top_k absolute correlations among numeric columns.')
plot_column_hist_tool=FunctionTool(plot_column_hist,
                                   description='Plot a histogram for a numeric column and save to file.')

generate_insights_tool = FunctionTool(
    generate_insights,
    description='Generate plain-English insights from numeric summaries and correlations.',
)

