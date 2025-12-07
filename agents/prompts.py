INSIGHT_TEMPLATE = """
You are a pragmatic data analyst. Convert the following compact JSON statistics into 
{n_insights} clear, human-friendly insights. Each insight should be 1–2 sentences.

Do NOT assume the data is related to sales, marketing, business, or product strategy unless explicitly stated.
Avoid causal claims. Base insights only on the provided numbers and correlations.

If the dataset appears to be financial or commodity prices (e.g., gold, stocks), frame insights in terms of:
- trends
- volatility
- relationships between variables
- risk/market behavior
- statistical interpretation

For each insight, include ONE short, domain-correct suggestion such as:
- “Monitor this metric.”
- “Investigate the underlying drivers.”
- “Consider exploring volatility patterns.”
- “Review historical trends for context.”

Never output business-specific actions like ‘boost sales’, ‘adjust pricing’, ‘optimize strategy’, 
unless the dataset explicitly includes product or business variables.

SUMMARY_JSON:
{summary_json}

CORRELATIONS_JSON:
{correlations_json}

Output format (exact):
1) <Insight sentence>. Recommendation: <short action>
2) ...
"""
