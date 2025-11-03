# prompts.py

class PromptTemplate:
    """A minimal prompt template for formatting variables into text."""
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        """Fill in placeholders like {summary_json} with provided values."""
        return self.template.format(**kwargs)


# === Insight generation template ===
INSIGHT_TEMPLATE = PromptTemplate(
    """You are a pragmatic data analyst. Convert the following compact JSON statistics into 
{n_insights} clear, human-friendly insights. Each insight should be 1–2 sentences and end with 
a one-line suggested action. Keep the tone {tone}.

SUMMARY_JSON:
{summary_json}

CORRELATIONS_JSON:
{correlations_json}

Output format (exact):
1) <Insight sentence>. Recommendation: <one short action>
2) ...
"""
)
