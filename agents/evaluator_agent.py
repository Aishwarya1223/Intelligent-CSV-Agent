import os,asyncio,json
from typing import Dict,Any,List
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


class EvaluatorAgent:
    def __init__(self,model: str='gpt-4o-mini'):
        model_client=OpenAIChatCompletionClient(model=model)
        self.assistant=AssistantAgent(
            name='evaluator_agent',
            model_client=model_client,
            description="An evaluator agent that scores insights for clarity, usefullness, and actionability."
        )
    def _build_prompt(self, insights_text: str) -> str:
        """Create structured evaluation prompt."""
        return f"""
            You are an expert evaluator assessing analytical insights.

            Evaluate each insight below on:
            - Clarity (is it understandable and well-written?)
            - Usefulness (does it provide a valuable observation?)
            - Actionability (does it give a concrete next step?)

            Each score: 1–5 (5 = excellent).

            Also, suggest improved phrasing if needed.

            Respond strictly in JSON with this format:

            {{
            "evaluations": [
                {{
                "insight": "...",
                "clarity": <1-5>,
                "usefulness": <1-5>,
                "actionability": <1-5>,
                "suggested_rewrite": "..."
                }},
                ...
            ],
            "average_scores": {{
                "clarity": <avg>,
                "usefulness": <avg>,
                "actionability": <avg>,
                "overall": <avg>
            }}
            }}

            Insights to evaluate:
            {insights_text}
                    """
    async def evaluate_async(self, insights: str) -> Dict[str, Any]:
        """Asynchronously evaluate given insights."""
        prompt = self._build_prompt(insights)
        result = await self.assistant.run(task=prompt)
        content = result.messages[-1].content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_output": content, "status": "json_parse_failed"}

    def evaluate(self, insights: str) -> Dict[str, Any]:
        """Sync wrapper for async evaluation."""
        coro = self.evaluate_async(insights)
        try:
            return asyncio.run(coro)
        except RuntimeError:
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)