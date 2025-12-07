# Evaluator_agent.py
import os
import json
import re
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEBUG_PREVIEW_LEN = 400
EVAL_PROMPT = (
    "You are an expert evaluator. Your job is to evaluate each insight strictly and return valid JSON only.\n\n"
    "Rules:\n"
    "- Return ONLY a single JSON object.\n"
    "- No markdown.\n"
    "- No code fences.\n"
    "- No text outside JSON.\n\n"
    "The JSON MUST follow this schema exactly:\n"
    "{{\n"
    "  \"evaluations\": [\n"
    "      {{\"insight\": \"<text>\", \"clarity\": 0, \"usefulness\": 0, \"actionability\": 0, \"suggested_rewrite\": \"<text>\"}}\n"
    "  ],\n"
    "  \"average_scores\": {{\"clarity\": 0, \"usefulness\": 0, \"actionability\": 0, \"overall\": 0}}\n"
    "}}\n\n"
    "INSIGHTS TO EVALUATE:\n{insights}\n\n"
    "If valid JSON cannot be produced, return exactly:\n"
    "{{\"evaluations\":[],\"average_scores\":{{\"clarity\":0,\"usefulness\":0,\"actionability\":0,\"overall\":0}}}}"
)

class EvaluatorAgent:
    """Compact evaluator for quick testing. Accepts extra kwargs to remain compatible with callers."""

    def __init__(self, model: str = "gpt-4o-mini", timeout: float = 30.0):
        self.model = model
        self.timeout = timeout
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.url = "https://api.openai.com/v1/chat/completions"

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        if not text:
            return text
        t = text.strip()
        if t.startswith("```") and t.endswith("```"):
            return "\n".join(t.splitlines()[1:-1]).strip()
        return re.sub(r"^`(.+)`$", r"\1", t).strip()

    @staticmethod
    def _find_first_json_block(text: str) -> Optional[str]:
        if not text:
            return None
        # find first { or [
        start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        open_ch = text[start]
        close_ch = "}" if open_ch == "{" else "]"
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            else:
                if ch == '"':
                    in_string = True
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        return None
    @staticmethod
    def _extract_array_after_key(text: str, key: str = '"evaluations"') -> Optional[str]:
        """
        If the response contains '"evaluations": [ ... ]' but no enclosing object,
        extract and return the array string (including brackets).
        """
        if not text:
            return None
        pattern = re.compile(rf'{re.escape(key)}\s*:\s*(\[[\s\S]*?\])', re.MULTILINE)
        m = pattern.search(text)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _cleanup_and_load_json(candidate: str) -> Any:
        if not candidate:
            raise json.JSONDecodeError("No candidate", "", 0)
        t = candidate.strip()
        if t.startswith("```") or t.endswith("```"):
            t = EvaluatorAgent._strip_code_fence(t)
        if "'" in t and '"' not in t:
            t = t.replace("'", '"')
        t = re.sub(r",\s*(\}|\])", r"\1", t)
        return json.loads(t)

    @staticmethod
    def _compute_averages(evals: List[Dict[str, Any]]) -> Dict[str, float]:
        if not evals:
            return {"clarity": 0, "usefulness": 0, "actionability": 0, "overall": 0}
        n = len(evals)
        c = sum(float(e.get("clarity", 0)) for e in evals) / n
        u = sum(float(e.get("usefulness", 0)) for e in evals) / n
        a = sum(float(e.get("actionability", 0)) for e in evals) / n
        return {
            "clarity": round(c, 2),
            "usefulness": round(u, 2),
            "actionability": round(a, 2),
            "overall": round((c + u + a) / 3, 2),
        }

    def _ensure_eval_shape(self, parsed: Any, raw_text: str = "") -> Dict[str, Any]:
        if isinstance(parsed, dict) and "evaluations" in parsed and isinstance(parsed["evaluations"], list):
            avg = self._compute_averages(parsed["evaluations"])
            return {"evaluations": parsed["evaluations"], "average_scores": avg}

        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            avg = self._compute_averages(parsed)
            return {"evaluations": parsed, "average_scores": avg}

        block = self._find_first_json_block(raw_text)
        if block:
            try:
                parsed2 = self._cleanup_and_load_json(block)
                return self._ensure_eval_shape(parsed2, raw_text=block)
            except Exception:
                pass
        return {"status": "parse_failed", "raw": raw_text[:_DEBUG_PREVIEW_LEN], "error": "cannot_shape"}

    async def _call_async(self, insights: str, model_override: Optional[str] = None) -> Dict[str, Any]:
        model = model_override or self.model
        if not self.api_key:
            return {"status": "agent_failed", "error": "no_api_key", "raw": ""}

        system_msg = {
            "role": "system",
            "content": "You are an expert evaluator. Reply with a single valid JSON object exactly matching the schema and nothing else."
        }
        user_msg = {"role": "user", "content": EVAL_PROMPT.format(insights=insights or "")}

        payload = {
            "model": str(model),
            "messages": [system_msg, user_msg],
            "temperature": 0.0,
            "max_tokens": 800,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    self.url, headers={"Authorization": f"Bearer {self.api_key}"}, json=payload
                )
            if r.status_code >= 400:
                return {"status": "agent_failed", "error": f"rest_error:{r.status_code}", "raw": r.text[:200]}
            resp = r.json()
        except Exception as e:
            return {"status": "agent_failed", "error": f"request_exception:{e}", "raw": ""}

        text = ""
        parsed = None
        try:
            choices = resp.get("choices", [])
            if choices:
                msg = choices[0].get("message", {}) or {}
                content = msg.get("content")
                if isinstance(content, (dict, list)):
                    parsed = content
                    text = json.dumps(content)
                elif isinstance(content, str):
                    text = content.strip()
        except Exception:
            text = str(resp)

        if parsed is None:
            try:
                parsed = self._cleanup_and_load_json(text)
            except Exception:
                block = self._find_first_json_block(text)
                if block:
                    try:
                        parsed = self._cleanup_and_load_json(block)
                    except Exception:
                        parsed = None
                else:
                    arr_text = self._extract_array_after_key(text, key='"evaluations"')
                    if arr_text:
                        try:
                            parsed = self._cleanup_and_load_json(arr_text)
                        except Exception:
                            parsed = None


        result = self._ensure_eval_shape(parsed, raw_text=text)
        status = "success" if result.get("evaluations") else result.get("status", "parse_failed")
        return {"status": status, "result": result, "raw": text[:_DEBUG_PREVIEW_LEN]}

    def evaluate(self,insights= None,models= None,ensemble_method= "mean",weights: Optional[Dict[str, float]] = None,**kwargs,) -> Dict[str, Any]:
       
        model_override = None
        if models:
            first = models[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(first[1], dict):
                model_override = first[1].get("model") or first[1].get("name")
            elif isinstance(first, str):
                model_override = first

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            def run_coro():
                return asyncio.new_event_loop().run_until_complete(
                    self._call_async(insights or "", model_override=model_override)
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(run_coro).result()
        else:
            return asyncio.run(self._call_async(insights or "", model_override=model_override))