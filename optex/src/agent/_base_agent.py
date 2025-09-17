import time
from typing import List, Tuple

from optex.src.buffer import BaseExemplarBuffer
from optex.src.episode import BaseEpisode
from optex.src.exemplars import BaseExemplar


class BaseAgent:
    """
    Replace the two TODOs:
      1) _call_llm_for_plan() – plan + tool arguments given instruction + exemplars
      2) _execute_actions() – call your environment's tools, return reward and action trace
    """

    def __init__(self, selector: BaseExemplar, k: int):
        self.selector = selector
        self.k = k

    @staticmethod
    def build_prompt(instruction: str, exemplars: List[BaseEpisode]) -> str:
        parts = []
        for i, e in enumerate(exemplars):
            act_preview = " | ".join(e.actions[:6])
            parts.append(
                f"# Exemplar {i + 1} (reward={e.reward:.2f})\n"
                f"Instruction: {e.instruction}\n"
                f"Plan: {e.plan}\n"
                f"Actions: {act_preview}\n"
                f"Outcome: {e.reward}\n"
            )
        parts.append(f"# Current\nInstruction: {instruction}\nPlan:")
        return "\n".join(parts)

    @staticmethod
    def _call_llm_for_plan(prompt: str) -> str:
        # TODO: Replace with your LLM API call (OpenAI/Claude/Gemini/etc.)
        # Return a short, structured "plan" string.
        # For now, we mock a trivial plan.
        return "think → search → filter → act"

    @staticmethod
    def _execute_actions(plan: str, instruction: str) -> Tuple[List[str], float]:
        # TODO: Replace with environment execution (WebShop, MiniWoB++, etc.)
        # Return (actions_list, reward). Here, we mock it deterministically/randomly.
        actions = plan.split(" → ")
        reward = 1.0 if "search" in plan else 0.0
        return actions, reward

    def run_episode(self, buffer: BaseExemplarBuffer, instruction: str) -> BaseEpisode:
        exemplars = self.selector.select(buffer, instruction)
        prompt = self.build_prompt(instruction, exemplars)
        plan = self._call_llm_for_plan(prompt)
        actions, reward = self._execute_actions(plan, instruction)
        return BaseEpisode(
            instruction=instruction,
            plan=plan,
            actions=actions,
            reward=reward,
            timestamp=time.time()
        )
