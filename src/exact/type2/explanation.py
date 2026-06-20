from __future__ import annotations

from exact.type2.schemas import SolutionTrace


class ExplanationComposer:
    def __init__(self, settings=None) -> None:
        self.settings = settings

    def compose(self, trace: SolutionTrace, answer: str, unit: str) -> str:
        explanation = self.compose_with_template(trace, answer, unit)
        if self.is_valid(explanation, answer, unit):
            return explanation
        return "The system could not determine a reliable answer from the given information."

    def compose_with_template(self, trace: SolutionTrace, answer: str, unit: str) -> str:
        sentences: list[str] = []
        for step in trace.steps:
            if step.kind == "verification":
                continue
            sentences.append(step.description)
            if step.substituted_expression:
                sentences.append(step.substituted_expression)
            elif step.expression and step.kind in {"formula", "network_reduction", "final_result"}:
                sentences.append(step.expression)
        final_text = f"The final answer is {answer} {unit}.".strip()
        sentences.append(final_text)
        return " ".join(part.strip() for part in sentences if part and part.strip())

    def is_valid(self, explanation: str, answer: str, unit: str) -> bool:
        if not explanation.strip():
            return False
        if answer and answer not in explanation:
            return False
        if unit and unit not in explanation:
            return False
        return True
