from dataclasses import dataclass
import re

from exact.common.schemas import PredictionRequest, QuestionType, TaskType

_OPTION_LABEL_RE = re.compile(r"(?:^|\n)\s*\(?([A-D])[.)]\s+", re.IGNORECASE)

# YNU stem: a modal/interrogative verb at (or near) the start of the stem.
# Allows an optional leading clause such as "Based on the premises," /
# "According to the premises," / "If <condition>," before the verb, because
# many dataset questions phrase the modal after an introductory clause
# (e.g. "If a student does X, will they Y?").
_YNU_STEM_RE = re.compile(
    r"^\s*(?:(?:based on|according to)[^,?.]*,\s*)?"
    r"(?:does|do|did|is|are|was|were|has|have|had|can|could|will|would|should|must|may)\b",
    re.IGNORECASE,
)

# Secondary YNU signal: a modal verb appears anywhere, OR the stem opens with
# a "which premises/statement/conclusion ... correct/true/support" framing whose
# gold answer is Yes/No/Uncertain in this dataset.  Used as a recall booster
# after MCQ detection so genuine multiple-choice questions are not captured.
_YNU_INLINE_RE = re.compile(
    r"\b(?:is|are|was|were|does|do|did|will|would|can|could|should|must)\b"
    r"[^?]*\?",
    re.IGNORECASE,
)
_YNU_FRAMING_RE = re.compile(
    r"\bwhich\s+(?:premise|premises|statement|conclusion|option)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteDecision:
    """Routing outcome for a prediction request.

    Attributes:
        task_type: Type 1 (logic) or Type 2 (physics).
        reason: Human-readable routing rationale for logs and debugging.
        question_type: Shape of the Type 1 question (MCQ, yes/no/uncertain, etc.).
            When routing cannot classify the stem, defaults to YES_NO_UNCERTAIN so
            Type 1 still attempts a solvable label instead of open-ended (see
            ``detect_question_type``).
    """

    task_type: TaskType
    reason: str
    question_type: QuestionType = QuestionType.UNKNOWN


# Backward-compatible alias for older imports.
# RouterDecision = RouteDecision


class TaskRouter:
    """Routes incoming requests to the Type 1 or Type 2 pipeline."""

    def route(self, request: PredictionRequest) -> RouteDecision:
        if request.premises_nl:
            # Premises present → Type 1 logic task.
            question_type = detect_question_type(request)

            return RouteDecision(
                task_type=TaskType.TYPE1_LOGIC,
                reason=f"premises_nl present; question_type={question_type.value}",
                question_type=question_type,
            )
        # No premises → Type 2 physics/numerical task.
        return RouteDecision(
            task_type=TaskType.TYPE2_PHYSICS,
            reason="premises_nl absent",
            question_type=QuestionType.NUMERICAL,
        )


def detect_question_type(request: PredictionRequest) -> QuestionType:
    """Classify a Type 1 question stem for pipeline dispatch.

    Detection order: MCQ (A+B option labels) → yes/no/uncertain heuristics →
    default YES_NO_UNCERTAIN. The Type 1 dataset has no free-text gold answers;
    defaulting unmatched stems to YES_NO_UNCERTAIN (rather than OPEN_ENDED)
    keeps them on a path that can produce Yes/No/Uncertain instead of scoring zero.

    Args:
        request: Prediction request whose ``question`` field is inspected.

    Returns:
        Detected ``QuestionType`` (MCQ, YES_NO_UNCERTAIN, or OPEN_ENDED if ever
        returned by an explicit rule).
    """
    question = request.question or ""

    # Question shape belongs in routing so Type 1 can focus on proving the routed goal type.
    # MCQ: at least two consecutive option labels starting from A (A+B, A+B+C, A+B+C+D).
    # The dataset contains 2- and 3-option MCQs whose gold is a letter, so requiring all
    # of A/B/C/D mis-routed them to open-ended. Requiring {A, B} captures every real MCQ
    # while staying safe: a Yes/No question with an enumerated "A. ... B. ..." body is in
    # fact a multiple-choice question in this dataset.
    labels = {match.group(1).upper() for match in _OPTION_LABEL_RE.finditer(question)}
    if {"A", "B"}.issubset(labels):
        return QuestionType.MCQ

    # YNU: modal verb at start, inline modal question, or "which premise/statement…" framing.
    if (
        _YNU_STEM_RE.search(question)
        or _YNU_INLINE_RE.search(question)
        or _YNU_FRAMING_RE.search(question)
    ):
        return QuestionType.YES_NO_UNCERTAIN

    # Empirically the Type 1 dataset has no free-text answers — every non-MCQ question
    # resolves to Yes/No/Uncertain. Default the fallback to YNU (rather than OPEN_ENDED)
    # so an unmatched question still attempts a solvable answer instead of scoring 0.
    return QuestionType.YES_NO_UNCERTAIN
