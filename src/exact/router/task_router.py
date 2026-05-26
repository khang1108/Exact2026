from dataclasses import dataclass
import re

from exact.common.schemas import PredictionRequest, QuestionType, TaskType

_OPTION_LABEL_RE = re.compile(r"(?:^|\n)\s*([A-D])\.\s+", re.IGNORECASE)
_YNU_STEM_RE = re.compile(
    r"^\s*(?:based on the above premises,\s*)?"
    r"(?:does|do|did|is|are|can|could|will|would|should)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteDecision:
    """
    Một schema dùng để lưu trữ các thông tin về quyết xem sẽ chuyển hướng tới loại nào.

    Args:
        task_type (TaskType): Kiểu dữ liệu của task
        reason (str): Lý do tại sao quyết định, dùng để debug.
        question_type (QuestionType): Kiểu dữ liệu của câu hỏi đó, nó là MCQ, YUN, hay là open-ended.
    """

    task_type: TaskType
    reason: str
    question_type: QuestionType = QuestionType.UNKNOWN


# Backward-compatible alias for older imports.
# RouterDecision = RouteDecision


class TaskRouter:
    """
    TaskRouter là một object để điều hướng và quyết định xem sẽ chuyển task tới module nào để xử lý.

    Methods:
        route(request: PredictionRequest) -> RouteDecision: Dùng để điều hướng request.
    """

    def route(self, request: PredictionRequest) -> RouteDecision:
        if request.premises_nl:
            # Nếu có premises, tức là đây là data Type 1.
            question_type = detect_question_type(request)

            return RouteDecision(
                task_type=TaskType.TYPE1_LOGIC,
                reason=f"premises_nl present; question_type={question_type.value}",
                question_type=question_type,
            )
        # Còn không sẽ là Type 2.
        return RouteDecision(
            task_type=TaskType.TYPE2_PHYSICS,
            reason="premises_nl absent",
            question_type=QuestionType.NUMERICAL,
        )


def detect_question_type(request: PredictionRequest) -> QuestionType:
    """
    Dùng để xác định loại câu hỏi.

    Args:
        request (PredictionRequest): Là một request được truyền tới.

    Returns:
        QuestionType: Là loại câu hỏi vừa xác định được (MCQ, YES_NO_UNCERTAIN, OPEN_ENDED)
    """
    question = request.question or ""

    # Question shape belongs in routing so Type 1 can focus on proving the routed goal type.
    labels = [match.group(1).upper() for match in _OPTION_LABEL_RE.finditer(question)]
    if {"A", "B", "C", "D"}.issubset(labels):
        return QuestionType.MCQ

    if _YNU_STEM_RE.search(question):
        return QuestionType.YES_NO_UNCERTAIN

    return QuestionType.OPEN_ENDED
