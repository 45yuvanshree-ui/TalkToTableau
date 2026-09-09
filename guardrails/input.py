import re

from pydantic import BaseModel


PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"disregard\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?prior\s+instructions",
    r"system\s+prompt",
    r"reveal\s+(your\s+)?instructions",
    r"show\s+(me\s+)?your\s+instructions",
    r"developer\s+message",
    r"jailbreak",
]


class InputValidationResult(BaseModel):
    is_valid: bool
    reason: str | None = None


def validate_user_input(user_input: str) -> tuple[bool, str | None]:
    """
    Validate basic user input before it enters the analytical workflow.
    """

    if not user_input or not user_input.strip():
        return False, "User input is empty."

    if len(user_input) > 10_000:
        return False, "User input exceeds the maximum allowed length."

    normalized = user_input.strip().lower()

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            return False, "Potential prompt injection detected."

    return True, None


def validate_input(user_input: str) -> InputValidationResult:
    """
    Structured input validation interface used by the orchestration workflow.
    """

    is_valid, reason = validate_user_input(user_input)

    return InputValidationResult(
        is_valid=is_valid,
        reason=reason,
    )