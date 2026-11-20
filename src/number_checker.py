"""Grade a model's answer to an Orca-Math problem: 1 if correct, 0 if wrong.

This is the GRPO reward, so it must never crash: anything it can't read scores 0.
Numbers are read with the same NUMBER pattern and clean_number as the cleaning
script, and compared as exact fractions, so "0.5", "1/2" and "2/4" are equal.
"""

import re
from fractions import Fraction

from src.prep_orca_math import NUMBER, clean_number

ANSWER_LINE = re.compile(r"answer\s*:(.*)", re.IGNORECASE)


def extract_final_answer(model_text: str) -> str | None:
    """Return the text after the LAST 'Answer:' in the model's output, or None
    if there isn't one."""
    matches = ANSWER_LINE.findall(model_text)
    return matches[-1].strip() if matches else None


def to_number(text: str) -> Fraction | None:
    """Turn answer text like '14', '$24', '1,000', '25%', '3/8' or '0.375' into
    an exact number, or None."""
    match = re.search(NUMBER, text)
    if not match:
        return None
    try:
        # Split on "/" ourselves: Fraction("3/8") works but Fraction("1/2.5") doesn't.
        numerator, _, denominator = clean_number(match.group()).partition("/")
        value = Fraction(numerator)
        return value / Fraction(denominator) if denominator else value
    except (ValueError, ZeroDivisionError):
        return None


def check(model_text: str, reference: str) -> int:
    """1 if the model's final answer equals the reference answer, else 0."""
    try:
        answer = extract_final_answer(model_text)
        if answer is None:
            return 0
        got, expected = to_number(answer), to_number(reference)
        if got is None or expected is None:
            return 0
        return int(got == expected)
    except Exception:
        return 0
