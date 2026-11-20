"""Tests for src/number_checker.py: one case per row of the team's spec, plus edge cases."""

from fractions import Fraction

import pytest

from src.number_checker import check, extract_final_answer, to_number


@pytest.mark.parametrize(
    "model_text, reference, expected",
    [
        ("23 - 9 = 14.\nAnswer: 14", "14", 1),
        ("Answer: 14.", "14", 1),  # trailing period
        ("Answer: 15", "14", 0),
        ("The answer is 14.", "14", 0),  # no "Answer:" line
        ("Answer: 1,000", "1000", 1),
        ("Answer: 0.5", "1/2", 1),
        ("Answer: 6/16", "3/8", 1),
        ("Answer: $24", "24", 1),
        ("Answer: 25%", "25", 1),
        ("Answer: 14 peaches.", "14", 1),
        ("Answer: -5", "-5", 1),
        ("Answer: 10\n...\nAnswer: 12", "12", 1),  # uses the last one
        ("answer: 14", "14", 1),  # lowercase
        ("Answer: fourteen", "14", 0),
        ("", "14", 0),
    ],
)
def test_check_spec_table(model_text, reference, expected):
    assert check(model_text, reference) == expected


@pytest.mark.parametrize(
    "model_text, reference, expected",
    [
        ("Answer: 4", "4.00", 1),  # references like "$4.00" were saved as "4.00"
        ("Answer: 108", "108.0", 1),
        ("Answer: 12\nAnswer: 10", "12", 0),  # an earlier correct answer doesn't count
        ("Answer:", "14", 0),  # empty answer
        ("Answer: 1/0", "14", 0),  # division by zero doesn't crash
        ("Answer: 14", "", 0),  # unreadable reference
    ],
)
def test_check_edge_cases(model_text, reference, expected):
    assert check(model_text, reference) == expected


def test_extract_final_answer_takes_rest_of_last_line():
    assert extract_final_answer("Answer: 10\nwork\nANSWER: 12 apples\nthanks") == "12 apples"
    assert extract_final_answer("no answer here") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("14", Fraction(14)),
        ("$24", Fraction(24)),
        ("1,000", Fraction(1000)),
        ("25%", Fraction(25)),
        ("3/8", Fraction(3, 8)),
        ("0.375", Fraction(3, 8)),
        ("1/2.5", Fraction(2, 5)),  # decimal in a fraction, found in 13 Orca-Math references
        ("-5", Fraction(-5)),
        ("fourteen", None),
    ],
)
def test_to_number(text, expected):
    assert to_number(text) == expected
