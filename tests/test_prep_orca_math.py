"""Tests for answer extraction in src/prep_orca_math.py.

The example lines are real final lines from Orca-Math solutions.
"""

import pandas as pd
import pytest

from src.prep_orca_math import assign_levels, calculation_steps, extract_answer, normalize_question


@pytest.mark.parametrize(
    "solution, expected",
    [
        ("13 + 55 = 68 peaches\n\nSo, there are 68 peaches now.", "68"),
        ("Therefore, the cost of the box of cookies and the board game together was $29 - $5 = $24.", "24"),
        ("d = 168 / 8\nd = 21\n\nSo, it takes 25 binders 21 days to bind 1400 books.", "21"),
        ("The total cost of the cards she bought, including tax, is $33.17.", "33.17"),
        ("The company sold 1,400 units.", "1400"),
        ("The temperature dropped to -5 degrees.", "-5"),
        ("So the fraction is \\( \\frac{3}{8} \\).", "3/8"),
        ("Each worker gets 3/8 of the pie.", "3/8"),
        ("The 24-hour store sold 90 items.", "90"),
        ("Thus, 2 moles of NH3 will react with 2 moles of HNO3 to produce 2 moles of NH4NO3.", "2"),
        (
            "16 + 1 = 17 peaches.\nJake has 6 fewer peaches than this person, so Jake has 17 - 6 = 11 peaches. "
            "However, you asked for the number of peaches the person has, not Jake. So the person has 17 peaches.",
            "17",
        ),
        ("The final bill is $33.17. That is what she paid.", "33.17"),
        ("So, B's share of the profit is Rs. 108.", "108"),
    ],
)
def test_extracts_final_number(solution, expected):
    assert extract_answer(solution) == (expected, "ok")


@pytest.mark.parametrize(
    "solution, reason",
    [
        ("Without additional information, we cannot determine the exact amounts.", "no_number"),
        ("", "no_number"),
        ("If we had p, we could use x = 4n to find x. But we cannot determine x.", "undetermined"),
        ("Geoff initially received approximately 0.52% of the votes.", "approximate"),
        ("Therefore, the ratio of the files organized this morning to the total files is 1:2.", "ratio"),
        ("They will meet at 8 a.m. + 4 hours = 12 p.m. (noon).", "time_of_day"),
        ("The train leaves at 7:45.", "time_of_day"),
        ("The salaries of A, B, and C are $3789.47, $1263.16, and $947.37, respectively.", "multiple_answers"),
        ("The total number of games is (n-1) * n / 2.", "formula"),
        ("Therefore, the area of the rhombus is 200√3 square centimeters.", "formula"),
        ("So, the annual interest rate of the second account is 10% (or 0.1 in decimal form).", "multiple_answers"),
        (
            "So, the quarterback should throw the ball at least twice to increase the probability "
            "of getting a completed pass at least once to 51%.",
            "number_word",
        ),
    ],
)
def test_drops_ungradable_answers(solution, reason):
    assert extract_answer(solution) == (None, reason)


def test_uses_only_the_last_line():
    solution = "Step 1: 10 + 5 = 15\nStep 2: 15 * 2 = 30\n\nThe answer is 30."
    assert extract_answer(solution) == ("30", "ok")


def test_normalize_question_ignores_case_and_spacing():
    assert normalize_question("  How many   Apples?\n") == normalize_question("how many apples?")


@pytest.mark.parametrize(
    "solution, expected",
    [
        ("10 + 5 = 15\nThe answer is 15.", 1),
        ("a = 2\nb = 3\nc = 5", 3),
        ("The answer is 15.", 0),
    ],
)
def test_calculation_steps(solution, expected):
    assert calculation_steps(solution) == expected


def test_assign_levels_makes_equal_thirds():
    assert assign_levels(pd.Series([1, 2, 3, 4, 5, 6])).tolist() == [1, 1, 2, 2, 3, 3]
