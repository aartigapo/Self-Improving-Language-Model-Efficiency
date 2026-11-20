from src.prep_calculus import (
    build_question,
    is_parseable,
    expression_size,
    limit_body,
)


def test_derivative_question():
    assert build_question("Derivatives", "x^{2}") == "Find the derivative of $x^{2}$ with respect to x."


def test_limit_question():
    assert build_question("Limits", "x^{2}") == "Evaluate the limit: $x^{2}$"


def test_integral_question():  # delete if you drop Integrals
    assert build_question("Integrals", "x^{2}") == "Find the indefinite integral of $x^{2}$ with respect to x."


def test_parseable():
    assert is_parseable("\\frac{1}{x}") is True


def test_unparseable():
    assert is_parseable("\\frac{1}{") is False


def test_size_ordering():
    assert expression_size("x") < expression_size("\\sin(x^{2}) + 3x")


def test_limit_body_extracts_inner_expression():
    assert limit_body(r"\lim {x \to 8 } {3 \sin{(x )} + 4}") == r"3 \sin{(x )} + 4"


def test_limit_body_leaves_other_expressions_alone():
    assert limit_body(r"\frac{1}{x}") == r"\frac{1}{x}"