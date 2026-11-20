"""Clean Orca-Math into the team's shared format: id, question, answer, level.

level is 1, 2 or 3: equal-sized thirds by number of calculation steps ('=' signs)
in the worked solution, 1 being the fewest.

Run from the repo root:
    python -m src.prep_orca_math

Prints the row count after every step and saves data/orca_math_clean.parquet
(data/ is gitignored; the final splits go to Hugging Face, not git).
"""

import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_ID = "microsoft/orca-math-word-problems-200k"
OUTPUT_PATH = Path("data/orca_math_clean.parquet")

# A number like 14, -3, 1,400, 0.5 or 3/8. Not digits glued to a letter, like the 3 in "NH3".
NUMBER = r"(?<![A-Za-z\d])-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?"

# Final lines we can't grade with a single number.
APPROXIMATE = re.compile(r"\b(approximately|approx\.?|about|roughly|around|nearly)\b", re.IGNORECASE)
UNDETERMINED = re.compile(
    r"\b(cannot|can't|can not|unable to)\b|not enough information|without (additional|more|further) information",
    re.IGNORECASE,
)
RATIO = re.compile(r"\d\s*:\s*\d")
TIME_OF_DAY = re.compile(r"\d\s*(a\.?\s*m\.?|p\.?\s*m\.?)(?!\w)|\d:\d\d", re.IGNORECASE)
# "...respectively", a list like "$3789.47, $1263.16, and $947.37", or an alternative like "10% (or 0.1 ...)".
MULTIPLE_ANSWERS = re.compile(
    r"\brespectively\b|" + NUMBER + r"[^.\d]*,\s*\$?" + NUMBER + r"[^.\d]*\band\b|\(or\s[^)]*\d",
    re.IGNORECASE,
)
# A one-letter variable next to an operator, e.g. "(n-1) * n / 2", or a root or pi
# like "200√3": a formula, not a plain number.
# One letter only, so hyphenated words like "24-hour" or "cross-section" don't match.
FORMULA = re.compile(
    r"(?<![a-zA-Z])[a-zA-Z]\s*[-+*/^]\s*[\d(]|[\d)]\s*[-+*/^]\s*[a-zA-Z](?![a-zA-Z])|√|π|\bsqrt\b|\bpi\b",
    re.IGNORECASE,
)
# An answer written as a word, e.g. "throw the ball at least twice ... to 51%".
NUMBER_WORD = re.compile(r"\b(once|twice|thrice)\b", re.IGNORECASE)


def last_line(solution: str) -> str:
    """The final non-empty line of a worked solution, where the answer is stated."""
    lines = [line.strip() for line in solution.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def final_sentence(line: str) -> str:
    """The last sentence of a line that has a number in it.

    A last line can hold several sentences: "...so Jake has 17 - 6 = 11 peaches.
    However, you asked about the person. So the person has 17 peaches." The answer
    is in the final one. Splits only before a capital letter, so "$33.17" and
    "Rs. 108" stay whole.
    """
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
    with_numbers = [s for s in sentences if re.search(r"\d", s)]
    return with_numbers[-1] if with_numbers else line


def extract_answer(solution: str) -> tuple[str | None, str]:
    """Return (answer, reason). answer is None when the solution can't be graded.

    reason is "ok" or the name of the rule that dropped it, so the script can
    report how many rows each rule removed.
    """
    solution = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", solution)  # \frac{3}{8} -> 3/8
    solution = solution.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    line = last_line(solution)

    if not re.search(r"\d", line):
        return None, "no_number"
    if UNDETERMINED.search(line):
        return None, "undetermined"
    if APPROXIMATE.search(line):
        return None, "approximate"
    if TIME_OF_DAY.search(line):
        return None, "time_of_day"
    if RATIO.search(line):
        return None, "ratio"
    if MULTIPLE_ANSWERS.search(line):
        return None, "multiple_answers"
    if FORMULA.search(line):
        return None, "formula"

    sentence = final_sentence(line)
    if NUMBER_WORD.search(sentence):
        return None, "number_word"

    numbers = [clean_number(n) for n in re.findall(NUMBER, sentence)]
    if not numbers:
        return None, "no_number"

    # The last number in the sentence isn't always the answer: "it takes 25 binders
    # 21 days to bind 1400 books". So prefer the solution's last calculated result
    # ("d = 21") when the final sentence repeats it.
    results = re.findall(r"=\s*\$?(" + NUMBER + ")", solution)
    if results and clean_number(results[-1]) in numbers:
        return clean_number(results[-1]), "ok"
    return numbers[-1], "ok"


def clean_number(text: str) -> str:
    """Strip thousands commas and spaces: "1,400" -> "1400", "3 / 8" -> "3/8"."""
    return text.replace(",", "").replace(" ", "")


def normalize_question(question: str) -> str:
    """Lowercase and collapse whitespace, so near-identical copies count as duplicates."""
    return re.sub(r"\s+", " ", question.lower()).strip()


def calculation_steps(solution: str) -> int:
    """Number of calculation steps in a worked solution, counted as '=' signs."""
    return solution.count("=")


def assign_levels(steps: pd.Series) -> pd.Series:
    """Split problems into 3 equal-sized groups by number of steps: 1 = fewest, 3 = most."""
    return pd.qcut(steps.rank(method="first"), 3, labels=[1, 2, 3]).astype(int)


def report(step: str, df: pd.DataFrame, before: int) -> None:
    print(f"{step:<40} {len(df):>8,} rows  ({len(df) - before:+,})")


def main() -> None:
    df = load_dataset(DATASET_ID, split="train").to_pandas()
    df["source_index"] = df.index
    print(f"{'Loaded ' + DATASET_ID:<40} {len(df):>8,} rows")

    # 1. Extract the reference answer into its own column.
    extracted = df["answer"].map(extract_answer)
    df["solution"] = df["answer"]
    df["answer"] = extracted.str[0]
    df["drop_reason"] = extracted.str[1]

    # 2. Drop problems whose answer can't be checked, one rule at a time.
    for reason in [
        "no_number", "undetermined", "approximate", "time_of_day", "ratio",
        "multiple_answers", "formula", "number_word",
    ]:
        before = len(df)
        df = df[df["drop_reason"] != reason]
        report(f"Drop: {reason}", df, before)

    # 3. Remove duplicate questions (case and spacing ignored), keeping the first copy.
    before = len(df)
    df = df[~df["question"].map(normalize_question).duplicated()]
    report("Drop: duplicate question", df, before)

    # 4. Difficulty label: 3 equal-sized levels by number of calculation steps.
    df["steps"] = df["solution"].map(calculation_steps)
    df["level"] = assign_levels(df["steps"])
    print(df.groupby("level")["steps"].agg(["count", "min", "max", "mean"]))

    # 5. Stable IDs from the original row number, so any row can be traced back.
    df["id"] = "orca_" + df["source_index"].astype(str).str.zfill(6)

    df = df[["id", "question", "answer", "level"]].reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows with columns {list(df.columns)} to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
