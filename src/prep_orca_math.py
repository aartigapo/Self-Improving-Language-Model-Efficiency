"""Clean Orca-Math into the team's shared format: id, question, answer, level.

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
MULTIPLE_ANSWERS = re.compile(r"\brespectively\b|" + NUMBER + r"[^.\d]*,\s*\$?" + NUMBER + r"[^.\d]*\band\b", re.IGNORECASE)
# A one-letter variable next to an operator, e.g. "(n-1) * n / 2": a formula, not a number.
# One letter only, so hyphenated words like "24-hour" or "cross-section" don't match.
FORMULA = re.compile(r"(?<![a-zA-Z])[a-zA-Z]\s*[-+*/^]\s*[\d(]|[\d)]\s*[-+*/^]\s*[a-zA-Z](?![a-zA-Z])")


def last_line(solution: str) -> str:
    """The final non-empty line of a worked solution, where the answer is stated."""
    lines = [line.strip() for line in solution.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


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

    numbers = [clean_number(n) for n in re.findall(NUMBER, line)]
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
    for reason in ["no_number", "undetermined", "approximate", "time_of_day", "ratio", "multiple_answers", "formula"]:
        before = len(df)
        df = df[df["drop_reason"] != reason]
        report(f"Drop: {reason}", df, before)

    # 3. Remove duplicate questions (case and spacing ignored), keeping the first copy.
    before = len(df)
    df = df[~df["question"].map(normalize_question).duplicated()]
    report("Drop: duplicate question", df, before)

    # 4. Difficulty label. Left empty until the team decides how to measure difficulty.
    df["level"] = pd.NA

    # 5. Stable IDs from the original row number, so any row can be traced back.
    df["id"] = "orca_" + df["source_index"].astype(str).str.zfill(6)

    df = df[["id", "question", "answer", "level"]].reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows with columns {list(df.columns)} to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
