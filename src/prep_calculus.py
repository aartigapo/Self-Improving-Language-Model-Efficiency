
"""Clean the calculus dataset into the team's shared format.

Output columns:
    id, question, answer, level

Run from repository root:
    python -m src.prep_calculus

Saves:
    data/calculus_clean.parquet
    data/calculus_cleaning_report.txt
"""

from pathlib import Path
import re

import pandas as pd
from sympy.parsing.latex import parse_latex


INPUT_PATH = Path("data/calculus_raw.csv")
OUTPUT_PATH = Path("data/calculus_clean.parquet")
REPORT_PATH = Path("data/calculus_cleaning_report.txt")

TOPICS = [
    "Derivatives",
    "Integrals",
    "Limits",
    "Differential Equations",
]


def report(step, before, after, reason=""):
    removed = before - after

    message = (
        f"{step:<40} {after:>8,} rows "
        f"({-removed:+,})"
    )

    if reason:
        message += f" | {reason}"

    print(message)


def assign_level(topic, question):
    """Assign provisional topic-based difficulty labels."""

    levels = {
        "Derivatives": 1,
        "Integrals": 2,
        "Limits": 3,
        "Differential Equations": 4,
    }

    return levels.get(topic, 5)


def classify_answer(answer):
    """Classify reference answers by mathematical format."""

    answer = str(answer).strip()

    if not answer:
        return "Empty"

    if r"\langle" in answer:
        return "Interval"

    if "f{(x )}" in answer:
        return "Differential equation"

    if r"\infty" in answer:
        return "Infinity"

    if re.fullmatch(r"[-+]?\d+(\.\d+)?", answer):
        return "Number"

    if r"\frac" in answer:
        return "Fraction"

    return "Symbolic expression"


def test_latex_parsing(df):
    """Inspect LaTeX parsing without deleting failed answers."""

    results = []

    for _, row in df.iterrows():
        answer = str(row["a"])

        try:
            parse_latex(answer)
            success = True
            error = ""

        except Exception as e:
            success = False
            error = str(e)

        results.append({
            "topic": row["type"],
            "answer": answer,
            "success": success,
            "error": error,
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("LATEX PARSING RESULTS")
    print("=" * 60)

    summary = results_df.groupby("topic")["success"].agg(
        total="count",
        successful="sum",
        failed=lambda x: (~x).sum(),
    )

    print(summary)

    print("\nOverall parsing results:")
    print("Total:", len(results_df))
    print("Successful:", results_df["success"].sum())
    print("Failed:", (~results_df["success"]).sum())

    failures = results_df[~results_df["success"]]

    print("\nSample parsing failures:")

    if failures.empty:
        print("No parsing failures.")

    else:
        print(
            failures[
                ["topic", "answer", "error"]
            ].head(30).to_string(index=False)
        )

    print("\nNo answers removed because of parser failures.")

    return results_df


def inspect_interval_answers(df):
    """Validate interval answer notation."""

    intervals = df[
        df["a"].astype(str).str.contains(
            r"\\langle",
            regex=True
        )
    ].copy()

    valid_pattern = (
        r"\\langle\s*"
        r"(-?\d+|-?\\infty)\s*,\s*"
        r"(-?\d+|\\infty)\s*"
        r"\\rangle"
    )

    intervals["valid_format"] = (
        intervals["a"]
        .astype(str)
        .str.fullmatch(valid_pattern)
    )

    print("\n" + "=" * 60)
    print("INTERVAL ANSWER INSPECTION")
    print("=" * 60)

    print("Total interval answers:", len(intervals))
    print("\nFormat results:")
    print(intervals["valid_format"].value_counts(dropna=False))

    print("\nIntervals by topic:")
    print(intervals["type"].value_counts())

    invalid = intervals[~intervals["valid_format"]]

    if invalid.empty:
        print("\nNo invalid interval formats found.")

    else:
        print("\nInvalid interval examples:")
        print(
            invalid[["type", "a"]]
            .head(30)
            .to_string(index=False)
        )

    print("\nNo interval answers removed.")


def inspect_cross_topic_conflicts(df):
    """Find identical questions with different answers across topics."""

    conflicts = (
        df.groupby("q")
        .agg(
            topic_count=("type", "nunique"),
            answer_count=("a", "nunique"),
            topics=("type", lambda x: list(x.unique())),
            answers=("a", lambda x: list(x.unique())),
        )
        .query("topic_count > 1 and answer_count > 1")
    )

    print("\n" + "=" * 60)
    print("CROSS-TOPIC CONFLICT INSPECTION")
    print("=" * 60)

    print("Conflicting questions:", len(conflicts))

    if not conflicts.empty:
        print("\nSample conflicts:")
        print(conflicts.head(30).to_string())

    return conflicts


def main():

    cleaning_log = []

    # 1. Load dataset
    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded calculus dataset: {len(df):,} rows")

    cleaning_log.append(f"Original rows: {len(df)}")

    # 2. Validate required columns
    required_columns = ["type", "q", "a"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 3. Drop missing values
    before = len(df)

    df = df.dropna(subset=required_columns).copy()

    report(
        "Drop: missing values",
        before,
        len(df),
        "Missing required fields"
    )

    cleaning_log.append(
        f"Missing values removed: {before - len(df)}"
    )

    # 4. Remove empty questions and answers
    before = len(df)

    df = df[
        df["q"].astype(str).str.strip().ne("")
        & df["a"].astype(str).str.strip().ne("")
    ].copy()

    report(
        "Drop: empty questions/answers",
        before,
        len(df),
        "Empty question or answer"
    )

    cleaning_log.append(
        f"Empty questions/answers removed: {before - len(df)}"
    )

    # 5. Validate topic names
    before = len(df)

    df = df[df["type"].isin(TOPICS)].copy()

    report(
        "Drop: invalid topics",
        before,
        len(df),
        "Topics outside approved list"
    )

    cleaning_log.append(
        f"Invalid topics removed: {before - len(df)}"
    )

    # 6. Inspect reference answers
    print("\n" + "=" * 60)
    print("REFERENCE ANSWER INSPECTION")
    print("=" * 60)

    sample_size = min(30, len(df))

    for i, answer in enumerate(
        df["a"].sample(
            n=sample_size,
            random_state=42
        ),
        start=1
    ):
        print(f"{i}. {answer}")

    # 7. Classify answer formats
    print("\n" + "=" * 60)
    print("ANSWER FORMAT DISTRIBUTION")
    print("=" * 60)

    answer_types = df["a"].apply(classify_answer)

    print(answer_types.value_counts())

    # 8. Inspect LaTeX parsing
    parsing_results = test_latex_parsing(df)

    # 9. Inspect interval answers
    inspect_interval_answers(df)

    # 10. Remove duplicate question-topic pairs
    before = len(df)

    df = df.drop_duplicates(
        subset=["type", "q"],
        keep="first",
    ).copy()

    report(
        "Drop: duplicate question-topic pairs",
        before,
        len(df),
        "Same question and topic"
    )

    cleaning_log.append(
        f"Duplicate question-topic pairs removed: {before - len(df)}"
    )

    # 11. Inspect cross-topic conflicts
    conflicts = inspect_cross_topic_conflicts(df)

    cleaning_log.append(
        f"Cross-topic conflicting questions found: {len(conflicts)}"
    )

    # 12. Preserve conflicts for manual review
    print("\n" + "=" * 60)
    print("CROSS-TOPIC CONFLICT HANDLING")
    print("=" * 60)

    print(
        "Conflicting questions are being preserved "
        "for manual review."
    )

    print(
        "No questions removed automatically because "
        "of cross-topic conflicts."
    )

    cleaning_log.append(
        "Cross-topic conflicts preserved for manual review."
    )

    # 13. Assign difficulty levels
    df["level"] = [
        assign_level(topic, question)
        for topic, question in zip(df["type"], df["q"])
    ]

    print("\nDifficulty distribution:")
    print(df["level"].value_counts().sort_index())

    # 14. Generate unique IDs
    df = df.reset_index(drop=True)

    df["id"] = [
        f"calculus_{i:06d}"
        for i in range(1, len(df) + 1)
    ]

    # 15. Rename shared columns
    df = df.rename(
        columns={
            "q": "question",
            "a": "answer",
        }
    )

    df = df[
        ["id", "question", "answer", "level"]
    ]

    # 16. Validate final output
    assert df["id"].is_unique

    assert not df.duplicated(
        subset=["question", "level"]
    ).any()

    assert df.columns.tolist() == [
        "id", "question", "answer", "level"
    ]

    # 17. Save final dataset
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # 18. Generate cleaning report
    cleaning_log.extend([
        f"Final rows: {len(df)}",
        f"Final columns: {df.columns.tolist()}",
        f"SymPy parsing successes: {parsing_results['success'].sum()}",
        f"SymPy parsing failures: {(~parsing_results['success']).sum()}",
        "Interval parser failures are not automatically removed.",
        "Difficulty labels are provisional topic-based labels.",
    ])

    REPORT_PATH.write_text(
        "\n".join(cleaning_log),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("FINAL DATASET")
    print("=" * 60)

    print("Final dataset shape:", df.shape)
    print("Final columns:", df.columns.tolist())
    print("Saved to:", OUTPUT_PATH)
    print("Cleaning report saved to:", REPORT_PATH)


if __name__ == "__main__":
    main()