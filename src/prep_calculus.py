"""Clean the calculus dataset into the team's shared format.

Output columns:
    id, question, answer, level

Run from repository root:
    python -m src.prep_calculus

Saves:
    data/calculus_clean.parquet  (not committed to Git)
"""
import os
import re

import pandas as pd
from datasets import load_dataset
from sympy import count_ops
from sympy.parsing.latex import parse_latex

TEMPLATES = {
    "Derivatives": "Find the derivative of ${expr}$ with respect to x.",
    "Limits": "Evaluate the limit: ${expr}$",
    "Integrals": "Find the indefinite integral of ${expr}$ with respect to x.",
}

# Placeholder until "Required change 3" is posted by the reviewer.
TOPICS_TO_KEEP = ["Derivatives", "Limits", "Integrals"]

_previous_rows = {"n": None}


def report(step, df, reason=""):
    """Print the row count after a step and how many rows it removed."""
    after = len(df)
    before = _previous_rows["n"] if _previous_rows["n"] is not None else after
    message = f"{step:<40} {after:>8,} rows ({after - before:+,})"
    if reason:
        message += f" | {reason}"
    print(message)
    _previous_rows["n"] = after


def build_question(topic: str, expression: str) -> str:
    """Turn a topic + LaTeX expression into a full question the model can answer."""
    if topic not in TEMPLATES:
        raise ValueError(f"Unknown topic: {topic}")
    return TEMPLATES[topic].format(expr=expression)


def is_parseable(latex: str) -> bool:
    """True if SymPy's parse_latex can read this LaTeX without an error."""
    try:
        parse_latex(latex)
        return True
    except Exception:
        return False


def limit_body(q: str) -> str:
    """For limit problems, return the expression inside \\lim {x \\to a } {...}.

    Other expressions are returned unchanged.
    """
    m = re.match(r"\s*\\lim\s*\{[^}]*\}\s*\{(.*)\}\s*$", q, flags=re.DOTALL)
    return m.group(1) if m else q


def expression_size(latex: str) -> int:
    """Number of operations in the expression, using SymPy's count_ops()."""
    return int(count_ops(parse_latex(latex)))


def inspect_latex_parsing(df):
    """Inspect LaTeX parsing without deleting failed answers (not called by main)."""
    results = []
    for _, row in df.iterrows():
        ok = is_parseable(str(row["a"]))
        results.append({"topic": row["type"], "success": ok})
    results_df = pd.DataFrame(results)
    print(results_df.groupby("topic")["success"].agg(total="count", successful="sum"))
    return results_df


def main():
    df = load_dataset("di-zhang-fdu/calculus-dataset", split="train").to_pandas()
    df["source_index"] = df.index
    report("Loaded", df)

    df = df[df["q"].notna() & df["a"].notna()]
    df = df[(df["q"].str.strip() != "") & (df["a"].str.strip() != "")]
    report("Drop missing/empty", df)

    df = df[df["type"].isin(TOPICS_TO_KEEP)]
    report("Keep chosen topics", df)

    df = df[~df["a"].str.contains(r"\langle", regex=False)]
    report("Drop interval answers", df)

    df = df.drop_duplicates(subset=["type", "q"])
    report("Drop duplicates", df)

    df = df[df["a"].map(is_parseable)]
    report("Drop unparseable answers", df)

    df = df.copy()
    df["body"] = df["q"].map(limit_body)
    df = df[df["body"].map(is_parseable)]
    report("Drop unparseable expressions", df)

    df = df.copy()
    df["size"] = df["body"].map(expression_size)
    df["level"] = pd.qcut(
        df["size"].rank(method="first"), 3, labels=[1, 2, 3]
    ).astype(int)
    print("\nRows per topic and level:")
    print(df.groupby(["type", "level"]).size())

    df["question"] = [build_question(t, q) for t, q in zip(df["type"], df["q"])]
    df["id"] = "calculus_" + df["source_index"].astype(str).str.zfill(6)
    df = df.rename(columns={"a": "answer"})[["id", "question", "answer", "level"]]

    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/calculus_clean.parquet", index=False)
    report("Final", df)


if __name__ == "__main__":
    main()