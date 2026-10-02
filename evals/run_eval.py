"""Score the agent against questions.json.

Usage:
    python evals/run_eval.py              run everything
    python evals/run_eval.py --limit 5    run the first five
    python evals/run_eval.py --only spark run ids containing "spark"

Every run costs real API calls - roughly two to six model calls per question.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402

from agent import answer  # noqa: E402

JUDGE_MODEL = "claude-opus-5"
HERE = Path(__file__).parent

JUDGE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["passed", "reason"],
        "additionalProperties": False,
    },
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def grade_contains(response: str, expected: list[str]) -> tuple[bool, str]:
    haystack = normalize(response)
    missing = [e for e in expected if normalize(e) not in haystack]
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, "all expected strings present"


def grade_judge(client, question: str, rubric: str, response: str) -> tuple[bool, str]:
    prompt = (
        f"Question asked:\n{question}\n\n"
        f"Grading rubric:\n{rubric}\n\n"
        f"Answer to grade:\n{response}\n\n"
        "Does the answer satisfy the rubric? Judge only against the rubric."
    )
    result = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1000,
        system="You are a strict grader. Apply the rubric literally.",
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": JUDGE_SCHEMA},
    )
    text = next(b.text for b in result.content if b.type == "text")
    verdict = json.loads(text)
    return verdict["passed"], verdict["reason"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", help="substring match against question id")
    args = parser.parse_args()

    cases = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if args.only in c["id"]]
    if args.limit:
        cases = cases[: args.limit]

    client = anthropic.Anthropic()
    records = []
    passed = 0

    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['id']} ... ", end="", flush=True)
        started = time.monotonic()

        try:
            response = answer(case["question"])
        except Exception as exc:
            print(f"ERROR ({exc})")
            records.append({**case, "response": None, "passed": False,
                            "reason": f"exception: {exc}"})
            continue

        if case["grade"] == "contains":
            ok, reason = grade_contains(response, case["expected"])
        else:
            ok, reason = grade_judge(client, case["question"], case["rubric"], response)

        passed += ok
        elapsed = time.monotonic() - started
        print(f"{'PASS' if ok else 'FAIL'} ({elapsed:.1f}s) - {reason}")

        records.append({**case, "response": response, "passed": ok, "reason": reason})

    total = len(records)
    score = passed / total * 100 if total else 0
    print(f"\nScore: {passed}/{total} ({score:.0f}%)")

    if failures := [r for r in records if not r["passed"]]:
        print("\nFailures:")
        for r in failures:
            print(f"  {r['id']}: {r['reason']}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = HERE / f"results-{stamp}.json"
    out.write_text(json.dumps(
        {"score": score, "passed": passed, "total": total, "results": records},
        indent=2,
    ), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
