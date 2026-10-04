"""
Target / OOD / retention ACCURACY evaluation, with a simple automatic
verifier (no human judgment needed) for math-style problems.

This is SEPARATE from feature_extractor.py on purpose: this measures the
CURRENT STATE of the model (shared across all 3 candidates at a state),
not a per-candidate feature. Run this ONCE per state, before building
candidates -- not once per candidate.

This is also more expensive than the probe features: it calls
model.generate() for every problem in each benchmark set, which is slow,
especially on CPU. Intended to run on GPU once available -- safe to try
on CPU with the small placeholder sets below, but expect it to be slow
on anything bigger.

IMPORTANT: the problem sets below are small PLACEHOLDERS (5-6 examples
each) just to prove the verifier/evaluation logic works end to end.
Swap TARGET_PROBLEMS for real GSM8K examples, OOD_PROBLEMS for real
MATH-500 examples, and RETENTION_PROBLEMS for a real fixed general-
knowledge set, per the project's dataset plan, once that's wired up.
"""

import re

import torch

from src.data.dataset import TextDataset


def extract_final_number(text):
    """
    Pulls the last number that appears in a piece of generated text.
    This is the simplest possible math verifier: GSM8K-style answers
    conventionally end with the final numeric answer.
    Returns None if no number is found.
    """
    matches = re.findall(r"-?\d+\.?\d*", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def verify_math_answer(generated_text, expected_answer, tolerance=1e-4):
    """
    The verifier: compares the model's extracted final number against
    the known-correct answer. No human judgment -- just a number compare,
    exactly like the project doc's definition of a verifier for math.
    """
    predicted = extract_final_number(generated_text)
    if predicted is None:
        return False
    return abs(predicted - expected_answer) < tolerance


def evaluate_accuracy(model, tokenizer, device, problems, max_new_tokens=64, verbose=False):
    """
    Runs the model on a fixed set of problems and returns the fraction
    answered correctly, using verify_math_answer() as the check.

    problems: list of {"question": str, "answer": float}
    """

    model.eval()

    correct = 0
    total = len(problems)

    with torch.no_grad():
        for problem in problems:
            inputs = tokenizer(
                problem["question"],
                return_tensors="pt",
            ).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            generated_text = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            is_correct = verify_math_answer(generated_text, problem["answer"])

            if verbose:
                print(f"Q: {problem['question']}")
                print(f"Generated: {generated_text.strip()[:120]}")
                print(f"Expected: {problem['answer']} | Correct: {is_correct}\n")

            if is_correct:
                correct += 1

    model.train()

    return correct / total if total > 0 else 0.0


def evaluate_current_state(model, tokenizer, device, target_problems=None, ood_problems=None, retention_problems=None):
    """
    Call this ONCE per state, on the untouched parent model, BEFORE
    building any candidates. Shared across all 3 candidates at that state
    -- do not call this separately for each candidate.

    Returns the three accuracy numbers that make up "current-state
    performance" in the feature schema.
    """

    target_problems = target_problems or TARGET_PROBLEMS
    ood_problems = ood_problems or OOD_PROBLEMS
    retention_problems = retention_problems or RETENTION_PROBLEMS

    return {
        "target_accuracy": evaluate_accuracy(model, tokenizer, device, target_problems),
        "ood_accuracy": evaluate_accuracy(model, tokenizer, device, ood_problems),
        "retention_accuracy": evaluate_accuracy(model, tokenizer, device, retention_problems),
    }


# ---------------------------------------------------------------------------
# PLACEHOLDER problem sets -- small, just to prove the pipeline works.
# Replace with real GSM8K / MATH-500 / general-knowledge examples.
# ---------------------------------------------------------------------------

TARGET_PROBLEMS = [
    {"question": "Q: If John has 5 apples and gives away 2, how many does he have left?\nA:", "answer": 3},
    {"question": "Q: A rectangle has length 4 and width 3. What is its area?\nA:", "answer": 12},
    {"question": "Q: Sarah saves $10 each week for 6 weeks. How much has she saved in total?\nA:", "answer": 60},
    {"question": "Q: There are 8 birds on a fence. 3 fly away. How many birds are left?\nA:", "answer": 5},
    {"question": "Q: A classroom has 4 rows with 6 chairs in each row. How many chairs in total?\nA:", "answer": 24},
]

OOD_PROBLEMS = [
    {"question": "Q: Solve for x: 3x + 7 = 22. What is x?\nA:", "answer": 5},
    {"question": "Q: What is the sum of the first 5 positive integers?\nA:", "answer": 15},
    {"question": "Q: A train travels 60 miles in 2 hours. What is its speed in miles per hour?\nA:", "answer": 30},
    {"question": "Q: What is 15% of 200?\nA:", "answer": 30},
    {"question": "Q: If a triangle has angles 50 and 60 degrees, what is the third angle in degrees?\nA:", "answer": 70},
]

RETENTION_PROBLEMS = [
    {"question": "Q: What is 12 plus 7?\nA:", "answer": 19},
    {"question": "Q: What is 9 times 3?\nA:", "answer": 27},
    {"question": "Q: What is 100 minus 45?\nA:", "answer": 55},
    {"question": "Q: What is 50 divided by 5?\nA:", "answer": 10},
    {"question": "Q: What is 6 squared?\nA:", "answer": 36},
]


if __name__ == "__main__":
    print("This module needs a real loaded model to run -- see evaluate_current_state().")
    print("Call it ONCE per state, before building candidates, e.g.:")
    print()
    print("  from evaluation import evaluate_current_state")
    print("  state_performance = evaluate_current_state(model, tokenizer, device)")
    print("  # -> {'target_accuracy': ..., 'ood_accuracy': ..., 'retention_accuracy': ...}")
