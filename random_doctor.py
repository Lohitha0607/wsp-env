"""🎲 The random baseline for Phase 2: a "doctor" with no brain, same tools and same 10-test budget.

Each try: pick a random room + a random fix, test it, keep it if surprise went down, else undo.
Graded by the same hidden exam as the AI doctors. Run: uv run python random_doctor.py
"""
import random

from wsp_llm import patient
from wsp_llm.taskset import QUESTIONS
from wsp_llm.toolset import FLOORS, TEST_BUDGET

FIXES = [0.1, 100.0, -1.0]      # undo loud (×10), dead (×0.01), flipped (×−1)
SEEDS = 10                      # 10 random doctors per patient


def random_doctor(bug: dict, rng: random.Random) -> list:
    """🎲 Try TEST_BUDGET random fixes; keep each one only if check surprise went down."""
    edits = []
    patient.apply(bug, edits)
    best = patient.loss(patient.CHECK_TEXT)
    for _ in range(TEST_BUDGET):
        move = [rng.randrange(FLOORS), rng.choice(list(patient.ROOMS)), rng.choice(FIXES)]
        patient.apply(bug, edits + [move])
        now = patient.loss(patient.CHECK_TEXT)
        if now < best:                              # ✅ helped → keep it
            edits, best = edits + [move], now
    return edits


def score(bug: dict, edits: list) -> float:
    """The same final mark as the real exam: 80% recovery + 20% clean cure."""
    if not edits:
        return 0.0                                  # did nothing → still sick
    r = patient.recovery(bug, edits)
    return 0.8 * r + 0.2 * r / (1 + patient.extra_edits(bug, edits))


if __name__ == "__main__":
    print("patient | average of 10 random doctors | best | found it")
    total = []
    for name, factor, floor in QUESTIONS:
        bug = {"floor": floor, "factor": factor}
        scores = [score(bug, random_doctor(bug, random.Random(seed))) for seed in range(SEEDS)]
        total += scores
        found = sum(s > 0.9 for s in scores)
        print(f"{name}_floor{floor} | {sum(scores) / SEEDS:.2f} | {max(scores):.2f} | {found}/{SEEDS}", flush=True)
    print(f"\n🎲 random doctor overall: {sum(total) / len(total):.2f}")
