"""Phase 2 one-shot mode (for free GRPO training with Unsloth): the student sees the floor overview once and
answers with its fix in one go. Same patient, same damage, same grader as the multi-turn exam.

Only damage you can SEE in the overview is used: loud (×10) and dead (×0.01). Flips need testing → multi-turn only.
"""
import math
import re

from wsp_llm import patient
from wsp_llm.patient import EXAM_FLOORS

VISIBLE_BUGS = (("loud", 10.0), ("dead", 0.01))
EXAM = [(bug, factor, floor) for floor in EXAM_FLOORS for bug, factor in VISIBLE_BUGS]            # 🔒 6
PRACTICE = [(bug, factor, floor) for floor in range(1, 27) if floor not in EXAM_FLOORS
            for bug, factor in VISIBLE_BUGS]                                                      # 🏋️ 46
MAX_EDITS = 3

INSTRUCTIONS = """You are an LLM doctor. A small language model (Qwen3 0.6B, 28 floors numbered 0-27) has been \
damaged on purpose: exactly one room was made too loud or almost silent. Each floor has a "looking" room \
(attention) and a "thinking" room (MLP). Floors naturally differ a little in loudness.
You get one look at the floor overview. First write ONE short line of reasoning: which room looks wrong and \
how many times louder or quieter it is than the same room on neighbouring floors. Then answer with the repair, \
as 1 to 3 lines like:
scale_floor(FLOOR, ROOM, FACTOR)
where ROOM is looking or thinking, and FACTOR multiplies that room (e.g. 0.1 makes it 10x quieter, 100 makes it \
100x louder). Score: how much healthy perplexity comes back on hidden text, minus a little for every room you \
change that was not damaged."""

FIX = re.compile(r"scale_floor\(\s*(\d+)\s*,\s*['\"]?(looking|thinking)['\"]?\s*,\s*([-+]?[\d.]+(?:e[-+]?\d+)?)\s*\)")


def question(bug: dict) -> str:
    """What the student sees: the damage symptoms + the floor overview of this patient."""
    patient.apply({"floor": 0, "factor": 1.0}, [])
    healthy = math.exp(patient.loss(patient.CHECK_TEXT))
    patient.apply(bug, [])
    sick = math.exp(patient.loss(patient.CHECK_TEXT))
    return (f"{INSTRUCTIONS}\n\nSurprise (perplexity) on a check text: healthy model {healthy:.2f}, this model "
            f"{sick:.2f}.\n\n{patient.overview()}\n\nYour repair:")


def parse(answer: str) -> list:
    """The student's edits, read from its answer (anything after a <think>...</think> block)."""
    answer = answer.split("</think>")[-1]
    edits = [[int(f), room, float(x)] for f, room, x in FIX.findall(answer)]
    return [e for e in edits if 0 <= e[0] < 28 and e[2] != 0][:MAX_EDITS]


def score(bug: dict, answer: str, use_quiz: bool = False) -> float:
    """🦴 80% recovery + 20% clean cure. No valid fix → 0.
    use_quiz=False = the FAST treat for training; True = the full exam grading (surprise + quiz)."""
    edits = parse(answer)
    if not edits:
        return 0.0
    r = patient.recovery(bug, edits, use_quiz=use_quiz)
    return 0.8 * r + 0.2 * r / (1 + patient.extra_edits(bug, edits))


def closeness(bug: dict, answer: str) -> float:
    """How near the damaged room ends up to its normal size: 1 = exactly normal, 0.5 = 10× off, 0 = 100× off or
    the damaged room wasn't touched. Rewards near-misses (×1000 beats ×10000) and punishes over-correction."""
    total = bug["factor"]
    touched = False
    for floor, room, factor in parse(answer):
        if floor == bug["floor"] and room == "thinking":
            total *= factor
            touched = True
    if not touched or total <= 0:
        return 0.0
    return max(0.0, 1 - abs(math.log10(total)) / 2)


def treat(bug: dict, answer: str) -> float:
    """🦴 The TRAINING treat (run 2): 70% the fast score + 30% closeness. Exams keep using score()."""
    return 0.7 * score(bug, answer) + 0.3 * closeness(bug, answer)


def dataset(split: str = "practice") -> list[dict]:
    """One row per patient: the question text + the hidden bug (the bug is for the grader only)."""
    rows = []
    for name, factor, floor in {"practice": PRACTICE, "exam": EXAM}[split]:
        bug = {"floor": floor, "factor": factor}
        rows.append({"name": f"{name}_floor{floor}", "prompt": question(bug), "floor": floor, "factor": factor})
    return rows


if __name__ == "__main__":
    # 🧪 test the test: perfect answers score 1, wrong or messy answers score ~0
    bug = {"floor": 7, "factor": 10.0}
    print(question(bug)[-400:])
    tries = {
        "perfect": "scale_floor(7, thinking, 0.1)",
        "perfect after thinking": "<think>floor 7 thinking is 10x louder</think>\nscale_floor(7, thinking, 0.1)",
        "half fix (x0.5)": "scale_floor(7, thinking, 0.5)",
        "wrong floor": "scale_floor(8, thinking, 0.1)",
        "right + 1 extra edit": "scale_floor(7, thinking, 0.1)\nscale_floor(3, looking, 2)",
        "gibberish": "I think floor 7 is broken, turn it down.",
    }
    for label, answer in tries.items():
        print(f"{label:22s} → reward {score(bug, answer):.2f}")
    print("practice:", len(PRACTICE), "exam:", len(EXAM))
