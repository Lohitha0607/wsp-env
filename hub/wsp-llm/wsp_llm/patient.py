"""The Phase 2 patient: Qwen3 0.6B (a real LLM), damaged on one floor. Shared by the tools and the grader."""
import json
import math
import os
from functools import cache
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "Qwen/Qwen3-0.6B"
DEVICE = os.environ.get("WSP_PATIENT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")   # e.g. "cuda:1"
DTYPE = getattr(torch, os.environ.get("WSP_PATIENT_DTYPE", "float32"))   # "float16" = lighter patient (training)
DATA = Path(__file__).parent / "data"
CHECK_TEXT = (DATA / "check.txt").read_text()     # 🧪 the doctor may test on this (test_surprise)
HIDDEN_TEXT = (DATA / "hidden.txt").read_text()   # 🔒 only the final mark uses this
QUIZ_CHECK = [json.loads(line) for line in (DATA / "quiz_check.jsonl").read_text().splitlines()]    # 🧪 20
QUIZ_HIDDEN = [json.loads(line) for line in (DATA / "quiz_hidden.jsonl").read_text().splitlines()]  # 🔒 65
QUIZ_COUNTS = 0.05                                # the quiz counts only if the damage cost it 5+ points
EXAM_FLOORS = (2, 6, 20)                          # 🔒 exam patients live here; practice uses every other floor
ROOMS = {"thinking": "mlp.down_proj", "looking": "self_attn.o_proj"}   # each room's "voice"


@cache
def load():
    """Load the patient once per process, plus a photo of every healthy room."""
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=DTYPE).to(DEVICE).eval()
    healthy = {(f, room): voice(model, f, room).data.clone()
               for f in range(len(model.model.layers)) for room in ROOMS}
    return tok, model, healthy


def voice(model, floor: int, room: str) -> torch.nn.Linear:
    return model.model.layers[floor].get_submodule(ROOMS[room]).weight


def factors(bug: dict, edits: list) -> dict[tuple[int, str], float]:
    """How much each room is multiplied by: the planted bug × every doctor edit."""
    total = {(bug["floor"], "thinking"): bug["factor"]}
    for floor, room, factor in edits:
        total[(floor, room)] = total.get((floor, room), 1.0) * factor
    return total


def apply(bug: dict, edits: list) -> None:
    """Set the loaded model to this patient: healthy, then the bug, then the doctor's edits."""
    _, model, healthy = load()
    for (floor, room), photo in healthy.items():
        voice(model, floor, room).data = photo * factors(bug, edits).get((floor, room), 1.0)


def loss(text: str) -> float:
    """Average "how wrong" per next word. Surprise (perplexity) = exp(loss)."""
    tok, model, _ = load()
    ids = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return model(**ids, labels=ids["input_ids"]).loss.item()


def quiz(questions: list[dict]) -> tuple[float, list[str]]:
    """✅ Fraction of quiz questions where the right choice is the likeliest, plus the prompts it got wrong."""
    tok, model, _ = load()
    wrong = []
    for q in questions:
        prompt = tok(q["prompt"])["input_ids"]
        scores = []
        for choice in q["choices"]:
            answer = tok(" " + choice)["input_ids"]
            with torch.no_grad():
                logp = model(torch.tensor([prompt + answer], device=DEVICE)).logits[0, :-1].log_softmax(-1)
            scores.append(sum(logp[len(prompt) - 1 + i, t].item() for i, t in enumerate(answer)))
        if max(range(len(scores)), key=scores.__getitem__) != q["answer"]:
            wrong.append(q["prompt"])
    return 1 - len(wrong) / len(questions), wrong


def write(prompt: str, words: int = 20) -> str:
    tok, model, _ = load()
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=words, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def overview() -> str:
    """How loud each floor's looking and thinking rooms are (average size of their voice numbers)."""
    _, model, _ = load()
    lines = ["floor | looking | thinking"]
    for f in range(len(model.model.layers)):
        sizes = [voice(model, f, room).abs().mean().item() for room in ("looking", "thinking")]
        lines.append(f"{f} | {sizes[0]:.4f} | {sizes[1]:.4f}")
    return "\n".join(lines)


def part(sick: float, healthy: float, fixed: float) -> float:
    """How much of the lost health came back: 0 = none, 1 = all (clipped)."""
    return min(1.0, max(0.0, (fixed - sick) / (healthy - sick)))


@cache
def baseline_loss(floor: int, factor: float) -> tuple[float, float]:
    """Hidden-text loss of the sick and the healthy patient."""
    apply({"floor": floor, "factor": factor}, [])
    sick = loss(HIDDEN_TEXT)
    apply({"floor": 0, "factor": 1.0}, [])
    return sick, loss(HIDDEN_TEXT)


@cache
def baseline_quiz(floor: int, factor: float) -> tuple[float, float]:
    """Hidden-quiz score of the sick and the healthy patient."""
    apply({"floor": floor, "factor": factor}, [])
    sick = quiz(QUIZ_HIDDEN)[0]
    apply({"floor": 0, "factor": 1.0}, [])
    return sick, quiz(QUIZ_HIDDEN)[0]


def recovery(bug: dict, edits: list, use_quiz: bool = True) -> float:
    """0 = still sick, 1 = fully healed, on hidden text + hidden quiz the doctor never saw.
    Half surprise, half quiz; the quiz only counts when the damage actually cost it QUIZ_COUNTS or more.
    use_quiz=False → surprise only (fast, for training)."""
    sick_loss, healthy_loss = baseline_loss(bug["floor"], bug["factor"])
    apply(bug, edits)
    surprise_part = part(-sick_loss, -healthy_loss, -loss(HIDDEN_TEXT))   # lower loss is healthier
    if not use_quiz:
        return surprise_part
    sick_quiz, healthy_quiz = baseline_quiz(bug["floor"], bug["factor"])
    if healthy_quiz - sick_quiz < QUIZ_COUNTS:
        return surprise_part
    apply(bug, edits)
    return (surprise_part + part(sick_quiz, healthy_quiz, quiz(QUIZ_HIDDEN)[0])) / 2


def closeness(bug: dict, edits: list) -> float:
    """🎯 Training bonus: how near the damaged room ends up to its normal size. 1 = exactly undone,
    0.5 = still 10× off, 0 = 100× off, sign still flipped, or the damaged room untouched."""
    key = (bug["floor"], "thinking")
    if not any((floor, room) == key for floor, room, _ in edits):
        return 0.0
    net = factors(bug, edits)[key]
    if net <= 0:
        return 0.0
    return max(0.0, 1 - abs(math.log10(net)) / 2)


def diagnosis(bug: dict, edits: list) -> float:
    """🔍 Training bonus: 1 if the doctor edited the damaged room at all (found the right floor and room)."""
    return float(any((floor, room) == (bug["floor"], "thinking") for floor, room, _ in edits))


def extra_edits(bug: dict, edits: list) -> int:
    """Rooms left changed that the bug never touched."""
    changed = {key for key, f in factors({"floor": bug["floor"], "factor": 1.0}, edits).items()
               if abs(f - 1.0) > 1e-3}
    return len(changed - {(bug["floor"], "thinking")})
