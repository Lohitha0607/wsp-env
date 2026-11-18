import json
import numpy as np
from pathlib import Path
from wsp_env.checker import DATA, accuracy

healthy = dict(np.load(DATA / "healthy.npz"))
rng = np.random.default_rng(42)            # fixed seed → the same exam paper every time
out = DATA / "dataset"
out.mkdir(exist_ok=True)

def copy(w):
    return {k: v.copy() for k, v in w.items()}

# each bug: spoil a copy, and return the HIDDEN answer (what was broken)
def sign_flip(w):
    helpers = rng.choice(32, 3, replace=False)
    w["0.weight"][helpers] *= -1
    return {"helpers": helpers.tolist()}

def dead_helper(w):
    helpers = rng.choice(32, 3, replace=False)
    w["0.bias"][helpers] = -100
    return {"helpers": helpers.tolist()}

def swap_bosses(w):
    a, b = rng.choice(10, 2, replace=False)
    w["2.weight"][[a, b]] = w["2.weight"][[b, a]]
    w["2.bias"][[a, b]] = w["2.bias"][[b, a]]
    return {"bosses": [int(a), int(b)]}

def too_loud(w):
    helper = int(rng.integers(32))
    w["0.weight"][helper] *= 10
    return {"helper": helper}

def rigged_boss(w):
    boss = int(rng.integers(10))
    w["2.bias"][boss] += 10
    return {"boss": boss}

def noise(w):
    w["0.weight"] += rng.normal(0, 0.3, w["0.weight"].shape)
    return {"strength": 0.3}

BUGS = [sign_flip, dead_helper, swap_bosses, too_loud, rigged_boss, noise]
healthy_acc = accuracy(healthy)
questions, answers = [], []

for bug in BUGS:
    made, tries = 0, 0
    while made < 10 and tries < 1000:                  # 10 questions per bug type (give up after 1000 tries)
        tries += 1
        sick = copy(healthy)
        secret = bug(sick)
        sick_acc = accuracy(sick)
        if healthy_acc - sick_acc < 0.03:              # barely sick → bad question, skip
            continue
        qid = f"{bug.__name__}_{made:02d}"
        np.savez(out / f"{qid}.npz", **sick)
        questions.append({"id": qid, "patient": f"{qid}.npz",
                          "task": "This digit-reading network is broken. Find the bug and fix its weights.",
                          "healthy_accuracy": round(healthy_acc, 3), "sick_accuracy": round(sick_acc, 3)})
        answers.append({"id": qid, "bug": bug.__name__, **secret})
        made += 1
    print(f"{bug.__name__:12s} {made} questions ({tries} tries)")

(out / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions))   # 👀 doctor sees
(out / "answers.jsonl").write_text("\n".join(json.dumps(a) for a in answers))       # 🔒 doctor NEVER sees
print(f"Made {len(questions)} questions in dataset/")
