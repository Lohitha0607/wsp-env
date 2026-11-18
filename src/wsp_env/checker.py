"""The rubric (marking scheme): gives a doctor's fix a score from 0 to 1."""
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent / "data"            # healthy.npz + dataset/ live here

# the hidden test pictures (the SAME split as train.py), cut in two:
digits = load_digits()
_, X_test, _, y_test = train_test_split(digits.data / 16.0, digits.target, test_size=0.3, random_state=42)
X_check, y_check = X_test[:140], y_test[:140]    # 🧪 140: the doctor may test on these (test_run)
X_final, y_final = X_test[140:], y_test[140:]    # 🔒 400: only the final mark uses these

def predict(w, X):
    """The patient's guess for each picture."""
    helpers = np.maximum(0, X @ w["0.weight"].T + w["0.bias"])   # 32 helpers + ReLU
    scores  = helpers @ w["2.weight"].T + w["2.bias"]            # 10 bosses
    return scores.argmax(axis=1)                                 # loudest boss wins

def accuracy(w, X=X_final, y=y_final):
    """% of pictures the patient reads correctly (the final 400 unless told otherwise)."""
    return (predict(w, X) == y).mean()

def per_digit(w, X=X_check, y=y_check):
    """Accuracy for each digit + the most common mix-ups (true → guessed)."""
    guesses = predict(w, X)
    lines = [f"digit {d}: {(guesses[y == d] == d).mean():.0%} right" for d in range(10)]
    mixups = {}
    for true, guess in zip(y, guesses):
        if true != guess:
            mixups[(int(true), int(guess))] = mixups.get((int(true), int(guess)), 0) + 1
    top = sorted(mixups.items(), key=lambda kv: -kv[1])[:5]
    lines.append("most common mix-ups: " + (", ".join(f"{t}→{g} ×{n}" for (t, g), n in top) or "none"))
    return "\n".join(lines)

def changed_rows(sick, fixed):
    """Which neurons (table, row) the doctor changed, counting a weight row or its bias."""
    rows = set()
    for t, n in (("0", 32), ("2", 10)):
        for r in range(n):
            same_w = np.allclose(sick[f"{t}.weight"][r], fixed[f"{t}.weight"][r], rtol=1e-3, atol=1e-6)
            same_b = np.isclose(sick[f"{t}.bias"][r], fixed[f"{t}.bias"][r], rtol=1e-3, atol=1e-6)
            if not (same_w and same_b):
                rows.add((t, r))
    return rows

def bug_rows(answer):
    """Which neurons the planted bug touched (from the secret answer)."""
    match answer["bug"]:
        case "sign_flip" | "dead_helper":
            return {("0", h) for h in answer["helpers"]}
        case "swap_bosses":
            return {("2", b) for b in answer["bosses"]}
        case "too_loud":
            return {("0", answer["helper"])}
        case "rigged_boss":
            return {("2", answer["boss"])}
        case _:                                                  # noise touched every helper
            return {("0", h) for h in range(32)}

def tidiness(recovery, extra_edits):
    """💊 Bonus for a clean cure: full when nothing extra was changed, shrinking with each unneeded edit.
    Multiplied by recovery, so doing nothing earns no tidiness bonus."""
    return recovery / (1 + extra_edits)

def score(healthy, sick, fixed):
    """How much of the lost accuracy came back: 0 = nothing, 1 = fully cured."""
    lost = accuracy(healthy) - accuracy(sick)
    if lost <= 0:
        return 1.0                                                     # the bug didn't hurt, nothing to cure
    recovered = accuracy(fixed) - accuracy(sick)
    return float(np.clip(recovered / lost, 0.0, 1.0))                  # keep it between 0 and 1

if __name__ == "__main__":
    healthy = dict(np.load(DATA / "healthy.npz"))
    sick = dict(np.load("sick.npz"))

    good_fix = {k: v.copy() for k, v in sick.items()}
    good_fix["0.weight"][5] *= -1                                      # the right fix
    wrong_fix = {k: v.copy() for k, v in sick.items()}
    wrong_fix["0.weight"][14] *= -1                                    # a wrong fix (breaks another helper)

    print(f"Did nothing: score {score(healthy, sick, sick):.2f}")
    print(f"Right fix:   score {score(healthy, sick, good_fix):.2f}")
    print(f"Wrong fix:   score {score(healthy, sick, wrong_fix):.2f}")
