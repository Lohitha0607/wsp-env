"""A rule-based baseline doctor: fixes only what LOOKS weird in the overview. An AI must beat this."""
import json
import numpy as np
from collections import defaultdict
from wsp_env.checker import DATA, score
from clinic import Clinic

healthy = dict(np.load(DATA / "healthy.npz"))      # used ONLY by the checker for marking, never by the doctor

def rule_doctor(clinic):
    s = clinic.stats()
    helper_size, helper_bias = s["0"]["size"], s["0"]["bias"]
    boss_bias = s["2"]["bias"]

    for i in range(32):
        if helper_bias[i] < -10:                                   # 💀 dead helper → normal bias
            clinic.set_bias("0", i, 0.0)
        if helper_size[i] > 3 * np.median(helper_size):           # 📢 too loud → quieter
            clinic.scale_row("0", i, 0.1)
    for i in range(10):
        if boss_bias[i] > 3:                                       # 🎯 rigged boss → normal bias
            clinic.set_bias("2", i, float(np.median(boss_bias)))
    clinic.test_run()                                              # check once before handing in
    return clinic.submit()

questions = [json.loads(line) for line in open(DATA / "dataset/questions.jsonl")]
marks = defaultdict(list)
for q in questions:
    sick = dict(np.load(DATA / "dataset" / q["patient"]))
    fixed = rule_doctor(Clinic(DATA / "dataset" / q["patient"]))
    marks[q["id"].rsplit("_", 1)[0]].append(score(healthy, sick, fixed))

print("Rule doctor's average mark per bug type:")
for bug, m in marks.items():
    print(f"  {bug:12s} {np.mean(m):.2f}")
print(f"  {'OVERALL':12s} {np.mean([x for m in marks.values() for x in m]):.2f}")
