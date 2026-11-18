"""The doctor's toolbox: the ONLY way a doctor can touch a patient."""
import numpy as np
from wsp_env.checker import accuracy, X_check, y_check   # the doctor's 140 check pictures

class Clinic:
    def __init__(self, patient_file, test_budget=10):
        self.w = dict(np.load(patient_file))
        self.tests_left = test_budget

    def stats(self):
        """Each row's size (average |weight|) and bias, for helpers ("0") and bosses ("2")."""
        return {t: {"size": np.abs(self.w[f"{t}.weight"]).mean(axis=1), "bias": self.w[f"{t}.bias"].copy()}
                for t in ("0", "2")}

    def overview(self):
        """📋 A quick X-ray of the patient, as text."""
        s = self.stats()
        lines = []
        for t, who in (("0", "helper"), ("2", "boss")):
            lines.append(f"{who} | size | bias")
            for i, (size, bias) in enumerate(zip(s[t]["size"], s[t]["bias"])):
                lines.append(f"{i:6d} | {size:.2f} | {bias:+.2f}")
        return "\n".join(lines)

    def look(self, table, row):
        """👀 The exact weights of one row."""
        return np.round(self.w[f"{table}.weight"][row], 2).tolist()

    def scale_row(self, table, row, factor):
        """✖️ Multiply a row (×−1 = flip back, ×0.1 = quieter)."""
        self.w[f"{table}.weight"][row] *= factor
        return f"{table}.weight row {row} × {factor}"

    def set_bias(self, table, row, value):
        """➕ Change one bias."""
        self.w[f"{table}.bias"][row] = value
        return f"{table}.bias[{row}] = {value}"

    def swap_rows(self, table, a, b):
        """🔀 Swap two rows (weights and bias)."""
        for part in ("weight", "bias"):
            t = self.w[f"{table}.{part}"]
            t[[a, b]] = t[[b, a]]
        return f"swapped {table} rows {a} and {b}"

    def test_run(self):
        """🧪 Accuracy on the 140 check pictures. Limited uses!"""
        if self.tests_left == 0:
            return None
        self.tests_left -= 1
        return accuracy(self.w, X_check, y_check)

    def submit(self):
        """✅ Hand in the patient."""
        return self.w
