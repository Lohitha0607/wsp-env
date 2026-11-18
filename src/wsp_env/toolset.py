"""The doctor's tools as a Verifiers tool server: the ONLY way to touch the patient."""
import numpy as np
from pydantic import Field

import verifiers.v1 as vf
from wsp_env.checker import X_check, accuracy, per_digit, y_check

TEST_BUDGET = 10


class PatientState(vf.State):
    """This attempt's patient: shared by the tools (which edit it) and the reward (which marks it)."""
    weights: dict[str, list] = Field(default_factory=dict)
    tests_left: int = TEST_BUDGET


def as_arrays(weights: dict[str, list]) -> dict[str, np.ndarray]:
    return {k: np.array(v) for k, v in weights.items()}


class ClinicToolset(vf.Toolset[vf.ToolsetConfig, PatientState]):
    """The doctor's tools: the ONLY way to touch the patient."""
    TOOL_PREFIX = None

    def _edit(self, change) -> None:
        w = as_arrays(self.state.weights)
        change(w)
        self.state.weights = {k: v.tolist() for k, v in w.items()}

    @vf.tool
    async def overview(self) -> str:
        """Size (average |weight|) and bias of every helper (table "0") and boss (table "2")."""
        w = as_arrays(self.state.weights)
        lines = []
        for t, who in (("0", "helper"), ("2", "boss")):
            lines.append(f"{who} | size | bias")
            for i, (size, bias) in enumerate(zip(np.abs(w[f"{t}.weight"]).mean(axis=1), w[f"{t}.bias"])):
                lines.append(f"{i} | {size:.2f} | {bias:+.2f}")
        return "\n".join(lines)

    @vf.tool
    async def look(self, table: str, row: int) -> list[float]:
        """The exact weights of one row. table is "0" (helpers) or "2" (bosses)."""
        return np.round(as_arrays(self.state.weights)[f"{table}.weight"][row], 2).tolist()

    @vf.tool
    async def scale_row(self, table: str, row: int, factor: float) -> str:
        """Multiply one row's weights by factor (e.g. -1 flips signs, 0.1 makes it quieter)."""
        def change(w):
            w[f"{table}.weight"][row] *= factor
        self._edit(change)
        return f"{table}.weight row {row} multiplied by {factor}"

    @vf.tool
    async def set_bias(self, table: str, row: int, value: float) -> str:
        """Set one neuron's bias."""
        def change(w):
            w[f"{table}.bias"][row] = value
        self._edit(change)
        return f"{table}.bias[{row}] set to {value}"

    @vf.tool
    async def swap_rows(self, table: str, a: int, b: int) -> str:
        """Swap two neurons (their weights and biases)."""
        def change(w):
            for part in ("weight", "bias"):
                t = w[f"{table}.{part}"]
                t[[a, b]] = t[[b, a]]
        self._edit(change)
        return f"swapped {table} rows {a} and {b}"

    @vf.tool
    async def test_run(self) -> str:
        """Accuracy on a small check set. Limited uses!"""
        if self.state.tests_left <= 0:
            return "No tests left."
        self.state.tests_left -= 1
        acc = accuracy(as_arrays(self.state.weights), X_check, y_check)
        return f"Check accuracy: {acc:.1%} ({self.state.tests_left} tests left)"

    @vf.tool
    async def test_by_digit(self) -> str:
        """Accuracy for each digit 0-9 and the most common mix-ups (true→guessed) on the check set.
        Counts as one test."""
        if self.state.tests_left <= 0:
            return "No tests left."
        self.state.tests_left -= 1
        return per_digit(as_arrays(self.state.weights)) + f"\n({self.state.tests_left} tests left)"


if __name__ == "__main__":
    ClinicToolset.run()
