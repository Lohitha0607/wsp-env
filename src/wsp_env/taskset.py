"""wsp-env as a Verifiers taskset: dataset (sick patients) + tools (the clinic) + rewards (recovery 80% + clean cure 20%)."""
import json

import numpy as np

import verifiers.v1 as vf
from wsp_env.checker import DATA, X_check, accuracy, bug_rows, changed_rows, score, tidiness, y_check
from wsp_env.toolset import TEST_BUDGET, ClinicToolset, PatientState, as_arrays

SYSTEM_PROMPT = f"""You are a weight doctor. A small neural network that reads handwritten digits (0-9) is broken: \
some of its weights were spoiled on purpose. Fix it by editing weights directly, using your tools.

The network: 64 pixel inputs -> 32 helper neurons (table "0": 0.weight is 32x64, 0.bias is 32) -> ReLU \
-> 10 boss neurons, one per digit (table "2": 2.weight is 10x32, 2.bias is 10). The loudest boss is the answer.

Tools: overview (size and bias of every row), look (one row's weights), scale_row, set_bias, swap_rows, \
test_run (overall accuracy on a small check set) and test_by_digit (accuracy per digit + the most common mix-ups). \
test_run and test_by_digit share a budget of {TEST_BUDGET} tests.
Possible bugs include flipped signs, silenced or over-loud neurons, rigged biases and swapped rows. \
Some neurons are naturally small or unused; not everything unusual is a bug.
Scoring: 80% for how much accuracy you restore, 20% for a clean cure. Every neuron you leave changed that \
was not part of the bug lowers the clean-cure part, so undo experiments that did not help.
When you are done, reply with a one-line summary of what you fixed."""


class WspData(vf.TaskData):
    patient: str
    healthy_accuracy: float
    sick_accuracy: float


class WspTaskConfig(vf.TaskConfig):
    clinic: vf.ToolsetConfig = vf.ToolsetConfig()


class WspTask(vf.Task[WspData, PatientState, WspTaskConfig]):
    @classmethod
    def toolsets(cls, config: WspTaskConfig) -> list[vf.Toolset]:
        return [ClinicToolset(config.clinic)]

    def sick(self) -> dict[str, np.ndarray]:
        return dict(np.load(DATA / "dataset" / self.data.patient))

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.state.weights = {k: v.tolist() for k, v in self.sick().items()}
        trace.state.tests_left = TEST_BUDGET

    def answer(self) -> dict:
        """The secret answer for this patient (only the grader reads it, never the doctor)."""
        answers = (json.loads(line) for line in (DATA / "dataset" / "answers.jsonl").read_text().splitlines())
        return next(a for a in answers if a["id"] == self.data.name)

    def _recovery(self, trace: vf.Trace) -> float:
        healthy = dict(np.load(DATA / "healthy.npz"))
        return score(healthy, self.sick(), as_arrays(trace.state.weights))

    def _extra_edits(self, trace: vf.Trace) -> int:
        return len(changed_rows(self.sick(), as_arrays(trace.state.weights)) - bug_rows(self.answer()))

    @vf.reward(weight=0.8)
    async def recovery(self, trace: vf.Trace) -> float:
        """0 = no lost accuracy came back, 1 = fully cured (on 400 pictures the doctor never saw)."""
        return self._recovery(trace)

    @vf.reward(weight=0.2)
    async def clean_cure(self, trace: vf.Trace) -> float:
        """💊 Full when the cure touched only the broken neurons; shrinks with each unneeded edit."""
        return tidiness(self._recovery(trace), self._extra_edits(trace))

    @vf.metric
    async def tests_used(self, trace: vf.Trace) -> float:
        return float(TEST_BUDGET - trace.state.tests_left)

    @vf.metric
    async def extra_edits(self, trace: vf.Trace) -> float:
        """Neurons left changed that the bug never touched."""
        return float(self._extra_edits(trace))

    async def validate(self, runtime: vf.Runtime) -> bool | None:
        """Oracle check: undo the planted bug with the secret answer → the score must be 1."""
        answer = self.answer()
        healthy = dict(np.load(DATA / "healthy.npz"))
        w = self.sick()
        match answer["bug"]:
            case "sign_flip":
                w["0.weight"][answer["helpers"]] *= -1
            case "dead_helper":
                w["0.bias"][answer["helpers"]] = healthy["0.bias"][answer["helpers"]]
            case "swap_bosses":
                a, b = answer["bosses"]
                for part in ("2.weight", "2.bias"):
                    w[part][[a, b]] = w[part][[b, a]]
            case "too_loud":
                w["0.weight"][answer["helper"]] /= 10
            case "rigged_boss":
                w["2.bias"][answer["boss"]] -= 10
            case _:
                return None                                        # noise: no clean fix exists
        return score(healthy, self.sick(), w) > 0.99


class WspConfig(vf.TasksetConfig):
    task: WspTaskConfig = WspTaskConfig()


class WspTaskset(vf.Taskset[WspTask, WspConfig]):
    def load(self) -> list[WspTask]:
        # quote accuracies on the SAME check set test_run uses, so the doctor's tests line up with the prompt
        healthy_check = accuracy(dict(np.load(DATA / "healthy.npz")), X_check, y_check)
        tasks = []
        for line in (DATA / "dataset" / "questions.jsonl").read_text().splitlines():
            q = json.loads(line)
            sick_check = accuracy(dict(np.load(DATA / "dataset" / q["patient"])), X_check, y_check)
            prompt = (f"{q['task']} On the check set that test_run uses, a healthy network scores "
                      f"{healthy_check:.1%}; this one scores {sick_check:.1%}. Start with overview.")
            tasks.append(WspTask(
                WspData(name=q["id"], prompt=prompt, system_prompt=SYSTEM_PROMPT, patient=q["patient"],
                        healthy_accuracy=q["healthy_accuracy"], sick_accuracy=q["sick_accuracy"]),
                self.config.task,
            ))
        return tasks
