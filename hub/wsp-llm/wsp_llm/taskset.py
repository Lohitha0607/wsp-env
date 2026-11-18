"""Phase 2 taskset: an AI doctor repairs a damaged real LLM (Qwen3 0.6B). Recovery 80% (surprise + fact quiz) + clean cure 20%."""
import math

import verifiers.v1 as vf
from wsp_llm import patient
from wsp_llm.toolset import FLOORS, TEST_BUDGET, LlmClinicToolset, LlmPatientState

SYSTEM_PROMPT = f"""You are an LLM doctor. A small language model (Qwen3 0.6B) has been damaged on purpose: \
one part of it was spoiled. Find the damage and repair it using your tools.

The model is a building of {FLOORS} floors (0-{FLOORS - 1}). Each floor has a "looking" room (attention) and a \
"thinking" room (MLP). Text passes up through every floor; the top guesses the next word.

Tools: floor_overview (how loud each room is), write (see what the model writes for a prompt), \
test_surprise (perplexity on a small check text: lower is healthier), test_quiz (% right on a small fact \
quiz, plus the questions it missed), skip_floor_test (perplexity with one floor \
switched off: if it DROPS, that floor was doing harm), scale_floor (multiply one room of one floor by a factor; \
edits stack). write, test_surprise, test_quiz and skip_floor_test share a budget of {TEST_BUDGET} tests.
Possible damage: a room made too loud, made almost silent, or with its signs flipped. \
Floors naturally differ in loudness; not everything unusual is damage. Switching off a healthy floor usually \
makes surprise go UP.
Scoring: 80% for how much health you restore (perplexity on hidden text, plus a hidden fact quiz when the \
damage hurt the quiz), 20% for a clean cure. Every room you \
leave changed that was not damaged lowers the clean-cure part, so undo experiments that did not help \
(e.g. scale by 1/factor).
When you are done, reply with a one-line summary of what you fixed."""

BUGS = (("loud", 10.0), ("flipped", -1.0), ("dead", 0.01))
QUESTIONS = [(bug, factor, floor) for floor in (2, 6, 20) for bug, factor in BUGS]     # 🔒 the exam (9 patients)
EXAM_FLOORS = patient.EXAM_FLOORS
EASY = [(bug, factor, floor) for floor in range(1, 27) if floor not in EXAM_FLOORS
        for bug, factor in BUGS if bug != "flipped"]                             # 🐣 practice without hidden flips (46)
PRACTICE = [(bug, factor, floor) for floor in range(1, 27) if floor not in EXAM_FLOORS
            for bug, factor in BUGS]                                              # 🏋️ practice only (69 patients)


class LlmData(vf.TaskData):
    floor: int
    factor: float
    fast: bool = False      # True → reward = surprise only (no quiz), for fast training
    closeness_weight: float = 0.0   # training bonus weight (0 = off)
    diagnosis_weight: float = 0.0   # training bonus weight (0 = off)


class LlmTaskConfig(vf.TaskConfig):
    clinic: vf.ToolsetConfig = vf.ToolsetConfig()


class LlmTask(vf.Task[LlmData, LlmPatientState, LlmTaskConfig]):
    @classmethod
    def toolsets(cls, config: LlmTaskConfig) -> list[vf.Toolset]:
        return [LlmClinicToolset(config.clinic)]

    def bug(self) -> dict:
        return {"floor": self.data.floor, "factor": self.data.factor}

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.state.bug = self.bug()
        trace.state.edits = []
        trace.state.tests_left = TEST_BUDGET

    @vf.reward(weight=0.8)
    async def recovery(self, trace: vf.Trace) -> float:
        """0 = still sick, 1 = fully healed (hidden text + hidden quiz the doctor never saw)."""
        return patient.recovery(self.bug(), trace.state.edits, use_quiz=not self.data.fast)

    @vf.reward(weight=0.2)
    async def clean_cure(self, trace: vf.Trace) -> float:
        """💊 Full when only the damaged room was touched; shrinks with each unneeded edit."""
        r = patient.recovery(self.bug(), trace.state.edits, use_quiz=not self.data.fast)
        return r / (1 + patient.extra_edits(self.bug(), trace.state.edits))

    @vf.reward(weight=1.0)
    async def closeness_bonus(self, trace: vf.Trace) -> float:
        """🎯 Optional training bonus (0 unless closeness_bonus > 0): partial credit for near-misses."""
        if not self.data.closeness_weight:
            return 0.0
        return self.data.closeness_weight * patient.closeness(self.bug(), trace.state.edits)

    @vf.reward(weight=1.0)
    async def diagnosis_bonus(self, trace: vf.Trace) -> float:
        """🔍 Optional training bonus (0 unless diagnosis_bonus > 0): credit for editing the damaged room."""
        if not self.data.diagnosis_weight:
            return 0.0
        return self.data.diagnosis_weight * patient.diagnosis(self.bug(), trace.state.edits)

    @vf.metric
    async def tests_used(self, trace: vf.Trace) -> float:
        return float(TEST_BUDGET - trace.state.tests_left)

    @vf.metric
    async def extra_edits(self, trace: vf.Trace) -> float:
        return float(patient.extra_edits(self.bug(), trace.state.edits))

    async def validate(self, runtime: vf.Runtime) -> bool | None:
        """Oracle check: undo the planted bug with the secret answer → the score must be 1."""
        return patient.recovery(self.bug(), [[self.data.floor, "thinking", 1 / self.data.factor]]) > 0.99


class LlmConfig(vf.TasksetConfig):
    task: LlmTaskConfig = LlmTaskConfig()
    split: str = "exam"                 # "exam" (floors 2, 6, 20) · "practice" (other floors 1–26) · "easy" (practice, no flips)
    fast_reward: bool = False           # True → surprise-only reward (skip the slow quiz); for TRAINING only
    closeness_bonus: float = 0.0        # TRAINING: extra reward × closeness (near-misses get partial credit)
    diagnosis_bonus: float = 0.0        # TRAINING: extra reward for editing the damaged room at all


class LlmTaskset(vf.Taskset[LlmTask, LlmConfig]):
    def load(self) -> list[LlmTask]:
        patient.apply({"floor": 0, "factor": 1.0}, [])
        healthy = math.exp(patient.loss(patient.CHECK_TEXT))
        healthy_quiz = patient.quiz(patient.QUIZ_CHECK)[0]
        tasks = []
        questions = {"exam": QUESTIONS, "practice": PRACTICE, "easy": EASY}[self.config.split]
        for bug, factor, floor in questions:
            patient.apply({"floor": floor, "factor": factor}, [])
            sick = math.exp(patient.loss(patient.CHECK_TEXT))
            sick_quiz = patient.quiz(patient.QUIZ_CHECK)[0]
            prompt = (f"This model is damaged. On the check text that test_surprise uses, a healthy model scores "
                      f"surprise {healthy:.2f}; this one scores {sick:.2f}. On the test_quiz questions, a healthy "
                      f"model gets {healthy_quiz:.0%} right; this one gets {sick_quiz:.0%}. Start with floor_overview.")
            tasks.append(LlmTask(
                LlmData(name=f"{bug}_floor{floor}", prompt=prompt, system_prompt=SYSTEM_PROMPT,
                        floor=floor, factor=factor, fast=self.config.fast_reward,
                        closeness_weight=self.config.closeness_bonus, diagnosis_weight=self.config.diagnosis_bonus),
                self.config.task,
            ))
        return tasks
