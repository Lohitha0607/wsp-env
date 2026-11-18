"""The Phase 2 doctor's tools: examine and repair a real LLM floor by floor."""
import math

from pydantic import Field

import verifiers.v1 as vf
from wsp_llm import patient
from wsp_llm.patient import CHECK_TEXT, QUIZ_CHECK, ROOMS

TEST_BUDGET = 10
FLOORS = 28


class LlmPatientState(vf.State):
    """This attempt's patient: the planted bug (hidden) and every edit the doctor made."""
    bug: dict = Field(default_factory=dict)
    edits: list[list] = Field(default_factory=list)
    tests_left: int = TEST_BUDGET


class LlmClinicToolset(vf.Toolset[vf.ToolsetConfig, LlmPatientState]):
    """The doctor's tools: the ONLY way to touch the patient."""
    TOOL_PREFIX = None

    def _ready(self) -> None:
        patient.apply(self.state.bug, self.state.edits)

    def _spend_test(self) -> str | None:
        if self.state.tests_left <= 0:
            return "No tests left."
        self.state.tests_left -= 1
        return None

    @vf.tool
    async def floor_overview(self) -> str:
        """How loud each floor's looking room and thinking room are (average size of their numbers)."""
        self._ready()
        return patient.overview()

    @vf.tool
    async def write(self, prompt: str) -> str:
        """Give the patient a prompt and read what it writes next (about 20 words). Uses one test."""
        if no := self._spend_test():
            return no
        self._ready()
        return f"{patient.write(prompt)}\n({self.state.tests_left} tests left)"

    @vf.tool
    async def test_surprise(self) -> str:
        """The patient's surprise (perplexity) on a small check text: lower is healthier. Uses one test."""
        if no := self._spend_test():
            return no
        self._ready()
        return f"Surprise: {math.exp(patient.loss(CHECK_TEXT)):.2f} ({self.state.tests_left} tests left)"

    @vf.tool
    async def test_quiz(self) -> str:
        """% right on a small fact quiz (e.g. "The capital of France is" → Paris) and the questions it missed.
        Uses one test."""
        if no := self._spend_test():
            return no
        self._ready()
        score, wrong = patient.quiz(QUIZ_CHECK)
        missed = "; ".join(wrong[:5]) or "none"
        return f"Quiz: {score:.0%} right. Missed: {missed} ({self.state.tests_left} tests left)"

    @vf.tool
    async def skip_floor_test(self, floor: int) -> str:
        """Surprise on the check text with one floor switched off (both rooms silent). If surprise DROPS,
        that floor was doing harm. Uses one test; the floor is switched back on afterwards."""
        if not 0 <= floor < FLOORS:
            return f"floor must be 0-{FLOORS - 1}"
        if no := self._spend_test():
            return no
        patient.apply(self.state.bug, self.state.edits + [[floor, room, 0.0] for room in ROOMS])
        return (f"Surprise with floor {floor} skipped: {math.exp(patient.loss(CHECK_TEXT)):.2f} "
                f"({self.state.tests_left} tests left)")

    @vf.tool
    async def scale_floor(self, floor: int, room: str, factor: float) -> str:
        """Multiply one room ("looking" or "thinking") of one floor by factor
        (e.g. -1 flips it, 0.1 makes it quieter, 10 makes it louder). Edits stack."""
        if room not in ROOMS:
            return 'room must be "looking" or "thinking"'
        if not 0 <= floor < FLOORS:
            return f"floor must be 0-{FLOORS - 1}"
        if factor == 0:
            return "factor 0 cannot be undone; use a small number like 0.01 instead"
        self.state.edits = self.state.edits + [[floor, room, factor]]
        return f"floor {floor} {room} room multiplied by {factor}"


if __name__ == "__main__":
    LlmClinicToolset.run()
