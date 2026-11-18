# wsp-llm: Weight Doctor for a real LLM

### Overview
- **Environment ID**: `wsp-llm`
- **Short description**: A small language model (**Qwen3-0.6B**, 28 layers) has been damaged on purpose: one layer's MLP output projection was scaled ×10 (too loud), ×0.01 (almost silent) or ×−1 (signs flipped). The agent is an "LLM doctor": using tools, it must **find the damaged layer and repair it by editing weights directly**, with no retraining. It is graded on **held-out text and a hidden fact quiz** it never sees.
- **Tags**: weight-surgery, model-editing, interpretability, multi-turn, tools, llm-repair, train, eval
- **Two kinds of "training" (don't mix them up):**

| | Patient (Qwen3-0.6B, the broken model) | Doctor (the agent model being tested) |
|---|---|---|
| Is it trained? | ❌ **Never.** The whole point is repair **without** training: no gradients, no fine-tuning, only direct weight edits through the tools | ✅ **Optionally**, with **reinforcement learning (RL)**: the doctor plays many repair episodes and is rewarded for good repairs, so it gets better at weight surgery |
| What changes | a few weights, edited by the doctor | the doctor's own weights, updated by an RL trainer (e.g. GRPO with prime-rl or Unsloth) |

  Everywhere below, "training" means **RL-training the doctor**. The patient is only ever repaired, never retrained.
- **Inspired by [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming)** (MIT, Nov 2026), where a frontier LLM repaired a small model by reading and editing its weights directly, with no training. This environment turns that idea into an open, reproducible, gradeable task (see **Credits** below).

### Task
- **Type**: multi-turn, tool use
- **The patient**: Qwen3-0.6B, loaded once per process. Damage = `layers[L].mlp.down_proj.weight × factor`. The doctor's edits are stored as a list of `(layer, room, factor)` and applied multiplicatively to cached healthy weights.
- **Visible vs hidden damage**: ×10 and ×0.01 change the layer's weight norm (visible in `floor_overview`); ×−1 keeps the norm identical, so it can only be found **behaviourally** (ablation, tests). This is the part that separates strong agents.
- **Tools** (`write`, `test_surprise`, `test_quiz` and `skip_floor_test` share a budget of **10 tests**):

| Tool | What it does |
|---|---|
| `floor_overview` | mean \|W\| of every layer's attention output (`o_proj`, "looking room") and MLP output (`down_proj`, "thinking room") |
| `write(prompt)` | greedy generation from the patient (~20 tokens) |
| `test_surprise` | perplexity on a 10-sentence check text (lower is healthier) |
| `test_quiz` | % right on a 20-question multiple-choice fact quiz, plus the questions missed |
| `skip_floor_test(layer)` | ablation: perplexity with one layer's attention + MLP outputs zeroed. If it **drops**, that layer was doing harm |
| `scale_floor(layer, room, factor)` | multiply one room of one layer; edits stack |

### Splits
| Split | Patients | Use |
|---|---|---|
| `exam` (default) | 9: layers **2, 6, 20** × {loud, dead, flipped} | evaluating doctors (held out: never use for training) |
| `practice` | 69: layers 1–26 except 2/6/20 × {loud, dead, flipped} | RL-training a doctor |
| `easy` | 46: like `practice`, without flips | RL-training a small doctor (easier curriculum) |

### Rubric
- **recovery (weight 0.8)**: how much healthy behaviour came back on **held-out** data. Computed in log-loss space on 30 hidden sentences (healthy perplexity ≈ 8.9), averaged with recovery on a **65-question hidden fact quiz** (healthy 88%) whenever the damage cost the quiz ≥ 5 points. 0 = still damaged, 1 = fully healed.
- **clean_cure (weight 0.2)**: recovery ÷ (1 + rooms changed that were not damaged). Rewards surgical fixes.
- Validated: the oracle fix scores 1 and doing nothing scores 0 on every patient (`vf-validate`).

### Quickstart
```bash
prime env install <owner>/wsp-llm
# evaluate a model (3 tries per exam patient)
vf-run wsp-llm -m <model> -r 3 \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.agent.max-turns 25 --env.agent.sampling.max-tokens 8000 \
  --env.taskset.task.clinic.timeout.startup 120
```
- **Hardware**: the patient runs locally. A GPU is strongly recommended (CPU works but grading is ~1 min per episode with the quiz). Set `WSP_PATIENT_DEVICE` (e.g. `cuda:1`) and `WSP_PATIENT_DTYPE=float16` to place a lighter patient on a spare GPU.
- Each concurrent episode loads its own patient (~1 GB RAM + GPU memory): cap concurrency accordingly.

### Environment arguments (`--env.taskset.<name>`)
| Arg | Default | Description |
|---|---|---|
| `split` | `"exam"` | `exam` · `practice` · `easy` |
| `fast_reward` | `false` | doctor training only: perplexity-only recovery (skips the slow quiz) |
| `closeness_bonus` | `0.0` | doctor training only: extra reward × how close the damaged room ends up to its normal size (1 = exact, 0.5 = 10× off, 0 = 100× off or still flipped) |
| `diagnosis_bonus` | `0.0` | doctor training only: extra reward for editing the damaged room at all |

The bonuses are off by default and never change the `exam` grading.

### Baselines (exam split, 3 tries per patient unless noted)
| Doctor | Overall | Visible damage (loud + dead) | Hidden flips |
|---|---|---|---|
| Nemotron 3.5 Super (free) | **0.82** | 18/18 | 4/9 |
| Claude Haiku 5.5 | 0.74 | 18/18 | ~2/9 |
| Nemotron 3.5 Lightning (free) | 0.67 | 16/18 | ~2/9 |
| Qwen3-4B (local) | 0.42 | partial | 0/9 |
| Random doctor (10 seeds, same tools and budget) | 0.11 | ~1 in 10 | partial |
| GPT-6 Astra (flips only, 1 try) | — | — | 1/3 |

### Training a doctor with RL (what we learned)
- **Single-turn variant** (`wsp_llm.oneshot`: see the overview table once, answer with the fix): GRPO with Unsloth on free Kaggle T4s, Qwen3-4B + LoRA, 150 steps → hidden exam **0.385 → 0.783**. Run 1 learned nothing because all samples in each group were identical (`frac_reward_zero_std = 1`); adding a short reasoning line, 8 samples per prompt, temperature 1.2 and a closeness bonus fixed it.
- That single-turn policy did **not transfer** to the multi-turn environment (0.42 → 0.34).
- **Multi-turn GRPO with prime-rl** (1× A100 80GB, Qwen3-4B-Instruct-2507 + LoRA, 6 small steps) ran end to end, but 67–92% of groups had zero reward variance (the base model rarely succeeds), so the result was inconclusive. For small models, start with `split="easy"` plus `closeness_bonus` / `diagnosis_bonus`, train for many more steps, and evaluate with 3+ tries per patient.

### Limitations
- 9 exam patients is small; report several tries per patient.
- Damage is a single scalar on one layer's `down_proj`; real-world damage is messier.
- The held-out sentences and quiz ship inside the package (the doctor has no file access, only the tools), so they are hidden from the agent but not secret: future models could see them in training data.
- Over-correcting a loud layer (e.g. ×0.01 instead of ×0.1) still scores well, because later layers compensate through the residual stream.

### Credits
- **Core idea:** [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming) (MIT, Nov 2026), on LLMs reading and editing neural-network weights directly, without training. Experiment logs: [github.com/RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs). The authors noted that a benchmark and harness for this were still missing; `wsp-llm` is an independent, open attempt at one.
- **Patient model:** [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Alibaba Qwen team).
- **Built with:** [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) and the Prime Intellect Environments Hub; training runs used [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) and [Unsloth](https://github.com/unslothai/unsloth).
