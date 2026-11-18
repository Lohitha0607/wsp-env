# 🩺 wsp-env: Weight Doctor

**RL environments where an AI "doctor" repairs a deliberately damaged neural network by editing its weights directly, with no retraining, graded on held-out data.**

Inspired by [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming) (MIT, Nov 2026), where a frontier LLM repaired a small model by reading and editing its weights. The authors noted that a benchmark and harness for this were still missing; this repo is an independent, open attempt at one.

- 🧩 **Published on the Prime Intellect Environments Hub:** [`loh/wsp-llm`](https://app.primeintellect.ai/dashboard/environments/loh/wsp-llm): `prime env install loh/wsp-llm`
- 🛠️ Built with [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) (v1)

> **Two kinds of "training":** the **patient** (the broken model) is never trained: it is only repaired by direct weight edits. "Training" in this repo always means **RL-training the doctor** (the agent) to get better at weight surgery.

---

## The two environments
| | 🔢 Phase 1: `wsp-env` | 🤖 Phase 2: `wsp-llm` (on the Hub) |
|---|---|---|
| Patient | a tiny digit classifier trained from scratch (`train.py`): 64 → 32 → 10 MLP, 2,410 parameters, 96.5% accuracy | **Qwen3-0.6B** (28 layers, 596M parameters) |
| Damage | per-neuron: sign flips, dead neurons (bias −100), ×10 rows, rigged output bias, swapped output rows, noise (60 patients) | one layer's `mlp.down_proj` × 10 (loud), × 0.01 (dead) or × −1 (flipped) (9 exam + 69 practice patients) |
| Doctor's tools | overview, look, scale_row, set_bias, swap_rows, test_run, test_by_digit (10-test budget) | floor_overview, write, test_surprise, test_quiz, skip_floor_test (ablation), scale_floor (10-test budget) |
| Grading | accuracy recovered on 400 held-out images | perplexity recovered on 30 held-out sentences + a 65-question held-out fact quiz |
| Reward | 0.8 × recovery + 0.2 × clean cure (penalises touching healthy neurons) | same |
| Validated | oracle fix = 1, no-op = 0 on every patient | same |

Key design idea: some damage is **visible** (it changes weight norms, so the overview shows it) and some is **hidden** (sign flips keep the norm identical), so it can only be found behaviourally. Hidden damage is what separates strong doctors from weak ones.

## Results (Phase 2, exam split, 3 tries per patient unless noted)
| Doctor | Overall | Visible damage | Hidden flips |
|---|---|---|---|
| Nemotron 3.5 Super (free) | **0.82** | 18/18 | 4/9 |
| Claude Haiku 5.5 | 0.74 | 18/18 | ~2/9 |
| Nemotron 3.5 Lightning (free) | 0.67 | 16/18 | ~2/9 |
| Qwen3-4B (local) | 0.42 | partial | 0/9 |
| Random doctor (10 seeds, same tools + budget) | 0.11 | ~1 in 10 | partial |
| GPT-6 Astra (flips only, 1 try) | — | — | 1/3 |

**Findings**
- Every model fixes visible damage; **hidden flips separate them**. In Phase 1, only GPT-6 Astra fixed hidden sign flips.
- The random doctor beat every model on flips: it *tried fixes*, while the models mostly only *ran tests*.
- One agent used a free tool as an unlimited test (a loophole, since closed); one claimed a fix it never made; one honestly reported failure.
- **Prompt optimisation (GEPA)** made a small student *worse* on the hidden exam (0.42 → 0.25) because of one wrong hint; fixing that hint gave 0.52.
- **Single-turn GRPO** (Unsloth, free Kaggle T4s, Qwen3-4B + LoRA): run 1 learned nothing (identical samples, zero advantage); run 2 (a reasoning line, 8 samples, temperature 1.2, closeness bonus) went **0.385 → 0.783** on held-out layers. It did **not transfer** to the multi-turn environment (0.42 → 0.34).
- **Multi-turn GRPO** (prime-rl, 1× A100 80GB, 6 small steps) ran end to end but was inconclusive (0.108 → 0.002 on 9 single tries; 67–92% of groups had no reward variance).

## Repo layout
| Path | What |
|---|---|
| `src/wsp_env/` | Phase 1 environment (digit patient): `checker.py` (grader), `toolset.py`, `taskset.py`, `data/` |
| `src/wsp_llm/` | Phase 2 environment (Qwen3-0.6B patient): `patient.py`, `toolset.py`, `taskset.py`, `oneshot.py`, `data/` |
| `train.py`, `make_dataset.py`, `break_it.py` | build the Phase 1 patient and its 60 damaged versions |
| `doctor.py`, `random_doctor.py` | rule-based (Phase 1) and random (Phase 2) baselines |
| `meet_patient.py`, `break_patient.py` | first look at the Qwen patient and its damage |
| `train_oneshot.py` | single-turn GRPO with Unsloth (Kaggle) |
| `gpu/` | multi-turn RL with prime-rl on a rented GPU (`setup.sh`, `start.sh`, configs) |
| `gepa/` | prompt cards from the GEPA experiment |
| `hub/wsp-llm/` | the package published on the Environments Hub |
| `docs/BUILD_LOG.md` | the full build diary |

## Quickstart
```bash
uv sync
uv run vf-validate wsp-env --runtime.type subprocess     # Phase 1 self-check
uv run vf-validate wsp-llm --runtime.type subprocess     # Phase 2 self-check
uv run vf-run wsp-llm -m <model> -r 3 --client.base-url <openai-compatible-url> --client.api-key-var <ENV_VAR> \
  --env.agent.harness.id null --env.agent.runtime.type subprocess --env.agent.max-turns 25
```
Phase 2 runs the patient locally; a GPU is strongly recommended.

## Credits
- Core idea: [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming) (MIT, Nov 2026) · logs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)
- Patient: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) · Built with [Verifiers](https://github.com/PrimeIntellect-ai/verifiers), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), [Unsloth](https://github.com/unslothai/unsloth)

## License
Apache-2.0
