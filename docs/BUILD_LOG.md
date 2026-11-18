# 🩺 wsp-env — Weight Doctor

**An exam (RL environment) that tests how well an AI fixes a broken neural network by editing its numbers directly** (no training).
Inspired by MIT's "Sparks of Weight-Space Programming" (Nov 2026), which showed a big AI repairing a model this way, but no open exam existed.
Full plan: `~/Desktop/notes/ai/projects/wsp-env.md`.

## 🏥 The idea (a doctors' exam)
| Exam part | In this project |
|---|---|
| 🧒 healthy patient | a tiny network that reads handwritten digits (~97% right) |
| 🤒 sick patients = the **questions** | copies of it with numbers spoiled on purpose |
| 🩺 doctor = who **takes** the exam | an AI model (DeepSeek, GPT, Claude…) using our tools |
| ✅ marking scheme = the **rubric** | score 0 → 1 = how much lost accuracy came back |

RL environment = 📄 **dataset** (60 sick patients) + ✍️ **rollout** (the AI fixes, via Verifiers) + ✅ **rubric** (recovery score).

## ▶️ How to run
```bash
uv run vf-validate wsp-env --runtime.type subprocess        # self-test: gold fix → 1, do nothing → 0 ($0)
uv run python doctor.py                                     # baseline rule doctor on all 60

# local + free: ollama pull qwen3:8b && ollama create qwen3-wsp -f ollama/Modelfile, then use
#   -m qwen3-wsp --client.base-url http://localhost:11434/v1 --client.api-key-var OLLAMA_API_KEY (OLLAMA_API_KEY=ollama)
set -a && source .env && set +a                             # loads OPENROUTER_API_KEY (never commit .env)
uv run vf-run wsp-env -n 1 -r 1 -v -m deepseek/deepseek-v4-flash \
  --client.base-url https://openrouter.ai/api/v1 --client.api-key-var OPENROUTER_API_KEY \
  --env.agent.harness.id null --env.agent.runtime.type subprocess --env.agent.max-turns 30 \
  --env.taskset.task.clinic.timeout.startup 60              # one AI doctor episode; traces in outputs/
```

## 📁 Files
**The environment (package `src/wsp_env/`, Verifiers id `wsp-env`)**
| File | Job |
|---|---|
| `taskset.py` | loads the 60 questions, sets up each patient in the state, the **reward** (`recovery`) + metric (`tests_used`), and `validate` (oracle check) |
| `toolset.py` | the doctor's tools as an MCP server: `overview`, `look`, `scale_row`, `set_bias`, `swap_rows`, `test_run` (max 10) |
| `checker.py` | ✅ the rubric: hidden 540 pictures split into 140 🧪 check (the doctor may test) + 400 🔒 final (marks only); `accuracy()` + `score()` |
| `data/healthy.npz` | the healthy patient (2,410 numbers) |
| `data/dataset/` | 60 sick patients + `questions.jsonl` 👀 (doctor sees) + `answers.jsonl` 🔒 (secret) |

**Scripts (top level)**
| File | Job |
|---|---|
| `look.py` | show one digit picture as numbers + blocks |
| `train.py` | train the patient: 64 → 32 helpers → ReLU → 10 bosses, 5% → 97% → `healthy.npz` |
| `break_it.py` | the first hand-made bug (flip helper 5) → `sick.npz` |
| `make_dataset.py` | 6 bug types × 10 → `data/dataset/` (skips bugs that drop < 3 points) |
| `clinic.py` + `doctor.py` | rule-based **baseline** doctor (fixes only what looks weird) → marks all 60 |

**Score (v2, 2026-11-09):** `0.8 × recovery + 0.2 × clean_cure`
- `recovery = (fixed − sick) / (healthy − sick)` on the final 400, clipped to 0–1. Do nothing → 0 · right fix → 1 · wrong fix → 0.
- 💊 `clean_cure = recovery / (1 + extra_edits)`, where `extra_edits` = neurons left changed that the bug never touched (from the secret answer). Clean fix → 1.00 · same fix + 3 unneeded edits → 0.85 total.
- Metrics logged too: `tests_used`, `extra_edits`.
- 🔍 New tool `test_by_digit`: per-digit accuracy + top mix-ups (shares the 10-test budget). A 2↔7 swap shows as "digit 2: 0%, digit 7: 0%, mix-ups 7→2 ×15, 2→7 ×13".
- ⚠️ Results above this line in the table were scored with v1 (recovery only, no `test_by_digit`) — not directly comparable.

## 🦠 The 6 bug types (10 questions each)
| Bug | What is spoiled | Visible in `overview`? |
|---|---|---|
| 🔄 sign flip | 3 helpers' weights × −1 | ❌ looks normal |
| 💀 dead helper | 3 helpers' bias → −100 | ✅ |
| 🔀 swapped bosses | two bosses swap rows | ❌ looks normal |
| 📢 too loud | one helper × 10 | ✅ |
| 🎯 rigged boss | one boss bias + 10 | ✅ |
| 🌫️ noise | small random changes everywhere | ❌ no clean fix |

## 📊 Results so far
| Doctor | dead | too loud | rigged | sign flip | swap | noise | **overall** |
|---|---|---|---|---|---|---|---|
| 🏁 rule baseline (`doctor.py`) | 0.98 | 1.00 | 1.00 | 0 | 0 | 0 | **0.50** |
| 🧠 deepseek-v4-flash (all 60, 30 turns, ≈ $1.2 actual) | 0.99 | 1.00 | 1.00 | 0.08 | 0.00 | 0.01 | **0.51** |
| 🌟 gpt-6-astra (2 questions only: sign_flip_00, swap_bosses_00; ≈ $0.51) | | | | 1.00 | 1.00 | | — |
| 🟠 claude-sonnet-5.5 (1 question: sign_flip_00; ≈ $0.11) | | | | 0.00 | | | — |
| 🟠 claude-fable-5.1 (2 questions) | | | | cut off | cut off | | — |
| 💻 qwen3:8b local via Ollama (3 questions: rigged_boss_00, sign_flip_00, swap_bosses_00; $0) | | | 1.00 | 0.00 | 0.00 | | — |
| 🆓 nemotron-3-ultra-550b:free (1 question: sign_flip_00; $0) | | | | 0.00 | | | — |
| **— exam v2 (clean-cure bonus + `test_by_digit`) —** | | | | | | | |
| 🟠 claude-haiku-5.5 (2 questions: rigged_boss_00, swap_bosses_00; ≈ $0.01) | | | 1.00 | | 1.00 | | — |

## 🏆 Leaderboard by tier (hidden-bug questions: `sign_flip_00` 🔄, `swap_bosses_00` 🔀)
Sizes in **bold** are published; others are vendor tiers (sizes not public). v1 = old exam, v2 = clean-cure bonus + `test_by_digit`.

| Tier | Model | Access | 🔄 Sign flip | 🔀 Swap | Exam | Cost |
|---|---|---|---|---|---|---|
| 🏔️ Frontier | openai/gpt-6-astra | paid | **1.00** | **1.00** | v1 | ~$0.25/q |
| 🏔️ Frontier | anthropic/claude-fable-5.1 | paid | cut off | cut off | v1 | (credits) |
| 🏢 Large | anthropic/claude-sonnet-5.5 | paid | 0.00 | — | v1 | ~$0.11 |
| 🏢 Large | nvidia/nemotron-3-ultra (**550B**, 55B active) | free | 0.00 (v1) · 0.00 (v2) | found 2↔7 ✅ then cut off (rate limit) | v1/v2 | $0 |
| 🏠 Mid | deepseek/deepseek-v4-flash | paid (cheap) | 0.00 | 0.00 | v1 | ~2¢/q |
| 🐜 Small/fast | anthropic/claude-haiku-5.5 | paid (cheapest Claude) | — | **1.00** | v2 | ~0.5¢/q |
| 🐜 Small | qwen3 (**8B**) on the MacBook | local | 0.00 | 0.00 | v1 | $0 |
| 🏠/🐜 not run yet | nemotron-3.5-super, gemma-4 (**31B**, **26B**-A4B), nemotron-3-nano (**30B**-A3B) | free | — | — | — | blocked: free daily limit |

**Reading it:** in v1 only the frontier model solved hidden bugs. In v2 the per-digit tool makes swaps solvable even for small models (Haiku 1.00; Nemotron found it too). Sign flips still separate the frontier from everyone else.

**Self-test (`vf-validate`):** 🔑 gold (correct fix → 1) 50/50 ✅ (10 noise = unchecked by design) · 🚫 noop (do nothing → 0) 60/60 ✅

## 📝 Findings
1. **The network is robust:** one flipped helper only drops 97% → 92% (8 flipped → 75%).
2. **Sign flips hide:** same size as healthy rows, so the doctor must reason or test.
3. **Natural tiny helpers** (0.07–0.09: helpers 0, 3, 17, 20, 24, 25, 29) are unused after training, not broken → a **false-alarm trap**. The first AI run fell straight into it 🪤.
4. **Exam-design bug found by reading the trace:** the prompt quotes final-set numbers (96% → 92%), but `test_run` reports check-set numbers (healthy 98.6%, this patient 97.1%). Different rulers confused the AI. → fix next.
5. **The AI ≈ the rule baseline (0.51 vs 0.50):** it fixes exactly what is visible in `overview` (dead, too loud, rigged: 30/30 wins) and none of the hidden bugs (sign flip, swap, noise: 0/30). It never hit its turn limit on visible bugs and always hit it on hidden ones.
6. **Over-treatment 💊:** in wins it also "fixes" the naturally tiny helpers (unneeded edits). The score doesn't penalise this yet → add the planned **minimality bonus**.
7. **Tool design limits what's solvable:** `test_run` only gives one overall %. A swapped 3↔8 shows up as "3s and 8s confused", but no tool reports per-digit accuracy, so swaps are nearly impossible to localise in 10 tests.
8. **Strong vs cheap model on the same hidden bugs:** deepseek-v4-flash → 0.00 after 30 turns; gpt-6-astra → 1.00 in 7–11 turns, ~10× fewer tokens. Astra found swapped bosses 2↔7 just by *reading* boss weights (1 test), and found sign flips by flipping candidates and checking (3 tests).
9. **OpenRouter + key limits:** Verifiers asks for 65,536 max tokens per turn by default; a key with a small remaining limit gets HTTP 402 before anything runs → pass `--env.agent.sampling.max-tokens 8000`.
10. **Claude runs (2026-11-09):** fable-5.1 was cut off mid-run by OpenRouter's *account* in-flight budget (402 `in_flight_budget_exhausted`, separate from the key limit) — no score; before the cutoff it correctly called the tiny helpers "naturally dead… probably not the bug". sonnet-5.5 on sign_flip_00: 0.00 in 15 turns / 9 tests — avoided the tiny-helper trap but spent tests on bias and scaling guesses, flipped only helper 10 (wrong), undid every change, and honestly reported "I didn't find the bug". Hidden sign flips need *systematic flip-and-test*, which only astra did.
11. **Local open model (qwen3:8b on an M5 MacBook, 16 GB, Ollama, 16k context, $0):** rigged boss 1.00 in 4 turns / 1 test, and *no* unneeded edits. Hidden bugs 0.00: random bias/scale/swap edits it never undid, flipped a naturally tiny helper, and on the swap it *said* "swapping rows 3 and 5 will fix it" without calling the tool → **narration ≠ state**: grading the weights, not the reply, is what catches this. ~13–16 min per hard question on the laptop.
12. **Free big model (nemotron-3-ultra-550b:free via OpenRouter):** sign flip 0.00 — used 15 of 20 turns on one `look` per turn, ran 1 test, then scaled the naturally tiny helpers ×5 (the same trap). Size alone didn't help; *strategy* (flip-and-test) is what separates Astra. Finding free tool-capable models: filter https://openrouter.ai/api/v1/models for ids ending `:free` with `tools` in `supported_parameters`.
13. **Exam v2 works as intended:** claude-haiku-5.5 (the cheapest Claude) solved the swap that every cheap model failed in v1 — one `test_by_digit` showed "2 and 7 completely swapped", one swap, done (6 turns, 3 tests, 0 extra edits, total 1.00). It also fixed the rigged boss with 0 extra edits and said "I made no other changes". The prompt's "not everything unusual is a bug" line + the clean-cure rule stopped the tiny-helper over-treatment.
14. **OpenRouter free models have a daily cap** ("free-models-per-day", 429): with < $5 of credits it's tiny (we hit it after ~1.5 runs); adding $5 credits raises it to 1,000 free requests/day.
15. **Verifiers gotcha:** a toolset module must end with `if __name__ == "__main__": MyToolset.run()`, or the tool server never reports its port (10-minute hang).

## 📍 Where I am
- [x] Healthy patient · rubric · 60-question dataset · tools · baseline (0.50)
- [x] Packaged as a Verifiers taskset (`wsp-env`), self-test passes
- [x] First real AI doctor episode (deepseek-v4-flash, 1 question, ≈ $0.005)
- [x] Fixed the prompt to quote check-set accuracies (the AI now says "97.1%, as expected")
- [x] Full run, all 60 (deepseek-v4-flash): **0.51** vs baseline 0.50 — solves every *visible* bug, fails every *hidden* one
- [x] Strong model test: **gpt-6-astra (the MIT post's model) solved both hidden bugs it was given** (sign flip 1.00 in 3 tests, swap 1.00 in 1 test), naming the exact neurons in the secret answers. The exam **separates models** ✅ (n = 2, so more runs needed)
- [ ] ⏭️ More astra questions for a fair score (needs budget) · minimality bonus · per-digit test tool
- [ ] Leaderboard: several models · tests-used vs score · per-bug breakdown
- [ ] Write-up (blog / README results) → publish on the Environments Hub
