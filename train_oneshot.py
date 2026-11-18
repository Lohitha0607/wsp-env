"""Way 1: free GRPO training (Unsloth) on the Phase 2 one-shot task. Made for Kaggle (2× T4):
GPU 0 = the student being trained, GPU 1 = the patient that grades it.

  python train_oneshot.py eval before    # 📸 the student BEFORE training, on the hidden exam floors (full grading)
  python train_oneshot.py train          # 🏋️ GRPO on the practice floors (fast treat) → saves grpo-lora-2/
  python train_oneshot.py eval after     # 🏆 the student AFTER training, same exam, same grading

Run 2 changes (run 1 learned nothing: all 4 attempts were identical every step):
  a short reasoning line before the answer · 8 attempts per patient · temperature 1.2 · closeness bonus in the treat
"""
import json
import os
import statistics
import sys

STAGE, TAG = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
os.environ.setdefault("WSP_PATIENT_DEVICE", "cuda:1")
# training: a lighter patient for speed · exams: full precision, comparable with every earlier result
os.environ.setdefault("WSP_PATIENT_DTYPE", "float16" if STAGE == "train" else "float32")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from unsloth import FastLanguageModel   # noqa: E402  (Unsloth must load before transformers / trl)

from wsp_llm import oneshot   # noqa: E402

STUDENT = "unsloth/Qwen3-4B"
LORA_DIR = "grpo-lora-2"
TRIES = 4             # answers per exam patient (sampled), so one lucky answer can't decide the score
MAX_STEPS = 150


def load(path: str):
    return FastLanguageModel.from_pretrained(path, max_seq_length=2048, load_in_4bit=True)


def chat(tok, text: str) -> str:
    """The question in the student's chat format, with thinking switched off (short, direct answers)."""
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def evaluate(tag: str) -> None:
    model, tok = load(LORA_DIR if tag == "after" else STUDENT)
    FastLanguageModel.for_inference(model)
    results = []
    for row in oneshot.dataset("exam"):
        bug = {"floor": row["floor"], "factor": row["factor"]}
        ids = tok(chat(tok, row["prompt"]), return_tensors="pt").to("cuda:0")
        outs = model.generate(**ids, max_new_tokens=256, do_sample=True, temperature=0.7, num_return_sequences=TRIES)
        for out in outs:
            answer = tok.decode(out[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            s = oneshot.score(bug, answer, use_quiz=True)                       # 🔒 full exam grading
            results.append({"patient": row["name"], "score": round(s, 3), "answer": answer[:200]})
            print(f"{row['name']:16s} {s:.2f}  {answer[:70]!r}", flush=True)
    mean = statistics.mean(r["score"] for r in results)
    print(f"\n📊 {tag.upper()}: mean exam score {mean:.3f} over {len(results)} answers")
    json.dump(results, open(f"oneshot-{tag}-run2.json", "w"), indent=1)


def train() -> None:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    model, tok = load(STUDENT)
    model = FastLanguageModel.get_peft_model(          # 📝 LoRA: train a small "sticky note", not the whole model
        model, r=16, lora_alpha=32, random_state=0, use_gradient_checkpointing="unsloth",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    rows = oneshot.dataset("practice")                 # 🏋️ practice floors only, the exam stays hidden
    data = Dataset.from_list([{"prompt": chat(tok, r["prompt"]), "floor": r["floor"], "factor": r["factor"]}
                              for r in rows])

    def treat(prompts, completions, floor, factor, **kwargs):
        """🦴 One treat per answer: fast score + closeness bonus (so near-misses still teach something)."""
        return [oneshot.treat({"floor": f, "factor": x}, c) for c, f, x in zip(completions, floor, factor)]

    args = GRPOConfig(
        output_dir="grpo-out-2", report_to="none", logging_steps=1, max_steps=MAX_STEPS, save_steps=50,
        learning_rate=5e-6, warmup_ratio=0.1, lr_scheduler_type="cosine", optim="adamw_8bit",
        per_device_train_batch_size=8, gradient_accumulation_steps=1,
        num_generations=8,                             # 8 attempts per patient → compare them → learn
        max_prompt_length=1024, max_completion_length=256, temperature=1.2, max_grad_norm=0.1)
    GRPOTrainer(model=model, processing_class=tok, reward_funcs=[treat], args=args, train_dataset=data).train()
    model.save_pretrained(LORA_DIR)
    tok.save_pretrained(LORA_DIR)
    print(f"✅ saved the trained student to {LORA_DIR}/")


if __name__ == "__main__":
    train() if STAGE == "train" else evaluate(TAG)
