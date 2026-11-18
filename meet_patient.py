"""Phase 2, Chunk 4: meet the real patient (Qwen3 0.6B) while it's healthy."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)                       # the word ↔ number dictionary
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)  # the patient (downloads once)
model.eval()                                                    # exam mode: no learning

# 🏢 1. the building
print("=== 1. THE BUILDING ===")
print(len(model.model.layers), "floors")
print(f"{sum(p.numel() for p in model.parameters()):,} numbers in total")
print(model.model.layers[0])                                    # one floor: looking room + thinking room

# 📝 2. watch it write
print("\n=== 2. WATCH IT WRITE ===")
ids = tok("The capital of France is", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=15, do_sample=False)  # always pick the likeliest word
print(tok.decode(out[0], skip_special_tokens=True))

# 😌 3. healthy surprise score (perplexity)
print("\n=== 3. SURPRISE SCORE ===")
text = ("The sun rises in the east and sets in the west. Water boils at one hundred degrees Celsius. "
        "Cats and dogs are popular pets in many countries. Paris is the capital of France.")
ids = tok(text, return_tensors="pt")
with torch.no_grad():
    loss = model(**ids, labels=ids["input_ids"]).loss           # average "how wrong" per next word
print(f"surprise (perplexity): {torch.exp(loss).item():.1f}")
