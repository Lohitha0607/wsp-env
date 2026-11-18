"""Phase 2, Chunk 5: break the real patient 3 ways and watch what happens."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
model.eval()

TEXT = ("The sun rises in the east and sets in the west. Water boils at one hundred degrees Celsius. "
        "Cats and dogs are popular pets in many countries. Paris is the capital of France.")

def write(prompt="The capital of France is"):
    """📝 What the patient writes (always the likeliest next word)."""
    ids = tok(prompt, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=15, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).replace("\n", " ")

def surprise():
    """😌 Perplexity on TEXT: lower = healthier."""
    ids = tok(TEXT, return_tensors="pt")
    with torch.no_grad():
        return torch.exp(model(**ids, labels=ids["input_ids"]).loss).item()

room = model.model.layers[14].mlp.down_proj.weight     # floor 14, thinking room, its "voice"
healthy = room.data.clone()                             # 📸 photo of the healthy numbers

print(f"✅ healthy               | {write()!r} | surprise {surprise():.1f}")
bugs = {
    "📢 loud floor (×10)   ": 10,
    "🔄 flipped floor (×-1)": -1,
    "💀 dead floor (×0.01) ": 0.01,
}
for bug, factor in bugs.items():
    room.data = healthy * factor                        # 🔨 break it
    print(f"{bug} | {write()!r} | surprise {surprise():.1f}")
    room.data = healthy.clone()                         # 🩹 heal before the next bug
