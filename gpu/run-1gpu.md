# 🖥️ Way 2 on ONE GPU: paste these, one block at a time

Machine: Prime Intellect On-Demand → 1× A100 80GB (spot ~$1.26/h), image Ubuntu 22 + CUDA 12.
⏱️ Budget ~3–4 h ≈ $4–5. **STOP THE MACHINE THE MOMENT WE'RE DONE.**

## 0. On your Mac: send the code to the machine (replace IP / port from the Prime page)
    scp -P <PORT> ~/Desktop/wsp-env-code.zip root@<IP>:~/
    ssh -p <PORT> root@<IP>

## 1. Setup (~15–30 min): installs uv, prime-rl, our environment
    unzip -oq ~/wsp-env-code.zip -d ~/wsp-env && bash ~/wsp-env/gpu/setup.sh

## 2. Check prime-rl's exact flag names (1 min, so nothing is guessed)
    cd ~/prime-rl && uv run inference --help | grep -i -E "gpu.memory|enable.lora|lora.rank|tool.call|sampling.mask|port" | head -20
    uv run rl --help | grep -i -E "num.infer|num.train|base.url" | head

## 3. Start the student server on GPU 0, using only ~35% of the memory (leave a 2nd terminal open)
    tmux new -s infer
    source ~/wsp.env && cd ~/prime-rl
    CUDA_VISIBLE_DEVICES=0 uv run inference --vllm.model Qwen/Qwen3-4B-Instruct-2507 \
      --vllm.enable-lora --vllm.max-lora-rank 8 --vllm.tool-call-parser hermes \
      --vllm.gpu-memory-utilization 0.35 --inference.enable-return-sampling-mask true
    # wait for "ready" / "Uvicorn running", then detach: Ctrl+B then D

## 4. Train (+ before/after exams built in) on the same GPU (~2–3 h)
    tmux new -s train
    source ~/wsp.env && cd ~/prime-rl
    uv run rl @ ~/wsp-env/gpu/rl-1gpu.toml --run.name wsp-1gpu 2>&1 | tee ~/train.log
    # detach: Ctrl+B then D · come back: tmux attach -t train

## 5. Read the results
    grep -i -E "eval|reward|step" ~/train.log | tail -40
    ls ~/prime-rl/outputs

## 6. Save everything to your Mac (run on your Mac)
    scp -P <PORT> -r root@<IP>:~/prime-rl/outputs ~/Desktop/wsp-env/gpu-results
    scp -P <PORT> root@<IP>:~/train.log ~/Desktop/wsp-env/gpu-results/

## 7. ⏹️ STOP the machine on the Prime page (Terminate). Billing stops only then.

### If step 3/4 fails
- Flag name errors → use the names printed in step 2.
- Out of memory → lower `--vllm.gpu-memory-utilization` to 0.3, or `initial_inflight` to 4 in rl-1gpu.toml.
- "num_infer_gpus must be ≥ 1" → single-GPU isn't allowed by this prime-rl version: stop the machine, wait for a 2-GPU offer (use rl.toml).
