#!/usr/bin/env bash
# ▶️ Start the student server, wait until it's ready, then start training (before/after exams built in).
# Both run in their OWN sessions (setsid), so closing the SSH connection can't stop them.
set -euo pipefail
source ~/wsp.env
cd ~/prime-rl
CUDA_VISIBLE_DEVICES=0 setsid nohup uv run inference --vllm.model Qwen/Qwen3-4B-Instruct-2507 \
  --vllm.max-model-len 8192 --vllm.enable-lora True --vllm.max-lora-rank 8 --vllm.tool-call-parser hermes \
  --vllm.gpu-memory-utilization 0.35 --enable-return-sampling-mask True > ~/infer.log 2>&1 < /dev/null &
echo "⏳ waiting for the student server..."
for i in $(seq 1 60); do
  sleep 10
  if curl -s localhost:8000/v1/models | grep -q Qwen; then echo "✅ server ready"; break; fi
  if grep -q "Engine core initialization failed" ~/infer.log; then echo "❌ server failed, see ~/infer.log"; exit 1; fi
done
setsid nohup uv run rl @ ~/wsp-env/gpu/rl-1gpu.toml --run.name wsp-1gpu-final --no-monitors.prime --orchestrator.eval.skip-first-step \
  --orchestrator.model.client.admin-base-url http://localhost:8100 > ~/train.log 2>&1 < /dev/null &
echo "✅ training started. Watch: tail -f ~/prime-rl/outputs/wsp-1gpu-final/logs/attempt_1/orchestrator.log"
