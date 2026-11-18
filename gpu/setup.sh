#!/usr/bin/env bash
# 🖥️ One-command setup on a rented GPU machine (Ubuntu 22 + CUDA 12+), for Way 2 (multi-turn RL with prime-rl).
# Includes every fix learned on the first machine (2026-11-17). Usage: bash ~/wsp-env/gpu/setup.sh
set -euo pipefail

# 1. system packages (Python headers are needed by Triton's GPU compiler; unzip for our code)
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq python3.12-dev build-essential unzip git >/dev/null 2>&1 || true

# 2. uv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3. our environment (the uploaded zip)
mkdir -p ~/wsp-env && unzip -oq ~/wsp-env-code.zip -d ~/wsp-env

# 4. prime-rl, with submodules over HTTPS (the machine has no GitHub SSH key)
git config --global url."https://github.com/".insteadOf "git@github.com:"
[ -d ~/prime-rl ] || git clone --depth 1 https://github.com/PrimeIntellect-ai/prime-rl.git ~/prime-rl
cd ~/prime-rl
for m in deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config; do
  GIT_TERMINAL_PROMPT=0 git submodule update --init --force -- "$m"
  if [ -z "$(ls -A "$m" 2>/dev/null)" ]; then          # an empty folder = a broken download → clone again
    name=$(basename "$m"); rm -rf "$m" ".git/modules/$name"
    GIT_TERMINAL_PROMPT=0 git submodule update --init --force -- "$m"
  fi
  echo "submodule $m: $(ls "$m" | wc -l) files"
done
uv sync --all-extras
uv pip install -e ~/wsp-env

# 5. settings: patients on GPU 0, half precision
cat > ~/wsp.env <<'ENV'
export PATH="$HOME/.local/bin:$PATH"
export WSP_PATIENT_DEVICE=cuda:0
export WSP_PATIENT_DTYPE=float16
ENV
source ~/wsp.env

# 6. checks
uv run python -c "import wsp_llm.taskset as t; print('✅ wsp-llm importable,', len(t.PRACTICE), 'practice patients')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "✅ setup done. Next: bash ~/wsp-env/gpu/start.sh"
