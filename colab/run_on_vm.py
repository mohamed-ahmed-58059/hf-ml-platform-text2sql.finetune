"""Runs ON the Colab VM, invoked from your laptop via:

    colab exec -f colab/run_on_vm.py --timeout 600

It expects your repo (with a filled-in .env) already uploaded as a tarball at
/content/repo.tgz — `colab/drive.sh` does that upload for you. This script then:

  1. extracts the repo to /content/text2sql.finetune
  2. installs the training-only deps (torch is already on the T4)
  3. launches train.py in the BACKGROUND with T4-safe overrides

It returns in well under the exec timeout; the fine-tune keeps running on the VM
after this call ends. Poll progress with:

    echo "tail -n 40 /content/train.log" | colab console

Checkpoints are pushed to the HF Hub every save (FULL mode), so a free-tier
disconnect never loses work — you resume by re-running with T2S_RESUME_FROM set.
"""
import os
import subprocess
import sys
import tarfile

# ---- toggle this for your two runs -------------------------------------------
# Start with SMOKE = True (≈ minutes) to prove the whole pipeline works on the T4,
# then flip to False for the real run.
SMOKE = True
# ------------------------------------------------------------------------------

REPO_DIR = "/content/text2sql.finetune"
TARBALL = "/content/repo.tgz"
LOG = "/content/train.log"
PID = "/content/train.pid"
REQS = os.path.join(REPO_DIR, "colab", "requirements-train.txt")


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


# 1. unpack the uploaded repo (overwrites any previous copy)
if not os.path.exists(TARBALL):
    sys.exit(f"ERROR: {TARBALL} not found. Run colab/drive.sh, which uploads it first.")
os.makedirs(REPO_DIR, exist_ok=True)
with tarfile.open(TARBALL) as t:
    t.extractall(REPO_DIR)
print(f"[vm] extracted repo -> {REPO_DIR}", flush=True)

# 2. install the training deps (torch already present on the T4)
sh(f"pip install -q -r {REQS}")

# 3. launch train.py in the background with T4-safe overrides
env = dict(os.environ)
env.update({
    "T2S_RESUME_FROM": "",                  # fresh run (the local checkpoint path won't exist here)
    "T2S_LOG_TO_WANDB": "false",            # no W&B key needed on the VM
    "T2S_SMOKE": "true" if SMOKE else "false",
    "T2S_PUSH_TO_HUB": "false" if SMOKE else "true",  # FULL run streams checkpoints to the Hub
    "T2S_SAVE_STEPS": "200",
})

# train.py calls load_dotenv(), so HF_TOKEN comes from the uploaded .env in REPO_DIR.
launch = (
    f"cd {REPO_DIR} && nohup python train.py "
    f"> {LOG} 2>&1 & echo $! > {PID}"
)
mode = "SMOKE" if SMOKE else "FULL"
print(f"[vm] launching train.py ({mode}) in background...", flush=True)
subprocess.run(launch, shell=True, check=True, env=env)

pid = open(PID).read().strip()
print(f"[vm] training started (pid {pid}). Logs -> {LOG}", flush=True)
print('[vm] poll with:  echo "tail -n 40 /content/train.log" | colab console', flush=True)
