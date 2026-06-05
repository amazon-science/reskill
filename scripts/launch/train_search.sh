#!/bin/bash
#SBATCH --job-name=reskill-search
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/ray_on_slurm_utils.sh"

# --- Ray cluster setup (single node) ---
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
port=6379

trap cleanup_ray EXIT

echo "Starting Ray head on $head_node ($head_node_ip:$port)"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus 96 --num-gpus 8 --include-dashboard=false &
sleep 15

check_ray_cluster 1

# --- Training ---
export WANDB_NAME="reskill_search_$(date +%m%d)"

cd "$REPO_DIR"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    python scripts/train.py \
    --config-name search \
    +ray_init.address="$head_node_ip:$port" \
    data.train_files="$REPO_DIR/data/search/train.parquet" \
    data.val_files="$REPO_DIR/data/search/val.parquet" \
    trainer.default_local_dir="$REPO_DIR/checkpoints/$WANDB_NAME" \
    trainer.experiment_name="$WANDB_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
