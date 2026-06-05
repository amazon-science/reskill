#!/bin/bash
#SBATCH --job-name=reskill-scienceworld
#SBATCH --nodes=2
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

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
port=6379

trap cleanup_ray EXIT
cleanup_stale_ray_sessions

echo "Starting Ray head on $head_node ($head_node_ip:$port)"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus 96 --num-gpus 8 --include-dashboard=false &
sleep 15

for node_i in "${nodes_array[@]:1}"; do
    echo "Starting Ray worker on $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address="$head_node_ip:$port" \
        --num-cpus 96 --num-gpus 8 --include-dashboard=false &
    sleep 5
done
sleep 10

check_ray_cluster ${#nodes_array[@]}

export WANDB_NAME="reskill_scienceworld_$(date +%m%d)"

cd "$REPO_DIR"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    python scripts/train.py \
    --config-name scienceworld \
    +ray_init.address="$head_node_ip:$port" \
    data.train_files="$REPO_DIR/data/scienceworld/train.parquet" \
    data.val_files="$REPO_DIR/data/scienceworld/val.parquet" \
    trainer.default_local_dir="$REPO_DIR/checkpoints/$WANDB_NAME" \
    trainer.experiment_name="$WANDB_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2
