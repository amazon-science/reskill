#!/bin/bash
# Ray cluster setup utilities for Slurm

cleanup_ray() {
    if [[ -n "${_RAY_CLEANED:-}" ]]; then
        return
    fi
    _RAY_CLEANED=1
    if [[ -n "${head_node:-}" ]]; then
        srun --nodes=1 --ntasks=1 -w "$head_node" ray stop >/dev/null 2>&1 || true
    fi
    if [[ ${#nodes_array[@]} -gt 1 ]]; then
        for node_i in "${nodes_array[@]:1}"; do
            srun --nodes=1 --ntasks=1 -w "$node_i" ray stop >/dev/null 2>&1 || true
        done
    fi
}

cleanup_stale_ray_sessions() {
    echo "Cleaning stale Ray sessions on all nodes..."
    for node in "${nodes_array[@]}"; do
        srun --nodes=1 --ntasks=1 -w "$node" bash -c \
            'find /tmp/ray/session_* -maxdepth 0 -mtime +4 -exec rm -rf {} + 2>/dev/null; echo "  $(hostname): $(df -h /tmp | tail -1 | awk "{print \$4}") free on /tmp"' \
            || true
    done
}

check_ray_cluster() {
    local expected_nodes=$1
    local max_attempts=5
    local attempt=1

    echo "Checking Ray cluster readiness (expecting $expected_nodes nodes)..."

    while [[ $attempt -le $max_attempts ]]; do
        echo "Attempt $attempt/$max_attempts: Checking Ray cluster status..."

        if ray_status_output=$(srun --nodes=1 --ntasks=1 -w "$head_node" ray status 2>/dev/null); then
            echo "$ray_status_output"
            alive_nodes=$(echo "$ray_status_output" | awk '
                /^Active:/ { in_active = 1; next }
                in_active && /^ 1 node_/ { count++ }
                in_active && /^[A-Za-z]/ && !/^ / { in_active = 0 }
                END { print (count ? count : 0) }
            ')
            if [[ ! "$alive_nodes" =~ ^[0-9]+$ ]]; then
                alive_nodes=0
            fi
            if [[ $alive_nodes -eq $expected_nodes ]]; then
                echo "Ray cluster is ready with $alive_nodes/$expected_nodes nodes"
                return 0
            else
                echo "Ray cluster has $alive_nodes/$expected_nodes nodes ready"
            fi
        else
            echo "Ray cluster is not responding"
        fi

        echo "Waiting 10 seconds before next check..."
        sleep 10
        ((attempt++))
    done

    echo "ERROR: Ray cluster failed to become ready after $max_attempts attempts"
    srun --nodes=1 --ntasks=1 -w "$head_node" ray status || echo "Ray status command failed"
    return 1
}
