# Requirements

The production veRL pin is recorded in `REQUIRED_VERL.txt`.

## ALFWorld


For the validated ALFWorld + vLLM CUDA 12.4 stack:

```bash
pip install -e ".[alfworld,vllm]" \
  -c requirements/constraints-alfworld-vllm-cu124.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

Set `ALFWORLD_DATA` to the downloaded ALFWorld data directory before running
ALFWorld experiments.

## Search

For the validated Search + vLLM CUDA 12.4 stack:

```bash
pip install -e ".[search,vllm]" \
  -c requirements/constraints-search-vllm-cu124.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124
```
