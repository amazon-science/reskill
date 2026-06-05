"""Action vocabulary for the registered search environment."""

ACTION_VOCABULARY = """
The agent operates in a multi-hop QA environment. Actions use XML tags:

| Action Template | Example | Description |
|----------------|---------|-------------|
| `<search>query</search>` | <search>capital of France</search> | Search for information |
| `<answer>text</answer>` | <answer>Paris</answer> | Submit final answer |

Trigger patterns should match these tags. For example:
- `<search>` or `<search>.*</search>` matches any search action
- `<answer>` matches the answer action
- `</search>` matches after a search completes (useful for post-search guidance)
"""
