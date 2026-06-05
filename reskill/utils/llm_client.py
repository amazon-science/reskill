"""LLM client abstraction for the ReSkill pipeline.

Supports two backends:
  - 'bedrock': AWS Bedrock Claude
  - 'openai':  OpenAI-compatible API (vLLM, local servers)

Usage:
    from reskill.utils.llm_client import create_llm_client
    client = create_llm_client({'backend': 'openai', 'model': 'Qwen/Qwen2.5-7B-Instruct',
                                'base_url': 'http://localhost:8000/v1'})
    text = client.generate("What is 2+2?", system_prompt="You are helpful.")
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_json(text: str) -> Optional[dict]:
    """Extract JSON object from LLM response text, handling markdown fences and thinking tags."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


class LLMClient(ABC):
    """Abstract LLM backend for the ReSkill pipeline stages."""

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """Send prompt to LLM, return text response or None on failure."""
        ...

    def generate_batch(
        self,
        prompts: List[Tuple[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
        max_workers: int = 16,
    ) -> List[Optional[str]]:
        """Parallel generation via ThreadPoolExecutor.

        Args:
            prompts: List of (system_prompt, user_prompt) pairs.
            max_tokens: Max tokens per response.
            temperature: Sampling temperature.
            max_workers: Concurrency limit.

        Returns:
            List of response strings (None for failed calls), same order as prompts.
        """
        results: List[Optional[str]] = [None] * len(prompts)

        def _call(idx: int, sys_prompt: str, usr_prompt: str) -> Tuple[int, Optional[str]]:
            return idx, self.generate(usr_prompt, system_prompt=sys_prompt,
                                      max_tokens=max_tokens, temperature=temperature)

        with ThreadPoolExecutor(max_workers=min(max_workers, len(prompts))) as executor:
            futures = {
                executor.submit(_call, i, sp, up): i
                for i, (sp, up) in enumerate(prompts)
            }
            for future in as_completed(futures):
                try:
                    idx, resp = future.result()
                    results[idx] = resp
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"[LLMClient] Batch call {idx} failed: {e}")
                    results[idx] = None

        n_ok = sum(1 for r in results if r is not None)
        logger.info(f"[LLMClient] Batch complete: {n_ok}/{len(prompts)} succeeded")
        return results


class BedrockClient(LLMClient):
    """AWS Bedrock Claude backend."""

    def __init__(
        self,
        model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region: str = "us-west-2",
        max_retries: int = 10,
    ):
        import boto3
        from botocore.config import Config

        self.model = model
        self.region = region
        self.max_retries = max_retries
        self.client = boto3.Session().client(
            "bedrock-runtime",
            region_name=self.region,
            config=Config(read_timeout=300, retries={"max_attempts": 3}),
        )
        logger.info(f"[BedrockClient] Initialized: model={model}, region={region}")

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Optional[str]:
        from botocore.exceptions import ClientError

        payload: Dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        for attempt in range(self.max_retries):
            try:
                resp = self.client.invoke_model(
                    body=json.dumps(payload),
                    modelId=self.model,
                )
                body = json.loads(resp["body"].read())
                return body["content"][0]["text"]
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") == "ThrottlingException":
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        f"[BedrockClient] Throttled, retry in {wait}s "
                        f"(attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait)
                else:
                    logger.error(f"[BedrockClient] Bedrock error: {e}")
                    return None
            except Exception as e:
                logger.error(f"[BedrockClient] Unexpected error: {e}")
                return None

        logger.error(f"[BedrockClient] Exhausted {self.max_retries} retries")
        return None


class OpenAIClient(LLMClient):
    """OpenAI-compatible backend (vLLM, local servers)."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        from openai import OpenAI

        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        logger.info(f"[OpenAIClient] Initialized: model={model}, base_url={base_url}")

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Optional[str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content
            except Exception as e:
                wait = min(2 ** attempt, 15)
                logger.warning(
                    f"[OpenAIClient] Error (attempt {attempt + 1}/{self.max_retries}): "
                    f"{e}, retry in {wait}s")
                time.sleep(wait)

        logger.error(f"[OpenAIClient] Exhausted {self.max_retries} retries")
        return None


class AnthropicClient(LLMClient):
    """Anthropic API backend (direct claude.ai API key, no AWS needed)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        api_key: Optional[str] = None,
        max_retries: int = 3,
    ):
        import anthropic as _anthropic

        self.model = model
        self.max_retries = max_retries
        self.client = _anthropic.Anthropic(api_key=api_key)  # falls back to ANTHROPIC_API_KEY
        logger.info(f"[AnthropicClient] Initialized: model={model}")

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> Optional[str]:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(**kwargs)
                return resp.content[0].text
            except Exception as e:
                wait = min(2 ** attempt, 30)
                logger.warning(
                    f"[AnthropicClient] Error (attempt {attempt + 1}/{self.max_retries}): "
                    f"{e}, retry in {wait}s")
                time.sleep(wait)

        logger.error(f"[AnthropicClient] Exhausted {self.max_retries} retries")
        return None

    def generate_with_usage(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> tuple:
        """Like generate() but returns (text, input_tokens, output_tokens)."""
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(**kwargs)
                text = resp.content[0].text
                return text, resp.usage.input_tokens, resp.usage.output_tokens
            except Exception as e:
                wait = min(2 ** attempt, 30)
                logger.warning(
                    f"[AnthropicClient] Error (attempt {attempt + 1}/{self.max_retries}): "
                    f"{e}, retry in {wait}s")
                time.sleep(wait)

        logger.error(f"[AnthropicClient] Exhausted {self.max_retries} retries")
        return None, 0, 0


def create_llm_client(config: dict) -> LLMClient:
    """Factory: create LLM client from config dict.

    Config keys:
        backend: 'bedrock' | 'openai' | 'anthropic' (default: 'bedrock')
        model: Model name/ID
        region: AWS region (bedrock only, default: 'us-west-2')
        base_url: API base URL (openai only, default: 'http://localhost:8000/v1')
        api_key: API key (openai and anthropic; anthropic falls back to ANTHROPIC_API_KEY)
    """
    backend = config.get('backend', 'bedrock')

    if backend == 'bedrock':
        return BedrockClient(
            model=config.get('model', 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
            region=config.get('region', 'us-west-2'),
        )
    elif backend == 'openai':
        return OpenAIClient(
            model=config.get('model', 'Qwen/Qwen2.5-7B-Instruct'),
            base_url=config.get('base_url', 'http://localhost:8000/v1'),
            api_key=config.get('api_key', 'dummy'),
        )
    elif backend == 'anthropic':
        return AnthropicClient(
            model=config.get('model', 'claude-sonnet-4-5-20250929'),
            api_key=config.get('api_key'),
        )
    else:
        raise ValueError(f"Unknown LLM backend: {backend!r}. Must be 'bedrock', 'openai', or 'anthropic'.")
