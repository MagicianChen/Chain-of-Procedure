import asyncio
import base64
import os

from openai import APIStatusError, AsyncOpenAI, AuthenticationError, BadRequestError


class LLMClient:
    def __init__(
        self,
        model,
        api_key=None,
        base_url=None,
        retries=6,
        max_tokens=None,
        reasoning_effort=None,
        enable_thinking=None,
        timeout=1800,
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY before running.")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self.model = model
        self.retries = retries
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    async def complete(self, prompt, image_path=None, temperature=0):
        content = []
        if image_path:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"},
                }
            )
        content.append({"type": "text", "text": prompt})

        extra = {}
        if self.enable_thinking is not None:
            extra["enable_thinking"] = self.enable_thinking
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if extra:
            kwargs["extra_body"] = extra

        delay = 2
        last_error = None
        for _ in range(self.retries):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                text = message.content or ""
                if not text.strip():
                    text = getattr(message, "reasoning_content", None) or ""
                return text
            except (AuthenticationError, BadRequestError):
                raise
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code not in (408, 409, 429, 500, 502, 503, 504):
                    raise
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
        raise last_error


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
