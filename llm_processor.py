import os
from abc import ABC, abstractmethod
import google.generativeai as genai
from openai import OpenAI, AsyncOpenAI
from typing import AsyncGenerator, Generator, Optional
import logging
import httpx

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LLMProcessor(ABC):
    @abstractmethod
    async def process_text(self, text: str, prompt: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        pass
    
    @abstractmethod
    def process_text_sync(self, text: str, prompt: str, model: Optional[str] = None) -> str:
        pass

class GeminiProcessor(LLMProcessor):
    def __init__(self, default_model: str = 'gemini-1.5-pro'):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=api_key)
        self.default_model = default_model

    async def process_text(self, text: str, prompt: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Using model: {model_name} for processing")
        logger.info(f"Prompt: {all_prompt}")
        genai_model = genai.GenerativeModel(model_name)
        response = await genai_model.generate_content_async(
            all_prompt,
            stream=True
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    def process_text_sync(self, text: str, prompt: str, model: Optional[str] = None) -> str:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Using model: {model_name} for sync processing")
        logger.info(f"Prompt: {all_prompt}")
        genai_model = genai.GenerativeModel(model_name)
        response = genai_model.generate_content(all_prompt)
        return response.text

class GPTProcessor(LLMProcessor):
    def __init__(self, api_key: Optional[str] = None):
        # Use provided API key or fall back to environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please provide an API key.")
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.sync_client = OpenAI(api_key=self.api_key)
        self.default_model = "gpt-4"

    async def process_text(self, text: str, prompt: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Using model: {model_name} for processing")
        logger.info(f"Prompt: {all_prompt}")
        response = await self.async_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": all_prompt}
            ],
            stream=True
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def process_text_sync(self, text: str, prompt: str, model: Optional[str] = None) -> str:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Using model: {model_name} for sync processing")
        logger.info(f"Prompt: {all_prompt}")
        response = self.sync_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": all_prompt}
            ]
        )
        return response.choices[0].message.content

class GLMProcessor(LLMProcessor):
    """
    GLM-4 Processor for self-hosted GLM models
    Compatible with OpenAI API format
    """
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        # 默认使用本地部署的 GLM-4
        self.base_url = base_url or os.getenv("GLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("GLM_API_KEY", "dummy-key")  # vLLM 不需要真实 key

        # 使用 OpenAI 兼容的客户端
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.sync_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.default_model = "glm-4"
        logger.info(f"GLMProcessor initialized with base_url: {self.base_url}")

    async def process_text(self, text: str, prompt: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Using GLM model: {model_name} for processing")

        try:
            response = await self.async_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": all_prompt}
                ],
                stream=True,
                temperature=0.7,
                max_tokens=2048
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"GLM processing error: {e}")
            raise

    def process_text_sync(self, text: str, prompt: str, model: Optional[str] = None) -> str:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Using GLM model: {model_name} for sync processing")

        try:
            response = self.sync_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": all_prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GLM sync processing error: {e}")
            raise

def get_llm_processor(model: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> LLMProcessor:
    model = model.lower()
    if model.startswith(('gemini', 'gemini-')):
        return GeminiProcessor(default_model=model)
    elif model.startswith(('gpt-', 'o1-')):
        return GPTProcessor(api_key=api_key)
    elif model.startswith(('glm', 'glm-')):
        return GLMProcessor(base_url=base_url, api_key=api_key)
    else:
        raise ValueError(f"Unsupported model type: {model}")
