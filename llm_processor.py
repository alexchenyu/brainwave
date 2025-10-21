import os
from abc import ABC, abstractmethod
import google.generativeai as genai # pip install google-generativeai
from openai import OpenAI, AsyncOpenAI
from typing import AsyncGenerator, Generator, Optional
import logging
from config_loader import get_llm_config, ModelConfig

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
    GLM Processor - 支持从 config.yml 加载的自托管 GLM 模型
    兼容 OpenAI API 格式
    """
    def __init__(self, model_name: str = "GLM-4.6-FP8", config: Optional[ModelConfig] = None,
                 host: Optional[str] = None, api_key: Optional[str] = None):
        """
        初始化 GLM 处理器

        Args:
            model_name: 模型名称（在 config.yml 中定义）
            config: 直接传入的模型配置（可选）
            host: GLM API host URL（如果提供，将覆盖 config）
            api_key: GLM API key（如果提供，将覆盖 config）
        """
        # 从配置文件加载或使用传入的配置
        if config is None and (host is None or api_key is None):
            llm_config = get_llm_config()
            config = llm_config.get_model(model_name)
            if config is None:
                raise ValueError(f"Model '{model_name}' not found in config.yml")

        # 如果直接提供了 host 和 api_key，优先使用它们
        if host and api_key:
            self.host = host
            self.api_key = api_key
            self.model_name = model_name
            self.token_limit = 200000  # 默认值
        elif config:
            self.host = config.host
            self.api_key = config.api_key
            self.model_name = model_name
            self.token_limit = config.token_limit
        else:
            raise ValueError("Must provide either config or both host and api_key")

        # 使用 OpenAI 兼容的客户端
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.host
        )
        self.sync_client = OpenAI(
            api_key=self.api_key,
            base_url=self.host
        )
        self.default_model = model_name

        logger.info(f"GLMProcessor initialized:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Host: {self.host}")
        logger.info(f"  Token limit: {self.token_limit}")

    async def process_text(self, text: str, prompt: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        异步流式处理文本

        Args:
            text: 输入文本
            prompt: 提示词
            model: 模型名称（可选，使用配置中的模型）

        Yields:
            生成的文本片段
        """
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Processing with GLM model: {model_name}")
        logger.debug(f"Prompt length: {len(all_prompt)} chars")

        try:
            response = await self.async_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": all_prompt}
                ],
                stream=True,
                temperature=0.7,
                max_tokens=min(4096, self.token_limit),  # 使用配置的限制
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}  # vLLM方式禁用推理
                }
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"GLM processing error: {e}", exc_info=True)
            raise

    def process_text_sync(self, text: str, prompt: str, model: Optional[str] = None) -> str:
        """
        同步处理文本

        Args:
            text: 输入文本
            prompt: 提示词
            model: 模型名称（可选）

        Returns:
            生成的完整文本
        """
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Processing (sync) with GLM model: {model_name}")
        logger.debug(f"Prompt length: {len(all_prompt)} chars")

        try:
            response = self.sync_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": all_prompt}
                ],
                temperature=0.7,
                max_tokens=min(4096, self.token_limit),
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}  # vLLM方式禁用推理
                }
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GLM sync processing error: {e}", exc_info=True)
            raise

def get_llm_processor(model: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
                     glm_host: Optional[str] = None, glm_api_key: Optional[str] = None) -> LLMProcessor:
    """
    获取 LLM 处理器

    Args:
        model: 模型名称或类型
            - "gemini-*": 使用 Google Gemini
            - "gpt-*" or "o1-*": 使用 OpenAI
            - "glm-*" or "GLM-*": 使用配置文件中的 GLM 模型
        api_key: API 密钥（用于 OpenAI/Gemini）
        base_url: 基础 URL（用于自定义端点）
        glm_host: GLM API host URL（用于自定义 GLM 端点）
        glm_api_key: GLM API key（用于自定义 GLM 认证）

    Returns:
        LLMProcessor 实例
    """
    model_lower = model.lower()

    if model_lower.startswith(('gemini', 'gemini-')):
        return GeminiProcessor(default_model=model)

    elif model_lower.startswith(('gpt-', 'o1-')):
        return GPTProcessor(api_key=api_key)

    elif model_lower.startswith('glm') or model.startswith('GLM'):
        # 如果提供了自定义的 GLM host 和 api_key，使用它们
        if glm_host and glm_api_key:
            logger.info(f"Using custom GLM endpoint: {glm_host}")
            return GLMProcessor(model_name=model, host=glm_host, api_key=glm_api_key)

        # 否则尝试从配置文件中查找匹配的模型
        llm_config = get_llm_config()

        # 精确匹配
        if llm_config.has_model(model):
            return GLMProcessor(model_name=model)

        # 模糊匹配（如果传入 "glm-4"，匹配 "GLM-4.6-FP8"）
        for config_model_name in llm_config.list_models():
            if model_lower in config_model_name.lower():
                logger.info(f"Matched '{model}' to config model '{config_model_name}'")
                return GLMProcessor(model_name=config_model_name)

        # 如果没找到，使用默认 GLM 模型
        logger.warning(f"Model '{model}' not found in config, using default")
        return GLMProcessor()

    else:
        raise ValueError(f"Unsupported model type: {model}. Supported: gemini-*, gpt-*, o1-*, glm-*, GLM-*")
