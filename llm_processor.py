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
    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found in environment variables")
        self.async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.sync_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
    def __init__(self, model_name: str = "GLM-4.6-FP8", config: Optional[ModelConfig] = None):
        """
        初始化 GLM 处理器

        Args:
            model_name: 模型名称（在 config.yml 中定义）
            config: 直接传入的模型配置（可选）
        """
        # 从配置文件加载或使用传入的配置
        if config is None:
            llm_config = get_llm_config()
            config = llm_config.get_model(model_name)
            if config is None:
                raise ValueError(f"Model '{model_name}' not found in config.yml")

        self.config = config
        self.model_name = model_name

        # 使用 OpenAI 兼容的客户端
        self.async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.host
        )
        self.sync_client = OpenAI(
            api_key=config.api_key,
            base_url=config.host
        )
        self.default_model = config.model

        logger.info(f"GLMProcessor initialized:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  API Model: {config.model}")
        logger.info(f"  Host: {config.host}")
        logger.info(f"  Token limit: {config.token_limit}")

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
                max_tokens=min(4096, self.config.token_limit),  # 使用配置的限制
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
                max_tokens=min(4096, self.config.token_limit),
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}  # vLLM方式禁用推理
                }
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GLM sync processing error: {e}", exc_info=True)
            raise

def get_llm_processor(model: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> LLMProcessor:
    """
    获取 LLM 处理器

    Args:
        model: 模型名称或类型
            - "gemini-*": 使用 Google Gemini
            - "gpt-*" or "o1-*": 使用 OpenAI
            - "glm-*" or "GLM-*": 使用配置文件中的 GLM 模型
        api_key: API 密钥（用于 OpenAI/Gemini）
        base_url: 基础 URL（用于自定义端点）

    Returns:
        LLMProcessor 实例
    """
    model_lower = model.lower()

    if model_lower.startswith(('gemini', 'gemini-')):
        return GeminiProcessor(default_model=model)

    elif model_lower.startswith(('gpt-', 'o1-')):
        return GPTProcessor()

    elif model_lower.startswith('glm') or model.startswith('GLM'):
        # 尝试从配置文件中查找匹配的模型
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
