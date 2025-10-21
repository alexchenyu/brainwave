"""
配置加载器 - 从 config.yml 读取 LLM 配置
"""
import yaml
import os
from typing import Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelConfig:
    """单个模型的配置"""
    def __init__(self, name: str, config: dict):
        self.name = name
        self.host = config.get('host')
        self.api_key = config.get('api_key')
        self.model = config.get('model', name)
        self.token_limit = config.get('token_limit', 128000)

    def __repr__(self):
        return f"ModelConfig(name={self.name}, model={self.model}, host={self.host})"


class LLMConfig:
    """LLM 配置管理器"""

    def __init__(self, config_path: str = "config.yml"):
        self.config_path = Path(config_path)
        self.models: Dict[str, ModelConfig] = {}
        self.default_model: Optional[str] = None
        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config or 'llm_models' not in config:
                logger.warning("No llm_models section in config")
                return

            # 加载所有模型配置
            for model_name, model_config in config['llm_models'].items():
                self.models[model_name] = ModelConfig(model_name, model_config)
                logger.info(f"Loaded model config: {model_name}")

            # 设置默认模型（第一个）
            if self.models:
                self.default_model = list(self.models.keys())[0]
                logger.info(f"Default model: {self.default_model}")

        except Exception as e:
            logger.error(f"Error loading config: {e}")

    def get_model(self, name: Optional[str] = None) -> Optional[ModelConfig]:
        """
        获取模型配置

        Args:
            name: 模型名称，如果为 None 则返回默认模型

        Returns:
            ModelConfig 或 None
        """
        if name is None:
            name = self.default_model

        return self.models.get(name)

    def list_models(self) -> list[str]:
        """列出所有可用模型"""
        return list(self.models.keys())

    def has_model(self, name: str) -> bool:
        """检查模型是否存在"""
        return name in self.models


# 全局配置实例
_llm_config: Optional[LLMConfig] = None


def get_llm_config(config_path: str = "config.yml") -> LLMConfig:
    """
    获取全局 LLM 配置实例（单例模式）

    Args:
        config_path: 配置文件路径

    Returns:
        LLMConfig 实例
    """
    global _llm_config
    if _llm_config is None:
        _llm_config = LLMConfig(config_path)
    return _llm_config


# 示例用法
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = get_llm_config()

    print("Available models:")
    for model_name in config.list_models():
        model = config.get_model(model_name)
        print(f"  - {model}")

    print(f"\nDefault model: {config.default_model}")

    # 获取 GLM-4.6-FP8 配置
    glm_config = config.get_model("GLM-4.6-FP8")
    if glm_config:
        print(f"\nGLM-4.6-FP8 config:")
        print(f"  Host: {glm_config.host}")
        print(f"  Model: {glm_config.model}")
        print(f"  Token limit: {glm_config.token_limit}")
