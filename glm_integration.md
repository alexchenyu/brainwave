# GLM-4 集成方案

## 当前代码中使用 LLM 的地方

### 1. 三个 API 端点（realtime_server.py）

| 端点 | 当前模型 | 用途 | 行号 |
|------|---------|------|------|
| `/api/v1/readability` | gpt-4o | 文本可读性优化 | 470, 474 |
| `/api/v1/ask_ai` | o1-mini | 回答问题 | 500, 502 |
| `/api/v1/correctness` | gpt-4o | 检查文本正确性 | 525, 529 |

### 2. llm_processor.py 结构

```python
LLMProcessor (抽象基类)
├── GeminiProcessor  # Google Gemini
├── GPTProcessor     # OpenAI GPT
└── GLMProcessor     # 需要新增！
```

---

## 集成步骤

### Step 1: 添加 GLMProcessor

创建新的 Processor 类支持你的 GLM-4.6-FP8。
