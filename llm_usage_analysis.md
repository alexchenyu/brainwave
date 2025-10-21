# 代码中 LLM 使用情况分析

## 📊 概览

代码分为两部分：
1. **语音处理** - 使用 OpenAI Realtime API（保持不变）
2. **文本处理** - 使用文本 LLM API（需要替换为 GLM）

---

## 🎤 语音处理部分（保持不变）

### 文件：`openai_realtime_client.py`
**用途**：语音转文字的实时处理
**使用模型**：OpenAI Realtime API（`gpt-4o-realtime-preview`）

```python
# openai_realtime_client.py:18
self.base_url = "wss://api.openai.com/v1/realtime"
```

**流程**：
```
用户语音 → OpenAI Realtime WebSocket → 实时转录文字
```

✅ **这部分不需要改动**

---

## 📝 文本处理部分（需要替换为 GLM）

### 1. 核心 LLM 处理器

#### 文件：`llm_processor.py`

**当前实现**：
- `GPTProcessor` - 使用 OpenAI GPT 模型
- `GeminiProcessor` - 使用 Google Gemini 模型

**使用的 OpenAI 模型**：
```python
# llm_processor.py:59
self.default_model = "gpt-4"
```

**API 调用方式**：
```python
# llm_processor.py:66-72 (异步流式)
response = await self.async_client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": all_prompt}],
    stream=True
)

# llm_processor.py:82-87 (同步)
response = self.sync_client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": all_prompt}]
)
```

---

### 2. 三个 API 端点

#### 文件：`realtime_server.py`

所有端点都在这个文件中，使用不同的模型：

| 端点 | 当前模型 | 调用方式 | 代码位置 | 用途 |
|------|---------|---------|---------|------|
| `/api/v1/readability` | **gpt-4o** | 异步流式 | 行 470, 474 | 优化文本可读性 |
| `/api/v1/ask_ai` | **o1-mini** | 同步 | 行 500, 502 | 回答问题 |
| `/api/v1/correctness` | **gpt-4o** | 异步流式 | 行 525, 529 | 检查文本正确性 |

---

## 🔍 详细代码分析

### 端点 1: `/api/v1/readability` - 文本可读性优化

**位置**：`realtime_server.py:455-481`

```python
@app.post("/api/v1/readability")
async def enhance_readability(request: ReadabilityRequest):
    prompt = PROMPTS.get('readability-enhance')

    # 获取 API Key
    api_key = request.api_key or OPENAI_API_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required.")

    # 创建处理器 - 使用 gpt-4o
    processor = get_llm_processor("gpt-4o", api_key=api_key)

    # 异步流式处理
    async def text_generator():
        async for part in processor.process_text(request.text, prompt, model="gpt-4o"):
            yield part

    return StreamingResponse(text_generator(), media_type="text/plain")
```

**特点**：
- ✅ 流式输出（逐字返回）
- ✅ 异步处理
- 使用 OpenAI gpt-4o 模型

---

### 端点 2: `/api/v1/ask_ai` - AI 问答

**位置**：`realtime_server.py:483-506`

```python
@app.post("/api/v1/ask_ai")
def ask_ai(request: AskAIRequest):
    prompt = PROMPTS.get('ask-ai')

    # 获取 API Key
    api_key = request.api_key or OPENAI_API_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required.")

    # 创建处理器 - 使用 o1-mini
    processor = get_llm_processor("o1-mini", api_key=api_key)

    # 同步处理
    answer = processor.process_text_sync(request.text, prompt, model="o1-mini")

    return AskAIResponse(answer=answer)
```

**特点**：
- ❌ 非流式（等待完整响应）
- ❌ 同步处理
- 使用 OpenAI o1-mini 模型

---

### 端点 3: `/api/v1/correctness` - 文本正确性检查

**位置**：`realtime_server.py:508-536`

```python
@app.post("/api/v1/correctness")
async def check_correctness(request: CorrectnessRequest):
    prompt = PROMPTS.get('correctness-check')

    # 获取 API Key
    api_key = request.api_key or OPENAI_API_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required.")

    # 创建处理器 - 使用 gpt-4o
    processor = get_llm_processor("gpt-4o", api_key=api_key)

    # 异步流式处理
    async def text_generator():
        async for part in processor.process_text(request.text, prompt, model="gpt-4o"):
            yield part

    return StreamingResponse(text_generator(), media_type="text/plain")
```

**特点**：
- ✅ 流式输出
- ✅ 异步处理
- 使用 OpenAI gpt-4o 模型

---

## 🔧 关键发现

### 1. 使用的 OpenAI 模型

| 模型 | 使用位置 | 数量 |
|------|---------|-----|
| **gpt-4o** | readability, correctness | 2 个端点 |
| **o1-mini** | ask_ai | 1 个端点 |
| gpt-4 | 默认（未使用） | 0 |

### 2. API 调用模式

| 模式 | 端点 | 优点 | 缺点 |
|------|------|------|------|
| **异步流式** | readability, correctness | 实时反馈，用户体验好 | 实现复杂 |
| **同步** | ask_ai | 实现简单 | 等待时间长 |

### 3. OpenAI Client 使用方式

```python
# llm_processor.py:57-58
self.async_client = AsyncOpenAI(api_key=self.api_key)
self.sync_client = OpenAI(api_key=self.api_key)
```

**好消息**：OpenAI SDK 支持 `base_url` 参数！

```python
# 可以这样改为 GLM
AsyncOpenAI(api_key=api_key, base_url="http://your-glm-server:8000/v1")
OpenAI(api_key=api_key, base_url="http://your-glm-server:8000/v1")
```

---

## 🎯 替换为 GLM 的方案

### 方案 1：最小改动（推荐）

在 `llm_processor.py` 中修改 `GPTProcessor`，添加 `base_url` 参数：

```python
class GPTProcessor(LLMProcessor):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")  # 新增

        # 使用自定义 base_url
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url  # 指向你的 GLM 服务
        )
        self.sync_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
```

**环境变量配置**：
```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"  # 你的 GLM 服务地址
export OPENAI_API_KEY="dummy-key"  # GLM 可能不需要真实 key
```

**优点**：
- ✅ 改动最小
- ✅ 兼容现有代码
- ✅ 可以随时切换回 OpenAI

---

### 方案 2：新增 GLMProcessor（更清晰）

创建独立的 `GLMProcessor` 类：

```python
class GLMProcessor(LLMProcessor):
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.base_url = base_url
        self.api_key = "dummy-key"

        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.sync_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.default_model = "glm-4"
```

**修改 `get_llm_processor`**：
```python
def get_llm_processor(model: str, api_key: Optional[str] = None) -> LLMProcessor:
    if model.startswith('glm'):
        return GLMProcessor(base_url=os.getenv("GLM_BASE_URL"))
    elif model.startswith(('gpt-', 'o1-')):
        return GPTProcessor(api_key=api_key)
    # ...
```

**修改端点**：
```python
# realtime_server.py
processor = get_llm_processor("glm-4")  # 改为 glm-4
```

**优点**：
- ✅ 代码结构清晰
- ✅ 容易维护
- ✅ 支持多种模型

---

## 📋 需要修改的文件清单

### 核心文件（必须改）

1. **`llm_processor.py`** - 添加 GLM 支持
   - 新增 `GLMProcessor` 类 或
   - 修改 `GPTProcessor` 添加 `base_url`

2. **`realtime_server.py`** - 修改模型调用
   - 行 58: 默认处理器
   - 行 470: readability 端点
   - 行 500: ask_ai 端点
   - 行 525: correctness 端点

### 不需要改的文件

- ✅ `openai_realtime_client.py` - 语音处理，保持不变
- ✅ `prompts.py` - 提示词，不需要改
- ✅ `static/main.js` - 前端代码，不需要改

---

## 🚀 GLM 部署要求

你的 GLM-4.6-FP8 需要提供 OpenAI 兼容的 API：

### 必须支持的端点

```
POST http://localhost:8000/v1/chat/completions
```

### 请求格式

```json
{
  "model": "glm-4",
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "stream": true,  // 流式响应
  "temperature": 0.7,
  "max_tokens": 2048
}
```

### 响应格式（流式）

```json
data: {"choices":[{"delta":{"content":"你"}}]}
data: {"choices":[{"delta":{"content":"好"}}]}
data: [DONE]
```

---

## 💡 推荐的替换步骤

1. **确认 GLM 服务正常运行**
   - 测试 API 端点
   - 确认兼容 OpenAI 格式

2. **修改 `llm_processor.py`**
   - 添加 `GLMProcessor` 类
   - 更新 `get_llm_processor` 函数

3. **修改 `realtime_server.py`**
   - 将 `"gpt-4o"` 改为 `"glm-4"`
   - 将 `"o1-mini"` 改为 `"glm-4"`

4. **配置环境变量**
   ```bash
   export GLM_BASE_URL="http://localhost:8000/v1"
   export USE_GLM="true"
   ```

5. **测试三个端点**
   - `/api/v1/readability`
   - `/api/v1/ask_ai`
   - `/api/v1/correctness`

---

## 📊 总结

### 语音处理（不改）
- ✅ `openai_realtime_client.py` - OpenAI Realtime API
- ✅ WebSocket 连接 - `wss://api.openai.com/v1/realtime`

### 文本处理（需要改为 GLM）
- 📝 `llm_processor.py` - 添加 GLM 支持
- 📝 `realtime_server.py` - 3 个端点改用 GLM
  - `/api/v1/readability` (gpt-4o → glm-4)
  - `/api/v1/ask_ai` (o1-mini → glm-4)
  - `/api/v1/correctness` (gpt-4o → glm-4)

### 改动量
- 核心修改：2 个文件
- 代码行数：约 50-100 行
- 影响范围：仅文本处理，语音不受影响

---

要我开始修改代码吗？还是你想先看看其他什么信息？
