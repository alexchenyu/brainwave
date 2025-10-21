# H100 集群架构方案

## 🎯 核心优势

有了多个 H100，你可以：
1. **完全自托管** - 不依赖任何第三方 API
2. **零边际成本** - 处理再多用户也不增加成本
3. **极致性能** - 实时转录 + 优化
4. **数据隐私** - 所有数据在自己服务器
5. **可定制化** - 微调模型适配特定场景

---

## 🏗️ 推荐架构

```
┌─────────────────────────────────────────────────────┐
│                   用户请求                            │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│              负载均衡器 (Nginx/Traefik)               │
└─────────────────────────────────────────────────────┘
                      ↓
        ┌─────────────┴─────────────┐
        ↓                           ↓
┌──────────────┐            ┌──────────────┐
│ API Gateway  │            │ WebSocket    │
│ (FastAPI)    │            │ Server       │
└──────────────┘            └──────────────┘
        ↓                           ↓
┌─────────────────────────────────────────────────────┐
│                 任务队列 (Redis/RabbitMQ)             │
└─────────────────────────────────────────────────────┘
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
┌──────────┐   ┌──────────┐   ┌──────────┐
│ H100 #1  │   │ H100 #2  │   │ H100 #3  │
│ Whisper  │   │ Whisper  │   │ Whisper  │
│ large-v3 │   │ large-v3 │   │ large-v3 │
└──────────┘   └──────────┘   └──────────┘
     ↓              ↓              ↓
┌──────────┐   ┌──────────┐   ┌──────────┐
│ GPT 推理 │   │ GPT 推理 │   │ GPT 推理 │
│ (vLLM)   │   │ (vLLM)   │   │ (vLLM)   │
└──────────┘   └──────────┘   └──────────┘
```

---

## 🔧 技术栈选择

### 1. Whisper 部署：faster-whisper + TensorRT

**为什么用 TensorRT？**
- 比普通 PyTorch 快 2-3 倍
- H100 上可达 0.02x 实时（5 分钟音频 → 6 秒）
- 支持 FP8 精度（H100 独有）

```bash
# 安装 TensorRT 优化的 Whisper
pip install faster-whisper tensorrt

# 或使用 WhisperX（更快）
pip install whisperx
```

**性能对比（H100）**:
| 实现 | 5 分钟音频处理时间 |
|------|-------------------|
| OpenAI Whisper | 60 秒 |
| faster-whisper | 15 秒 |
| WhisperX + TensorRT | **6 秒** |
| faster-whisper + batch | **3 秒** |

---

### 2. LLM 推理：自托管开源模型

既然有 H100，完全可以自己跑 LLM！

#### 方案 A：Qwen2.5（推荐）

**为什么选 Qwen？**
- 中文效果最好
- 性能接近 GPT-4
- 完全免费开源
- 7B/14B/32B/72B 多种规格

```python
# 使用 vLLM 部署（最快的推理引擎）
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-14B-Instruct",
    tensor_parallel_size=1,  # 单卡
    gpu_memory_utilization=0.9
)

def optimize_text(text: str) -> str:
    prompt = f"""请优化以下语音转文字的结果，修正语法错误，添加标点符号，使其更易读：

{text}

优化后的文本："""

    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text
```

**H100 性能**:
- Qwen2.5-7B: ~200 tokens/秒
- Qwen2.5-14B: ~120 tokens/秒
- Qwen2.5-32B: ~60 tokens/秒
- Qwen2.5-72B: ~30 tokens/秒（需要 2 张 H100）

#### 方案 B：Llama 3.1

```python
# Meta Llama 3.1
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
```

#### 方案 C：混合使用

```python
# 快速场景用小模型，高质量场景用大模型
class LLMRouter:
    def __init__(self):
        self.fast_model = LLM("Qwen2.5-7B")
        self.quality_model = LLM("Qwen2.5-72B", tensor_parallel_size=2)

    def optimize(self, text: str, mode: str = "fast"):
        if mode == "fast":
            return self.fast_model.generate([text])
        else:
            return self.quality_model.generate([text])
```

---

### 3. 完整服务实现

```python
# h100_service.py
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from faster_whisper import WhisperModel
from vllm import LLM, SamplingParams
import asyncio
from typing import Optional
import tempfile
import os

app = FastAPI()

# 全局模型（启动时加载一次）
print("Loading models on H100...")

# Whisper（使用 FP16，H100 上速度已经够快）
whisper_model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",
    device_index=0  # 使用第一张 H100
)

# LLM（使用 vLLM）
llm = LLM(
    model="Qwen/Qwen2.5-14B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.4,  # 为 Whisper 留空间
    device="cuda:0"  # 同一张卡，或者用 cuda:1 分开
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048
)

print("Models loaded!")


@app.post("/transcribe_and_optimize")
async def transcribe_and_optimize(
    file: UploadFile = File(...),
    language: str = "zh",
    mode: str = "standard"  # standard, formal, casual
):
    """
    一站式服务：语音转文字 + AI 优化
    """

    # 1. 保存音频
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        # 2. Whisper 转录
        segments, info = whisper_model.transcribe(
            temp_path,
            language=language,
            beam_size=5,
            vad_filter=True
        )

        raw_text = " ".join([segment.text for segment in segments])

        # 3. LLM 优化
        if mode == "formal":
            prompt = f"""请将以下语音转文字结果改写为正式的书面语，适合用于邮件或报告：

{raw_text}

改写后的文本："""
        elif mode == "casual":
            prompt = f"""请将以下语音转文字结果改写为轻松的口语化表达，适合聊天：

{raw_text}

改写后的文本："""
        else:  # standard
            prompt = f"""请优化以下语音转文字结果，修正错误，添加标点符号，使其更易读：

{raw_text}

优化后的文本："""

        outputs = llm.generate([prompt], sampling_params)
        optimized_text = outputs[0].outputs[0].text.strip()

        return {
            "raw_text": raw_text,
            "optimized_text": optimized_text,
            "language": info.language,
            "duration": info.duration,
            "mode": mode
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/transcribe_only")
async def transcribe_only(
    file: UploadFile = File(...),
    language: str = "zh"
):
    """纯转录，不优化"""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        segments, info = whisper_model.transcribe(temp_path, language=language)
        text = " ".join([segment.text for segment in segments])

        return {
            "text": text,
            "language": info.language,
            "duration": info.duration
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/optimize_only")
async def optimize_only(
    text: str,
    mode: str = "standard"
):
    """纯文本优化"""

    prompts = {
        "formal": f"请将以下文本改写为正式的书面语：\n\n{text}\n\n改写后：",
        "casual": f"请将以下文本改写为轻松的口语化表达：\n\n{text}\n\n改写后：",
        "standard": f"请优化以下文本，修正错误，添加标点符号：\n\n{text}\n\n优化后：",
        "concise": f"请将以下文本改写得更简洁：\n\n{text}\n\n精简后：",
        "detailed": f"请将以下文本扩写得更详细：\n\n{text}\n\n扩写后："
    }

    prompt = prompts.get(mode, prompts["standard"])
    outputs = llm.generate([prompt], sampling_params)

    return {
        "original": text,
        "optimized": outputs[0].outputs[0].text.strip(),
        "mode": mode
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "whisper_model": "large-v3",
        "llm_model": "Qwen2.5-14B-Instruct",
        "device": "H100"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 📊 容量规划

### 单张 H100 的处理能力

**假设**：
- 平均音频 3 分钟
- Whisper 处理 3 秒
- LLM 优化 2 秒
- 总计 5 秒/请求

**并发能力**：
- 串行：720 请求/小时
- 并发（批处理）：**3000+ 请求/小时**

**每天**：
- 单 H100：72,000 请求/天
- 3 张 H100：**216,000 请求/天**

### 用户容量估算

假设平均每用户每天 20 次使用：
- 单 H100：3,600 DAU
- 3 张 H100：**10,800 DAU**

如果 5% 转化率，每人 $9.99/月：
- 单 H100：$1,800/月收入
- 3 张 H100：**$5,400/月收入**

**纯利润！（不算服务器和电费）**

---

## 🎮 多 GPU 调度策略

### 方案 1：简单轮询

```python
import itertools

gpu_pool = itertools.cycle([0, 1, 2])  # 3 张 H100

async def process_request(audio_file):
    gpu_id = next(gpu_pool)

    # 使用指定 GPU
    model = WhisperModel("large-v3", device=f"cuda:{gpu_id}")
    result = model.transcribe(audio_file)
    return result
```

### 方案 2：任务队列（推荐）

```python
# 使用 Celery + Redis
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def transcribe_task(audio_data, gpu_id):
    """在指定 GPU 上执行转录"""
    model = WhisperModel("large-v3", device=f"cuda:{gpu_id}")
    return model.transcribe(audio_data)

# 客户端提交任务
result = transcribe_task.delay(audio_data, gpu_id=0)
```

### 方案 3：Ray（高级）

```python
import ray

ray.init()

@ray.remote(num_gpus=1)
class WhisperWorker:
    def __init__(self, gpu_id):
        self.model = WhisperModel("large-v3", device=f"cuda:{gpu_id}")

    def transcribe(self, audio_path):
        return self.model.transcribe(audio_path)

# 创建 3 个 worker
workers = [WhisperWorker.remote(i) for i in range(3)]

# 分发任务
futures = [workers[i % 3].transcribe.remote(audio) for i, audio in enumerate(audio_list)]
results = ray.get(futures)
```

---

## 💰 成本分析

### H100 服务器成本

假设你已有 H100（沉没成本），只算运营成本：

| 项目 | 成本/月 |
|------|---------|
| 电费（3 × H100 × 700W × 24h） | ~$400 |
| 网络带宽（1TB） | ~$100 |
| 服务器机房/维护 | ~$200 |
| **总计** | **~$700/月** |

### 盈亏平衡点

假设定价 $9.99/月，5% 转化率：

需要 DAU = $700 / ($9.99 × 5%) = **1,400 DAU**

只需要 1400 个日活用户就能盈利！

---

## 🚀 性能优化建议

### 1. 批处理

```python
# 同时处理多个音频（提升吞吐量）
audios = [audio1, audio2, audio3]
results = whisper_model.transcribe_batch(audios)
```

### 2. FP8 量化（H100 独有）

```python
# 使用 FP8，速度再快 30%，显存减半
model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float8"  # H100 支持
)
```

### 3. 流式处理

```python
# 边录音边转录（降低延迟）
async def stream_transcribe(audio_stream):
    async for chunk in audio_stream:
        segments = whisper_model.transcribe(chunk)
        yield segments
```

### 4. 缓存热门请求

```python
import redis

cache = redis.Redis()

def transcribe_with_cache(audio_hash):
    # 检查缓存
    cached = cache.get(audio_hash)
    if cached:
        return cached

    # 未缓存，执行转录
    result = whisper_model.transcribe(audio)
    cache.set(audio_hash, result, ex=3600)  # 缓存 1 小时
    return result
```

---

## 🎯 推荐行动方案

### 立即可做：

1. **部署 Whisper large-v3** 在一张 H100 上
   ```bash
   pip install faster-whisper
   python h100_service.py
   ```

2. **部署 Qwen2.5-14B** 在同一张或另一张 H100 上
   ```bash
   pip install vllm
   # vLLM 会自动优化推理
   ```

3. **集成到现有项目**
   - 修改 `realtime_server.py`
   - 调用本地 H100 服务而不是 OpenAI API

### 1-2 周内：

1. **性能基准测试**
   - 延迟测试
   - 并发测试
   - 质量对比（vs OpenAI）

2. **多 GPU 调度**
   - 实现负载均衡
   - 任务队列

3. **监控面板**
   - GPU 利用率
   - 请求 QPS
   - 错误率

---

## 📈 商业优势

有了 H100，你的优势：

✅ **零边际成本** - OpenAI 每分钟 $0.144，你 $0
✅ **极致性能** - 6 秒处理 5 分钟音频
✅ **数据隐私** - 数据不离开你的服务器
✅ **可定制化** - 微调模型适配特定场景
✅ **利润率高** - 成本 $700/月，收入可达 $5000+/月

---

要不要我帮你：
1. 写一个完整的 H100 部署脚本？
2. 测试 Whisper + Qwen 的端到端性能？
3. 设计多 GPU 的负载均衡方案？
