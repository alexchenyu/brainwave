# Whisper 自部署方案

## 模型选择

### Whisper 模型大小对比

| 模型 | 参数量 | VRAM 需求 | 速度 | 准确度 | 推荐场景 |
|------|--------|-----------|------|--------|----------|
| tiny | 39M | ~1GB | 32x | ⭐⭐⭐ | 实时预览 |
| base | 74M | ~1GB | 16x | ⭐⭐⭐⭐ | 快速转录 |
| small | 244M | ~2GB | 6x | ⭐⭐⭐⭐ | 平衡选择 |
| medium | 769M | ~5GB | 2x | ⭐⭐⭐⭐⭐ | 高质量 |
| large-v3 | 1550M | ~10GB | 1x | ⭐⭐⭐⭐⭐ | 最高质量 |

**推荐**：`medium` 或 `large-v3`
- medium: 性能和质量平衡，大多数场景够用
- large-v3: 最新最好，中文效果显著提升

---

## 硬件需求

### GPU 服务器选择

#### 方案 A：云服务器（按需付费）
| 提供商 | GPU | VRAM | 价格 | 适合场景 |
|--------|-----|------|------|----------|
| **RunPod** | RTX 4090 | 24GB | $0.69/h | 开发测试 |
| **Vast.ai** | RTX 3090 | 24GB | $0.25-0.50/h | 成本敏感 |
| **Lambda Labs** | A100 | 40GB | $1.10/h | 高并发 |
| **腾讯云 GPU** | T4 | 16GB | ¥5.4/h | 国内用户 |
| **阿里云 GPU** | V100 | 16GB | ¥10/h | 国内用户 |

**月成本估算**（24/7 运行）:
- Vast.ai RTX 3090: $180-360/月
- RunPod RTX 4090: $497/月
- 腾讯云 T4: ¥3,888/月 (~$540)

#### 方案 B：自己买 GPU
| GPU | 价格 | VRAM | 适合模型 | ROI |
|-----|------|------|----------|-----|
| RTX 3060 | $300 | 12GB | medium | 2 个月回本 |
| RTX 3090 | $1000 | 24GB | large-v3 | 3 个月回本 |
| RTX 4090 | $1600 | 24GB | large-v3 | 4 个月回本 |

**电费**：约 $30-50/月（300W × 24h）

---

## 部署方案对比

### 方案 1: faster-whisper（推荐）

**优点**：
- 速度快 4-5 倍
- 内存占用低
- 支持 GPU 和 CPU
- 支持流式转录

**安装**：
```bash
pip install faster-whisper
```

**代码示例**：
```python
from faster_whisper import WhisperModel

# 加载模型（首次会下载）
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# 转录
segments, info = model.transcribe("audio.mp3", language="zh")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

**性能**：
- RTX 3090 + large-v3: 约 0.5x 实时（5 分钟音频 → 2.5 分钟处理）
- RTX 4090 + large-v3: 约 0.3x 实时（5 分钟音频 → 1.5 分钟处理）

---

### 方案 2: whisper.cpp（最快）

**优点**：
- 纯 C++ 实现
- 速度最快
- CPU 也能跑
- 支持量化（INT8/INT4）

**安装**：
```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make
```

**性能**：
- CPU (16 核): medium 约 1x 实时
- RTX 3090: large-v3 约 0.2x 实时

---

### 方案 3: OpenAI Whisper（原版）

**缺点**：
- 速度慢
- 内存占用高
- 不适合生产环境

**不推荐用于生产**

---

## 完整 API 服务实现

### 使用 FastAPI + faster-whisper

```python
# whisper_service.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from faster_whisper import WhisperModel
import tempfile
import os
from typing import Optional

app = FastAPI()

# 全局加载模型（避免每次请求都加载）
print("Loading Whisper model...")
model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",
    download_root="./models"  # 模型缓存目录
)
print("Model loaded!")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = "zh",  # zh, en, ja, etc.
    task: Optional[str] = "transcribe"  # transcribe or translate
):
    """
    转录音频文件

    参数：
    - file: 音频文件（支持 mp3, wav, m4a, ogg, webm 等）
    - language: 语言代码（zh, en, ja 等）
    - task: transcribe（转录）或 translate（翻译成英文）
    """

    # 保存上传的文件到临时位置
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        # 转录
        segments, info = model.transcribe(
            temp_path,
            language=language,
            task=task,
            beam_size=5,  # 提高质量
            vad_filter=True,  # 过滤静音
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        # 收集结果
        result_text = ""
        segments_list = []

        for segment in segments:
            result_text += segment.text
            segments_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })

        return {
            "text": result_text.strip(),
            "segments": segments_list,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/transcribe_stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    language: Optional[str] = "zh"
):
    """
    流式转录（逐句返回）
    """
    from fastapi.responses import StreamingResponse
    import json

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    async def generate():
        try:
            segments, info = model.transcribe(
                temp_path,
                language=language,
                beam_size=5,
                vad_filter=True
            )

            for segment in segments:
                yield json.dumps({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end
                }, ensure_ascii=False) + "\n"

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "model": "large-v3"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 部署脚本

```bash
# install_whisper.sh
#!/bin/bash

# 安装依赖
pip install faster-whisper fastapi uvicorn python-multipart

# 创建模型缓存目录
mkdir -p models

# 下载模型（可选，首次运行时会自动下载）
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', download_root='./models')"

# 运行服务
uvicorn whisper_service:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## 性能优化

### 1. 使用量化模型（降低显存）

```python
# float16: 10GB VRAM
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# int8: 5GB VRAM（速度稍慢，质量略降）
model = WhisperModel("large-v3", device="cuda", compute_type="int8")
```

### 2. 批处理（提高吞吐量）

```python
# 同时处理多个音频
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

def process_audio(audio_path):
    segments, info = model.transcribe(audio_path)
    return segments

# 并行处理
futures = [executor.submit(process_audio, audio) for audio in audio_list]
results = [f.result() for f in futures]
```

### 3. VAD（语音活动检测）

```python
# 自动过滤静音部分
segments, info = model.transcribe(
    audio_path,
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=500,  # 最小静音时长
        speech_pad_ms=200  # 语音前后填充
    )
)
```

### 4. 使用 Flash Attention（需要较新的 GPU）

```python
# 需要安装 flash-attn
pip install flash-attn

# 启用 Flash Attention（自动）
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
```

---

## 集成到现有项目

### 修改 realtime_server.py

```python
# realtime_server.py
import httpx

async def transcribe_audio_local(audio_data: bytes, language: str = "zh"):
    """调用本地 Whisper 服务"""

    files = {"file": ("audio.wav", audio_data, "audio/wav")}
    data = {"language": language}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://localhost:8000/transcribe",
            files=files,
            data=data
        )
        response.raise_for_status()
        return response.json()

# 在 WebSocket 处理中使用
@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    # ... 现有代码 ...

    # 收到音频数据后
    audio_bytes = await websocket.receive_bytes()

    # 使用本地 Whisper 转录
    result = await transcribe_audio_local(audio_bytes)
    transcript_text = result["text"]

    # 发送转录结果给客户端
    await websocket.send_json({
        "type": "transcription",
        "text": transcript_text
    })
```

---

## 成本对比

### 场景：10,000 分钟/月转录量

| 方案 | 月成本 | 年成本 | 备注 |
|------|--------|--------|------|
| **OpenAI Whisper API** | $60 | $720 | $0.006/分钟 |
| **云 GPU (Vast.ai)** | $180-360 | $2,160-4,320 | 24/7 运行 |
| **自购 RTX 3090** | $80 | $960 | 电费 + 分摊硬件成本 |
| **自购 RTX 3090 (仅高峰)** | $40 | $480 | 按需开机 |

### 结论

- **< 10K 分钟/月**：OpenAI API 更划算
- **> 10K 分钟/月**：自部署开始划算
- **> 50K 分钟/月**：自部署显著省钱

---

## 效果对比

### 中文语音识别准确率（非官方测试）

| 服务 | 准确率 | 标点 | 延迟 |
|------|--------|------|------|
| Whisper large-v3 | ⭐⭐⭐⭐ | ✅ | 中 |
| OpenAI Whisper API | ⭐⭐⭐⭐ | ✅ | 中 |
| 讯飞语音 | ⭐⭐⭐⭐⭐ | ✅ | 低 |
| 阿里达摩院 | ⭐⭐⭐⭐⭐ | ✅ | 低 |
| Azure Speech | ⭐⭐⭐⭐ | ✅ | 低 |

**Whisper 的优势**：
- 多语言支持好
- 开源可控
- 成本可控

**Whisper 的劣势**：
- 中文不如国内专业服务
- 延迟较高（非实时）
- 方言支持差

---

## 推荐方案

### 最佳组合：Whisper + 国内语音服务

```python
# 混合方案
async def transcribe_smart(audio_data: bytes, language: str):
    if language == "zh":
        # 中文用讯飞/阿里（便宜且准确）
        return await iflytek_transcribe(audio_data)
    else:
        # 其他语言用 Whisper
        return await whisper_transcribe(audio_data)
```

### 国内语音服务价格

| 服务 | 价格 | 免费额度 |
|------|------|---------|
| **讯飞语音** | ¥0.0004/次 (~$0.00006) | 5000 次/天 |
| **腾讯云** | ¥0.00125/秒 (~$0.0045/分钟) | 50 万次/月 |
| **阿里云** | ¥0.0022/分钟 | 3 个月免费 |
| **百度语音** | ¥0.00008/次 | 50000 次/天 |

**比 Whisper API 便宜 100 倍！**

---

## 下一步行动

### 立即可做：
1. [ ] 部署本地 Whisper 服务（测试）
2. [ ] 对比 Whisper vs 讯飞 效果
3. [ ] 测试处理速度和准确率

### 1 周内：
1. [ ] 集成到现有项目
2. [ ] 实现混合方案（中文用讯飞，英文用 Whisper）
3. [ ] 性能基准测试

### 1 个月内：
1. [ ] 优化延迟
2. [ ] 实现流式转录
3. [ ] 部署到生产环境

---

要我帮你部署一个本地 Whisper 服务吗？
