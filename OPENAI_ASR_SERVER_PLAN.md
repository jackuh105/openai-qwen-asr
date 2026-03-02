# OpenAI API 相容 ASR 伺服器（FastAPI + mlx-qwen3-asr）實作計畫

## 目標與範圍

### 核心目標
使用 `mlx-qwen3-asr` 作為 ASR 引擎，建立 OpenAI API 相容的轉錄伺服器

### 支援的 API
1. **Audio Transcriptions API** (`POST /v1/audio/transcriptions`)
   - 非串流模式：客戶端上傳完整音訊，伺服器回傳完整轉錄結果
   - SSE 串流模式：客戶端上傳完整音訊，伺服器以 SSE 串流回傳增量文字
2. **Realtime API** (`wss://<host>/v1/realtime`)
   - WebSocket 雙向通訊，支援即時音訊輸入與文字增量輸出
   - 遵循 OpenAI Realtime API 事件格式（僅實作 ASR 相關事件）

### 確認的設計決策
| 決策項目 | 決策內容 |
|---------|---------|
| Model 參數處理 | `whisper-1` 映射至 `Qwen/Qwen3-ASR-1.7B`，同時接受直接 Qwen ID |
| 不支援參數處理 | `temperature`、`prompt` 靜默忽略 |
| Timestamps | `srt`/`vtt` 格式自動啟用 timestamps |
| 預設模型 | `Qwen/Qwen3-ASR-1.7B` |
| JSON 回應格式 | 嚴格 OpenAI 相容：`{"text": "..."}` |
| Language 參數 | 直接傳遞給 mlx-qwen3-asr（支援 ISO code 與完整名稱） |
| Realtime 音訊格式 | base64 編碼 PCM16 |
| 身份驗證 | 無驗證 |
| 量化支援 | 可透過環境變數設定 |

---

## 1. 專案結構

```
server/
├── __init__.py
├── app.py                 # FastAPI 入口、lifespan、路由註冊
├── config.py              # 環境變數、配置資料類別
├── models.py              # Pydantic 請求/回應 schema
├── errors.py              # OpenAI 相容錯誤格式
├── asr/
│   ├── __init__.py
│   ├── engine.py          # mlx-qwen3-asr 封裝（非串流）
│   ├── streaming.py       # SSE 串流狀態管理
│   └── realtime.py        # WebSocket Realtime API 處理
├── routes/
│   ├── __init__.py
│   ├── transcriptions.py  # /v1/audio/transcriptions 端點
│   └── realtime.py        # /v1/realtime WebSocket 端點
└── utils/
    ├── __init__.py
    ├── audio.py           # 音訊載入、格式轉換
    └── model_mapping.py   # OpenAI model → Qwen model 映射
tests/
├── conftest.py            # pytest fixtures
├── test_transcriptions.py # 非串流端點測試
├── test_streaming.py      # SSE 串流測試
└── test_realtime.py       # WebSocket Realtime API 測試
```

---

## 2. 配置與環境變數

### `server/config.py`

```python
from dataclasses import dataclass, field
from typing import Optional, Literal
import os

@dataclass
class ServerConfig:
    model_id: str = "Qwen/Qwen3-ASR-1.7B"
    dtype: str = "fp16"
    quantize_bits: Optional[int] = None
    quantize_group_size: int = 64
    
    sample_rate: int = 16000
    max_file_size_mb: int = 100
    
    chunk_size_sec: float = 2.0
    max_context_sec: float = 30.0
    
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 4
    
    max_new_tokens: int = 4096
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            model_id=os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-1.7B"),
            dtype=os.getenv("DTYPE", "fp16"),
            quantize_bits=int(b) if (b := os.getenv("QUANTIZE_BITS")) else None,
            quantize_group_size=int(os.getenv("QUANTIZE_GROUP_SIZE", "64")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "100")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "4")),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            chunk_size_sec=float(os.getenv("CHUNK_SIZE_SEC", "2.0")),
            max_context_sec=float(os.getenv("MAX_CONTEXT_SEC", "30.0")),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "4096")),
        )
```

### 環境變數
| 變數名稱 | 預設值 | 說明 |
|---------|--------|------|
| `MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | 預設模型 ID |
| `DTYPE` | `fp16` | 資料型別 |
| `QUANTIZE_BITS` | (空) | 量化位元數（4 或 8） |
| `QUANTIZE_GROUP_SIZE` | `64` | 量化群組大小 |
| `MAX_FILE_SIZE_MB` | `100` | 最大上傳檔案大小（MB） |
| `MAX_CONCURRENT_REQUESTS` | `4` | 最大併發請求數 |
| `HOST` | `0.0.0.0` | 綁定位址 |
| `PORT` | `8000` | 綁定埠號 |
| `CHUNK_SIZE_SEC` | `2.0` | 串流切塊大小（秒） |
| `MAX_CONTEXT_SEC` | `30.0` | 串流最大上下文（秒） |
| `MAX_NEW_TOKENS` | `4096` | 最大生成 token 數 |

---

## 3. Model 映射

### `server/utils/model_mapping.py`

```python
MODEL_MAPPING = {
    "whisper-1": "Qwen/Qwen3-ASR-1.7B",
    "whisper": "Qwen/Qwen3-ASR-1.7B",
    "qwen-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
}

def resolve_model(model_param: str) -> str:
    """
    解析 model 參數：
    1. 檢查是否為映射表中的別名
    2. 若否，假設為直接 Qwen ID
    """
    return MODEL_MAPPING.get(model_param.lower(), model_param)
```

---

## 4. Audio Transcriptions API

### 4.1 端點規格

**路徑**: `POST /v1/audio/transcriptions`

**Content-Type**: `multipart/form-data`

### 4.2 請求參數

| 參數 | 類型 | 必要 | 說明 |
|------|------|------|------|
| `file` | file | ✓ | 音訊檔案 |
| `model` | string | ✓ | 模型名稱（`whisper-1` 或 Qwen ID） |
| `language` | string | | 語言（如 `en`、`zh`、`English`） |
| `response_format` | string | | 輸出格式：`json`、`text`、`srt`、`vtt`、`verbose_json` |
| `stream` | boolean | | 是否啟用 SSE 串流（預設 false） |
| `temperature` | float | | （忽略） |
| `prompt` | string | | （忽略） |

### 4.3 回應格式

#### `response_format=json` (預設)
```json
{
  "text": "Hello, world."
}
```

#### `response_format=text`
```
Hello, world.
```

#### `response_format=srt`
```
1
00:00:00,000 --> 00:00:01,000
Hello, world.
```

#### `response_format=vtt`
```
WEBVTT

00:00:00.000 --> 00:00:01.000
Hello, world.
```

#### `response_format=verbose_json`
```json
{
  "task": "transcribe",
  "language": "english",
  "duration": 5.2,
  "text": "Hello, world.",
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "world", "start": 0.6, "end": 1.0}
  ]
}
```

### 4.4 SSE 串流模式 (`stream=true`)

**Content-Type**: `text/event-stream`

```
event: transcript.partial
data: {"type": "transcript.partial", "text": "Hello"}

event: transcript.partial
data: {"type": "transcript.partial", "text": "Hello world"}

event: transcript.final
data: {"type": "transcript.final", "text": "Hello world."}

data: [DONE]
```

---

## 5. Realtime API (WebSocket)

### 5.1 端點規格

**路徑**: `wss://<host>/v1/realtime`

**通訊協定**: WebSocket

**訊息格式**: JSON（文字訊息）

### 5.2 實作的事件類型

#### 客戶端 → 伺服器

| 事件 | 說明 |
|------|------|
| `session.update` | 更新 session 設定（model, language, etc.） |
| `input_audio_buffer.append` | 追加音訊資料（base64 PCM16） |
| `input_audio_buffer.commit` | 標記音訊結束，觸發轉錄 |
| `input_audio_buffer.clear` | 清除音訊緩衝區 |

#### 伺服器 → 客戶端

| 事件 | 說明 |
|------|------|
| `session.created` | Session 建立成功 |
| `session.updated` | Session 設定已更新 |
| `input_audio_buffer.committed` | 音訊緩衝區已提交 |
| `input_audio_buffer.speech_started` | 偵測到語音開始 |
| `input_audio_buffer.speech_stopped` | 偵測到語音結束 |
| `response.created` | 回應開始產生 |
| `response.audio_transcript.delta` | 增量文字 |
| `response.audio_transcript.done` | 文字完成 |
| `response.done` | 回應完成 |
| `error` | 錯誤事件 |

### 5.3 事件流程範例

```
Client                                Server
  |                                     |
  |---------- session.update ---------->|  (可選，設定 model/language)
  |<--------- session.updated ---------|
  |                                     |
  |------ input_audio_buffer.append --->|  (base64 PCM16)
  |------ input_audio_buffer.append --->|  (多次)
  |------ input_audio_buffer.append --->|
  |------ input_audio_buffer.commit --->|
  |                                     |
  |<----- response.created -------------|
  |<----- response.audio_transcript.delta --|
  |<----- response.audio_transcript.delta --|  (增量文字)
  |<----- response.audio_transcript.done ---|
  |<----- response.done ---------------|
```

### 5.4 音訊格式

- **格式**: PCM16 (16-bit signed integer, little-endian)
- **取樣率**: 16000 Hz
- **聲道**: 單聲道 (mono)
- **編碼**: Base64

### 5.5 事件 Schema

#### `session.update` (client → server)
```json
{
  "type": "session.update",
  "session": {
    "model": "whisper-1",
    "input_audio_format": "pcm16"
  }
}
```

#### `session.created` (server → client)
```json
{
  "type": "session.created",
  "session": {
    "id": "sess_abc123",
    "model": "whisper-1"
  }
}
```

#### `input_audio_buffer.append` (client → server)
```json
{
  "type": "input_audio_buffer.append",
  "audio": "<base64-encoded-pcm16>"
}
```

#### `input_audio_buffer.commit` (client → server)
```json
{
  "type": "input_audio_buffer.commit"
}
```

#### `response.audio_transcript.delta` (server → client)
```json
{
  "type": "response.audio_transcript.delta",
  "response_id": "resp_xyz789",
  "delta": "Hello"
}
```

#### `response.audio_transcript.done` (server → client)
```json
{
  "type": "response.audio_transcript.done",
  "response_id": "resp_xyz789",
  "transcript": "Hello world."
}
```

#### `error` (server → client)
```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "Invalid audio format",
    "code": "invalid_audio"
  }
}
```

---

## 6. ASR 引擎封裝

### 6.1 `server/asr/engine.py`

```python
from mlx_qwen3_asr import Session
from typing import Optional
import mx.array as mx

class ASREngine:
    def __init__(self, config: "ServerConfig"):
        self.config = config
        self.session: Optional[Session] = None
    
    def load_model(self):
        self.session = Session(
            model=self.config.model_id,
            dtype=self.config.dtype
        )
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        return_timestamps: bool = False
    ):
        return self.session.transcribe(
            audio_path,
            language=language,
            return_timestamps=return_timestamps
        )
```

### 6.2 `server/asr/streaming.py`

```python
from mlx_qwen3_asr.streaming import init_streaming, feed_audio, finish_streaming
from typing import AsyncIterator
import mx.array as mx

class StreamingTranscriber:
    def __init__(self, session: Session, config: "ServerConfig"):
        self.session = session
        self.config = config
    
    async def transcribe_stream(
        self,
        audio: mx.array,
        language: Optional[str] = None
    ) -> AsyncIterator[str]:
        state = init_streaming(
            chunk_size_sec=self.config.chunk_size_sec,
            max_context_sec=self.config.max_context_sec
        )
        
        chunk_samples = int(self.config.chunk_size_sec * 16000)
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            state = feed_audio(chunk, state, session=self.session)
            yield state.text
        
        state = finish_streaming(state, session=self.session)
        yield state.text
```

### 6.3 `server/asr/realtime.py`

```python
from mlx_qwen3_asr.streaming import init_streaming, feed_audio, finish_streaming
import base64
import numpy as np
import mx.array as mx

class RealtimeSession:
    def __init__(self, session: Session, config: "ServerConfig"):
        self.session = session
        self.config = config
        self.streaming_state = None
        self.current_text = ""
    
    def start(self):
        self.streaming_state = init_streaming(
            chunk_size_sec=self.config.chunk_size_sec,
            max_context_sec=self.config.max_context_sec
        )
    
    def append_audio(self, base64_audio: str) -> Optional[str]:
        pcm_bytes = base64.b64decode(base64_audio)
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        pcm_float = pcm_int16.astype(np.float32) / 32768.0
        audio_chunk = mx.array(pcm_float)
        
        self.streaming_state = feed_audio(
            audio_chunk,
            self.streaming_state,
            session=self.session
        )
        
        if self.streaming_state.text != self.current_text:
            delta = self.streaming_state.text[len(self.current_text):]
            self.current_text = self.streaming_state.text
            return delta
        return None
    
    def commit(self) -> str:
        self.streaming_state = finish_streaming(
            self.streaming_state,
            session=self.session
        )
        return self.streaming_state.text
```

---

## 7. 錯誤處理

### 7.1 錯誤格式

```json
{
  "error": {
    "message": "The model 'invalid-model' does not exist",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

### 7.2 錯誤代碼

| HTTP Status | Code | 說明 |
|-------------|------|------|
| 400 | `invalid_file` | 音訊檔案格式不支援 |
| 400 | `file_too_large` | 檔案超過大小限制 |
| 400 | `invalid_model` | 模型名稱無效 |
| 400 | `invalid_response_format` | 不支援的 response_format |
| 500 | `transcription_failed` | 轉錄過程失敗 |
| 503 | `server_busy` | 超過併發限制 |

---

## 8. 請求處理流程

### 8.1 非串流轉錄

```
POST /v1/audio/transcriptions
        │
        ▼
┌─────────────────────────────────┐
│ 1. 解析 multipart form data      │
│    - file, model, language, etc. │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ 2. 驗證                          │
│    - 檔案大小 ≤ limit           │
│    - 檔案格式支援               │
│    - model 參數解析             │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ 3. 載入音訊                      │
│    - load_audio() → 16kHz mono  │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ 4. 轉錄                          │
│    - stream=true → SSE          │
│    - srt/vtt → timestamps=True  │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ 5. 格式化輸出                    │
│    - json/text/srt/vtt/verbose  │
└─────────────────────────────────┘
```

### 8.2 Realtime API

```
WebSocket /v1/realtime 連線
        │
        ▼
┌─────────────────────────────────┐
│ 1. 建立 RealtimeSession         │
│    - init_streaming()           │
│    - 發送 session.created       │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ 2. 事件循環                      │
│    - append → feed_audio        │
│    - commit → finish_streaming  │
│    - update → 更新設定          │
└─────────────────────────────────┘
```

---

## 9. 效能與資源控制

### 9.1 模型預載
使用 FastAPI lifespan 在啟動時預載模型。

### 9.2 併發控制
使用 `asyncio.Semaphore` 限制併發請求數。

### 9.3 記憶體管理
- 串流模式使用 `max_context_sec` 限制 KV cache
- 定期清理已完成的 session

---

## 10. 測試計畫

### 10.1 單元測試
- model_mapping: model 參數解析
- audio_utils: 音訊載入與格式轉換
- response_formats: 各格式輸出
- error_handling: 錯誤回應格式

### 10.2 整合測試
- transcriptions: 完整檔案上傳轉錄
- streaming: SSE 串流事件
- realtime: WebSocket 事件

---

## 11. 部署

### 11.1 啟動指令
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 自訂設定
MODEL_ID=Qwen/Qwen3-ASR-0.6B QUANTIZE_BITS=4 uvicorn server.app:app
```

### 11.2 Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server/ ./server/

ENV MODEL_ID=Qwen/Qwen3-ASR-1.7B
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 11.3 Dependencies
```
fastapi>=0.109.0
uvicorn>=0.27.0
mlx-qwen3-asr>=0.1.0
python-multipart>=0.0.6
websockets>=12.0
pydantic>=2.0.0
pytest>=8.0.0
httpx>=0.26.0
```

---

## 12. 里程碑

### Phase 1: 基礎架構與非串流轉錄 ✅ COMPLETE
- [x] 建立專案結構與配置
  - [x] Create `server/` directory with subdirectories (`asr/`, `routes/`, `utils/`)
  - [x] Create `tests/` directory
  - [x] Create `__init__.py` files
  - [x] Implement `server/config.py` - ServerConfig with env vars and dtype mapping
  - [x] Implement `server/utils/model_mapping.py` - MODEL_MAPPING dict and resolve_model()
  - [x] Create `pyproject.toml` - Project config with uv
- [x] 實作 ASREngine 封裝
  - [x] Create `server/asr/engine.py`
  - [x] Implement ASREngine singleton with Session initialization
  - [x] Handle mlx dtype conversion (string -> mx.Dtype)
- [x] 實作 Audio Transcriptions API（非串流）
  - [x] Create `server/models.py` - Pydantic request/response schemas
  - [x] Create `server/errors.py` - OpenAI-compatible error format
  - [x] Create `server/utils/audio.py` - Audio loading and SRT/VTT formatting
  - [x] Create `server/routes/transcriptions.py` - POST /v1/audio/transcriptions
  - [x] Create `server/app.py` - FastAPI app with lifespan
- [x] 支援 json, text, srt, vtt, verbose_json 回應格式
- [x] 錯誤處理與 OpenAI 格式對齊
- [x] 單元測試 (45 tests passing)

### Phase 2: SSE 串流與字幕格式 ✅ COMPLETE
- [x] 實作 SSE 串流轉錄
  - [x] Create `server/asr/streaming.py` - StreamingTranscriber class
  - [x] Implement `transcribe_stream()` and `transcribe_stream_with_deltas()` async generators
  - [x] Add SSE event models to `server/models.py` - TranscriptPartialEvent, TranscriptFinalEvent
  - [x] Update `server/routes/transcriptions.py` with `stream` parameter
  - [x] Implement SSE StreamingResponse with proper headers
- [x] 支援 srt, vtt 回應格式（已在 Phase 1 完成）
- [x] 自動啟用 timestamps for srt/vtt（已在 Phase 1 完成）
- [x] 串流測試 (54 tests passing)

### Phase 3: Realtime API (WebSocket) ✅ COMPLETE
- [x] 實作 WebSocket 端點
  - [x] Create `server/asr/realtime.py` - RealtimeSessionState and RealtimeTranscriber
  - [x] Create `server/routes/realtime.py` - WebSocket endpoint at /v1/realtime
  - [x] Add realtime event models to `server/models.py`
  - [x] Register realtime router in `server/app.py`
- [x] 實作 session.update, session.created
- [x] 實作 input_audio_buffer.append/commit/clear
- [x] 實作 response.audio_transcript.delta/done
- [x] 錯誤事件處理
- [x] Realtime API 測試 (21 tests, 75 total)

### Phase 4: 強化與優化
- [ ] 併發控制與資源管理
- [ ] 效能監控
- [ ] 文件與 API 範例

### Phase 5: 部署
- [ ] Dockerfile
- [ ] 部署文件
- [ ] 壓力測試