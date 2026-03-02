# Architecture Decisions

## Decision Log

### ADR-001: Model Parameter Mapping
**Decision**: Map `whisper-1` to `Qwen/Qwen3-ASR-1.7B` while also accepting direct Qwen model IDs.

**Rationale**: 
- Provides OpenAI compatibility for existing clients
- Allows power users to specify exact Qwen models
- Simple mapping table for maintenance

**Alternatives Considered**:
- Only accept OpenAI names (too restrictive)
- Only accept Qwen IDs (breaks compatibility)

---

### ADR-002: Unsupported Parameter Handling
**Decision**: Silently ignore `temperature` and `prompt` parameters.

**Rationale**:
- mlx-qwen3-asr doesn't support these parameters
- Silently ignoring maintains compatibility with OpenAI clients
- No error that would break existing integrations

---

### ADR-003: Timestamp Auto-Enable
**Decision**: Automatically enable timestamps when `response_format=srt` or `vtt`.

**Rationale**:
- srt/vtt formats require timestamps by definition
- Users expect these formats to work without additional parameters
- Simplifies client implementation

---

### ADR-004: Default Model Selection
**Decision**: Use `Qwen/Qwen3-ASR-1.7B` as default.

**Rationale**:
- Higher accuracy than 0.6B model
- Users can opt for 0.6B if speed is priority
- Matches OpenAI's quality-first default approach

---

### ADR-005: Strict JSON Response Format
**Decision**: Match OpenAI's exact response format `{"text": "..."}` for `json` format.

**Rationale**:
- Ensures compatibility with OpenAI SDKs
- Predictable response structure
- `verbose_json` available for extended data

---

### ADR-006: Realtime API Scope
**Decision**: Implement only ASR-related events for Realtime API.

**Rationale**:
- mlx-qwen3-asr is ASR-only (no TTS)
- Reduces implementation complexity
- Can be extended later if needed

---

### ADR-007: No Authentication
**Decision**: Do not require authentication.

**Rationale**:
- Intended for local/trusted network deployment
- Simplifies development and testing
- Can be added via reverse proxy if needed

---

### ADR-008: Configurable Quantization
**Decision**: Allow quantization configuration via environment variables.

**Rationale**:
- Users can optimize for speed vs accuracy
- 4-bit/8-bit provides significant speedup
- Useful for resource-constrained environments

---

### ADR-009: Dtype String to mx.Dtype Conversion
**Decision**: Implement `DTYPE_MAP` and `get_mlx_dtype()` method in config to convert string config values to proper `mx.Dtype` objects.

**Rationale**:
- `mlx_qwen3_asr.Session` expects `mx.Dtype` objects (like `mx.float16`), NOT strings like `"fp16"`
- Passing strings causes `TypeError: astype(): incompatible function arguments`
- Valid dtype values: `mx.float16`, `mx.bfloat16`, `mx.float32`
- Environment variables are strings, need conversion before passing to Session

---

### ADR-010: TranscriptionResult Attribute Access
**Decision**: Access `TranscriptionResult` attributes directly (e.g., `result.text`) instead of dict methods.

**Rationale**:
- `mlx_qwen3_asr.Session.transcribe()` returns a `TranscriptionResult` dataclass, not a dict
- Has attributes: `text`, `language`, `segments`, `chunks`, `speaker_segments`
- Dict methods like `.get()` will fail