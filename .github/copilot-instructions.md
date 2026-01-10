# AI Coding Agent Instructions for STT-Phi4 Multimodal ASR Project

## Project Overview
This is a **Google Colab-optimized Jupyter notebook** for automatic speech recognition (ASR) and multilingual translation using Microsoft's **Phi-4-multimodal-instruct** model. The notebook runs on T4 GPUs with CUDA 12.x and handles audio input/output with a Gradio web UI exposed via Cloudflared tunneling.

## Architecture & Critical Components

### Core Processing Pipeline
The notebook follows this sequential flow:

1. **Environment Setup** → GPU/CUDA detection, dependency installation (transformers 4.46.0, accelerate, soundfile), FFmpeg audio support
2. **Model Loading** → Phi-4-multimodal-instruct with `attn_implementation='eager'` (critical for T4 compatibility)
3. **Speech Adapter** → Loads "speech-lora" adapter for audio tasks (fallback to base model if unavailable)
4. **ASR Function** → `phi4_transcribe_or_translate()` processes audio with language-specific prompts
5. **Fallback Translation** → Uses M2M-100 (facebook/m2m100_418M) cached model for Bengali-only output
6. **Gradio UI** → Interactive interface with audio input (microphone/upload), language selectors, output display
7. **Cloudflared Tunnel** → Exposes local Gradio server publicly via TryCloudflare (automatic URL detection)

### Key Model Configuration Decisions

**Attention Implementation:**
- **Must use `attn_implementation='eager'`** — FlashAttention2/SDPA unavailable on T4
- Patching is applied pre-model-load via monkey-patch to `Phi4MMModel` class in sys.modules
- If patching fails during load, a second attempt with manual instance-level patch occurs

**Dtype & Device Mapping:**
- GPU: float16 (T4 standard); CPU fallback: float32
- Offload folder: `/tmp/` to manage memory on limited GPU VRAM
- Cache disabled: `use_cache=False` to prevent `DynamicCache.get_usable_length` errors

**Adapter Loading:**
```python
model.load_adapter(MODEL_ID, adapter_name='speech', adapter_kwargs={'subfolder': 'speech-lora', 'offload_folder': '/tmp/'})
model.set_adapter('speech')  # Critical: must activate adapter
```

### ASR Function: `phi4_transcribe_or_translate()`

**Input Processing Pipeline:**

The function `phi4_transcribe_or_translate(audio_path: str, in_lang_ui: str, out_lang_ui: str)` processes audio in this order:

1. **Audio Loading** (`sf.read(audio_path)`)
   - Reads any format supported by soundfile (WAV, MP3, FLAC, OGG)
   - Returns tuple: (audio_array, sample_rate)
   - Sample rate preserved as-is; processor handles resampling internally

2. **Stereo to Mono Conversion**
   ```python
   if audio_array.ndim > 1:
       audio_array = audio_array.mean(axis=1)
   ```
   - Averages across channels to single mono track
   - Phi-4 audio processing expects mono input
   - Reduces computational overhead (half the samples)

3. **Duration Limiting (30 seconds max)**
   ```python
   max_seconds = 30
   max_samples = int(sr * max_seconds)
   if audio_array.shape[0] > max_samples:
       audio_array = audio_array[:max_samples]
   ```
   - Truncates at 30s to prevent:
     - T4 GPU timeout (long sequences exhaust VRAM/inference time)
     - User experience delays (inference typically 5-15s on T4)
     - `DynamicCache` errors from excessive token generation
   - Silent truncation with no error; user gets output from first 30s

4. **Float32 Normalization**
   ```python
   audio_array = np.asarray(audio_array, dtype=np.float32)
   ```
   - Ensures float32 precision for processor consistency
   - Handles integer audio (PCM16) → float conversion automatically by soundfile
   - Critical for numerical stability in audio embeddings

5. **Processor Multimodal Call**
   ```python
   processor(return_tensors='pt', text=[prompt], audios=[(audio_array, sr)])
   ```
   - Processor expects tuple: (waveform_array, sample_rate)
   - Converts to feature embeddings internally (Mel-spectrogram-like)
   - Moves tensors to `DEVICE` (GPU/CPU) with proper dtype (float16 on T4)

6. **Placeholder Token Selection**
   - Tries both `<|audio|>` and `<|audio_1|>` tokens in the prompt
   - Some versions of Phi-4 processor respond to one or the other
   - Loop with error catching; first successful token used
   - If both fail, returns error with both attempted placeholders listed

**Language-Specific Prompt Construction:**
- Same-language (e.g., Portuguese → Portuguese): `"<|audio|>\nTranscribe the audio clip into text in Portuguese."`
- Cross-language (e.g., Portuguese → Spanish): `"<|audio|>\nTranscribe the audio to text in Portuguese, and then translate the audio to Spanish. Use <sep> as a separator between the original transcript and the translation."`
- Bengali special case: Always transcribes source → uses M2M-100 fallback for translation (not Phi-4's multilingual capability)

**Language Support:**
- Input: Portuguese, Spanish, English (mapped to "Portuguese", "Spanish", "English" in prompts)
- Output: Same as input + Bengali (Bengali uses fallback M2M-100 translation)
- Same-language transcription: simpler prompt; cross-language: uses `<sep>` separator in output

**Generation Parameters (Fixed for Stability):**
```python
max_new_tokens=256
do_sample=False  # Deterministic
temperature=0.0
top_p=1.0
num_beams=1  # No beam search on T4 for speed
early_stopping=True
use_cache=False
```

**M2M Caching Pattern:**
```python
_M2M_CACHE = {'tokenizer': None, 'model': None}
# Load once, reuse globally — critical for performance
```

## Colab-Specific Patterns & Workarounds

### Dependency Conflicts
- **Pillow 9.5.0 must be installed AFTER all other deps** to prevent Gradio version conflicts
- Use `pip install --force-reinstall --no-cache-dir --no-deps pillow==9.5.0`
- Install without upgrade flag: `pip install -q --no-warn-conflicts ...`

### Gradio Monkey-Patch
Gradio 4.44.1 on Colab has a schema parsing bug with boolean types:
```python
from gradio_client import utils as gc_utils
def patched_get_type(schema):
    if isinstance(schema, bool):
        return "any"
    return original_get_type(schema)
gc_utils.get_type = patched_get_type
```

### Cloudflared Tunnel
- Downloaded via `wget` in setup cell, made executable
- Port selection: dynamic (check GRADIO_SERVER_PORT env var, else find free port)
- URL extracted via regex: `https://[a-z0-9-]+\.trycloudflare\.com`
- Subprocess stdout read with 30s timeout; restart logic on previous tunnel termination

## Audio Preprocessing Edge Cases & Debugging

**Silent/Quiet Audio:**
- No special detection; Phi-4 will attempt transcription even on near-silence
- May return hallucinated text or empty output
- Solution: Check audio levels before upload; recommend SNR > 10dB

**Very Short Audio (< 0.5s):**
- Processor may fail to extract meaningful features
- Fallback: returns error message from processor
- Recommendation: minimum 1-2 seconds of continuous speech

**Audio Distortion or Clipping:**
- No preprocessing normalization applied (intentional — preserve original dynamics)
- High-amplitude audio may cause processor overflow
- Solution: Users should normalize audio to -3dB peak before upload

**Sample Rate Extremes:**
- 8 kHz (telephony): Works but lower quality (too downsampled for ASR)
- 48 kHz (professional): Works; processor resamples to internal frequency (typically 16 kHz)
- Recommendation: 16 kHz optimal (standard ASR rate); 44.1 kHz also acceptable

**Stereo Music/Multi-Speaker:**
- Averaged to mono; loses channel separation
- May produce garbled transcription if speakers overlap heavily
- Solution: Pre-mix or use single-channel source

## Error Handling & Debugging

### Common Failure Points & Solutions

**`prepare_inputs_for_generation` Missing:**
- Model class lacks method during generation
- **Fix**: Pre-patch via sys.modules before `from_pretrained`, or add to instance via `types.MethodType`
- Includes fallback: return minimal dict with input_ids + optional tensors

**Audio Processing Failures:**
- Placeholder token mismatch (try both `<|audio|>` and `<|audio_1|>`)
- Audio too long → auto-truncate to 30s
- **Return strategy**: Multi-line error string with all attempted placeholders

**Decoding Failures:**
- `_safe_decode_text()` includes 3 fallback layers: processor.batch_decode → tokenizer.batch_decode → str() conversion
- Prevents silent failures in output generation

**M2M Translation Fallback:**
- Only used for Bengali output (other langs translate via Phi-4's multi-lingual capability)
- If M2M fails, returns transcription with error message

## File Structure & Editing Guidelines

- **Single-file structure**: All code in `phi4_asr_colab_demo_v2.ipynb`
- **Markdown cell sections**: Use for documentation (currently Portuguese)
- **Global variables**: `DEVICE`, `DTYPE`, `MODEL_ID`, `_M2M_CACHE`, `CF_PUBLIC_URL`, `CF_PROC`
- **Module-level imports**: Deferred until needed (except torch, soundfile at top)

### Adding Features
1. New language support: Update `SUPPORTED_SPEECH` dict and `INPUT_LANGS`/`OUTPUT_LANGS` Gradio dropdowns
2. New audio preprocessing: Modify audio loading in `phi4_transcribe_or_translate()` before processor call
3. UI changes: Edit Gradio blocks in cell 5 (button logic, markdown descriptions)

## Testing & Validation

**Manual Test Workflow:**
1. Run setup cell → verify Pillow version, check GPU availability
2. Run model load cell → confirm adapter activation, check dtype
3. Provide test audio < 30s → verify `<|audio|>` placeholder works
4. Check output with different language pairs (same-language vs. cross-language)
5. Verify Cloudflared tunnel publishes correct TryCloudflare URL

**Expected Outputs:**
- Transcription in source language
- Translation in target language (cross-language mode)
- Separate transcription + translation with `<sep>` token visible in debug (stripped for UI)

## Performance Optimization Notes

- **Float16 vs Float32**: T4 requires float16 for speed; CPU fallback uses float32
- **Offload Folder**: Critical for T4 memory management when loading 7B+ parameter models
- **Batch Size**: Always 1 (implicit in processor call); GPU cannot handle larger batches with audio
- **No KV Cache**: Simplifies generation, prevents custom cache errors
- **M2M Caching**: Single load per notebook session; reuse across multiple transcriptions saves ~5-10s per call
