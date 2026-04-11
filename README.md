# vLLM Worker for AKS + KEDA

Minimal event-driven worker for vLLM models using Azure Service Bus.

**Supports both GPU and CPU inference modes.**

## What It Does
- Consumes queue messages from Azure Service Bus.
- Routes by task:
    - `transcription` (audio_url required, prompt optional)
    - `generate` (prompt required)
- Reuses the currently loaded model when the next message uses the same model.
- **Auto-detects GPU/CPU and adapts performance settings accordingly.**

## Queue Payloads

Transcription:

```json
{
    "task": "transcription",
    "model": "openai/whisper-large-v3",
    "audio_url": "file:///workspace/audio/sample1.wav"
}
```

Generation:

```json
{
    "task": "generate",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "Summarize this text.",
    "max_tokens": 256,
    "temperature": 0.0
}
```

## Local Run

### GPU Mode (Default)

Build image:

```bash
docker build -t vllm-worker .
```

Run container:

```bash
docker run --rm --runtime nvidia --gpus all \
    --ipc=host \
    -e SERVICEBUS_CONNECTION_STRING="<connection-string>" \
    -e SERVICEBUS_QUEUE_NAME="<queue-name>" \
    -e MODEL_NAME="openai/whisper-large-v3" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/workspace \
    -w /workspace \
    vllm-worker
```

### CPU Mode

Build CPU-only image:

```bash
docker build -f Dockerfile.cpu -t vllm-worker:cpu .
```

Run container (no GPU required):

```bash
docker run --rm \
    -e SERVICEBUS_CONNECTION_STRING="<connection-string>" \
    -e SERVICEBUS_QUEUE_NAME="<queue-name>" \
    -e MODEL_NAME="openai/whisper-large-v3" \
    -e DEVICE=cpu \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/workspace \
    -w /workspace \
    vllm-worker:cpu
```

### Without Docker (Direct Python)

1. **Setup virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **GPU mode (requires NVIDIA GPU):**
   ```bash
   export SERVICEBUS_CONNECTION_STRING="<connection-string>"
   export SERVICEBUS_QUEUE_NAME="<queue-name>"
   export MODEL_NAME="openai/whisper-large-v3"
   # DEVICE auto-detects GPU if available
   python src/worker.py
   ```

4. **CPU mode:**
   ```bash
   export SERVICEBUS_CONNECTION_STRING="<connection-string>"
   export SERVICEBUS_QUEUE_NAME="<queue-name>"
   export MODEL_NAME="openai/whisper-large-v3"
   export DEVICE=cpu
   python src/worker.py
   ```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICEBUS_CONNECTION_STRING` | Required | Azure Service Bus connection string |
| `SERVICEBUS_QUEUE_NAME` | Required | Queue name to consume messages from |
| `MODEL_NAME` | `openai/whisper-large-v3` | Default vLLM model to load |
| `DEVICE` | `auto` | Device to use: `auto`, `cuda`, `gpu`, or `cpu` |

## Device Auto-Detection & Configuration

The worker automatically detects the available hardware:

- **GPU available**: Uses GPU mode with optimized memory management
- **No GPU**: Falls back to CPU mode automatically
- **Explicit override**: Set `DEVICE=cpu` to force CPU mode even if GPU is available

## Performance Notes

### GPU Mode
- Optimal for large models (13B-70B)
- Uses ~30% of available VRAM by default
- Adjusts automatically if VRAM is constrained
- Supports all quantization methods (FP8, INT8, INT4, GPTQ, AWQ)

### CPU Mode
- Recommended for smaller models (7B-13B)
- Uses conservative 20% memory utilization for stability
- Slower inference (~5-10x vs GPU) but lower power consumption
- Suitable for:
  - Development and testing
  - Environments without GPU
  - Low-power deployments (edge/embedded systems)
  - Cost-sensitive scenarios

**For CPU mode, consider using quantized models** (GPTQ, AWQ, INT8) to reduce memory footprint and speed up inference.

## AKS + KEDA Deploy

1. Enable KEDA in AKS.
2. Create secret with Service Bus connection string.
3. Apply [k8s/scaledjob.yaml](k8s/scaledjob.yaml).

```bash
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/scaledjob.yaml
```

### CPU-Only Deployment

For CPU-only AKS nodes, update the deployment manifest to use the CPU image and remove GPU requests:

```yaml
spec:
  template:
    spec:
      containers:
      - name: vllm-worker
        image: vllm-worker:cpu  # CPU variant
        env:
        - name: DEVICE
          value: "cpu"
        resources:
          requests:
            cpu: "4"        # Adjust based on available cores
            memory: "8Gi"   # 8GB for 7B-13B models
          limits:
            cpu: "8"
            memory: "16Gi"
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Tests include:
- GPU/CPU memory utilization logic
- Device detection
- Model loading and switching
- Message queue handling
- Error routing (dead-letter vs retry)
