# vLLM Worker for AKS + KEDA

Minimal event-driven worker for vLLM models using Azure Service Bus.

## What It Does
- Consumes queue messages from Azure Service Bus.
- Routes by task:
    - `transcription` (audio_url required, prompt optional)
    - `generate` (prompt required)
- Reuses the currently loaded model when the next message uses the same model.

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

## AKS + KEDA Deploy

1. Enable KEDA in AKS.
2. Create secret with Service Bus connection string.
3. Apply [k8s/scaledjob.yaml](k8s/scaledjob.yaml).

```bash
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/scaledjob.yaml
```
