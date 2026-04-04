FROM vllm/vllm-openai:latest

# Install system dependencies required for handling audio
RUN apt-get update && apt-get install -y ffmpeg

# Copy and install python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy the worker script
COPY src/worker.py /workspace/src/worker.py

# Keep backwards compatibility for manually running the `vllm run-batch`
# or defaulting to our new queue processor when running as KEDA ScaledJob
ENTRYPOINT ["python", "/workspace/src/worker.py"]
