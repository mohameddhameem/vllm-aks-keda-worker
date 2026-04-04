import os
import gc
import json
import torch
import logging
from typing import Any, Dict, Optional, Tuple
from azure.servicebus import ServiceBusClient
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


SUPPORTED_TASKS = {"transcription", "generate"}

def get_target_memory_utilization(target_ratio=0.30):
    if not torch.cuda.is_available():
        return target_ratio
    
    device = torch.cuda.current_device()
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    free_ratio = free_memory / total_memory
    
    logger.info(f"GPU Memory Free: {free_ratio:.1%} ({free_memory/1e9:.2f}GB / {total_memory/1e9:.2f}GB)")
    
    if free_ratio >= target_ratio:
        logger.info(f"Sufficient memory available. Allocating target utilization: {target_ratio:.1%}")
        return target_ratio
    else:
        # If less than 30% is available, we allocate slightly less than the free ratio to avoid OOM
        safe_ratio = max(0.1, free_ratio - 0.05)
        logger.warning(f"Memory is constrained. Allocating safe utilization: {safe_ratio:.1%}")
        return safe_ratio

def load_vllm_model(model_name):
    utilization_ratio = get_target_memory_utilization(target_ratio=0.30)
    logger.info(f"Loading vLLM model {model_name} with gpu_memory_utilization={utilization_ratio:.2f}...")
    return LLM(
        model=model_name, 
        trust_remote_code=True,
        gpu_memory_utilization=utilization_ratio,
        # Default limits for audio capabilities (ignored safely by text models like Qwen)
        limit_mm_per_prompt={"audio": 1}
    )

def unload_vllm_model(llm_instance):
    if llm_instance is not None:
        logger.info("Unloading previous vLLM model to free GPU memory...")
        destroy_model_parallel()
        del llm_instance
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def decode_message_body(msg: Any) -> Dict[str, Any]:
    if hasattr(msg, "body"):
        body_raw = b"".join(list(msg.body))
        return json.loads(body_raw.decode("utf-8"))
    return json.loads(str(msg))


def resolve_task(body: Dict[str, Any], default_model_name: str) -> Tuple[str, str]:
    task = str(body.get("task", "transcription")).strip().lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task}'. Supported tasks: {sorted(SUPPORTED_TASKS)}")

    model_name = body.get("model", default_model_name)
    if not model_name:
        raise ValueError("Missing model name and MODEL_NAME default is not set")

    return task, model_name


def build_sampling_params(body: Dict[str, Any]) -> SamplingParams:
    return SamplingParams(
        temperature=float(body.get("temperature", 0.0)),
        max_tokens=int(body.get("max_tokens", 256)),
    )


def run_transcription(llm: Any, body: Dict[str, Any]) -> str:
    audio_url = body.get("audio_url")
    if not audio_url:
        raise ValueError("Missing required field 'audio_url' for transcription task")

    prompt_text = body.get("prompt", "")
    payload = {
        "prompt": prompt_text,
        "multi_modal_data": {"audio": audio_url},
    }
    outputs = llm.generate([payload], sampling_params=build_sampling_params(body))
    return outputs[0].outputs[0].text


def run_generate(llm: Any, body: Dict[str, Any]) -> str:
    prompt_text = body.get("prompt")
    if not prompt_text:
        raise ValueError("Missing required field 'prompt' for generate task")

    outputs = llm.generate([prompt_text], sampling_params=build_sampling_params(body))
    return outputs[0].outputs[0].text


def ensure_model_loaded(
    current_llm: Optional[Any],
    current_model_name: Optional[str],
    target_model_name: str,
) -> Tuple[Any, str]:
    if current_model_name == target_model_name and current_llm is not None:
        logger.info("Reusing already loaded model: %s", current_model_name)
        return current_llm, current_model_name

    if current_model_name is not None:
        logger.info("Switching model from %s to %s", current_model_name, target_model_name)
    unload_vllm_model(current_llm)
    new_llm = load_vllm_model(target_model_name)
    return new_llm, target_model_name

def main():
    connection_str = os.getenv("SERVICEBUS_CONNECTION_STRING")
    queue_name = os.getenv("SERVICEBUS_QUEUE_NAME")
    default_model_name = os.getenv("MODEL_NAME", "openai/whisper-large-v3")

    if not connection_str or not queue_name:
        logger.error("Missing Service Bus connection environment variables.")
        return

    logger.info("Connecting to Azure Service Bus...")
    
    current_llm = None
    current_model_name = None

    with ServiceBusClient.from_connection_string(connection_str) as client:
        with client.get_queue_receiver(queue_name=queue_name, max_wait_time=5) as receiver:
            for msg in receiver:
                try:
                    # {"task": "transcription", "model": "openai/whisper-large-v3", "audio_url": "file:///workspace/audio/a.wav"}
                    # {"task": "generate", "model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "Summarize this text"}
                    body = decode_message_body(msg)
                    task, target_model_name = resolve_task(body, default_model_name)

                    logger.info(
                        "Processing message id %s with task=%s model=%s",
                        msg.message_id,
                        task,
                        target_model_name,
                    )

                    current_llm, current_model_name = ensure_model_loaded(
                        current_llm,
                        current_model_name,
                        target_model_name,
                    )

                    if task == "transcription":
                        result_text = run_transcription(current_llm, body)
                    else:
                        result_text = run_generate(current_llm, body)

                    logger.info("Result [%s|%s]: %s", task, target_model_name, result_text)
                    # TODO: Push result to output topic/database here.
                    
                    # Complete message to remove it from the queue
                    receiver.complete_message(msg)
                    logger.info("Successfully processed and completed message id %s", msg.message_id)

                except Exception as e:
                    logger.error("Error processing message: %s", e, exc_info=True)
                    # Abandon message so it can be retried or dead-lettered
                    receiver.abandon_message(msg)

if __name__ == "__main__":
    main()
