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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_TASKS = {"transcription", "generate"}
NON_RETRYABLE_EXCEPTIONS = (
    ValueError,
    json.JSONDecodeError,
    UnicodeDecodeError,
    TypeError,
)


def get_target_memory_utilization(target_ratio: float = 0.30) -> float:
    if not torch.cuda.is_available():
        return target_ratio

    device = torch.cuda.current_device()
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    free_ratio = free_memory / total_memory

    logger.info(
        "GPU Memory Free: %.1f%% (%.2fGB / %.2fGB)",
        free_ratio * 100,
        free_memory / 1e9,
        total_memory / 1e9,
    )

    if free_ratio >= target_ratio:
        logger.info(
            "Sufficient memory available. Allocating target utilization: %.1f%%",
            target_ratio * 100,
        )
        return target_ratio

    # If less than target is available, allocate slightly less than free ratio to avoid OOM.
    safe_ratio = max(0.1, free_ratio - 0.05)
    logger.warning("Memory is constrained. Allocating safe utilization: %.1f%%", safe_ratio * 100)
    return safe_ratio


def load_vllm_model(model_name: str):
    utilization_ratio = get_target_memory_utilization(target_ratio=0.30)
    logger.info(
        "Loading vLLM model %s with gpu_memory_utilization=%.2f...",
        model_name,
        utilization_ratio,
    )
    return LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=utilization_ratio,
        # Default limits for audio capabilities (ignored safely by text models).
        limit_mm_per_prompt={"audio": 1},
    )


def unload_vllm_model(llm_instance: Optional[Any]) -> None:
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
        parts = []
        for chunk in msg.body:
            if isinstance(chunk, bytes):
                parts.append(chunk.decode("utf-8"))
            elif isinstance(chunk, str):
                parts.append(chunk)
            else:
                raise TypeError(f"Unsupported message body chunk type: {type(chunk)}")
        return json.loads("".join(parts))

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


def safe_complete(receiver: Any, msg: Any) -> None:
    try:
        receiver.complete_message(msg)
        logger.info("Successfully processed and completed message id %s", msg.message_id)
    except Exception:
        logger.exception("Failed to complete message id %s", getattr(msg, "message_id", "<unknown>"))


def safe_abandon(receiver: Any, msg: Any) -> None:
    try:
        receiver.abandon_message(msg)
        logger.warning("Abandoned message id %s for retry", getattr(msg, "message_id", "<unknown>"))
    except Exception:
        logger.exception("Failed to abandon message id %s", getattr(msg, "message_id", "<unknown>"))


def safe_dead_letter(receiver: Any, msg: Any, reason: str, description: str) -> None:
    try:
        receiver.dead_letter_message(
            msg,
            reason=reason[:128],
            error_description=description[:1024],
        )
        logger.warning(
            "Dead-lettered message id %s reason=%s",
            getattr(msg, "message_id", "<unknown>"),
            reason,
        )
    except Exception:
        logger.exception("Failed to dead-letter message id %s", getattr(msg, "message_id", "<unknown>"))


def main() -> None:
    connection_str = os.getenv("SERVICEBUS_CONNECTION_STRING")
    queue_name = os.getenv("SERVICEBUS_QUEUE_NAME")
    default_model_name = os.getenv("MODEL_NAME", "openai/whisper-large-v3")

    if not connection_str or not queue_name:
        logger.error("Missing Service Bus connection environment variables.")
        return

    logger.info("Connecting to Azure Service Bus...")

    current_llm = None
    current_model_name = None

    try:
        with ServiceBusClient.from_connection_string(connection_str) as client:
            with client.get_queue_receiver(queue_name=queue_name, max_wait_time=5) as receiver:
                for msg in receiver:
                    try:
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

                        preview = result_text[:160].replace("\n", " ")
                        logger.info(
                            "Result summary task=%s model=%s chars=%d preview=%r",
                            task,
                            target_model_name,
                            len(result_text),
                            preview,
                        )

                        # TODO: Push result to output topic/database here.
                        safe_complete(receiver, msg)

                    except Exception as e:
                        logger.error(
                            "Error processing message id %s: %s",
                            getattr(msg, "message_id", "<unknown>"),
                            e,
                            exc_info=True,
                        )
                        if isinstance(e, NON_RETRYABLE_EXCEPTIONS):
                            safe_dead_letter(receiver, msg, reason="InvalidMessage", description=str(e))
                        else:
                            safe_abandon(receiver, msg)
    finally:
        unload_vllm_model(current_llm)


if __name__ == "__main__":
    main()