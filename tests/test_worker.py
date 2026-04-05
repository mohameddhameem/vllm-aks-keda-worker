import json
import os
import pytest
from unittest.mock import patch, MagicMock

# Attempt to mock vLLM and azure imports before loading the worker module
# so we can run tests without requiring a GPU or massive dependencies locally.
import sys

# Add src folder to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

sys.modules["torch"] = MagicMock()
sys.modules["vllm"] = MagicMock()
sys.modules["vllm.distributed.parallel_state"] = MagicMock()
sys.modules["azure.servicebus"] = MagicMock()

import worker


def _build_servicebus_mocks(messages):
    receiver = MagicMock()
    receiver.__iter__.return_value = iter(messages)

    receiver_cm = MagicMock()
    receiver_cm.__enter__.return_value = receiver
    receiver_cm.__exit__.return_value = False

    client = MagicMock()
    client.get_queue_receiver.return_value = receiver_cm

    client_cm = MagicMock()
    client_cm.__enter__.return_value = client
    client_cm.__exit__.return_value = False

    return client_cm, receiver


def _make_msg(message_id, payload):
    msg = MagicMock()
    msg.message_id = message_id
    if isinstance(payload, list):
        msg.body = payload
    else:
        msg.body = [json.dumps(payload).encode("utf-8")]
    return msg


def test_resolve_task_defaults_to_transcription():
    task, model = worker.resolve_task({}, "openai/whisper-large-v3")
    assert task == "transcription"
    assert model == "openai/whisper-large-v3"


def test_resolve_task_rejects_unsupported_task():
    with pytest.raises(ValueError):
        worker.resolve_task({"task": "unknown"}, "openai/whisper-large-v3")


def test_ensure_model_loaded_reuses_same_model():
    llm = MagicMock()
    out_llm, out_model = worker.ensure_model_loaded(
        llm, "openai/whisper-large-v3", "openai/whisper-large-v3"
    )
    assert out_llm is llm
    assert out_model == "openai/whisper-large-v3"


@patch("worker.load_vllm_model")
@patch("worker.unload_vllm_model")
def test_ensure_model_loaded_switches_model(mock_unload, mock_load):
    old_llm = MagicMock()
    new_llm = MagicMock()
    mock_load.return_value = new_llm

    out_llm, out_model = worker.ensure_model_loaded(
        old_llm, "openai/whisper-large-v3", "Qwen/Qwen2.5-7B-Instruct"
    )

    mock_unload.assert_called_once_with(old_llm)
    mock_load.assert_called_once_with("Qwen/Qwen2.5-7B-Instruct")
    assert out_llm is new_llm
    assert out_model == "Qwen/Qwen2.5-7B-Instruct"


def test_run_transcription_uses_optional_prompt_default():
    llm = MagicMock()
    llm.generate.return_value = [MagicMock(outputs=[MagicMock(text="hello")])]

    result = worker.run_transcription(llm, {"audio_url": "file:///workspace/audio/a.wav"})

    call_args = llm.generate.call_args[0][0][0]
    assert call_args["prompt"] == ""
    assert call_args["multi_modal_data"]["audio"] == "file:///workspace/audio/a.wav"
    assert result == "hello"


@patch("worker.torch.cuda")
def test_get_target_memory_utilization_no_gpu(mock_cuda):
    mock_cuda.is_available.return_value = False
    ratio = worker.get_target_memory_utilization(0.30)
    assert ratio == 0.30


@patch("worker.torch.cuda")
def test_get_target_memory_utilization_sufficient_memory(mock_cuda):
    mock_cuda.is_available.return_value = True
    mock_cuda.current_device.return_value = 0
    mock_cuda.mem_get_info.return_value = (8_000_000_000, 10_000_000_000)
    ratio = worker.get_target_memory_utilization(0.30)
    assert ratio == 0.30


@patch("worker.torch.cuda")
def test_get_target_memory_utilization_constrained_memory(mock_cuda):
    mock_cuda.is_available.return_value = True
    mock_cuda.current_device.return_value = 0
    mock_cuda.mem_get_info.return_value = (2_500_000_000, 10_000_000_000)
    ratio = worker.get_target_memory_utilization(0.30)
    assert abs(ratio - 0.20) < 0.001


@patch("worker.get_target_memory_utilization")
@patch("worker.LLM")
def test_load_vllm_model(mock_llm, mock_get_util):
    mock_get_util.return_value = 0.25
    model_name = "test/fake-model"

    worker.load_vllm_model(model_name)

    mock_llm.assert_called_once_with(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.25,
        limit_mm_per_prompt={"audio": 1},
    )


def test_decode_message_body_with_bytes_chunks():
    msg = _make_msg("m1", {"task": "generate", "prompt": "hello"})
    out = worker.decode_message_body(msg)
    assert out["task"] == "generate"
    assert out["prompt"] == "hello"


def test_decode_message_body_with_str_chunks():
    msg = MagicMock()
    msg.body = ['{"task":"generate","prompt":"hello"}']
    out = worker.decode_message_body(msg)
    assert out["task"] == "generate"
    assert out["prompt"] == "hello"


def test_decode_message_body_invalid_chunk_type_raises():
    msg = MagicMock()
    msg.body = [123]
    with pytest.raises(TypeError):
        worker.decode_message_body(msg)


def test_safe_complete_swallow_settlement_error():
    receiver = MagicMock()
    msg = MagicMock(message_id="m1")
    receiver.complete_message.side_effect = RuntimeError("lock lost")
    worker.safe_complete(receiver, msg)


def test_safe_abandon_swallow_settlement_error():
    receiver = MagicMock()
    msg = MagicMock(message_id="m2")
    receiver.abandon_message.side_effect = RuntimeError("lock lost")
    worker.safe_abandon(receiver, msg)


def test_safe_dead_letter_truncates_and_calls_receiver():
    receiver = MagicMock()
    msg = MagicMock(message_id="m3")
    worker.safe_dead_letter(receiver, msg, reason="R" * 200, description="D" * 2000)

    receiver.dead_letter_message.assert_called_once()
    _, kwargs = receiver.dead_letter_message.call_args
    assert len(kwargs["reason"]) == 128
    assert len(kwargs["error_description"]) == 1024


@patch.dict(
    os.environ,
    {"SERVICEBUS_CONNECTION_STRING": "Endpoint=sb://x/", "SERVICEBUS_QUEUE_NAME": "q", "MODEL_NAME": "m"},
    clear=True,
)
@patch("worker.ServiceBusClient.from_connection_string")
@patch("worker.unload_vllm_model")
@patch("worker.ensure_model_loaded")
def test_main_non_retryable_goes_dead_letter(mock_ensure_loaded, mock_unload, mock_from_conn):
    msg = _make_msg("nr1", {"task": "generate"})
    client_cm, receiver = _build_servicebus_mocks([msg])

    mock_from_conn.return_value = client_cm
    mock_ensure_loaded.return_value = (MagicMock(), "m")

    worker.main()

    receiver.dead_letter_message.assert_called_once()
    receiver.abandon_message.assert_not_called()
    receiver.complete_message.assert_not_called()
    mock_unload.assert_called_once()


@patch.dict(
    os.environ,
    {"SERVICEBUS_CONNECTION_STRING": "Endpoint=sb://x/", "SERVICEBUS_QUEUE_NAME": "q", "MODEL_NAME": "m"},
    clear=True,
)
@patch("worker.ServiceBusClient.from_connection_string")
@patch("worker.unload_vllm_model")
@patch("worker.ensure_model_loaded")
@patch("worker.run_generate")
def test_main_retryable_goes_abandon(
    mock_run_generate, mock_ensure_loaded, mock_unload, mock_from_conn
):
    msg = _make_msg("r1", {"task": "generate", "prompt": "hello"})
    client_cm, receiver = _build_servicebus_mocks([msg])

    mock_from_conn.return_value = client_cm
    mock_ensure_loaded.return_value = (MagicMock(), "m")
    mock_run_generate.side_effect = RuntimeError("transient backend error")

    worker.main()

    receiver.abandon_message.assert_called_once()
    receiver.dead_letter_message.assert_not_called()
    receiver.complete_message.assert_not_called()
    mock_unload.assert_called_once()


@patch.dict(
    os.environ,
    {"SERVICEBUS_CONNECTION_STRING": "Endpoint=sb://x/", "SERVICEBUS_QUEUE_NAME": "q", "MODEL_NAME": "m"},
    clear=True,
)
@patch("worker.ServiceBusClient.from_connection_string")
@patch("worker.unload_vllm_model")
def test_main_malformed_json_goes_dead_letter(mock_unload, mock_from_conn):
    msg = _make_msg("badjson", [b"{not-valid-json"])
    client_cm, receiver = _build_servicebus_mocks([msg])

    mock_from_conn.return_value = client_cm

    worker.main()

    receiver.dead_letter_message.assert_called_once()
    receiver.abandon_message.assert_not_called()
    receiver.complete_message.assert_not_called()
    mock_unload.assert_called_once()


@patch.dict(os.environ, {}, clear=True)
@patch("worker.ServiceBusClient.from_connection_string")
def test_main_missing_env_returns_early(mock_from_conn):
    worker.main()
    mock_from_conn.assert_not_called()
