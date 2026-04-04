import pytest
from unittest.mock import patch, MagicMock

# Attempt to mock vLLM and azure imports before loading the worker module
# so we can run tests without requiring a GPU or massive dependencies locally
import sys
import os

# Add src folder to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

sys.modules['torch'] = MagicMock()
sys.modules['vllm'] = MagicMock()
sys.modules['vllm.distributed.parallel_state'] = MagicMock()
sys.modules['azure.servicebus'] = MagicMock()

# Now import the worker explicitly
import worker


def test_resolve_task_defaults_to_transcription():
    task, model = worker.resolve_task({}, "openai/whisper-large-v3")
    assert task == "transcription"
    assert model == "openai/whisper-large-v3"


def test_resolve_task_rejects_unsupported_task():
    with pytest.raises(ValueError):
        worker.resolve_task({"task": "unknown"}, "openai/whisper-large-v3")


def test_ensure_model_loaded_reuses_same_model():
    llm = MagicMock()
    out_llm, out_model = worker.ensure_model_loaded(llm, "openai/whisper-large-v3", "openai/whisper-large-v3")
    assert out_llm is llm
    assert out_model == "openai/whisper-large-v3"


@patch("worker.load_vllm_model")
@patch("worker.unload_vllm_model")
def test_ensure_model_loaded_switches_model(mock_unload, mock_load):
    old_llm = MagicMock()
    new_llm = MagicMock()
    mock_load.return_value = new_llm

    out_llm, out_model = worker.ensure_model_loaded(old_llm, "openai/whisper-large-v3", "Qwen/Qwen2.5-7B-Instruct")

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

@patch('worker.torch.cuda')
def test_get_target_memory_utilization_no_gpu(mock_cuda):
    # Arrange
    mock_cuda.is_available.return_value = False
    
    # Act
    ratio = worker.get_target_memory_utilization(0.30)
    
    # Assert
    assert ratio == 0.30

@patch('worker.torch.cuda')
def test_get_target_memory_utilization_sufficient_memory(mock_cuda):
    # Arrange
    mock_cuda.is_available.return_value = True
    mock_cuda.current_device.return_value = 0
    # Simulate 8GB free out of 10GB total (80% free, which is >= 30%)
    mock_cuda.mem_get_info.return_value = (8_000_000_000, 10_000_000_000)
    
    # Act
    ratio = worker.get_target_memory_utilization(0.30)
    
    # Assert
    assert ratio == 0.30

@patch('worker.torch.cuda')
def test_get_target_memory_utilization_constrained_memory(mock_cuda):
    # Arrange
    mock_cuda.is_available.return_value = True
    mock_cuda.current_device.return_value = 0
    # Simulate 2.5GB free out of 10GB total (25% free, < 30%)
    mock_cuda.mem_get_info.return_value = (2_500_000_000, 10_000_000_000)
    
    # Act
    ratio = worker.get_target_memory_utilization(0.30)
    
    # Assert
    # Safe ratio is max(0.1, 0.25 - 0.05) -> 0.20
    assert abs(ratio - 0.20) < 0.001

@patch('worker.get_target_memory_utilization')
@patch('worker.LLM')
def test_load_vllm_model(mock_llm, mock_get_util):
    # Arrange
    mock_get_util.return_value = 0.25
    model_name = "test/fake-model"
    
    # Act
    worker.load_vllm_model(model_name)
    
    # Assert
    mock_llm.assert_called_once_with(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.25,
        limit_mm_per_prompt={"audio": 1}
    )
