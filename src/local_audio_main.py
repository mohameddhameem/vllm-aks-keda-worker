import argparse
from pathlib import Path
from typing import Literal

import torch
from vllm import LLM, SamplingParams


def detect_device(requested: str) -> Literal["cuda", "cpu"]:
    value = requested.strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value in {"cuda", "gpu"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return "cpu"
    raise ValueError("--device must be one of: auto, cuda, gpu, cpu")


def get_memory_utilization(device: Literal["cuda", "cpu"]) -> float:
    if device == "cpu":
        return 0.20

    free_mem, total_mem = torch.cuda.mem_get_info(torch.cuda.current_device())
    free_ratio = free_mem / total_mem
    target_ratio = 0.30
    if free_ratio >= target_ratio:
        return target_ratio
    return max(0.1, free_ratio - 0.05)


def to_file_url(audio_path: str) -> str:
    resolved = Path(audio_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved}")
    return resolved.as_uri()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local vLLM transcription on one audio file")
    parser.add_argument("--audio", required=True, help="Path to local audio file")
    parser.add_argument("--model", default="openai/whisper-large-v3", help="Model name")
    parser.add_argument("--prompt", default="", help="Optional transcription prompt")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto | cuda | gpu | cpu")
    args = parser.parse_args()

    device = detect_device(args.device)
    audio_url = to_file_url(args.audio)

    llm_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "device": device,
    }
    if device == "cuda":
        llm_kwargs["gpu_memory_utilization"] = get_memory_utilization(device)
        llm_kwargs["limit_mm_per_prompt"] = {"audio": 1}

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    payload = {
        "prompt": args.prompt,
        "multi_modal_data": {"audio": audio_url},
    }
    outputs = llm.generate([payload], sampling_params=sampling)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()