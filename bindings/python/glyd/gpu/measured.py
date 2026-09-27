# Glyd's measured sizes, from getglyd.com's data/sizes.json and scripts/site_data.py's MODELS (2026-09-27).
RATIOS = {  # the tiered layout's bytes over bf16's, measured on the model's Linear layers
    "HuggingFaceTB/SmolLM3-3B": 0.671,
    "meta-llama/Llama-3.2-3B-Instruct": 0.672,
    "Qwen/Qwen3-4B-Instruct-2507": 0.678,
    "mistralai/Mistral-7B-Instruct-v0.3": 0.673,
    "meta-llama/Llama-3.1-8B-Instruct": 0.672,
    "google/gemma-3-12b-it": 0.671,
    "microsoft/phi-4": 0.671,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 0.681,
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": 0.67,
    "google/gemma-4-26B-A4B-it": 0.672,
    "google/gemma-3-27b-it": 0.672,
    "Qwen/Qwen3.8-27B": 0.672,
    "meta-models/Muse-Glimmer-30B": 0.672,
    "Qwen/Qwen3-30B-A3B": 0.673,
    "meta-llama/Llama-3.3-70B-Instruct": 0.671,
    "Qwen/Qwen3-Next-80B-A3B-Instruct": 0.678,
    "zai-org/GLM-4.5-Air": 0.67,
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": 0.671,
}
TOTALS = {  # an end-to-end run's weights, bf16 and Glyd, GB
    "Qwen/Qwen3-8B": (16.38, 11.15),
    "Qwen/Qwen3-32B": (65.5, 44.5),
    "Qwen/Qwen2.5-72B-Instruct": (145.4, 97.8),
}
