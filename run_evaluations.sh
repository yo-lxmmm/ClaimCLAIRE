#!/bin/bash
# Commands to run each ablation (A0-A4) on the full dataset
# Usage: Run each command separately, or use this script as a reference
#
# IMPORTANT: Model Usage in Ablations
# ====================================
# The ablations use DIFFERENT models for different parts (same as full pipeline):
# 1. Decomposition (A2+ only): Hardcoded to claude-sonnet-4-20250514 (Anthropic)
#    - This is NOT configurable via --engine parameter
# 2. Main Agent/ReAct/Component Evaluation/Gap-filling: Uses --engine parameter
#    - Default: gemini-2.5-flash (configurable)
# 3. Reranking: Uses the same model as --engine parameter
#
# The --engine parameter only affects the main agent, not decomposition.

# Set common parameters
DATA_PATH="data/dev.json"
OUTPUT_DIR="results"
ENGINE="gemini-2.5-flash"  # Only affects main agent, not decomposition
MODEL_PROVIDER="google_genai"
NUM_RESULTS=10

# A0: Baseline RAG (simple search, no features)
python evaluate_ablations.py \
  --data-path "${DATA_PATH}" \
  --ablation A0 \
  --output-path "${OUTPUT_DIR}/ablation_A0_full.json" \
  --engine "${ENGINE}" \
  --model-provider "${MODEL_PROVIDER}" \
  --num-results "${NUM_RESULTS}"

# A1: A0 + ReAct Agent (adaptive evidence gathering)
python evaluate_ablations.py \
  --data-path "${DATA_PATH}" \
  --ablation A1 \
  --output-path "${OUTPUT_DIR}/ablation_A1_full.json" \
  --engine "${ENGINE}" \
  --model-provider "${MODEL_PROVIDER}" \
  --num-results "${NUM_RESULTS}"

# A2: A1 + Iterative Decomposition (with validation) [Contribution A]
python evaluate_ablations.py \
  --data-path "${DATA_PATH}" \
  --ablation A2 \
  --output-path "${OUTPUT_DIR}/ablation_A2_full.json" \
  --engine "${ENGINE}" \
  --model-provider "${MODEL_PROVIDER}" \
  --num-results "${NUM_RESULTS}"

# A3: A2 + Trust Weighting (source credibility scoring)
python evaluate_ablations.py \
  --data-path "${DATA_PATH}" \
  --ablation A3 \
  --output-path "${OUTPUT_DIR}/ablation_A3_full.json" \
  --engine "${ENGINE}" \
  --model-provider "${MODEL_PROVIDER}" \
  --num-results "${NUM_RESULTS}"

# A4: A3 + Gap-Filling + Support Verification [Contribution B] (full system)
python evaluate_ablations.py \
  --data-path "${DATA_PATH}" \
  --ablation A4 \
  --output-path "${OUTPUT_DIR}/ablation_A4_full.json" \
  --engine "${ENGINE}" \
  --model-provider "${MODEL_PROVIDER}" \
  --num-results "${NUM_RESULTS}"

