#!/usr/bin/env bash
# GPU classes for the card and matrix scripts (source this): a class is a list of Runpod Secure Cloud GPU types in the order to try them.
# Prices: USD an hour, Secure Cloud, from Runpod's GPU type list on 2026-10-09 (they move). Used for estimates only.
class_types() {
  case $1 in
    4090) echo "NVIDIA GeForce RTX 4090";;
    5090) echo "NVIDIA GeForce RTX 5090";;   # 32 GB, PCIe 5, sm_120 (CUDA 12.8 on)
    3090) echo "NVIDIA GeForce RTX 3090";;
    16gb) printf '%s\n' "NVIDIA RTX 2000 Ada Generation" "NVIDIA RTX A4000";;
    l4) echo "NVIDIA L4";;
    a6000) echo "NVIDIA RTX A6000";;
    l40) printf '%s\n' "NVIDIA L40" "NVIDIA L40S";;
    a100) echo "NVIDIA A100-SXM4-80GB";;   # 80 GB: Nemotron 3 Super packed at wren (tools/cards/nh_run_check_job.sh)
    h200) printf '%s\n' "NVIDIA H200" "NVIDIA H200 NVL";;   # 141 GB: Nemotron 3 Super packed at linnet (tools/cards/nh_run_check_job.sh)
    # 4-GPU pods (class_gpus) for the routing-skew job (tools/cards/glm_skew.sh): GLM-5.3-Flash is 328 GB of FP8, so 4 x 80 GB (320 GB) does not hold it
    glm) printf '%s\n' "NVIDIA RTX PRO 6000 Blackwell Server Edition" "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" "NVIDIA H100 NVL" "NVIDIA H200 NVL" "NVIDIA H200";;
    glmh) printf '%s\n' "NVIDIA H100 NVL" "NVIDIA H200 NVL" "NVIDIA H200";;   # Hopper only, if the FP8 kernels fail on the Blackwell cards
    nemoa) printf '%s\n' "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe";;   # nemo without the H100s: a run held to the A100 price
    nemo) printf '%s\n' "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 PCIe" "NVIDIA H100 80GB HBM3";;   # Nemotron 3 Super bf16: 247 GB
    *) return 1;;
  esac
}
# GPUs a pod of the class has (rp_class.sh passes it as RP_GPUS; the prices below are per GPU)
class_gpus() { case $1 in glm|glmh|nemo|nemoa) echo 4;; *) echo 1;; esac; }
type_price() {
  case $1 in
    "NVIDIA GeForce RTX 4090") echo 0.89;;
    "NVIDIA GeForce RTX 5090") echo 1.19;;
    "NVIDIA GeForce RTX 3090") echo 0.50;;
    "NVIDIA RTX A4000") echo 0.25;;
    "NVIDIA RTX 2000 Ada Generation") echo 0.24;;
    "NVIDIA L4") echo 0.59;;
    "NVIDIA RTX A6000") echo 0.59;;
    "NVIDIA L40S") echo 1.09;;
    "NVIDIA L40") echo 0.82;;
    "NVIDIA A100-SXM4-80GB") echo 1.79;;
    "NVIDIA A100 80GB PCIe") echo 1.79;;
    "NVIDIA H100 PCIe") echo 2.89;;
    "NVIDIA H100 80GB HBM3") echo 3.99;;
    "NVIDIA H100 NVL") echo 3.19;;
    "NVIDIA H200 NVL") echo 3.79;;
    "NVIDIA H200") echo 5.29;;
    "NVIDIA RTX PRO 6000 Blackwell Server Edition") echo 2.49;;
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition") echo 2.59;;
  esac
}
# the hourly price of a class's first type, for all its GPUs (what a dry run quotes)
class_price() { awk -v p="$(type_price "$(class_types "$1" | head -1)")" -v g="$(class_gpus "$1")" 'BEGIN{printf "%.2f", p * g}'; }
