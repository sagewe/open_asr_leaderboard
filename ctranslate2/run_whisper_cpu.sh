#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=(
  "small"
  "/home/sage/faster-whisper-server/models/Systran/faster-distil-whisper-large-v3"
  "/home/sage/open_asr_leaderboard/models/distil-v3-cwj"
  "/home/sage/faster-whisper-server/models/deepdml/faster-whisper-large-v3-turbo-ct2"
  "/home/sage/faster-whisper-server/models/Systran/faster-whisper-large-v3"
)
DEVICE_INDEX=0
DEVICE="cpu"

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="google/fleurs" \
        --dataset="ar_eg" \
        --split="test" \
        --device_index=${DEVICE_INDEX} \
        --device=${DEVICE} \
        --no-streaming \
        --cpu_threads 4 \
        --max_eval_samples -1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="mozilla-foundation/common_voice_15_0" \
        --dataset="ar" \
        --split="test" \
        --device_index=${DEVICE_INDEX} \
        --device=${DEVICE} \
        --no-streaming \
        --cpu_threads 4 \
        --max_eval_samples -1

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
