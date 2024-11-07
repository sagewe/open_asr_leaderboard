#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=(
  "../models/Systran/faster-distil-whisper-large-v3"
  "../models/deepdml/faster-whisper-large-v3-turbo-ct2"
  "../models/Systran/faster-whisper-large-v3"
)
DEVICE_INDEX=0
DEVICE="cuda"

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    ../.venv/bin/python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="../dataset/google/fleurs" \
        --split="test" \
        --device=${DEVICE} \
        --device_index=${DEVICE_INDEX} \
        --no-streaming \
        --max_eval_samples -1

    ../.venv/bin/python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="../dataset/mozilla-foundation/common_voice_15_0" \
        --split="test" \
        --device=${DEVICE} \
        --device_index=${DEVICE_INDEX} \
        --no-streaming \
        --max_eval_samples -1

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    ./.venv/bin/python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
