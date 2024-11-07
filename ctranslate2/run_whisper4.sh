#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=(
  #"/home/sage/faster-whisper-server/models/Systran/faster-distil-whisper-large-v3"
  #"/home/sage/open_asr_leaderboard/models/distil-v3-cwj"
  "/home/sage/faster-whisper-server/models/deepdml/faster-whisper-large-v3-turbo-ct2"
  #"/home/sage/faster-whisper-server/models/Systran/faster-whisper-large-v3"
)
DEVICE_INDEX=0

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}


    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
