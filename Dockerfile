FROM cschranz/gpu-jupyter:v1.7_cuda-12.2_ubuntu-22.04_python-only

WORKDIR ${HOME}
RUN mkdir open_asr_leaderboard
COPY ./requirements open_asr_leaderboard/requirements

RUN ${CONDA_DIR}/bin/conda create \
  --prefix ${HOME}/open_asr_leaderboard/.venv \
  --yes \
  python=3.9

RUN ./open_asr_leaderboard/.venv/bin/pip install --no-cache -r open_asr_leaderboard/requirements/requirements.txt && \
  ./open_asr_leaderboard/.venv/bin/pip install --no-cache -r open_asr_leaderboard/requirements/requirements_ctranslate2.txt

COPY ./normalizer open_asr_leaderboard/normalizer
COPY ./ctranslate2 open_asr_leaderboard/ctranslate2
