FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output
USER algorithm

WORKDIR /opt/algorithm
COPY --chown=algorithm:algorithm weights/ /opt/algorithm/weights/
# COPY staff/honeyk/.cache/torch/hub/checkpoints/adv-efficientnet-b5-86493f6b.pth /home/algorithm/.cache/torch/hub/checkpoints/adv-efficientnet-b5-86493f6b.pth

ENV PATH="/home/algorithm/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

RUN python3 -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python3 -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm model.py /opt/algorithm/

ENTRYPOINT ["python3", "process.py"]

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=picai_unet_baseline_processor
