FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Configuration
RUN echo "PYTHONUNBUFFERED=1" >> /etc/environment && \
    echo "OMP_NUM_THREADS=1" >> /etc/environment

# Update apt-get
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
WORKDIR /opt/algorithm
COPY . /opt/algorithm/ITUNet-for-PICAI-2022-Challenge
RUN pip install --upgrade pip && \
    pip install -e /opt/algorithm/ITUNet-for-PICAI-2022-Challenge
