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
RUN git clone https://github.com/Yukiya-Umimi/ITUNet-for-PICAI-2022-Challenge
ENV PATH="/opt/algorithm/ITUNet-for-PICAI-2022-Challenge:${PATH}"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /opt/algorithm/ITUNet-for-PICAI-2022-Challenge/segmentation/requirements.txt

RUN chmod +x /opt/algorithm/ITUNet-for-PICAI-2022-Challenge/train_2_segmentation.py && \
    ln -s /opt/algorithm/ITUNet-for-PICAI-2022-Challenge/train_2_segmentation.py /usr/local/bin/train
