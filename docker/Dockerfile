FROM nvcr.io/nvidia/nvhpc:23.5-devel-cuda_multi-ubuntu22.04
ENV DEBIAN_FRONTEND="noninteractive"
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y \
  python3-pip \
  git \
  wget \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir --upgrade pip

WORKDIR /app

# clone the repository and get model
RUN git clone https://github.com/jspenmar/slowtv_monodepth.git slow-tv
RUN mkdir slow-tv/models && wget --output-document slow-tv/models/kbr.ckpt 'https://www.dropbox.com/s/o8j4wyhnhgvh4o7/kbr.ckpt?dl=0'

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH /app/slow-tv
ENV DEVICE cpu

VOLUME /images

ENTRYPOINT python3 /app/slow-tv/api/quickstart/run.py \
  --ckpt-file /app/slow-tv/models/kbr.ckpt \
  --device $DEVICE \
  --img-dir /images/in \
  --out-dir /images/out