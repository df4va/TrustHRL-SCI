FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.10 python3-pip && apt-get clean
WORKDIR /opt/trusthrl
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python3.10 -m pip install --no-deps .
ENTRYPOINT ["trusthrl-train"]

