FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Python 별칭 설정
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# 필요한 Python 패키지 설치
RUN pip3 install --no-cache-dir --upgrade pip

# 필수 패키지 설치
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

# Demucs 설치
RUN pip3 install --no-cache-dir demucs

# 작업 디렉토리 생성
WORKDIR /app

# 애플리케이션 파일 복사
COPY serverless.py /app/
COPY runpod_handler.py /app/

# 필요한 경우 모델 사전 다운로드
RUN python3 -c "from demucs.pretrained import get_model; get_model('htdemucs')"

# 실행 명령
CMD ["python", "-u", "runpod_handler.py"] 