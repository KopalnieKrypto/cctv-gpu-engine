FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip curl ffmpeg \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY test_video_pose.py run_test.sh ./
RUN chmod +x run_test.sh

CMD ["./run_test.sh"]
