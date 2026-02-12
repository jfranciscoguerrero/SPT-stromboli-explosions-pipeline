FROM tensorflow/tensorflow:2.9.0

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip && \
    grep -vE '^(tensorflow|keras)=' requirements.txt > /tmp/requirements_no_tf.txt && \
    pip install --no-cache-dir -r /tmp/requirements_no_tf.txt

CMD ["bash"]
