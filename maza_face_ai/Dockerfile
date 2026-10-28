# Debian slim base: apt available, manylinux wheels for OpenCV available
FROM python:3.12-slim-bookworm

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VIRTUAL_ENV_DISABLE_PROMPT=1

# libgomp1 required by OpenCV; curl for model download
# Home Assistant add-on image labels (picked up by Supervisor)
ARG BUILD_VERSION="0.1.3"
ARG BUILD_ARCH="amd64"
LABEL io.hass.version="${BUILD_VERSION}" \
      io.hass.type="app" \
      io.hass.arch="${BUILD_ARCH}"
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 curl ca-certificates ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# YuNet + SFace with pinned SHA-256 verification
RUN mkdir -p /app/models \
 && curl -fsSL -o /app/models/face_detection_yunet_2023mar.onnx \
      https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
 && curl -fsSL -o /app/models/face_recognition_sface_2021dec.onnx \
      https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
 && echo "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  /app/models/face_detection_yunet_2023mar.onnx" > /tmp/sums \
 && echo "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79  /app/models/face_recognition_sface_2021dec.onnx" >> /tmp/sums \
 && sha256sum -c /tmp/sums \
 && rm -f /tmp/sums

COPY run.sh /run.sh
COPY app /app
COPY tests /app/tests
RUN chmod a+x /run.sh

WORKDIR /app
CMD [ "/run.sh" ]
