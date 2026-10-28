ARG BUILD_FROM
FROM ${BUILD_FROM}

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apk add --no-cache python3 py3-pip libstdc++ libgomp curl \
 && python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip

COPY requirements.txt /tmp/
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt

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
RUN chmod a+x /run.sh

WORKDIR /app
CMD [ "/run.sh" ]
