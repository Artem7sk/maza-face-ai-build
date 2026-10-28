#!/bin/sh
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

LOG_LEVEL="info"
MAX_RUNTIME="60"
MODE="synthetic"
INPUT_PATH=""

if [ -f /data/options.json ]; then
  LOG_LEVEL=$(python3 -c "import json;print(json.load(open('/data/options.json')).get('log_level','info'))" 2>/dev/null || echo info)
  MAX_RUNTIME=$(python3 -c "import json;print(json.load(open('/data/options.json')).get('max_runtime_sec',60))" 2>/dev/null || echo 60)
  MODE=$(python3 -c "import json;print(json.load(open('/data/options.json')).get('mode','synthetic'))" 2>/dev/null || echo synthetic)
  INPUT_PATH=$(python3 -c "import json;print(json.load(open('/data/options.json')).get('input_path','') or '')" 2>/dev/null || echo "")
fi

echo "[INFO] MAZA FACE AI start (log=${LOG_LEVEL}, max_runtime=${MAX_RUNTIME}s, mode=${MODE})"
[ -n "${INPUT_PATH}" ] && echo "[INFO] input_path=${INPUT_PATH}"

export MAZA_LOG_LEVEL="${LOG_LEVEL}" MAZA_MAX_RUNTIME="${MAX_RUNTIME}"
export MAZA_MODE="${MODE}" MAZA_INPUT_PATH="${INPUT_PATH}"
exec python3 /app/main.py
