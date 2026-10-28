#!/bin/sh
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

LOG_LEVEL="info"
MAX_RUNTIME="60"

# Home Assistant writes addon options to /data/options.json
if [ -f /data/options.json ]; then
  LOG_LEVEL=$(python3 -c "import json;print(json.load(open('/data/options.json')).get('log_level','info'))" 2>/dev/null || echo info)
  MAX_RUNTIME=$(python3 -c "import json;print(json.load(open('/data/options.json')).get('max_runtime_sec',60))" 2>/dev/null || echo 60)
fi

echo "[INFO] MAZA FACE AI start (log=${LOG_LEVEL}, max_runtime=${MAX_RUNTIME}s)"
export MAZA_LOG_LEVEL="${LOG_LEVEL}" MAZA_MAX_RUNTIME="${MAX_RUNTIME}"
exec python3 /app/main.py
