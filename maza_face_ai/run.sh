#!/usr/bin/with-contenv bashio
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG_LEVEL=$(bashio::config 'log_level' 'info')
MAX_RUNTIME=$(bashio::config 'max_runtime_sec' '60')
bashio::log.info "MAZA FACE AI start (log=${LOG_LEVEL}, max_runtime=${MAX_RUNTIME}s)"
export MAZA_LOG_LEVEL="${LOG_LEVEL}" MAZA_MAX_RUNTIME="${MAX_RUNTIME}"
exec /opt/venv/bin/python /app/main.py
