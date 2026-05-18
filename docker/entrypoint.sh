#!/bin/sh
set -e

export PYTHONPATH="${PYTHONPATH:-/app/src}"

case "${1:-test}" in
  test)
    exec pytest -q /app/tests "$@"
    ;;
  lint)
    exec ruff check /app/tests /app/src/scripts/mlops "$@"
    ;;
  streamlit)
    exec streamlit run /app/src/app/ui/streamlit_app.py \
      --server.port="${STREAMLIT_PORT:-8501}" \
      --server.address=0.0.0.0 \
      --browser.gatherUsageStats=false \
      "$@"
    ;;
  train)
    exec python /app/src/scripts/mlops/train.py "$@"
    ;;
  validate)
    exec python /app/src/scripts/mlops/validate.py "$@"
    ;;
  monitor)
    exec python /app/src/scripts/mlops/monitor.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
