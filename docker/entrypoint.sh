#!/bin/sh
set -e

export PYTHONPATH="${PYTHONPATH:-/app/src}"

case "${1:-test}" in
  test)
    shift
    exec pytest -q /app/tests "$@"
    ;;
  lint)
    shift
    exec ruff check /app/tests /app/src/scripts/mlops "$@"
    ;;
  streamlit)
    shift
    exec streamlit run /app/src/app/ui/streamlit_app.py \
      --server.port="${STREAMLIT_PORT:-8501}" \
      --server.address=0.0.0.0 \
      --browser.gatherUsageStats=false \
      "$@"
    ;;
  train)
    shift
    exec python /app/src/scripts/mlops/train.py "$@"
    ;;
  validate)
    shift
    exec python /app/src/scripts/mlops/validate.py "$@"
    ;;
  monitor)
    shift
    exec python /app/src/scripts/mlops/monitor.py "$@"
    ;;
  daily-report)
    shift
    exec python /app/src/scripts/gcp/daily_report.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
