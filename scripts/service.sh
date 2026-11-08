#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
pid_file="logs/server.pid"
is_running() {
  [[ -f "$pid_file" ]] || return 1
  local server_pid
  server_pid="$(cat "$pid_file")"
  [[ "$server_pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$server_pid" 2>/dev/null || return 1
  [[ -r "/proc/$server_pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$server_pid/cmdline" | grep -q 'ragkb.main'
}
case "${1:-status}" in
  start)
    if is_running; then
      echo "Knowledge base is already running (PID $(cat "$pid_file"))."
      exit 0
    fi
    nohup bash scripts/start.sh >> logs/server.log 2>&1 &
    echo "$!" > "$pid_file"
    echo "Starting knowledge base (PID $(cat "$pid_file")); see logs/server.log."
    ;;
  stop)
    if is_running; then
      server_pid="$(cat "$pid_file")"
      kill -TERM "$server_pid"
      for _ in {1..30}; do
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$server_pid" 2>/dev/null; then
        echo "Still completing active work; no forced termination performed."
        exit 1
      fi
      rm -f "$pid_file"
      echo "Stopped."
    else
      echo "No managed server is running."
    fi
    ;;
  restart)
    bash "$0" stop
    bash "$0" start
    ;;
  status)
    if is_running; then
      echo "Running (PID $(cat "$pid_file"))."
    else
      echo "No managed server is running."
    fi
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 2
    ;;
esac

