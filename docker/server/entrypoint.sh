#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Register once, then run the GitHub Actions runner. On restart the persisted .runner
# credentials are reused, so a fresh token is only needed for the very first launch.
set -euo pipefail

cd /opt/runner

if [ ! -f .runner ]; then
  : "${RUNNER_URL:?set RUNNER_URL, e.g. https://github.com/ultralytics/inference}"
  : "${RUNNER_TOKEN:?set RUNNER_TOKEN to a runner registration token}"

  # Runner groups are org/enterprise-only; pass --runnergroup solely when RUNNER_GROUP is set,
  # otherwise a repo-scoped registration errors.
  group_args=()
  [ -n "${RUNNER_GROUP:-}" ] && group_args=(--runnergroup "${RUNNER_GROUP}")

  ./config.sh \
    --url "${RUNNER_URL}" \
    --token "${RUNNER_TOKEN}" \
    --name "${RUNNER_NAME:-gpu-$(hostname)}" \
    --labels "${RUNNER_LABELS:-rust-gpu-runner}" \
    "${group_args[@]}" \
    --work _work \
    --unattended --replace
fi

# exec so run.sh becomes PID 1 and receives SIGTERM directly for a graceful stop.
exec ./run.sh
