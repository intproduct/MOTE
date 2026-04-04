#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

CONFIG_JSON="${1:-${ROOT_DIR}/fitmotn_config.example.json}"

if [[ ! -f "${CONFIG_JSON}" ]]; then
  echo "Config file not found: ${CONFIG_JSON}" >&2
  exit 1
fi

export PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH:-}"

echo "[FitMoTN] workspace: ${WORKSPACE_DIR}"
echo "[FitMoTN] config: ${CONFIG_JSON}"

python3 -m MOTN.fitmotn.cli.train --config_json "${CONFIG_JSON}"
