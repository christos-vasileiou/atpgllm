#!/usr/bin/env bash
# Compatibility shim — prefer scripts/train/submit_training_code.sh
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/train/submit_training_code.sh" "$@"
