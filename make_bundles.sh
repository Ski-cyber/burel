#!/usr/bin/env bash
# make_bundles.sh — regenerate the self-contained reproducibility zips for each experiment.
#
# Each zip carries exactly the source an experiment needs, so a contributor can drop it in
# Google Drive and run the matching Colab notebook without cloning the repo. Run this after
# changing the code so the bundles don't go stale:  bash make_bundles.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

strip_pycache() { find "$1" -name __pycache__ -type d -prune -exec rm -rf {} + ; }

# --- 01: TTT ablation — burel package + ablation script + config ---
B1="$TMP/burel_ablation"
mkdir -p "$B1/scripts" "$B1/configs"
cp -R "$ROOT/burel" "$B1/"
cp "$ROOT/scripts/ablation_ttt.py" "$B1/scripts/"
cp "$ROOT/configs/config.yaml" "$B1/configs/"
strip_pycache "$B1"
rm -f "$ROOT/experiments/01_ttt_ablation/burel_ablation.zip"
( cd "$TMP" && zip -rq "$ROOT/experiments/01_ttt_ablation/burel_ablation.zip" burel_ablation )

# --- 02: nested vs vanilla — burel + study_vanilla + scripts + configs ---
B2="$TMP/burel_vanilla_ab"
mkdir -p "$B2/scripts"
cp -R "$ROOT/burel" "$ROOT/study_vanilla" "$ROOT/configs" "$B2/"
cp "$ROOT/scripts/ablation_ttt.py" "$ROOT/scripts/compare_three.py" "$B2/scripts/"
strip_pycache "$B2"
rm -f "$ROOT/experiments/02_vanilla_ab/burel_vanilla_ab.zip"
( cd "$TMP" && zip -rq "$ROOT/experiments/02_vanilla_ab/burel_vanilla_ab.zip" burel_vanilla_ab )

echo "bundles rebuilt:"
echo "  experiments/01_ttt_ablation/burel_ablation.zip"
echo "  experiments/02_vanilla_ab/burel_vanilla_ab.zip"
