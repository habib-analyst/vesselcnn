#!/usr/bin/env bash
# Fetch the VesselMNIST3D source file and verify it against the
# medmnist package's published MD5 before any use.
# VesselMNIST3D is part of MedMNIST v2 (Yang et al., Scientific Data
# 2023), distributed via Zenodo record 10519652 under CC BY 4.0.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -L -o data/vesselmnist3d.npz \
  "https://zenodo.org/records/10519652/files/vesselmnist3d.npz?download=1"
echo "b41fd4f7e7e2feedddb201585ecafa1b  data/vesselmnist3d.npz" | md5sum -c -
sha256sum data/vesselmnist3d.npz
# Expected SHA-256 (computed in the foreground at build time, 2026-10-29):
# cc00c6b0869bf1f68cf7c13e52e8585eb2f38991acab5ad7abe1af1814a83206
