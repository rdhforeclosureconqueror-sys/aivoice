#!/usr/bin/env bash
set -e

echo "==> Installing system packages..."
apt-get update
apt-get install -y --no-install-recommends \
  ffmpeg \
  pkg-config \
  libavformat-dev \
  libavcodec-dev \
  libavdevice-dev \
  libavutil-dev \
  libavfilter-dev \
  libswscale-dev \
  libswresample-dev

echo "==> Upgrading pip tooling..."
python -m pip install -U pip setuptools wheel

echo "==> Installing Python requirements..."
pip install -r requirements.txt
