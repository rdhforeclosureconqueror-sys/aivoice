#!/usr/bin/env bash
set -o errexit

# Install ffmpeg (Render's base image supports apt-get)
apt-get update
apt-get install -y ffmpeg

# Upgrade pip
python -m pip install --upgrade pip

# Install Python deps
pip install -r requirements.txt
