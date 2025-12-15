#!/usr/bin/env bash
set -o errexit  # stop on first error

# Upgrade pip safely
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
