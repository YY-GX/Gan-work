#!/bin/sh
pip install -r requirements.txt &
pip install --upgrade torch torchvision &
python use_memory_FREE.py

