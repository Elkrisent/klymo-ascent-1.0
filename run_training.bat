@echo off
REM Clear any existing Python processes
taskkill /F /IM python.exe 2>nul

REM Wait a moment for GPU to clear
timeout /t 2 /nobreak >nul

REM Set environment variable for better memory management
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Run training
python train_swinir.py
