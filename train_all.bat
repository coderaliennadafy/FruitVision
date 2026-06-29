@echo off
REM Run complete training pipeline
cd /d %~dp0src
echo === Export YOLO artifacts ===
python export_yolo_artifacts.py
echo === Train ViT ===
python -u train_vit.py
echo === Train SAM ===
python -u train_sam.py
echo === Done ===
pause
