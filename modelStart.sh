#!/bin/bash

# 실행할 Python 파일과 로그 파일 경로
PYTHON_FILE="run_funkSVD.py"
LOG=/dev/null

# 백그라운드에서 Flask 애플리케이션 실행
nohup python3 $PYTHON_FILE > $LOG 2>&1 &
echo "Model server started."
