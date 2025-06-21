#!/bin/bash

# Flask 애플리케이션 PID 검색
MODEL_PID=$(ps -ef | grep python | grep run_funkSVD.py | awk '{print $2}')

# 프로세스가 실행 중인지 확인 후 종료
if [ -z "$MODEL_PID" ]; then
    echo "Model server is not running."
else
    kill -9 $MODEL_PID
    echo "Model server stopped."
fi
