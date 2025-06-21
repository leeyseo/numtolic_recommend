#!/bin/bash

nyumtolic_PID=$(ps -ef | grep java | grep nyumtolic | awk '{print $2}')

if [ -z "$nyumtolic_PID" ];
then
    echo "nyumtolic is not running"
else
    kill -9 $nyumtolic_PID
    echo "nyumtolic stopped."
fi
