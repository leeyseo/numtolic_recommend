#!/bin/bash

JAR=nyumtolic-0.0.1-SNAPSHOT.jar
LOG=/dev/null
export spring_profiles_active=prod

nohup java -jar $JAR > $LOG 2>&1 &
