#!/bin/bash
# A simple shell script to parse task IDs and durations from Ray worker logs.
# Author: samyu@berkeley.edu
cat /tmp/ray/session_latest/logs/python-core-worker-* | grep -rn "duration" | awk -F ":" '{print $(NF)}' | awk -F " " '{print $5, $7}' > $1
