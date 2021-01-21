#!/bin/bash

# 'failure' flag can be omitted to achieve a failure-free execution.

# 'recovery' flag has three possible values: app_lose_frames (app-level
# recovery that tolerates missing frames), app_keep_frames (app-level recovery
# that does not tolerate missing frames), and checkpoint (the default, which
# simulates global checkpointing + rollback).

# DO NOT set the checkpoint-freq flag unless you want checkpoint recovery.

# ----- Run benchmarks -----
# Simulate one failure about halfway through processing and with
# simulated application-level recovery that tolerates missing frames.
python3 video_benchmark_debugging.py --local --output app_lose_frames_timeline.txt --num-nodes 3 --video-path videos/husky.mp4 --failure --recovery app_lose_frames

# Simulate one failure about halfway through processing and with
# simulated application-level recovery that does not allow missing frames.
python3 video_benchmark_debugging.py --local --output app_keep_frames_timeline.txt --num-nodes 3 --video-path videos/husky.mp4 --failure --recovery app_keep_frames

# Simulate one failure about halfway through processing and with
# simulated global checkpointing for recovery. To vary the checkpoint frequency,
# simply specify a different number with the checkpoint-freq flag (the
# command below sets it to one checkpoint per ten frames processed).
python3 video_benchmark_debugging.py --local --output checkpoint_timeline.txt --num-nodes 3 --video-path videos/husky.mp4 --failure --checkpoint-freq 10 --recovery checkpoint