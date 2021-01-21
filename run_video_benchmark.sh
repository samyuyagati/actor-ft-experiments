#!/bin/bash

python3 video_benchmark_debugging.py --local --output app_lose_frames_timeline.txt --num-nodes 3 --video-path videos/husky.mp4 --failure --recovery app_lose_frames

python3 video_benchmark_debugging.py --local --output app_keep_frames_timeline.txt --num-nodes 3 --video-path videos/husky.mp4 --failure --recovery app_keep_frames

python3 video_benchmark_debugging.py --local --output checkpoint_timeline.txt --num-nodes 3 --video-path videos/husky.mp4 --failure --checkpoint-freq 10 --recovery checkpoint
