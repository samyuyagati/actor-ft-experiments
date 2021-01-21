import asyncio
import cv2
import numpy as np
import os
import os.path
from ray import profiling
import ray
import ray.cluster_utils
import signal
import sys
import time

from collections import defaultdict

# Application-level semantics: drops some frames in the middle (skip failed
#    frame; Sink gets error but ignore it and keep executing). Default should be
#    to not retry tasks, so there should just be an error. Should work out of
#    the box. Need to make sure actor is auto-restarted. Set max_restarts > 0 
#    but max_retries = 0.
# App-level but keep all frames: set max_retries > 0. May process out of order,
#    but ok b/c of idempotence.
# DONE Checkpointing: kill all actors and restart from last frame.
#    DONE fake checkpoint that just records frame number (way to sync processes)
#    just do this at the application level (wait on Sink to see when it receives
#    frame k and take the checkpoint to do a checkpoint every k frames) (L217 of
#    Stephanie's code). 
#    Restart after fail should kill all then read checkpoint
#    frame from file, and start processing from there. 
# Logging: restart resizing actor and simulate replay from last checkpoint.
#    for logging, use existing simulate replay code but add in checkpoint idea.
# DONE For failure: kill resizer actor and restart it. 
# Kill actor w/ PID for failure. Ideally would probably want large single node
# with multiple frames.

# TODO should I make a process_videos or process_chunk actor so that the
# kill-all thing is easier to make sense of? TODO where does execution resume?
# Do I set task retries to 0 for all and manually resubmit??

NUM_WORKERS_PER_VIDEO = 1

# Decoder class uses OpenCV library to decode individual video frames.
# Can be instantiated as an actor.
# TODO: same as Resizer, see comment above resizer class.
class Decoder:
    def __init__(self, filename, start_frame):
        print("Starting decoder init")
        self.v = cv2.VideoCapture(filename)
        self.v.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.start_frame_idx = 0
        # If restarting from checkpoint
        if os.path.exists("checkpoint.txt"):
            print("Decoder recovering from checkpoint")
            with open("checkpoint.txt", "r") as f:
                self.start_frame_idx = int(f.read())
        print("Decoder start frame idx: ", self.start_frame_idx)

    # Decode frame using cv2 utilities.
    def decode(self, frame, frame_timestamp):
        if frame != self.v.get(cv2.CAP_PROP_POS_FRAMES):
            print("next frame", frame, ", at frame", self.v.get(cv2.CAP_PROP_POS_FRAMES))
            self.v.set(cv2.CAP_PROP_POS_FRAMES, frame)
        grabbed, frame = self.v.read()
        assert grabbed
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        return frame

    # Utility fn used to simulate failures.
    def get_pid(self):
        return os.getpid()

    # Utility function to be used with ray.get to check that
    # Decoder actor has been fully initialized.
    def ready(self):
        print("Decoder ready")
        return


# Resizer class uses OpenCV library to resize individual video frames.
# TODO make max_restarts, max_task_retries parameters (L232 in Stephanie's
# original benchmark) so they can be set depending on what kind of recovery
# we want. 
class Resizer:
    def __init__(self, scale_factor=0.5):
        print("initializing resizer")
        self.scale_factor = scale_factor
        self.start_frame_idx = 0
        # If restarting from checkpoint
        if os.path.exists("checkpoint.txt"):
            print("Resizer recovering from checkpoint")
            with open("checkpoint.txt", "r") as f:
                self.start_frame_idx = int(f.read())
        print("Resizer start frame idx: ", self.start_frame_idx)

    def transformation(self, frame):
        with ray.profiling.profile("resize"):
            new_width = int(self.scale_factor * frame.shape[1])
            new_height = int(self.scale_factor * frame.shape[0])
            return (new_width, new_height)

    def get_start_frame_idx(self):
        return self.start_frame_idx

    # Utility fn used to simulate failures.
    def get_pid(self):
        return os.getpid()

    # Utility function to be used with ray.get to check that
    # Resizer actor has been fully initialized.
    def ready(self):
        print("Resizer ready")
        return


@ray.remote(num_cpus=0)
class SignalActor:
    def __init__(self, num_events):
        self.ready_event = asyncio.Event()
        self.num_events = num_events

    def send(self):
        assert self.num_events > 0
        self.num_events -= 1
        if self.num_events == 0:
            self.ready_event.set()

    async def wait(self, should_wait=True):
        if should_wait:
            await self.ready_event.wait()

    def ready(self):
        return


@ray.remote(num_cpus=0)
class Viewer:
    def __init__(self, video_pathname):
#	self.out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))
        self.video_pathname = video_pathname
        self.v = cv2.VideoCapture(video_pathname)

    def send(self, transformation):
        success, frame = self.v.read()
        assert success
        resized = cv2.resize(frame, transformation, interpolation=cv2.INTER_AREA)

        # Pad resized image so it can be concatenated w/ original
        top_padding = int((frame.shape[0] - resized.shape[0]) / 2)
        bottom_padding = frame.shape[0] - top_padding - resized.shape[0]
        left_padding = int((frame.shape[1] - resized.shape[1]) / 2)
        right_padding = frame.shape[1] - left_padding - resized.shape[1]
        resized_padded = cv2.copyMakeBorder(resized, top_padding,
                bottom_padding, left_padding, right_padding,
                borderType=cv2.BORDER_CONSTANT)

        # Generate concatenated output frame.
        frame_out = cv2.hconcat([frame, resized_padded])
        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(1) # Leave image up until user presses a key.

    def ready(self):
        return


@ray.remote(num_cpus=0)
class Sink:
    # Default checkpoint_frequency=0 --> no checkpointing.
    def __init__(self, signal, viewer, checkpoint_frequency=0):
        self.signal = signal
        self.num_frames_left = {}
        self.latencies = defaultdict(dict)
        self.checkpoint_frequency = checkpoint_frequency

        self.viewer = viewer
        self.last_view = None
        print("initializing sink")

    def set_expected_frames(self, video_index, num_frames):
        self.num_frames_left[video_index] = num_frames

    def send(self, video_index, frame_index, transform, timestamp):
        # Update checkpoint
        if self.checkpoint_frequency > 0 and frame_index % self.checkpoint_frequency == 0:
            with open("checkpoint.txt", "w") as f:
                f.write(str(frame_index)) 

        with ray.profiling.profile("Sink.send"):
            # Duplicate frame.
            if frame_index in self.latencies[video_index]:
                print("Received duplicate frame ", frame_index)
                return
            # Record latency.
            self.latencies[video_index][frame_index] = time.time() - timestamp

            if frame_index % 100 == 0:
                print("Expecting", self.num_frames_left[video_index] - frame_index, "more frames from video", video_index)

	    # Only view the first video to check for correctness.
            if self.viewer is not None and video_index == 0:
                self.last_view = self.viewer.send.remote(transform)

            if frame_index == self.num_frames_left[video_index] - 1:
                print("DONE")
                if self.last_view is not None:
                    ray.get(self.last_view)
                self.signal.send.remote()


    def latencies(self):
        latencies = []
        for video_index in sorted(self.latencies.keys()):
            video = self.latencies[video_index]
            for i in sorted(video.keys()):
                latencies.append((video_index, i, video[i]))
        return latencies


    # Used to ensure actor is fully initialized.
    def ready(self):
        return 


class FailureSimError(Exception):
    def __init__(self, message):
        super().__init__(message)


@ray.remote
def process_chunk(video_index, video_pathname, sink, num_frames, fps, start_timestamp, simulate_failure=False, recovery="checkpoint"):
    if recovery == "checkpoint":
        try:
            final_frame = process_chunk_helper.remote(video_index, 
                            video_pathname, 
                            sink, num_frames, fps,
                            start_timestamp, simulate_failure, recovery=recovery)
            ray.get(final_frame)
        except ray.exceptions.WorkerCrashedError:
            final_frame = process_chunk_helper.remote(video_index, 
                            video_pathname, 
                            sink, num_frames, fps,
                            start_timestamp, False, recovery=recovery)
            ray.get(final_frame)
    else:
        final_frame = process_chunk_helper.remote(video_index, 
                        video_pathname, 
                        sink, num_frames, fps,
                        start_timestamp, simulate_failure, recovery=recovery)
        ray.get(final_frame)
    print(final_frame)
    return final_frame 


# Uses decoder and frame resizer to process each frame in the specified video.
@ray.remote(max_retries=0)
def process_chunk_helper(video_index, video_pathname, sink, num_frames, fps, start_timestamp, simulate_failure, recovery="checkpoint"):
    print("num frames", num_frames)
    # Set ray remote parameters for the actors
    # Default: checkpoint recovery/manual restart
    cls_args = {"max_restarts": 0, "max_task_retries": 0}
    if recovery == "app_lose_frames":
        cls_args = {
            "max_restarts": -1,
            "max_task_retries": 0}
    else: 
        cls_args = {
            "max_restarts": -1,
            "max_task_retries": -1,
            }

    # Create the decoder actor.
    decoder_cls = ray.remote(**cls_args)(Decoder)
    print("Made decoder cls")
    decoder = decoder_cls.remote(video_pathname, 0)
    print("Decoder ID: ", decoder)
    print("decoder init started")
    ray.get(decoder.ready.remote())
    decoder_pid = ray.get(decoder.get_pid.remote())
    print("Decoder PID: ", decoder_pid)

    # Create the frame resizing (i.e., processing) actor.
    resizer_cls = ray.remote(**cls_args)(Resizer)
    resizer = resizer_cls.remote()
    print("Resizer ID: ", resizer)
    ray.get(resizer.ready.remote())
    resizer_pid = ray.get(resizer.get_pid.remote())
    print("Resizer PID: ", resizer_pid)

    # Set start point
    start_frame_idx = ray.get(resizer.get_start_frame_idx.remote())
    print("Start frame idx: ", start_frame_idx)

    # If failure simulation: determine point to fail
    fail_point = (num_frames - start_frame_idx) // 2 + 15
    print("Fail point: ", fail_point)

    # Process each frame.
    for i in range(0, num_frames - start_frame_idx - 1):
        # Simulate failure at failure point.
        if simulate_failure and i == fail_point:
            print("Killing resizer and decoder")
            if recovery == "checkpoint":
                sys.exit()
                # Raise error to break out of loop.
                raise FailureSimError("Simulated failure occurred; resizer and decoder died.")
            else:
                os.kill(resizer_pid, signal.SIGKILL)
                #verify_killed(resizer_pid, "Resizer")

        # Calculate frame timestamp
        frame_timestamp = start_timestamp + (start_frame_idx + i + 1) / fps
        # Simulate an incoming video stream; need to sleep to output one
        # frame per 1/fps seconds in order to do this.
        diff = frame_timestamp - time.time()
        if diff > 0:
            time.sleep(diff)

        # Process the frame
        frame = decoder.decode.remote(start_frame_idx + i + 1, frame_timestamp)	
        transformation = resizer.transformation.remote(frame)
        final = sink.send.remote(video_index, start_frame_idx + i, 
                                 transformation, frame_timestamp)

    # Block on processing of final frame so that latencies include all frames
    # processed.
    ray.wait([final], num_returns=1)


def verify_killed(pid, name):
    killed = False
    while not killed:
        try:
            os.kill(pid, 0)
        except OSError:
            print("{} PID {} is unassigned".format(name, pid))
            killed = True
        else:
            print("{} PID {} is in use; retrying kill".format(name, pid))
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
    return True


def process_videos(video_pathnames, max_frames, num_sinks, checkpoint_frequency, 
    simulate_failure=False, recovery="checkpoint", view=False):
    # Set up sinks. 
    signal = SignalActor.remote(len(video_pathnames))
    ray.get(signal.ready.remote())

    viewer = None
    if view:
        viewer = Viewer.remote(video_pathnames[0])

    sinks = [Sink.remote(signal, viewer, checkpoint_frequency) for i in range(num_sinks)]
    ray.get([sink.ready.remote() for sink in sinks])

    for i, video_pathname in enumerate(video_pathnames):
	# Set expected frames per sink.
        v = cv2.VideoCapture(video_pathname)
        num_total_frames = int(min(v.get(cv2.CAP_PROP_FRAME_COUNT), max_frames))
        print(video_pathname, "FRAMES", num_total_frames)
        ray.get(sinks[i % len(sinks)].set_expected_frames.remote(i, num_total_frames - 1))

    # Give the actors some time to start up.
    start_timestamp = time.time() + 5

    # Process each video.
    for i, video_pathname in enumerate(video_pathnames):
        v = cv2.VideoCapture(video_pathname)
        num_total_frames = int(min(v.get(cv2.CAP_PROP_FRAME_COUNT), max_frames))
        fps = v.get(cv2.CAP_PROP_FPS)
        process_chunk.remote( 
                i, video_pathnames[i],
                sinks[i % len(sinks)], num_total_frames,
                int(fps), start_timestamp,
                simulate_failure if i == 0 else False, recovery=recovery)

    # Wait for all videos to finish processing.
    ray.get(signal.wait.remote())

    # Calculate latencies.
    latencies = []
    for sink in sinks:
        sink_latencies = ray.get(sink.latencies.remote())
        latencies += sink_latencies

        j = 0
        for video_index, i, _ in latencies:
            if i != j:
                print("Sink missing frame", j, "for video", video_index)
                j = i
            j += 1

    output_filename = "{}.txt".format(recovery)
    with open(output_filename, 'w') as f:
        for i, t, l in latencies:
            f.write("{} {} {}\n".format(i, t, l))
    latencies = [l for _, _, l in latencies]
    print("Mean latency:", np.mean(latencies))
    print("Max latency:", np.max(latencies))


def main(args):
    ray.init(_system_config={
            "task_retry_delay_ms": 100,
            })

    # Delete checkpoint file to have a clean start.
    if os.path.exists("checkpoint.txt"):
        os.remove("checkpoint.txt")

    process_videos(args.video_path,
            args.max_frames,
            args.num_sinks, args.checkpoint_freq,
            simulate_failure=args.failure, recovery=args.recovery_type,
            view=args.view)

    if args.timeline:
        ray.timeline(filename="{}.json".format(args.recovery))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the video benchmark.")

    parser.add_argument("--video-path", required=True, nargs='+', type=str)
    parser.add_argument("--failure", action="store_true")
    parser.add_argument("--timeline", default=None, type=str)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--max-frames", default=600, type=int)
    parser.add_argument("--num-sinks", default=1, type=int)
    parser.add_argument("--num-owners-per-node", default=1, type=int)
    parser.add_argument("--centralized", action="store_true")
    parser.add_argument("--checkpoint-freq", default=0, type=int)
    # Options: "checkpoint", "app_lose_frames", "app_keep_frames"
    parser.add_argument("--recovery-type", default="checkpoint", type=str)
    args = parser.parse_args()
    main(args)
