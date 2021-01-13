import asyncio
import cv2
import numpy as np
import os
import os.path
from ray import profiling
import ray
import ray.cluster_utils
import signal
import time

from collections import defaultdict

# Application-level semantics: drops some frames in the middle (skip failed
#    frame; Sink gets error but ignore it and keep executing). Default should be
#    to not retry tasks, so there should just be an error. Should work out of
#    the box. Need to make sure actor is auto-restarted. Set max_restarts > 0 
#    but max_retries = 0.
# App-level but keep all frames: set max_retries > 0. May process out of order,
#    but ok b/c of idempotence.
# Checkpointing: kill all actors and restart from last frame.
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
@ray.remote(max_restarts=-1, max_task_retries=-1)
class Decoder:
    def __init__(self, filename, start_frame):
        self.v = cv2.VideoCapture(filename)
        self.v.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def decode(self, frame, frame_timestamp):
        # Simulate an incoming video stream; need to sleep to output one
        # frame per 1/fps seconds in order to do this.
        diff = frame_timestamp - time.time()
        if diff > 0:
            time.sleep(diff)

        # Decode frame
        if frame != self.v.get(cv2.CAP_PROP_POS_FRAMES):
            print("next frame", frame, ", at frame", self.v.get(cv2.CAP_PROP_POS_FRAMES))
            self.v.set(cv2.CAP_PROP_POS_FRAMES, frame)
        grabbed, frame = self.v.read()
        assert grabbed
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        return frame

    # Utility function to be used with ray.get to check that
    # Decoder actor has been fully initialized.
    def ready(self):
        return


# Resizer class uses OpenCV library to resize individual video frames.
@ray.remote(max_restarts=-1, max_task_retries=-1)
class Resizer:
    def __init__(self, scale_factor=0.5):
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
        return


@ray.remote(num_cpus=0, resources={"head": 1})
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


@ray.remote(num_cpus=0, resources={"head": 1})
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
        self.latencies = defaultdict(list)
        self.checkpoint_frequency = checkpoint_frequency

        self.viewer = viewer
        self.last_view = None

    def set_expected_frames(self, video_index, num_frames):
        self.num_frames_left[video_index] = num_frames

    def send(self, video_index, frame_index, transform, timestamp):
        with ray.profiling.profile("Sink.send"):
            if frame_index < len(self.latencies[video_index]):
                return
            assert frame_index == len(self.latencies[video_index]), frame_index

            self.latencies[video_index].append(time.time() - timestamp)

            self.num_frames_left[video_index] -= 1
            if self.num_frames_left[video_index] % 100 == 0:
                print("Expecting", self.num_frames_left[video_index], "more frames from video", video_index)

            if self.num_frames_left[video_index] == 0:
                print("DONE")
                if self.last_view is not None:
                    ray.get(self.last_view)
                self.signal.send.remote()

        # Update checkpoint
        if self.checkpoint_frequency > 0 and frame_index % self.checkpoint_frequency == 0:
            with open("checkpoint.txt", "w") as f:
                f.write(str(frame_index)) 

	    # Only view the first video to check for correctness.
            if self.viewer is not None and video_index == 0:
                self.last_view = self.viewer.send.remote(transform)


    def latencies(self):
        latencies = []
        for video in self.latencies.values():
            for i, l in enumerate(video):
                latencies.append((i, l))
        return latencies


    # Used to ensure actor is fully initialized.
    def ready(self):
        return 


@ray.remote(num_cpus=0)
def process_chunk(video_index, video_pathname, sink, num_frames, fps, resource, start_timestamp, simulate_failure=False):
    # Create the decoder actor.
    decoder = Decoder.remote(video_pathname, 0)
    ray.get(decoder.ready.remote())

    # Create the frame resizing (i.e., processing) actor.
    resizer = Resizer.options(resources={resource: 1}).remote()
    ray.get(resizer.ready.remote())
    resizer_pid = ray.get(resizer.get_pid.remote())
    print("Resizer PID: ", resizer_pid)

    # Set start point
    start_frame_idx = ray.get(resizer.get_start_frame_idx.remote())
    print("Start frame idx: ", start_frame_idx)

    # If failure simulation: determine point to fail
    fail_point = int(((num_frames - 1) - start_frame_idx) / 2)
    print("Fail point: ", fail_point)

    for i in range(start_frame_idx, num_frames - 1):
        # start_frame_idx == 0 indicates first execution; don't want to
        # fail on re-execution.
        if simulate_failure and i == fail_point and start_frame_idx == 0:
            print("Killing resizer")
            os.kill(resizer_pid, signal.SIGKILL) 
        
        # Need to set s_f_i again because resizer actor would pick up from
        # here on restart (I think). NO -- restarts whole submitted task,
        # which is why we need the timestamps.
#        start_frame_idx = ray.get(resizer.get_start_frame_idx.remote())
        frame_timestamp = start_timestamp + (start_frame_idx + i + 1) / fps

        # Process the frame 
        frame = decoder.decode.remote(start_frame_idx + i + 1, frame_timestamp) 	
        transformation = resizer.transformation.remote(frame)
        final = sink.send.remote(video_index, start_frame_idx + i, 
                                 transformation, frame_timestamp)

    # Block on processing of final frame so that latencies include all frames
    # processed.
    return ray.get(final)


def process_videos(video_pathnames, output_filename, resources, owner_resources, 
	sink_resources, max_frames, num_sinks, checkpoint_frequency, 
    simulate_failure=False, view=False):
    # Set up sinks. 
    signal = SignalActor.remote(len(video_pathnames))
    ray.get(signal.ready.remote())

    viewer = None
    if view:
        viewer = Viewer.remote(video_pathnames[0])

    sinks = [Sink.options(resources={
        sink_resources[i % len(sink_resources)]: 1
        }).remote(signal, viewer, checkpoint_frequency) for i in range(num_sinks)]
    ray.get([sink.ready.remote() for sink in sinks])

    for i, video_pathname in enumerate(video_pathnames):
	# Set expected frames per sink.
        v = cv2.VideoCapture(video_pathname)
        num_total_frames = int(min(v.get(cv2.CAP_PROP_FRAME_COUNT), max_frames))
        print(video_pathname, "FRAMES", num_total_frames)
        ray.get(sinks[i % len(sinks)].set_expected_frames.remote(i, num_total_frames - 1))

    # TODO what is start_timestamp? Start of processing or start of video?
    # Give the actors some time to start up.
    start_timestamp = time.time() # + 5

    # Process each video.
    for i, video_pathname in enumerate(video_pathnames):
        v = cv2.VideoCapture(video_pathname)
        num_total_frames = int(min(v.get(cv2.CAP_PROP_FRAME_COUNT), max_frames))
        fps = v.get(cv2.CAP_PROP_FPS)
        owner_resource = owner_resources[i % len(owner_resources)]
        worker_resource = resources[i % len(resources)]
        print("Placing owner of video", i, "on node with resource", owner_resource)
        process_chunk.options(resources={owner_resource: 1}).remote( 
                i, video_pathnames[i],
                sinks[i % len(sinks)], num_total_frames,
                int(fps), worker_resource, start_timestamp,
                simulate_failure)

    # Wait for all videos to finish processing.
    ray.get(signal.wait.remote())

    # Calculate latencies.
    latencies = []
    for sink in sinks:
        latencies += ray.get(sink.latencies.remote())
    if output_filename:
        with open(output_filename, 'w') as f:
            for t, l in latencies:
                f.write("{} {}\n".format(t, l))
    else:
        for latency in latencies:
            print(latency)
    latencies = [l for _, l in latencies]
    print("Mean latency:", np.mean(latencies))
    print("Max latency:", np.max(latencies))


def main(args):
    video_resources = ["video:{}".format(i) for i in range(len(args.video_path))]

    num_owner_nodes = len(args.video_path) // args.num_owners_per_node
    if len(args.video_path) % args.num_owners_per_node:
        num_owner_nodes += 1
    owner_resources = ["video_owner:{}".format(i) for i in range(num_owner_nodes)]

    num_sink_nodes = len(video_resources) # For now, one sink per video. 
    sink_resources = ["video_sink:{}".format(i) for i in range(num_sink_nodes)]
    
    num_required_nodes = len(args.video_path) + num_owner_nodes + num_sink_nodes
    assert args.num_nodes >= num_required_nodes, ("Requested {} nodes, need {}".format(args.num_nodes, num_required_nodes)) 

    # Just do local for now, add remote later.
    if args.local:
        cluster = ray.cluster_utils.Cluster()
        # Create head node.
        cluster.add_node(num_cpus=0, resources={"head": 100})
        # Add remaining nodes.
        for _ in range(args.num_nodes):
            cluster.add_node()
        cluster.wait_for_nodes()
        address = cluster.address
    else:
        address = "auto"

    ray.init(address=address)

    nodes = [node for node in ray.nodes() if node["Alive"]]
    while len(nodes) < args.num_nodes + 1:
        time.sleep(1)
        print("{} nodes found, waiting for nodes to join".format(len(nodes)))
        nodes = [node for node in ray.nodes() if node["Alive"]]

    # TODO add in non-local

    for node in nodes:
        for resource in node["Resources"]:
            if resource.startswith("video"):
                ray.experimental.set_resource(resource, 0, node["NodeID"])

    nodes = [node for node in ray.nodes() if node["Alive"]]
    print("All nodes joined")
    for node in nodes:
        print("{}:{}".format(node["NodeManagerAddress"], node["NodeManagerPort"]))

    head_node = [node for node in nodes if "head" in node["Resources"]]
    assert len(head_node) == 1
    head_ip = head_node[0]["NodeManagerAddress"]
    nodes.remove(head_node[0])
    assert len(nodes) >= len(video_resources) + num_owner_nodes, ("Found {} nodes, need {}".format(len(nodes), len(video_resources) + num_owner_nodes))

    worker_ip = None
    worker_resource = None
    owner_ip = None
    owner_resource = None
    node_index = 0
    for node, resource in zip(nodes, sink_resources + owner_resources + video_resources):
        if "CPU" not in node["Resources"]:
            continue

        print("Assigning", resource, "to node", node["NodeID"], node["Resources"])
        ray.experimental.set_resource(resource, 100, node["NodeID"])

        if "owner" in resource:
            owner_resource = resource
            owner_ip = node["NodeManagerAddress"]
        elif "video:" in resource:
            worker_resource = resource
            worker_ip = node["NodeManagerAddress"]
    process_videos(args.video_path, args.output,
            video_resources, owner_resources, sink_resources, args.max_frames,
            args.num_sinks, args.checkpoint_freq,
            simulate_failure=args.failure, view=args.view)
    # Delete checkpoint file to have a clean start next run.
    if os.path.exists("checkpoint.txt"):
        os.remove("checkpoint.txt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the video benchmark.")

    parser.add_argument("--num-nodes", required=True, type=int)
    parser.add_argument("--video-path", required=True, nargs='+', type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--failure", action="store_true")
    parser.add_argument("--timeline", default=None, type=str)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--max-frames", default=600, type=int)
    parser.add_argument("--num-sinks", default=1, type=int)
    parser.add_argument("--num-owners-per-node", default=1, type=int)
    parser.add_argument("--centralized", action="store_true")
    parser.add_argument("--checkpoint-freq", default=0, type=int)
    args = parser.parse_args()
    main(args)
