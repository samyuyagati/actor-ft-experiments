import asyncio
import cv2
import numpy as np
import os.path
from ray import profiling
import ray
import ray.cluster_utils

NUM_WORKERS_PER_VIDEO = 1

# Decoder class uses OpenCV library to decode individual video frames.
# Can be instantiated as an actor.
@ray.remote(max_restarts=-1, max_task_retries=-1)
class Decoder:
    def __init__(self, filename, start_frame):
        self.v = cv2.VideoCapture(filename)
        self.v.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def decode(self, frame):
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
    # TODO should it have other state?
    def __init__(self, scale_factor=0.5):
	self.scale_factor = 0.5

    def transformation(self, frame):
        with ray.profiling.profile("resize"):
	    new_width = int(self.scale_factor * frame.shape[1])
	    new_height = int(self.scale_factor * frame.shape[0])
	    return (new_width, new_height)

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
	frame_out = cv2.hconcat([frame, resized])
	cv2.imshow("Before and After", frame_out)
	cv2.waitKey(1) # Leave image up until user presses a key.

    def ready(self):
	return


@ray.remote(num_cpus=0)
class Sink:
    def __init__(self, signal, viewer):
        self.signal = signal
        self.num_frames_left = {}
        self.latencies = defaultdict(list)

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

	    # Only view the first video to check for correctness.
	    if self.viewer is not None and video_index == 0:
                self.last_view = self.viewer.send.remote(transform)


    def latencies(self):
        latencies = []
        for video in self.latencies.values():
            for i, l in enumerate(video):
                latencies.append((i, l))
        return latencies    


def process_chunk(video_index, video_pathname, sink, num_frames, fps, resource, start_timestamp):
    # Create the decoder actor.
    decoder = Decoder.remote(video_pathname, 0))
    ray.get(decoder.ready.remote())

    # Create the frame resizing (i.e., processing) actor.
    resizer = ray.get(Resizer.remote())
    ray.get(resizer.ready.remote())

    # Process frames
    start_frame_idx = 0
    prev_frame = decoder.decode.remote(start_frame_idx)

    for i in range(start_frame_idx, num_frames - 1):
	frame_timestamp = start_timestamp + (start_frame_idx + i + 1) / fps 
	frame = decoder.decode.remote(start_frame_idx + i + 1) # Why not just i + 1?	
	transformation = resizer.transformation.options(resources={resource: 1}).remote(frame)
        final = sink.send.remote(video_index, i, transformation, frame_timestamp)

    # Block on processing of final frame so that latencies include all frames
    # processed.
    return ray.get(final)


def process_videos(video_pathnames, num_sinks, owner_resources, resources, view=False):
    # Set up sinks. 
    signal = SignalActor.remote(len(video_pathnames))
    ray.get(signal.ready.remote())

    viewer = None
    if view:
	viewer = Viewer.remote(video_pathnames[0])

    sinks = [Sink.options(resources={
        sink_resources[i % len(sink_resources)]: 1
        }).remote(signal, viewer, checkpoint_interval) for i in range(num_sinks)]
    ray.get([sink.ready.remote() for sink in sinks])

    for i, video_pathname in enumerate(video_pathnames):
	# Set expected frames per sink.
        v = cv2.VideoCapture(video_pathname)
        num_total_frames = int(min(v.get(cv2.CAP_PROP_FRAME_COUNT), max_frames))
        print(video_pathname, "FRAMES", num_total_frames)
        ray.get(sinks[i % len(sinks)].set_expected_frames.remote(i, num_total_frames - 1))

    # TODO what is start_timestamp? Start of processing or start of video?
    # Give the actors some time to start up.
    start_timestamp = time.time() + 5

    # Process each video.
    for i, video_pathname in enumerate(video_pathnames):
        v = cv2.VideoCapture(video_pathname)
        num_total_frames = int(min(v.get(cv2.CAP_PROP_FRAME_COUNT), max_frames))
        fps = v.get(cv2.CAP_PROP_FPS)
        owner_resource = owner_resources[i % len(owner_resources)]
        worker_resource = resources[i % len(resources)]
        print("Placing owner of video", i, "on node with resource", owner_resource)
        process_chunk.options(resources={owner_resource: 1}).remote(i, video_pathnames[i],
                sinks[i % len(sinks)], num_total_frames,
                int(fps), worker_resource, start_timestamp)

    # Wait for all videos to finish processing.
    ray.get(signal.wait.remote())

    # Calculate latencies.
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
    # TODO Finish this function (based on Stephanie's code, but try to take just relevant parts).
