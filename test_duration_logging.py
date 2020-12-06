import ray
import time
import os, signal
import sys

# Author: samyu@berkeley.edu
# Toy Python program that creates an actor Actor, has it do some tasks,
# fail, restart, and do more tasks. Intended to test that logging + simulated
# recovery happens as expected.

@ray.remote(max_restarts=-1, max_task_retries=-1)
class Actor:
	def __init__(self):
		self.count = 0

	def add(self):
		self.count += 1
		#time.sleep()
		return self.count

	def get_pid(self):
		return os.getpid()

	def increment_and_possibly_fail(self):
		if self.count == 10:
			os._exit(0)
		self.count += 1
		return self.count

ray.init(ignore_reinit_error=True)
actor = Actor.remote()
pid = ray.get(actor.get_pid.remote())
print("pid:", pid)

start1 = time.time()
count = ray.get([actor.increment_and_possibly_fail.remote() for i in range(8)])
print(count)
end1 = time.time()

#os.kill(pid, signal.SIGSTOP)
#ray.get(actor.increment_and_possibl.remote())
#actor.kill.remote()

start = time.time()
count = ray.get([actor.increment_and_possibly_fail.remote() for i in range(8)])
print(count)
end = time.time()

print("no recovery total time:", float(end1) - float(start1)) 
print("w/ recovery total time:", float(end) - float(start))   # Not precise since could switch threads between kill
						# and start of timing

