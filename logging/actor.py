import datetime
import ray
import time

@ray.remote(max_restarts=-1, max_task_retries=-1)
class RayActor:
	def __init__(self):
		pass
	def request(self):
		print('hello')
		self.checkpoint()
		time.sleep(5)
		
		print('fail now')
		time.sleep(10)
		return

	def checkpoint(self):
		myfile = "/home/ubuntu/ray_source/checkpoint_time.txt"
		with open(myfile, "w") as f:
			f.write(str(int(time.time())) + "\n")
		print("checkpoint")

ray.init()

actor = RayActor.remote()

# for i in range(10):
print('starting')
ray.get(actor.request.remote())
print(int(time.time()))
