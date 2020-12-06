import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
import sys
import time

# Author: accheng@berkeley.edu
# TODO@accheng: File description.

@ray.remote(num_gpus=0, max_restarts=-1, max_task_retries=-1)
class RayActor:
	def __init__(self, num_workers):
		self.config = ppo.DEFAULT_CONFIG.copy()
		# self.config["num_cpus"] = 0
		self.config["num_gpus"] = 0
		self.config["num_workers"] = num_workers
		# self.config["num_cpus_per_worker"] = 1
		self.config["ignore_worker_failures"] = True
		# config["eager"] = False
		self.trainer = ppo.PPOTrainer(config=self.config, env="CartPole-v0")
		self.restarted = False

	def request(self, restart, fail_time):
		tstart = time.time()
		data = {}
		num_iters = 0
		tcheck = time.time()
		for i in range(100):
			# Restart training
			# if tcheck - tstart >= fail_time and restart and not self.restarted:
			# 	self.trainer._stop()
			# 	self.trainer = ppo.PPOTrainer(config=self.config, env="CartPole-v0")
			# 	self.restarted = True
			# 	print("restarted **************************************************")
			# 	print("restarted **************************************************")

		   # Perform one iteration of training the policy with PPO
			result = self.trainer.train()
			num_iters += 1
			print('episode_reward_mean: ', result['episode_reward_mean'], 'episodes_total: ', result['episodes_total'])
		   # print(pretty_print(result))
		   # CartPole-v0 defines "solving" as getting average reward of 195.0
			if result['episode_reward_mean'] >= 195.0:
				print('training complete')
				break

			if i % 1 == 0:
			   checkpoint = self.trainer.save()
			   print("checkpoint saved at", checkpoint)
			tcheck = time.time()

		tstop = time.time()
		data[num_iters] = tstop-tstart
		self.trainer._stop()
		return data

	# def kill_actor(self):
	# 	print("killing actor")
	# 	sys.exit()


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Run the video benchmark.")

	parser.add_argument("--num-workers", default=1, type=int)
	parser.add_argument("--global-restart", default=False, type=bool)
	parser.add_argument("--fail-time", default=1, type=int)
	args = parser.parse_args()

	ray.init()
	actor = RayActor.remote(args.num_workers)
	backup_actor = RayActor.remote(args.num_workers)

	tstart = time.time()
	data = ray.get(actor.request.remote(args.global_restart, args.fail_time))
	# if args.global_restart:
	# 	time.sleep(5)
	# 	ray.get(actor.kill_actor.remote())
	# 	print('exited')

	tstop = time.time()
	# data = ray.get(actor.request.remote())
	num_iters = list(data.keys())[0]
	runtime = list(data.values())[0]
	print("time: ", tstop-tstart)
	print("runtime: ", runtime)
	print("iterations: ", num_iters)
	print("avg. time per iteration: ", runtime / num_iters)
