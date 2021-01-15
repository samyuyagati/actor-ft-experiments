import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
import sys
import time

@ray.remote(num_gpus=0, max_restarts=-1, max_task_retries=-1)
class RayActor:
	def __init__(self, num_workers, global_restart, resources_out, resource_wait):
		self.config = ppo.DEFAULT_CONFIG.copy()
		self.config["num_gpus"] = 0
		self.config["num_workers"] = num_workers
		self.config["ignore_worker_failures"] = True
		self.config["resources_out"] = resources_out
		self.config["resource_wait"] = resource_wait

		# Pendulum-v0
		self.config["framework"] = "tf"
		self.config["train_batch_size"] = 512
		self.config["vf_clip_param"] = 10.0
		# self.config["num_envs_per_worker"] = 20
		self.config["lambda"] = 0.1
		self.config["gamma"] = 0.95
		self.config["lr"] = 0.0003
		self.config["sgd_minibatch_size"] = 64
		self.config["num_sgd_iter"] = 6
		self.config["model"] = {"fcnet_hiddens": [256, 256]}
		self.config["observation_filter"] = "MeanStdFilter"

		# Pendulum-v0, CartPole-v0
		self.trainer =  ppo.PPOTrainer(config=self.config, env='Pendulum-v0')
		self.global_restart = global_restart
		self.backup_trainer = None
		if global_restart:
			self.backup_trainer = ppo.PPOTrainer(config=self.config, env="Pendulum-v0")
			self.backup_trainer_2 = ppo.PPOTrainer(config=self.config, env="Pendulum-v0")
		self.restarted = False
		self.count = 0
		self.start_time = 0
		self.recovered = False

	def request(self):
		tstart = time.time()
		data = {}
		num_iters = 0
		tcheck = time.time()
		t1 = time.time()
		for i in range(1000):
			if i == 85 and self.config["resources_out"] > 0:
				self.start_time = time.time()
				self.trainer.remove_workers(self.config["resources_out"])
				# print("********removed workers**********")
				# Global checkpoint requires all workers to wait when one worker stalls
			if i == 95 and self.global_restart:
				while time.time() - self.start_time < self.config["resource_wait"]:
					print("waiting for worker")
					time.sleep(1)
				self.trainer._stop()
				self.trainer = self.backup_trainer
				print("restarted ************************************************** count: ", self.count)

			if i == 180 and self.config["resources_out"] > 0:
				self.start_time = time.time()
				self.trainer.remove_workers(self.config["resources_out"])
				# print("********removed workers**********")
				# Global checkpoint requires all workers to wait when one worker stalls
			if i == 185 and self.global_restart:
				while time.time() - self.start_time < self.config["resource_wait"]:
					print("waiting for worker")
					time.sleep(1)
				self.trainer._stop()
				self.trainer = self.backup_trainer_2
				print("restarted 2 ************************************************** count: ", self.count)

			if self.config["resources_out"] > 0 and self.start_time != 0 and time.time() - self.start_time > self.config["resource_wait"] and not self.recovered:
				self.trainer.recover_workers(self.config["resources_out"])
				self.recovered = True
				# print("---------recovered workers---------")
			# print('hello')
			# Restart training
			# if i == 100 and self.global_restart and self.count < 1: #and not self.restarted 
			# 	self.trainer._stop()
			# 	self.trainer = self.backup_trainer
			# 	# self.restarted = True
			# 	self.count += 1
			# 	# print("restarted ----------------------------------------------------")
			# 	self.trainer._stop()
			# 	self.trainer = self.backup_trainer_2
			# 	print("restarted ************************************************** count: ", self.count)
			# elif i == 200 and self.global_restart and self.count < 2:
			# 	# self.restarted = True
			# 	self.count += 1
			# 	# print("restarted ----------------------------------------------------")
			# 	print("restarted ************************************************** count: ", self.count)				

		   # Perform one iteration of training the policy with PPO
			result = self.trainer.train()
			num_iters += 1
			# print('episode_reward_mean: ', result['episode_reward_mean'], 'episodes_total: ', result['episodes_total'],
			# 	'timesteps_total', result['timesteps_total'])
			# print(pretty_print(result))
		   # CartPole-v0 defines "solving" as getting average reward of 195.0
		   # Pendulum-v0 does not have a specified reward threshold at which it's considered solved
			if result['episode_reward_mean'] >= -150.0:
				print('training complete')
				break

			if i % 20 == 0:
			   checkpoint = self.trainer.save()
			   print("checkpoint saved at", checkpoint)
			   # t2 = time.time()
			   # print("checkpoint took ", t2 - t1)
			   # t1 = t2
			tcheck = time.time()

		tstop = time.time()
		data[num_iters] = tstop-tstart
		self.trainer._stop()
		return data

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Run the video benchmark.")

	parser.add_argument("--num-workers", default=1, type=int)
	parser.add_argument("--global-restart", default=False, type=bool)
	parser.add_argument("--resources-out", default=0, type=int)
	parser.add_argument("--resource-wait", default=0, type=int)
	args = parser.parse_args()

	ray.init()
	actor = RayActor.remote(args.num_workers, args.global_restart, args.resources_out, args.resource_wait)

	tstart = time.time()
	data = ray.get(actor.request.remote())
	# if args.global_restart:
	# 	time.sleep(5)
	# 	ray.get(actor.kill_actor.remote())
	# 	print('exited')

	tstop = time.time()
	num_iters = list(data.keys())[0]
	runtime = list(data.values())[0]
	print("finished **************************************************")
	print("time: ", tstop-tstart)
	print("runtime: ", runtime)
	print("iterations: ", num_iters)
	print("avg. time per iteration: ", runtime / num_iters)
	print("finished **************************************************")
