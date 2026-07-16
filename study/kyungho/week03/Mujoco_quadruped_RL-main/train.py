import gym
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import custom_callback as callback
from quadruped_robot.envs import quadruped_robot_env

render = True
retrain = False
num_envs = 8 #
env_name = 'quadruped_robot-v0'

log_dir = "PPO_trained_model"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_dir = log_dir + '/' + env_name+ '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

start_time = datetime.now().replace(microsecond=0)

env = gym.make(env_name)
env.settings(rend = render, train = True)
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
if retrain == True:
    model = PPO.load(log_dir + "ppo_" + env_name, env=vec_env, learning_rate=0.0002)
else:
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0002)

model_callback = callback.Custom_callback(model, vec_env, env_name)
model.learn(total_timesteps=100_00, callback=model_callback)

log_dir = "PPO_trained_model"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_dir = log_dir + '/' + env_name+ '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

model.save(log_dir + "ppo_" + env_name)
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
vec_env.save(stats_path)

print("modelsaved at" + log_dir)

end_time = datetime.now().replace(microsecond=0)

print("--------------------------------------------------------------------------------------------")
print("Started training at : ", start_time)
print("Finished training at : ", end_time)
print("Total training time : ", end_time - start_time)
print("--------------------------------------------------------------------------------------------")

env = gym.make(env_name)
env.settings(rend = True, train = False)
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize.load(stats_path, vec_env)
vec_env.training = False
vec_env.norm_reward = False

# obs = vec_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)