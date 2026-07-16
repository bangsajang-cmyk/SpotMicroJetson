import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

class Custom_callback(BaseCallback):

    def __init__(self, model, env, env_name, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.num_timesteps = 0
        self.env_name = env_name
        self.model = model
        self.env = env


    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        self.num_timesteps += 1
        if self.num_timesteps % 10000 == 0:
            log_dir = "PPO_pretrained_model/" + "time_steps_" + str(self.num_timesteps)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_dir = log_dir + '/' + self.env_name+ '/'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.model.save(log_dir + "ppo_" + self.env_name)
            stats_path = os.path.join(log_dir, "vec_normalize.pkl")
            self.env.save(stats_path)
            print("modelsaved at" + log_dir)
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass