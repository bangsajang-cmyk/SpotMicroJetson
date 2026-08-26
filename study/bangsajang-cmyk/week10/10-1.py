from playground import registry
from playground.config import Config
from playground.algorithms.ppo import PPO

# 환경 생성
env_name = "G1JoystickFlatTerrain"
env_config = registry.get_default_config(env_name)
env = registry.make(env_name, env_config)

# PPO 설정
ppo_config = Config(
    num_envs=4096,
    num_steps=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    total_timesteps=10_000_000,
)

# 학습 실행
ppo = PPO(env, ppo_config)
ppo.train()

# 학습된 모델 저장
ppo.save("g1_walking_policy.pkl")