import gymnasium as gym
import numpy as np
from gymnasium import spaces


class UnitreeG1Env(gym.Env):
    def __init__(self):
        super().__init__()

        # 관측값: 예시로 48차원 상태값 사용
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(48,),
            dtype=np.float32
        )

        # 행동값: 예시로 12개 관절 제어
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        self.state = np.zeros(48, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(48, dtype=np.float32)

        return self.state, {}

    def step(self, action):
        # 간단한 예제용 상태 변화
        self.state[:12] += action * 0.01

        # 보상값 예시
        reward = 1.0 - np.mean(np.square(action))

        terminated = False
        truncated = False

        info = {}

        return self.state, reward, terminated, truncated, info


if __name__ == "__main__":
    env = UnitreeG1Env()

    observation, info = env.reset()

    print("Observation shape:", observation.shape)
    print("Action space:", env.action_space)

    for i in range(10):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        print(
            f"Step {i + 1}: "
            f"reward={reward:.4f}"
        )

    env.close()