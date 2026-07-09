from gym.envs.registration import register

register(
    id='quadruped_robot-v0',
    entry_point='quadruped_robot.envs:Quadruped_robot',
)