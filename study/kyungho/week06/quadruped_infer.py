import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
from mujoco.glfw import glfw

import jax
from jax import numpy as jp

from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks


# ------------------------------------------------------------------
# 학습 환경(BarkourEnv)과 반드시 일치해야 하는 상수들
# ------------------------------------------------------------------
ACTION_SCALE = 0.3
OBS_DIM = 31
HISTORY_LEN = 15
CONTROL_DT = 0.02  # 50 Hz — 학습 시 self._dt와 동일
POLICY_HIDDEN_LAYER_SIZES = (128, 128, 128, 128)

# 키보드 커맨드 크기 (barkour 학습 범위: vx[-0.6,1.5], vy[-0.8,0.8], wz[-0.7,0.7] 안쪽으로)
CMD_LIN_VEL_X = 1.0   # m/s
CMD_LIN_VEL_Y = 0.6   # m/s
CMD_ANG_VEL = 0.6     # rad/s


def quat_rotate_inverse(quat: np.ndarray, v: np.ndarray) -> np.ndarray:
    """월드 프레임 벡터 v를 quat=(w,x,y,z)이 나타내는 로컬(base) 프레임으로 회전.

    brax의 math.rotate(v, math.quat_inv(x.rot))와 동일한 연산.
    """
    w = quat[0]
    q_vec = quat[1:4]
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


class KeyboardCommand:
    """W/A/S/D, Q/E, SPACE, R 키로 (vx, vy, wz) 커맨드를 조종."""

    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.reset_requested = False

    def key_callback(self, keycode: int) -> None:
        if keycode == glfw.KEY_UP:
            self.vx = CMD_LIN_VEL_X
            return
        if keycode == glfw.KEY_DOWN:
            self.vx = -CMD_LIN_VEL_X
            return
        if keycode == glfw.KEY_LEFT:
            self.vy = CMD_LIN_VEL_Y
            return
        if keycode == glfw.KEY_RIGHT:
            self.vy = -CMD_LIN_VEL_Y
            return

        try:
            key = chr(keycode).upper()
        except ValueError:
            return

        if key == "W":
            self.vx = CMD_LIN_VEL_X
        elif key == "S":
            self.vx = -CMD_LIN_VEL_X
        elif key == "A":
            self.vy = CMD_LIN_VEL_Y
        elif key == "D":
            self.vy = -CMD_LIN_VEL_Y
        elif key == "Q":
            self.wz = CMD_ANG_VEL
        elif key == "E":
            self.wz = -CMD_ANG_VEL
        elif key == " ":
            self.vx = self.vy = self.wz = 0.0
        elif key == "R":
            self.reset_requested = True


class MiniCheetahSim:
    """실제 mujoco 물리 + observation 구성을 담당."""

    def __init__(self, scene_path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))

        self.model.opt.timestep = 0.004
        self.model.actuator_gainprm[:, 0] = 35.0
        self.model.actuator_biasprm[:, 1] = -35.0
        self.model.dof_damping[6:] = 0.5239

        self.data = mujoco.MjData(self.model)
        self.nu = self.model.nu  # 액추에이터(관절) 개수, mini_cheetah는 12

        home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key_id == -1:
            raise RuntimeError(
                f"'{scene_path}'에 'home' keyframe이 없습니다. "
                "학습 시 사용한 씬 파일과 동일한 파일을 지정했는지 확인하세요."
            )
        home_qpos = self.model.key_qpos[home_key_id]
        self.default_pose = np.array(home_qpos[7:7 + self.nu], dtype=np.float32)
        self._home_key_id = home_key_id

        # base free joint(qpos[0:7]) 다음 self.nu개가 다리 관절이라는 가정.
        self.lowers = self.model.jnt_range[1:1 + self.nu, 0].astype(np.float32)
        self.uppers = self.model.jnt_range[1:1 + self.nu, 1].astype(np.float32)

        self.n_substeps = max(1, int(round(CONTROL_DT / self.model.opt.timestep)))

        self.obs_history = np.zeros(OBS_DIM * HISTORY_LEN, dtype=np.float32)
        self.last_action = np.zeros(self.nu, dtype=np.float32)

        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        mujoco.mj_forward(self.model, self.data)
        self.last_action[:] = 0.0

        obs = self._compute_obs(0.0, 0.0, 0.0)
        self.obs_history[:] = 0.0
        self.obs_history[:OBS_DIM] = obs  # 나머지 history는 0으로 (학습 시 reset과 동일)

    def _compute_obs(self, cmd_vx: float, cmd_vy: float, cmd_wz: float) -> np.ndarray:
        d = self.data

        # yaw rate: free joint의 qvel[3:6]은 이미 로컬(base) 프레임 각속도.
        yaw_rate = d.qvel[5] * 0.25

        # projected gravity: 월드 중력 벡터를 base 로컬 프레임으로 회전.
        base_quat = d.qpos[3:7]  # (w, x, y, z)
        projected_gravity = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))

        command = np.array(
            [cmd_vx * 2.0, cmd_vy * 2.0, cmd_wz * 0.25], dtype=np.float32
        )

        joint_angle = d.qpos[7:7 + self.nu] - self.default_pose

        obs = np.concatenate(
            [[yaw_rate], projected_gravity, command, joint_angle, self.last_action]
        ).astype(np.float32)
        return np.clip(obs, -100.0, 100.0)

    def step(self, action: np.ndarray, cmd_vx: float, cmd_vy: float, cmd_wz: float) -> np.ndarray:
        motor_targets = self.default_pose + np.asarray(action, dtype=np.float32) * ACTION_SCALE
        motor_targets = np.clip(motor_targets, self.lowers, self.uppers)
        self.data.ctrl[: self.nu] = motor_targets

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._compute_obs(cmd_vx, cmd_vy, cmd_wz)
        # 최신 obs를 맨 앞에 두고, 나머지를 뒤로 밀어 오래된 마지막 블록을 버림.
        self.obs_history = np.concatenate([obs, self.obs_history[:-OBS_DIM]])
        self.last_action = np.asarray(action, dtype=np.float32)
        return self.obs_history.copy()


def build_inference_fn(obs_size: int, action_size: int, ckpt_path: Path):
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    params = brax_model.load_params(str(ckpt_path))
    inference_fn = make_policy(params, deterministic=True)
    return jax.jit(inference_fn)


def main():
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Mini Cheetah 키보드 조종 시뮬레이션")
    parser.add_argument(
        "--scene",
        type=Path,
        default=base_dir / "xml" / "scene_mjx.xml",
        help="xml 폴더 안의 씬 파일 경로 (기본값: xml/scene_mjx.xml)",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=base_dir / "mjx_brax_quadruped_policy",
        help="brax model.save_params로 저장한 정책 체크포인트 경로",
    )
    args = parser.parse_args()

    if not args.scene.exists():
        raise FileNotFoundError(
            f"씬 파일을 찾을 수 없습니다: {args.scene}\n"
            "xml 폴더 안의 실제 파일명으로 --scene 옵션을 지정하세요."
        )
    if not args.ckpt.exists():
        raise FileNotFoundError(f"정책 체크포인트를 찾을 수 없습니다: {args.ckpt}")

    sim = MiniCheetahSim(args.scene)
    obs_size = OBS_DIM * HISTORY_LEN
    inference_fn = build_inference_fn(obs_size, sim.nu, args.ckpt)

    cmd = KeyboardCommand()
    rng = jax.random.PRNGKey(0)

    print("조작: W/A/S/D 또는 방향키로 이동, Q/E 회전, SPACE 정지, R 리셋. 뷰어 창을 닫으면 종료됩니다.")

    with mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=cmd.key_callback
    ) as viewer:
        obs = sim.obs_history.copy()

        while viewer.is_running():
            step_start = time.time()

            if cmd.reset_requested:
                sim.reset()
                obs = sim.obs_history.copy()
                cmd.reset_requested = False

            rng, act_rng = jax.random.split(rng)
            action, _ = inference_fn(jp.asarray(obs), act_rng)
            action = np.asarray(action)

            obs = sim.step(action, cmd.vx, cmd.vy, cmd.wz)

            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = CONTROL_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()