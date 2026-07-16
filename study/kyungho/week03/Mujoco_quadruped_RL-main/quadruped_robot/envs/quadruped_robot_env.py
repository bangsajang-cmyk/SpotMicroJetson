import os
import gym
from gym import spaces
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import pylab

# =============================================================================
# 전역 설정
# =============================================================================

env_name  = "quadruped_robot-v0"
xml_path  = 'unitree_a1/scene.xml'

button_left   = False
button_middle = False
button_right  = False
lastx = 0
lasty = 0

scores, episodes = [], []
_overlay = {}

# =============================================================================
# GLFW 오버레이 유틸리티
# =============================================================================

def add_overlay(gridpos, text1, text2):
    if gridpos not in _overlay:
        _overlay[gridpos] = ["", ""]
    _overlay[gridpos][0] += text1 + "\n"
    _overlay[gridpos][1] += text2 + "\n"

def create_overlay(model, data, episode, total_return):
    bottomleft = mj.mjtGridPos.mjGRID_BOTTOMLEFT
    add_overlay(bottomleft, "episode", str(episode))
    add_overlay(bottomleft, "Time",    "%.2f" % data.time)
    add_overlay(bottomleft, "reward",  "%.2f" % total_return)

# =============================================================================
# MuJoCo 컨트롤러 콜백 (미사용)
# =============================================================================

def init_controller(model, data):
    pass

def controller(model, data):
    pass

# =============================================================================
# GLFW 입력 콜백
# =============================================================================

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left   = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)   == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right  = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT)  == glfw.PRESS)
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx = xpos - lastx
    dy = ypos - lasty
    lastx, lasty = xpos, ypos

    if not (button_left or button_middle or button_right):
        return

    _, height = glfw.get_window_size(window)
    mod_shift = (
        glfw.get_key(window, glfw.KEY_LEFT_SHIFT)  == glfw.PRESS or
        glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    )

    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)

def scroll(window, xoffset, yoffset):
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)

# =============================================================================
# MuJoCo 전역 초기화
# =============================================================================

dirname  = os.path.dirname(__file__)
xml_path = os.path.join(dirname, xml_path)

model = mj.MjModel.from_xml_path(xml_path)
data  = mj.MjData(model)
cam   = mj.MjvCamera()
opt   = mj.MjvOption()

init_controller(model, data)
mj.set_mjcb_control(controller)

glfw.init()
window = glfw.create_window(1200, 900, env_name, None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
glfw.set_key_callback(window,           keyboard)
glfw.set_cursor_pos_callback(window,    mouse_move)
glfw.set_mouse_button_callback(window,  mouse_button)
glfw.set_scroll_callback(window,        scroll)

scene   = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

cam.azimuth   = 90
cam.elevation = -30
cam.distance  = 3
cam.lookat    = np.array([0.0, 0.0, 0.0])


# =============================================================================
# 환경 클래스
# =============================================================================

class Quadruped_robot(gym.Env):
    """
    Unitree A1 quadruped locomotion environment (MuJoCo).

    Reference:
        Argo-Robot/quadrupeds_locomotion
        https://github.com/Argo-Robot/quadrupeds_locomotion

    -------------------------------------------------------------------------
    Observation (dim = 48)
    -------------------------------------------------------------------------
        [0:3]   base linear  velocity (body frame)   vx, vy, vz
        [3:6]   base angular velocity (body frame)   wx, wy, wz
        [6:8]   orientation angles                    roll, pitch
        [8:20]  joint positions                       q_1 .. q_12
        [20:32] joint velocities                      dq_1 .. dq_12
        [32:44] previous actions                      a_{t-1}
        [44]    reference linear  vel x               vx_ref
        [45]    reference linear  vel y               vy_ref
        [46]    reference angular vel z               wz_ref
        [47]    reference altitude                    z_ref

    -------------------------------------------------------------------------
    Action (dim = 12)
    -------------------------------------------------------------------------
        Residual offsets from homing pose.
        data.ctrl = Q_HOMING + action

    -------------------------------------------------------------------------
    Reward
    -------------------------------------------------------------------------
        r_lin_vel      =  w * exp(-||v_xy_ref - v_xy||^2)   xy 속도 추종
        r_ang_vel      =  w * exp(-(wz_ref - wz)^2)         yaw 추종
        r_height       = -w * (z - z_ref)^2                 높이 유지
        r_pose         = -w * ||q - q_default||^2            기본 자세 유지
        r_action_rate  = -w * ||a_t - a_{t-1}||^2           부드러운 제어
        r_lin_vel_z    = -w * vz^2                          수직 속도 억제
        r_roll_pitch   = -w * (roll^2 + pitch^2)            기울기 억제

    -------------------------------------------------------------------------
    Termination
    -------------------------------------------------------------------------
        |roll|  > roll_th  or
        |pitch| > pitch_th or
        z       < z_min    or
        steps   >= max_steps
    """

    # homing 관절 위치 (residual action의 기준점)
    # [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
    #  RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
    Q_HOMING = np.array([
        0,  0.78, -1.53,
        0,  0.78, -1.53,
        0,  0.78, -1.53,
        0,  0.78, -1.53,
    ], dtype=np.float32)

    def __init__(self):

        # =====================================================================
        # Action / Observation spaces
        # =====================================================================

        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(12,), dtype=np.float32
        )
        # 3 + 3 + 2 + 12 + 12 + 12 + 4 = 48
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(48,), dtype=np.float32
        )

        # =====================================================================
        # 시뮬레이션 파라미터
        # =====================================================================

        self.dt = 1.0 / 60.0

        # =====================================================================
        # User command (목표값)
        # =====================================================================

        self.vx_ref = 0.5       # 전진 속도 [m/s]   (body x축)
        self.vy_ref = 0.0       # 횡방향 속도 [m/s]
        self.wz_ref = 0.0       # yaw 속도 [rad/s]
        self.z_ref  = 0.29      # 목표 base 높이 [m]

        # =====================================================================
        # 보상 가중치  (Argo-Robot 레퍼런스)
        # =====================================================================

        self.w_lin_vel     =  1.0
        self.w_ang_vel     =  0.5
        self.w_height      = -1.0
        self.w_pose        = -0.01
        self.w_action_rate = -0.01
        self.w_lin_vel_z   = -2.0
        self.w_roll_pitch  = -0.5

        # =====================================================================
        # 종료 조건 임계값
        # =====================================================================

        self.roll_th   = 0.8        # [rad]
        self.pitch_th  = 0.8        # [rad]
        self.z_min     = 0.15       # [m]
        self.max_steps = 1200       # 20 s × 60 Hz

        # =====================================================================
        # 에피소드 상태
        # =====================================================================

        self.episode      = 0
        self.curr_step    = 0
        self.train        = False
        self.rend         = True
        self.total_return = 0.0
        self.score_avg    = 0.0

        # =====================================================================
        # 이전 스텝 기록
        # =====================================================================

        self.last_action = np.zeros(12, dtype=np.float32)

        # reset의 기준이 되는 초기 qpos/qvel 저장
        self.qpos_init = data.qpos.copy()
        self.qvel_init = data.qvel.copy()

    # =========================================================================
    # 내부 헬퍼
    # =========================================================================

    def _get_body_rot(self) -> np.ndarray:
        """trunk body rotation matrix R (world 기준, 3×3)"""
        body_id = model.body("trunk").id
        return data.xmat[body_id].reshape(3, 3)

    def _get_rpy(self):
        """quaternion (qpos[3:7]) → roll, pitch [rad]"""
        qw = data.qpos[3]
        qx = data.qpos[4]
        qy = data.qpos[5]
        qz = data.qpos[6]
        roll  = np.arctan2(2.0 * (qw*qx + qy*qz),
                           1.0 - 2.0 * (qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2.0 * (qw*qy - qz*qx), -1.0, 1.0))
        return float(roll), float(pitch)

    # =========================================================================
    # Observation  (Argo-Robot 3.2)
    # =========================================================================

    def _get_obs(self, action: np.ndarray) -> np.ndarray:
        """
        48차원 관측 벡터를 반환.

        body-frame 변환:  v_body = R^T @ v_world
        → world 좌표계가 아닌 로봇의 '앞 방향' 기준으로 속도를 표현.
          어떤 방향으로 놓여 있어도 vx_body = 전진 속도가 되므로
          정책이 방향에 무관하게 '앞으로 가기'를 학습할 수 있다.
        """
        R = self._get_body_rot()

        v_body = R.T @ np.array(data.qvel[0:3])    # body-frame 선속도
        w_body = R.T @ np.array(data.qvel[3:6])    # body-frame 각속도

        roll, pitch = self._get_rpy()

        obs = np.concatenate([
            v_body,                                         # [0:3]
            w_body,                                         # [3:6]
            [roll, pitch],                                  # [6:8]
            data.qpos[7:19],                                # [8:20]  관절 위치
            data.qvel[6:18],                                # [20:32] 관절 속도
            action,                                         # [32:44] 이전 action
            [self.vx_ref, self.vy_ref,
             self.wz_ref, self.z_ref],                      # [44:48] user command
        ]).astype(np.float32)

        return obs

    # =========================================================================
    # Reward  (Argo-Robot 3.3)
    # =========================================================================

    def _compute_reward(self, action: np.ndarray) -> float:
        """
        7가지 보상 항목 합산.
        모두 body-frame 기준 속도를 사용 → 전진 방향이 항상 body x축.
        """
        R = self._get_body_rot()

        v_body = R.T @ np.array(data.qvel[0:3])
        w_body = R.T @ np.array(data.qvel[3:6])
        vx, vy, vz = v_body
        wz = w_body[2]

        roll, pitch = self._get_rpy()
        z = float(data.qpos[2])

        # 1. xy 속도 추종: exp(-||v_xy_ref - v_xy||^2)
        vel_err   = (vx - self.vx_ref)**2 + (vy - self.vy_ref)**2
        r_lin_vel = self.w_lin_vel * np.exp(-vel_err)

        # 2. yaw 추종: exp(-(wz_ref - wz)^2)
        r_ang_vel = self.w_ang_vel * np.exp(-(self.wz_ref - wz)**2)

        # 3. 높이 페널티: -(z - z_ref)^2
        r_height = self.w_height * (z - self.z_ref)**2

        # 4. 자세 유사도: -||q - q_default||^2
        q_curr = np.array(data.qpos[7:19], dtype=np.float32)
        r_pose = self.w_pose * float(np.sum((q_curr - self.Q_HOMING)**2))

        # 5. action rate: -||a_t - a_{t-1}||^2
        r_action_rate = self.w_action_rate * float(
            np.sum((action - self.last_action)**2)
        )

        # 6. 수직 속도 페널티: -vz^2
        r_lin_vel_z = self.w_lin_vel_z * (vz**2)

        # 7. roll/pitch 페널티: -(roll^2 + pitch^2)
        r_roll_pitch = self.w_roll_pitch * (roll**2 + pitch**2)

        return float(
            r_lin_vel + r_ang_vel + r_height
            + r_pose + r_action_rate + r_lin_vel_z + r_roll_pitch
        )

    # =========================================================================
    # 종료 조건  (Argo-Robot 3.4)
    # =========================================================================

    def _is_done(self, roll: float, pitch: float) -> bool:
        z = float(data.qpos[2])
        return (
            abs(roll)        > self.roll_th  or
            abs(pitch)       > self.pitch_th or
            z                < self.z_min    or
            self.curr_step  >= self.max_steps
        )

    # =========================================================================
    # 렌더링 헬퍼
    # =========================================================================

    def _render_frame(self):
        create_overlay(model, data, self.episode, self.total_return)
        vw, vh = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, vw, vh)
        mj.mjv_updateScene(
            model, data, opt, None, cam,
            mj.mjtCatBit.mjCAT_ALL.value, scene
        )
        mj.mjr_render(viewport, scene, context)
        for gridpos, [t1, t2] in _overlay.items():
            mj.mjr_overlay(
                mj.mjtFontScale.mjFONTSCALE_150,
                gridpos, viewport, t1, t2, context
            )
        glfw.swap_buffers(window)
        glfw.poll_events()
        _overlay.clear()

    # =========================================================================
    # step
    # =========================================================================

    def step(self, action: np.ndarray):
        """
        action: residual joint position offsets (shape 12, range [-0.5, 0.5])
                실제 제어값 = Q_HOMING + action
        반환: (obs, reward, done, info)
        """
        action = np.asarray(action, dtype=np.float32)

        # 제어 입력 적용
        data.ctrl[:] = self.Q_HOMING + action

        # 시뮬레이션 1 제어 주기 진행
        time_prev = data.time
        while data.time - time_prev < self.dt:
            mj.mj_step(model, data)

        self.curr_step += 1

        # 관측 / 보상 / 종료
        obs    = self._get_obs(action)
        reward = self._compute_reward(action)

        self.total_return += reward
        self.last_action   = action.copy()

        roll, pitch = self._get_rpy()
        done = self._is_done(roll, pitch)

        if done:
            self.plot(self.train)
            self.episode += 1

        if self.rend:
            self._render_frame()

        return obs, reward, done, {}

    # =========================================================================
    # reset  (Argo-Robot 3.5)
    # =========================================================================

    def reset(self) -> np.ndarray:
        """
        초기 위치/속도에 균일 노이즈를 추가하여 재시작.
        yaw = 0 고정 → body x축이 항상 전진 방향.
        """
        self.total_return = 0.0
        self.curr_step    = 0
        self.last_action  = np.zeros(12, dtype=np.float32)

        mj.mj_resetData(model, data)

        # 균일 노이즈로 초기 상태 다양화
        data.qpos[:] = self.qpos_init + np.random.uniform(
            -0.05, 0.05, size=model.nq
        )
        data.qvel[:] = self.qvel_init + np.random.uniform(
            -0.05, 0.05, size=model.nv
        )

        # yaw = 0 고정: quaternion [w, x, y, z] = [1, 0, 0, 0]
        data.qpos[3] = 1.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0

        # 초기 제어값 = homing 자세
        data.ctrl[:] = self.Q_HOMING

        mj.mj_forward(model, data)

        return self._get_obs(self.last_action)

    # =========================================================================
    # settings (렌더링/학습 모드 전환)
    # =========================================================================

    def settings(self, rend: bool, train: bool):
        self.train = train
        self.rend  = rend
        if not self.rend:
            glfw.terminate()

    # =========================================================================
    # 학습 곡선 저장
    # =========================================================================

    def plot(self, enable: bool):
        if not enable:
            return
        self.score_avg = (
            0.9 * self.score_avg + 0.1 * self.total_return
            if self.episode != 0 else self.total_return
        )
        scores.append(self.score_avg)
        episodes.append(self.episode)
        pylab.plot(episodes, scores, 'b')
        pylab.xlabel("episode")
        pylab.ylabel("average score")
        pylab.savefig("PPO_reward.png")