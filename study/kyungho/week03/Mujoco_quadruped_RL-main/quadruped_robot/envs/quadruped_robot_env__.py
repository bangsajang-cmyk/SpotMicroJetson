import os
import gym
from gym import spaces
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import math
import pylab

env_name = "quadruped_robot-v0"
xml_path = 'unitree_a1/scene.xml'
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0
scores, episodes = [], []
_overlay = {}

def add_overlay(gridpos, text1, text2):
    if gridpos not in _overlay:
        _overlay[gridpos] = ["", ""]
    _overlay[gridpos][0] += text1 + "\n"
    _overlay[gridpos][1] += text2 + "\n"

def create_overlay(model, data, episode, total_return):
    topleft = mj.mjtGridPos.mjGRID_TOPLEFT
    topright = mj.mjtGridPos.mjGRID_TOPRIGHT
    bottomleft = mj.mjtGridPos.mjGRID_BOTTOMLEFT
    bottomright = mj.mjtGridPos.mjGRID_BOTTOMRIGHT
    add_overlay(bottomleft, "episode", str(episode),)
    add_overlay(bottomleft, "Time", '%.2f' % data.time,)
    add_overlay(bottomleft, "reward", '%.2f' % total_return,)

def init_controller(model, data):
    pass

def controller(model, data):
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if (not button_left) and (not button_middle) and (not button_right):
        return

    width, height = glfw.get_window_size(window)

    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

init_controller(model, data)
mj.set_mjcb_control(controller)

glfw.init()
window = glfw.create_window(1200, 900, env_name, None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
cam.azimuth = 90
cam.elevation = -30
cam.distance = 3
cam.lookat = np.array([0.0, 0, 0])


class Quadruped_robot(gym.Env):
    def __init__(self):
        super().__init__()
        # =========================
        # simulation params
        # =========================
        self.dt = 1.0 / 60.0
        # =========================
        # reward params (논문 Table VI 기준)
        # =========================
        # 목표 속도: body frame x축 방향 전진
        self.target_lin_vel = 0.5   # [m/s]  body x축 전진 속도
        self.target_ang_vel = 0.0   # [rad/s] yaw 유지
        self.tracking_sigma = 0.25  # 가우시안 추종 폭
        # =========================
        # action
        # =========================
        self.action = np.zeros(12, dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        # =========================
        # foot contact tracking (foot airtime bonus)
        # =========================
        self.feet_air_time = np.zeros(4, dtype=np.float32)
        self.last_feet_contact = np.zeros(4, dtype=bool)
        # =========================
        # observation
        # =========================
        #
        # state1: joint position        = 12
        # state2: body linear vel       = 3   (body frame)
        #         body angular vel      = 3   (body frame)
        #         projected gravity     = 3   (body frame)
        #         joint vel             = 12
        # state3: sensor                = 4
        # total = 37
        #
        # =========================
        self.sensorlist = [0, 1, 5, 8]
        self.observation_space = spaces.Box(
            low=-100,
            high=100,
            shape=(37,),
            dtype=np.float32
        )
        # =========================
        # action space
        # =========================
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )
        self.episode = 0
        self.train = False
        self.rend = False
        self.total_return = 0.0
        self.score_avg = 0.0
    # =========================================================
    # rotation matrix helper
    # =========================================================
    def _get_body_rot(self):
        """trunk body의 rotation matrix (world → body 변환용: R.T)"""
        body_id = model.body("trunk").id
        R = data.xmat[body_id].reshape(3, 3)
        return R
    # =========================================================
    # foot contact
    # =========================================================
    def _get_foot_contacts(self):
        """
        4개 발의 접촉 여부를 반환.
        MuJoCo contact 배열에서 발 geom과 바닥 geom 간 충돌을 확인.
        geom 이름이 모델에 따라 다를 수 있으므로 필요 시 수정.
        """
        foot_geom_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        contact_flags = np.zeros(4, dtype=bool)
        foot_geom_ids = []
        for name in foot_geom_names:
            try:
                foot_geom_ids.append(model.geom(name).id)
            except Exception:
                foot_geom_ids.append(-1)
        for i in range(data.ncon):
            c = data.contact[i]
            for fi, gid in enumerate(foot_geom_ids):
                if gid == -1:
                    continue
                if c.geom1 == gid or c.geom2 == gid:
                    contact_flags[fi] = True
        return contact_flags
    # =========================================================
    # observation
    # =========================================================
    def _get_obs(self):
        R = self._get_body_rot()
        # body frame 선속도 (world → body)
        v_world = np.array(data.qvel[0:3])
        v_body = R.T @ v_world
        # body frame 각속도 (world → body)
        w_world = np.array(data.qvel[3:6])
        w_body = R.T @ w_world
        # body frame 중력 벡터 (projected gravity)
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_body = R.T @ gravity_world
        # 관절 위치
        state1 = data.qpos[7:19].astype(np.float32)
        # 속도 관련 state
        state2 = np.concatenate([
            v_body,
            w_body,
            gravity_body,
            data.qvel[6:18]
        ]).astype(np.float32)
        # 센서 state
        state3 = np.array(
            [data.sensordata[i] for i in self.sensorlist],
            dtype=np.float32
        )
        return np.concatenate([state1, state2, state3])

    def _compute_reward(self):
        R = self._get_body_rot()
        # -------------------------------------------------------
        # body frame 속도 변환
        # -------------------------------------------------------
        v_world = np.array(data.qvel[0:3])
        v_body = R.T @ v_world        # [vx_body, vy_body, vz_body]
        w_world = np.array(data.qvel[3:6])
        w_body = R.T @ w_world        # [wx_body, wy_body, wz_body]
        vx = v_body[0]   # 전진 속도 (body x축)
        vy = v_body[1]   # 횡방향 속도
        vz = v_body[2]   # 수직 속도
        wx = w_body[0]   # roll 각속도
        wy = w_body[1]   # pitch 각속도
        wz = w_body[2]   # yaw 각속도
        # -------------------------------------------------------
        # projected gravity (자세 기울기 측정)
        # -------------------------------------------------------
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_body = R.T @ gravity_world
        gx = gravity_body[0]
        gy = gravity_body[1]
        # -------------------------------------------------------
        # [Task] xy 속도 추종 — 논문: exp(-|v_xy - v_cmd|² / σ)
        #
        # target_lin_vel = body x축 전진 속도 목표
        # vy 목표는 0 (옆으로 가지 않도록)
        # → "앞으로 걸어가기" 학습의 핵심 항목
        # -------------------------------------------------------
        vel_error_sq = (vx - self.target_lin_vel) ** 2 + vy ** 2
        r_vxy = np.exp(-vel_error_sq / self.tracking_sigma)
        # -------------------------------------------------------
        # [Task] yaw 추종 — 논문: exp(-(wz - wz_cmd)² / σ)
        #
        # target_ang_vel = 0 → yaw가 바뀌지 않도록 억제
        # yaw가 변하면 "앞"의 방향이 달라지므로 반드시 필요
        # -------------------------------------------------------
        yaw_error_sq = (wz - self.target_ang_vel) ** 2
        r_wz = np.exp(-yaw_error_sq / self.tracking_sigma)
        # -------------------------------------------------------
        # [Stability] z 속도 페널티 — 논문: -0.04 * vz²
        # -------------------------------------------------------
        r_vz = -0.04 * (vz ** 2)
        # -------------------------------------------------------
        # [Stability] roll/pitch 각속도 페널티 — 논문: -0.001 * |ω_xy|²
        # -------------------------------------------------------
        r_wp = -0.001 * (wx ** 2 + wy ** 2)
        # -------------------------------------------------------
        # [Stability] base orientation 페널티 — 논문: -0.002 * |g_ori_xy|²
        #
        # projected gravity의 xy 성분 → 몸체가 기울수록 증가
        # -------------------------------------------------------
        r_bori = -0.002 * (gx ** 2 + gy ** 2)
        # -------------------------------------------------------
        # [Smoothness] action rate 페널티 — 논문: -2e-4 * |a_{t-1} - a_t|²
        # -------------------------------------------------------
        r_arate = -2e-4 * np.sum((self.action - self.last_action) ** 2)
        # -------------------------------------------------------
        # [Smoothness] 관절 토크 페널티 — 논문: -2e-7 * |τ|²
        # -------------------------------------------------------
        torques = np.array(data.actuator_force)
        r_tau = -2e-7 * np.sum(torques ** 2)
        # -------------------------------------------------------
        # [Smoothness] foot airtime 보너스 — 논문: 0.02 * Σ(t_air * 1[new_contact])
        #
        # 발이 공중에 있던 시간만큼, 착지 순간에만 보너스 지급
        # → 보행 리듬(gait) 형성을 유도
        # -------------------------------------------------------
        feet_contact = self._get_foot_contacts()
        r_air = 0.0
        for i in range(4):
            if feet_contact[i] and not self.last_feet_contact[i]:
                # 이번 스텝에 새로 착지 → 체공 시간만큼 보너스
                r_air += 0.02 * self.feet_air_time[i]
                self.feet_air_time[i] = 0.0
            elif not feet_contact[i]:
                # 공중에 있는 동안 타이머 누적
                self.feet_air_time[i] += self.dt
        self.last_feet_contact = feet_contact.copy()
        
        reward = (
            1.0  * r_vxy    # 전진 속도 추종 (가장 중요)
            + 0.5  * r_wz   # yaw 유지
            + r_vz           # z 속도 억제
            + r_wp           # 기울기 억제
            + r_bori         # 자세 유지
            + r_arate        # 부드러운 action
            + r_tau          # 에너지 절약
            + r_air          # gait 리듬
        )

        return float(reward)

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
    
    def _is_done(self):
        # base height가 너무 낮으면 넘어진 것으로 판정
        if data.qpos[2] < 0.15:
            return True
        return False

    def step(self, action):
        # action 갱신 (rate 페널티를 위해 순서 중요)
        self.last_action = self.action.copy()
        self.action = action.copy()

        # 제어 입력 적용
        data.ctrl[:] = action

        # 시뮬레이션 스텝
        time_prev = data.time
        while (data.time - time_prev < self.dt):
            mj.mj_step(model, data)

        obs = self._get_obs()
        reward = self._compute_reward()
        self.total_return = reward
        terminated = self._is_done()
        truncated = False
        info = {}
        
        if terminated:
            self.plot(self.train)
            self.episode += 1
        if self.rend:
            self._render_frame()
        
        return obs, reward, terminated, truncated, info

    def settings(self, rend, train):
        self.train = train
        self.rend = rend
        if not self.rend:
            glfw.terminate()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mj.mj_resetData(model, data)

        # -------------------------------------------------------
        # [수정] yaw 초기화: 0으로 고정
        #
        # 기존: np.random.uniform(-π, π) → world x/y축 중 하나로
        #       학습되는 원인. body frame 변환을 써도 초기 방향이
        #       랜덤이면 "body x = 앞"이라는 개념이 일관되지 않음.
        #
        # 수정: yaw=0 고정 → 항상 world x축 방향이 body 앞(+x)
        #       학습이 안정화된 후 yaw 랜덤화를 재도입하면
        #       더 일반적인 방향성 학습도 가능.
        # -------------------------------------------------------

        yaw = 0.0   # 고정 (학습 초기 안정성 우선)
        # yaw = np.random.uniform(-np.pi, np.pi)  # 방향 일반화 시 주석 해제

        quat = np.array([
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2)
        ])
        data.qpos[3:7] = quat

        # 관절 위치/속도 소량 노이즈 (자세 다양성)
        data.qpos[7:19] += np.random.uniform(-0.02, 0.02, size=12)
        data.qvel[:] += np.random.uniform(-0.02, 0.02, size=model.nv)

        mj.mj_forward(model, data)

        # 상태 변수 초기화
        self.action[:] = 0.0
        self.last_action[:] = 0.0
        self.feet_air_time[:] = 0.0
        self.last_feet_contact[:] = False

        obs = self._get_obs()
        
        for i in range(10):
            time_prev = data.time
            while (data.time - time_prev < 1.0/60.0):
                mj.mj_step(model, data)
            if self.rend == True:
                create_overlay(model,data, self.episode, self.total_return)
                viewport_width, viewport_height = glfw.get_framebuffer_size(window)
                viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
                mj.mjv_updateScene(model, data, opt, None, cam,
                mj.mjtCatBit.mjCAT_ALL.value, scene)
                mj.mjr_render(viewport, scene, context)
                for gridpos, [t1, t2] in _overlay.items():
                    mj.mjr_overlay(
                        mj.mjtFontScale.mjFONTSCALE_150, gridpos, viewport, t1, t2, context)
                glfw.swap_buffers(window)
                glfw.poll_events()
                _overlay.clear()
        return obs, {}

    def plot(self, enable):
        if enable:
            self.score_avg = (0.9 * self.score_avg + 0.1 * self.total_return
                              if self.episode != 0 else self.total_return)
            scores.append(self.score_avg)
            episodes.append(self.episode)
            pylab.plot(episodes, scores, 'b')
            pylab.xlabel("episode")
            pylab.ylabel("average score")
            pylab.savefig("PPO_reward.png")
            