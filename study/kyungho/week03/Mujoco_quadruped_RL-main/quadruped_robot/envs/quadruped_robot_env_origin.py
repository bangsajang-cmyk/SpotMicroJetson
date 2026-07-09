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
    add_overlay(bottomleft, "episode", str(episode) ,)
    add_overlay(bottomleft, "Time", '%.2f' % data.time,)
    add_overlay(bottomleft, "reward", '%.2f' % total_return,)

def init_controller(model,data):
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

init_controller(model,data)
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
cam.distance =  3
cam.lookat =np.array([0.0 , 0 , 0])

class Quadruped_robot(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(12, ), dtype="float32") #set action space size, range
        self.observation_space = spaces.Box(low=-10**2, high=10**2, shape=(34,), dtype="float32") ##set observation space size, range
        self.done = False
        self.episode = 0
        self.train = False
        self.rend = True
        self.total_return = 0
        self.score_avg = 0
        self.target_base = 0.29
        self.target_lin_vel = 0.8
        self.feet_time = [0, 0, 0, 0]
        self.feet_state = [0, 0, 0, 0]
        self.sensorlist = [0, 1, 5, 8]
        self.last_qvel = [0 for i in range(18)]
        self.res = [0, 0.78, -1.53, 0, 0.78, -1.53, 0, 0.78, -1.53, 0, 0.78, -1.53]

    def step(self, action):
        for i in range(12):
            data.ctrl[i] = action[i] + self.res[i]
        time_prev = data.time
        while (data.time - time_prev < 1.0/60.0):
            mj.mj_step(model, data)
        state1 = [data.qpos[i] for i in range(7, 19)]
        state2 = [data.qvel[i] for i in range(18)]
        state3 = [data.sensordata[i] for i in self.sensorlist]
        #state4 = [self.target_lin_vel, self.target_ang_vel]
        state = state1 + state2 + state3
        velocity = data.qvel[0] * state3[0] + data.qvel[1] * state3[1]
        base_height = state3[2]
        rvel = 5 * math.exp(-3*((self.target_lin_vel - velocity)**2)) - 3
        rbase = 5 * math.exp(-80*((self.target_base - base_height)**2)) - 3
        rzvel = -7 * abs(data.qvel[2])
        rang_vel = -0.3 * math.sqrt(data.qvel[3]**2 + data.qvel[4]**2)
        renergy = 0
        for i in range(12):
            renergy += -0.002 * abs(data.qvel[i + 6] * (data.qvel[i + 6] - self.last_qvel[i + 6]))
        reward = rvel + rbase + rzvel + rang_vel + renergy
        self.last_qvel = state2
        self.total_return  = self.total_return + reward
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
        if data.time >= 20 and self.train == True: #end episode
            self.done = True
            self.plot(self.train)
            self.episode = self.episode + 1
        elif state3[3] <= 0:
            self.done = True
            reward = reward - 10000 * (20 - data.time)/20 - 1000
            self.total_return  = self.total_return + reward
            self.plot(self.train)
            self.episode = self.episode + 1
        _overlay.clear()
        return state, reward, self.done, {}
    
    def settings(self, rend, train):
        self.train = train
        self.rend = rend
        if self.rend == False:
            glfw.terminate()

    def reset(self):
        self.total_return = 0
        self.done = False
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        for i in range(12):
            data.ctrl[i] = self.res[i]
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
        state1 = [data.qpos[i] for i in range(7, 19)]
        state2 = [data.qvel[i] for i in range(18)]
        state3 = [data.sensordata[i] for i in self.sensorlist]
        state = state1 + state2 + state3
        return state

    def plot(self, enable): #plot score graph
        if enable == True:
            self.score_avg = 0.9 * self.score_avg + 0.1 * self.total_return if self.episode != 0 else self.total_return 
            scores.append(self.score_avg)
            episodes.append(self.episode)
            pylab.plot(episodes, scores, 'b')
            pylab.xlabel("episode")
            pylab.ylabel("average score")
            pylab.savefig("PPO_reward.png") 