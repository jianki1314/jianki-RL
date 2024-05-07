import gym
from gym import spaces
import traci
import sumolib
import numpy as np
from scipy.spatial import distance
from math import atan2, degrees
from collections import deque
import math
import os

class SumoEnv(gym.Env):
    def __init__(self):
        #车辆信息相关变量
        self.vehicle = '0'
        self.presence = False #车辆存在性判断
        self.step_length = 1
        self.pos = (0,0)
        self.posx = 0 #x轴位置
        self.posy = 0 #y轴位置
        self.speed_x = 0 #x轴速度
        self.speed_y = 0 #y轴速度
        self.lane_heading_difference = 0 #车道航向差
        self.curr_lane = '' #车辆当前车道名称
        self.lane_distance = 0 #车辆与车道中心线的距离
        self.throttle = 0 #加速度输入
        self.action_space = spaces.Box(low=-3.0, high=4.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, -1000, -1000, -10, -10, -180, -10]), 
                                    high=np.array([1, 1000, 1000, 13.89, 13.89, 180, 10]), 
                                    dtype=np.float32)
        self.steering  = 0 #方向盘输入
        self.acc = 0 #车辆加速度
        self.acc_history = deque([0, 0], maxlen=2) #存储加速度值
        self.angle = 0 #车辆的角度
        self.gui = False #仿真界面启动开关值
        self.done = False

    
    def start(self,gui=False,network_cfg="networks/sumoconfig.sumo.cfg",network_net="networks/rouabout.net.xml"):
        self.gui = gui
        self.sumocfg = network_cfg
        self.net_xml = network_net
        self.done = False

        #starting sumo
        #home = os.getenv("HOME") or os.getenv("USERPROFILE")  # 兼容Windows环境
        home = "D:/sumo"
        print("home is:",home)
        if self.gui:
            sumoBinary = os.path.join(home,"bin/sumo-gui")
        else:
            sumoBinary = os.path.join(home,"bin/sumo")
        sumoCmd = [sumoBinary,"-c",self.sumocfg]
        traci.start(sumoCmd)
        print("Using SUMO binary:", sumoBinary)

        self.update_params()

    def update_params(self):
        # 获取基本车辆信息
        self.pos = traci.vehicle.getPosition(self.vehicle)
        self.posx,self.posy = self.pos
        self.curr_lane = traci.vehicle.getLaneID(self.vehicle)
        
        if self.curr_lane != '':
            self.speed_limit = traci.lane.getMaxSpeed(self.curr_lane)
            lane_angle = traci.lane.getShape(self.curr_lane)[-1][-1]  # 假设航向为车道形状末点的角度
            self.lane_heading_difference = lane_angle - self.angle
            self.lane_distance = traci.vehicle.getLateralLanePosition(self.vehicle)  
            # 车道的子标签（如果需要）
            self.curr_sublane = int(self.curr_lane.split("_")[1]) if "_" in self.curr_lane else 0
        else:
            self.speed_limit = 13.89
            self.lane_heading_difference = 0
            self.lane_distance = 0

        # 获取车辆速度及其他信息
        #self.speed_limit = traci.lane.getMaxSpeed(self.curr_lane)
        #self.target_speed = traci.vehicle.getAllowedSpeed(self.vehicle)
        self.speed_x = traci.vehicle.getSpeed(self.vehicle)
        self.speed_y = traci.vehicle.getLateralSpeed(self.vehicle)  # 如果需要横向速度
        self.acc = traci.vehicle.getAcceleration(self.vehicle)
        self.acc_history.append(self.acc)  # 您可能需要初始化 acc_history作为一个列表
        self.angle = traci.vehicle.getAngle(self.vehicle)

    def log_state_to_file(self,state):
        with open("vehicle_states.long","a") as file:
            file.write(f"{state}\n")
            
    def get_state(self):
        all_vehicle  =  traci.vehicle.getIDList()
        states = []
        for veh in all_vehicle:
            self.presence = True
            self.vehicle = veh
            self.update_params() #更新车辆信息
            state = (
                self.presence,
                self.posx,
                self.posy,
                self.speed_x,
                self.speed_y,
                self.lane_heading_difference,
                self.lane_distance         
            )
            states.append(state)
            #保存打印信息
            #elf.log_state_to_file(states)
            return np.array(states)
        
    def reset(self,gui=False):
        print("Resetting environment with GUI set to:", gui)
        self.start(gui)
        return self.get_state()

    def step(self,action):
        #将action中的加速度值输入到sumo中
        traci.vehicle.setAcceleration(self.vehicle, action, self.step_length)
        traci.simulationStep()
        reward = self.compute_reward()
        self.update_params()
        next_state = self.get_state()
        done = True
        
        return next_state, reward, done
    
    def compute_jerk(self):
        return (self.acc_history[1] - self.acc_history[0])/self.step_length
    
    def compute_reward(self):
        #安全奖励函数
        #lane_id = traci.vehicle.getLaneID(self.vehicle)
        L_width = traci.lane.getWidth(self.curr_lane)
        L_lateral = traci.vehicle.getLateralLanePosition(self.vehicle)
        R_lc = 1 - math.pow(L_lateral/L_width,2)

        leader = traci.vehicle.getLeader(self.vehicle)
        if leader is not None:
            leader_id, gap = leader
            leader_speed = traci.vehicle.getSpeed(leader_id)
            speed_diff = self.speed_x - leader_speed
            # 仅当本车速度大于领导车速度且gap大于0时，计算ttc
            if speed_diff > 0 and gap > 0:
                ttc = gap / speed_diff
            else:
                # 当本车速度小于等于领导车速度或gap不大于0时，设置ttc为一个很大的值
                ttc = float('inf')
        else:
            # 如果没有领导车，设置ttc为一个很大的值
            ttc = float('inf')
        # 如果ttc是无穷大，R_ttc应该设置为最大奖励值1
        if ttc == float('inf'):
            R_ttc = 1
        else:
            R_ttc = 1 - 3/ttc if ttc > 0 else -1 
        R_safe = 0.7 * R_lc + 0.3 * R_ttc

        #效率奖励
        speed_max = traci.vehicle.getMaxSpeed(self.vehicle)
        if self.speed_x > self.speed_limit:
            R_efficient = self.speed_x / self.speed_limit
        else:
            R_efficient = 1 - ( (self.speed_x - self.speed_limit) / (speed_max - self.speed_limit) )

        #舒适度计算
        jerk = self.compute_jerk()
        R_comfort = 1 - jerk/2

        #能耗奖励计算
        E_consumed = float(traci.vehicle.getParameter(self.vehicle, "device.battery.totalEnergyConsumed"))
        E_max = float(traci.vehicle.getParameter(self.vehicle, "device.battery.maximumBatteryCapacity"))
        E_regenerated= float(traci.vehicle.getParameter(self.vehicle, "device.battery.totalEnergyRegenerated"))
        R_energy = 1 - ( (E_consumed - E_regenerated) / E_max ) 
        
        R_total = 0.3*R_safe + 0.25*R_efficient + 0.1*R_comfort + 0.35*R_energy

        return R_total
    
    def close(self):
        traci.close()
