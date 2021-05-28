# Adapted from: https://github.com/veryprofessionalusername/FYP-trafficDensityDetectionandManagement

import glob
import math as m
import os
import random
import sys
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import traci.constants as tc
from sumo_constuctor import construct_sim


class sumoEnv(object):
    def __init__(self,
                folder = "",
                gui = False,
                maxEnvSteps = 3600,
                demand = 1.0,
                greenPhaseTime = 60,
                seed = -1,
                constant_demand = False,
                vehicle_rate = 1.0):

        if seed == -1:
            randomNumber = random.randint(random.randint(0, 100000), random.randint(100000, 200000))
        else:
            randomNumber = seed
        randomNumber = str(randomNumber)
            
        self.constant_demand = constant_demand    
        self.vehicle_rate = vehicle_rate    
            
        if gui:
            sumoBinary = "sumo-gui"
        else:
            sumoBinary = "sumo"

        self.sumoFolder = "sumo_files"

        if len(folder): self.sumoFolder = os.path.join(self.sumoFolder, folder)

        os.chdir(self.sumoFolder)
        cfgFile = glob.glob("*.sumocfg")[0]
        addFile = glob.glob("*.add.xml")[0]
        os.chdir("../..")

        sumocfgPath = os.path.join(self.sumoFolder, cfgFile)
        sumoAddPath = os.path.join(self.sumoFolder, addFile)

        self.sumoCmd = [sumoBinary,  "-c", sumocfgPath,
                                "-a", sumoAddPath,
                                "--random-depart-offset", "5",
                                "--seed", randomNumber,
                                "--random",
                                "-v", "false",
                                "--no-warnings",
                                "--start",
                                "--time-to-teleport","300",
                                "--waiting-time-memory", "10000",
                                "--no-step-log", "True",
                                "--scale", str(demand),
                                "--eager-insert"]

        self.greenPhaseTime = greenPhaseTime
        self.yellowPhaseTime = 3
        self.vehicles = dict()
        self.networkDict = self.create_env()
        self.intersections = list(self.networkDict.keys())

        self.agentNum = len(self.intersections)

        self.past_action = -1
        self.past_actions = []

        self.maxSteps = maxEnvSteps
        self.previousSpeedModeChanged = []

    def create_env(self):
        networkDict = construct_sim.createNetworkDict(folder=self.sumoFolder)
        return networkDict

    def get_network_states(self, lanes):
        max_wait_time = []
        num_emergency = []
        num_car = []
        num_motor = []
        for lane in lanes:
            veh_list = traci.edge.getLastStepVehicleIDs(lane)
            max_wait_time_temp = 0
            num_emergency_temp = 0
            num_car_temp = 0
            num_motor_temp = 0
            for veh in veh_list:
                if max_wait_time_temp < traci.vehicle.getWaitingTime(veh):
                    max_wait_time_temp = traci.vehicle.getWaitingTime(veh)
                if traci.vehicle.getTypeID(veh) == 'emergency':
                    num_emergency_temp += 1
                elif traci.vehicle.getTypeID(veh) == 'car':
                    num_car_temp += 1
                else:
                    num_motor_temp += 1
            max_wait_time.append(max_wait_time_temp)
            num_emergency.append(num_emergency_temp)
            num_car.append(num_car_temp)
            num_motor.append(num_motor_temp)
        current_phase = traci.trafficlight.getPhase(self.intersections[0])
        return np.array([current_phase] + max_wait_time + num_emergency + num_car + num_motor)

    def get_network_rewards(self, lanes, threshold = 0.4):
        wait_time = 0
        emergency_wait_time = 0
        veh_num = 0
        for lane in lanes:
            veh_num += traci.edge.getLastStepHaltingNumber(lane)
            veh_list = traci.edge.getLastStepVehicleIDs(lane)
            for veh in veh_list:
                veh_type = traci.vehicle.getTypeID(veh)
                veh_lane = traci.vehicle.getLaneID(veh)
                veh_lane = veh_lane.split("_",1)[0]
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.vehicles:
                    self.vehicles[veh] = {veh_lane: acc}
                else:
                    self.vehicles[veh][veh_lane] = acc - sum([self.vehicles[veh][lane] for lane in self.vehicles[veh].keys() if lane != veh_lane])
                    
                wait_time += self.vehicles[veh][veh_lane]
                      
                if veh_type == 'emergency':
                    emergency_wait_time += traci.vehicle.getWaitingTime(veh)
                    veh_num += 4
                elif veh_type == 'motor':
                    veh_num -= 0.5     
                    
        reward = -(wait_time/2 + 2*emergency_wait_time + veh_num)
        return reward

    def create_env_connection(self):
        self.vehicles = dict()
        traci.start(self.sumoCmd)

    def close_env_connection(self):
        traci.close()

    def restart_env(self, seed=-1):
        self.vehicles = dict()
        if self.constant_demand:
            for n in range(1, 10, 1):
                begin = self.maxSteps*(n-1)/9
                end = self.maxSteps*(n)/9
                os.system("python3 /usr/share/sumo/tools/randomTrips.py -o ./sumo_files/single_intersection/trips.trips.xml -n ./sumo_files/single_intersection/single_intersection.net.xml" +
                        " -a ./sumo_files/single_intersection/single_intersection.add.xml --trip-attribute=\"type=\\\"dist0\\\" departLane=\\\"best\\\" departSpeed=\\\"max\\\" departPos=\\\"random\\\"\"" +
                        " --route-file ./sumo_files/single_intersection/single_intersection" + str(n) + ".rou.xml -L --random --vtype-output vtypeout.xml --prefix a" + str(n) + "a -p " + str(1/self.vehicle_rate) + " -b " + str(begin) + " -e " + str(end))
        else:
            vehicle_rate = 0.4
            for n in range(1, 10, 1):
                if n > 1 and n <= 5:
                    vehicle_rate += (3-0.4)/4
                elif n > 5:
                    vehicle_rate -= (3-0.4)/4
                begin = self.maxSteps*(n-1)/9
                end = self.maxSteps*(n)/9
                if seed == -1:
                    os.system("python3 /usr/share/sumo/tools/randomTrips.py -o ./sumo_files/single_intersection/trips.trips.xml -n ./sumo_files/single_intersection/single_intersection.net.xml" +
                        " -a ./sumo_files/single_intersection/single_intersection.add.xml --trip-attribute=\"type=\\\"dist0\\\" departLane=\\\"best\\\" departSpeed=\\\"max\\\" departPos=\\\"random\\\"\"" +
                        " --route-file ./sumo_files/single_intersection/single_intersection" + str(n) + ".rou.xml -L --random --vtype-output vtypeout.xml --prefix a" + str(n) + "a -p " + str(1/vehicle_rate) + 
                        " -b " + str(begin) + " -e " + str(end))
                else:
                    os.system("python3 /usr/share/sumo/tools/randomTrips.py -o ./sumo_files/single_intersection/trips.trips.xml -n ./sumo_files/single_intersection/single_intersection.net.xml" +
                        " -a ./sumo_files/single_intersection/single_intersection.add.xml --trip-attribute=\"type=\\\"dist0\\\" departLane=\\\"best\\\" departSpeed=\\\"max\\\" departPos=\\\"random\\\"\"" +
                        " --route-file ./sumo_files/single_intersection/single_intersection" + str(n) + ".rou.xml -L --random -s " + str(seed) + " --vtype-output vtypeout.xml --prefix a" + str(n) + "a -p " + str(1/vehicle_rate) + 
                        " -b " + str(begin) + " -e " + str(end))
            
        traci.load(self.sumoCmd[1:])
        self.past_action = -1

    def step(self, lanes, frames):
        new_state_1 = 0 
        state_1_batch = []
        reward_batch = []
        done = False

        for _ in range(frames):
            self.sumo_step()
            if traci.simulation.getTime() >= self.maxSteps:
                done = True
            new_state_1 = self.get_network_states(lanes)
            state_1_batch.append(new_state_1.copy())
            reward_batch.append(self.get_network_rewards(lanes))

        reward = reward_batch[-1]

        state_1_array = np.array(state_1_batch)
        state_1_array = np.expand_dims(state_1_array, axis=0)

        input_list = [state_1_array]
        return input_list, reward, done

    def sumo_step(self):
        traci.simulationStep()
        self.speedModeChanged = []
        allVehicleID = traci.vehicle.getIDList()
        for x in allVehicleID:
            if traci.vehicle.getTypeID(x) == 'emergency':
                traci.vehicle.setSpeedMode(x,7)
                if (traci.vehicle.getLeader(x,25) != None):
                    y,_ = traci.vehicle.getLeader(x,25)
                    traci.vehicle.setSpeedMode(y,7)
                    self.speedModeChanged.append(y)
            elif x not in self.speedModeChanged:
                if self.previousSpeedModeChanged:
                    traci.vehicle.setSpeedMode(x,31)
        self.previousSpeedModeChanged = self.speedModeChanged[:]
        
    def perform_actions(self, action, intersection, lanes, past_action=-100):
        yellow_light_flag = False

        if past_action == -100:
            past_action = self.past_action;

        if action != past_action:
            current_phase = traci.trafficlight.getPhase(intersection)
            traci.trafficlight.setPhase(intersection, current_phase + 1)
        
            for _ in range(self.yellowPhaseTime):
               self.sumo_step()
            
            if  action == 0:
                traci.trafficlight.setPhase(intersection, 6)
            elif  action == 1:
                traci.trafficlight.setPhase(intersection, 0)
            elif action == 2:
                traci.trafficlight.setPhase(intersection, 2)
            elif action == 3:
                traci.trafficlight.setPhase(intersection, 4)
            
        else:
            if action == 0:
                traci.trafficlight.setPhase(intersection, 6)
            elif  action == 1:
                traci.trafficlight.setPhase(intersection, 0)
            elif action == 2:
                traci.trafficlight.setPhase(intersection, 2)
            elif action == 3:
                traci.trafficlight.setPhase(intersection, 4)
   
        self.past_action = action
 

