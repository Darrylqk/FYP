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

class sumoEnv(object):
    def __init__(self,
                folder = "",
                gui = False,
                maxEnvSteps = 3600,
                demand = 1.0,
                greenPhaseTime = 60,
                constant_demand = True,
                vehicle_rate = 1.0):

        randomNumber = random.randint(random.randint(0, 100000), random.randint(100000, 200000))
        randomNumber = str(randomNumber)
        if gui:
            sumoBinary = "sumo-gui"
        else:
            sumoBinary = "sumo"

        self.sumoFolder = "sumo_files"

        self.constant_demand = constant_demand
        self.vehicle_rate = vehicle_rate
        
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

        self.networkDict = {'jt1': ['-l1.2', 'l2.2', 'l3.1'], 'jt2': ['-l3.3', 'l4.2', 'l5.2']}
        self.networkDictExits = {'jt1': ['l1', '-l2', '-l3.1', '-l3.2', '-l3.3'], 'jt2': ['-l4', '-l5', 'l3.3', 'l3.2', 'l3.1']}
        
        self.intersections = list(self.networkDict.keys())

        self.agentNum = len(self.intersections)

        self.past_action = {intersection: -1 for intersection in self.networkDict}

        #define step parameter
        self.maxSteps = maxEnvSteps
        self.previousSpeedModeChanged = []

    def get_network_states(self, intersection, lanes):
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
        current_phase = traci.trafficlight.getPhase(intersection)
        return np.array([current_phase] + max_wait_time + num_emergency + num_car + num_motor)

    def get_network_rewards(self, assigned_intersection, lanes):
        wait_time = 0
        emergency_wait_time = 0
        veh_num = 0
        for lane in lanes:
            num_lane = traci.edge.getLaneNumber(lane)
            veh_num += traci.edge.getLastStepHaltingNumber(lane)/num_lane
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
                    
                wait_time += self.vehicles[veh][veh_lane]/num_lane
                      
                if veh_type == 'emergency':
                    emergency_wait_time += traci.vehicle.getWaitingTime(veh)/num_lane
                    veh_num += 4/num_lane
                elif veh_type == 'motor':
                    veh_num -= 0.5/num_lane
                    
        reward = -(wait_time/2 + 2*emergency_wait_time + veh_num)
        
        responsible_intersections = []
        for intersection in self.networkDict:
            for controlled_lanes in self.networkDict[assigned_intersection]:
                if controlled_lanes in self.networkDictExits[intersection]:
                    responsible_intersections.append(intersection)
            
        return responsible_intersections, reward

    def create_env_connection(self):
        self.vehicles = dict()
        traci.start(self.sumoCmd)

    def close_env_connection(self):
        traci.close()

    def restart_env(self):
        self.vehicles = dict()
        vehicle_rate = 2
        for n in range(1, 10, 1):
            if self.constant_demand:
                vehicle_rate = self.vehicle_rate
            else:
                if n > 1 and n <= 5:
                    vehicle_rate += (10-2)/4
                elif n > 5:
                    vehicle_rate -= (10-2)/4
            begin = self.maxSteps*(n-1)/9
            end = self.maxSteps*(n)/9
            os.system("python3 /usr/share/sumo/tools/randomTrips.py -o ./sumo_files/monash_intersection/trips.trips.xml -n ./sumo_files/monash_intersection/monash_intersection.net.xml" +
                " -a ./sumo_files/monash_intersection/monash_intersection.add.xml --trip-attribute=\"type=\\\"dist0\\\" departLane=\\\"best\\\" departSpeed=\\\"max\\\" departPos=\\\"random\\\"\"" +
                " --route-file ./sumo_files/monash_intersection/monash_intersection" + str(n) + ".rou.xml -L --random --vtype-output vtypeout.xml --prefix a" + str(n) + "a -p " + str(1/vehicle_rate) + 
                " -b " + str(begin) + " -e " + str(end))
        traci.load(self.sumoCmd[1:])
        self.past_action = {intersection: -1 for intersection in self.networkDict}

    def step(self, frames):
        state_dict = {x: [] for x in self.networkDict}
        reward_local_dict = {x: 0 for x in self.networkDict}
        responsibilty_dict = {x: 0 for x in self.networkDict}
        reward_combined_dict = {x: 0 for x in self.networkDict}
        done = False

        for _ in range(frames):
            self.sumo_step()
            if traci.simulation.getTime() >= self.maxSteps:
                done = True

            for intersection in self.networkDict:
                new_state = self.get_network_states(intersection, self.networkDict[intersection])
                state_dict[intersection].append(new_state.copy())
            
        for intersection in self.networkDict:
            responsibilty_dict[intersection], reward_local_dict[intersection] = self.get_network_rewards(intersection, self.networkDict[intersection])

        for intersection in self.networkDict:
            responsibilty_reward = 0
            for i in responsibilty_dict:
                if intersection == responsibilty_dict[i]:
                    responsibilty_reward += 0.5*reward_local_dict[i]

            reward_combined_dict[intersection] = reward_local_dict[intersection] #+ responsibilty_reward
        
        input_dict = {x: [np.expand_dims(i, axis=0)] for x,i in zip(self.networkDict, list(state_dict.values()))}
        
        return input_dict, reward_combined_dict, done
 
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

    def perform_actions(self, action_dict, benchmark=False):
        yellow_light_flag = False

        for intersection, action in zip(action_dict, list(action_dict.values())):
            if action != self.past_action[intersection]:
                current_phase = traci.trafficlight.getPhase(intersection)
                if current_phase == 0 and action == 2:
                    traci.trafficlight.setPhase(intersection, current_phase + 2)
                elif current_phase == 3 and action == 1:
                    traci.trafficlight.setPhase(intersection, current_phase + 2)
                else:
                    traci.trafficlight.setPhase(intersection, current_phase + 1)
                    
                for _ in range(self.yellowPhaseTime):
                    self.sumo_step()
                   
                if  action == 0:
                    traci.trafficlight.setPhase(intersection, 6)
                elif action == 1:
                    traci.trafficlight.setPhase(intersection, 0)
                elif action == 2:
                    traci.trafficlight.setPhase(intersection, 3)
                        
            else:
                if  action == 0:
                    traci.trafficlight.setPhase(intersection, 6)
                elif action == 1:
                    traci.trafficlight.setPhase(intersection, 0)
                elif action == 2:
                    traci.trafficlight.setPhase(intersection, 3)

                self.past_action[intersection] = action


