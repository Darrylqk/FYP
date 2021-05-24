from  RL_agent.algorithms.MA2C_simple.ma2c import MA2C
from Interface.sumo_interface import sumoEnv
from utils.write_to_csv import write_to_csv

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import csv

from keras.backend.tensorflow_backend import set_session
from keras import backend as K

K.clear_session()
config = tf.ConfigProto()
sess = tf.Session(config=config)

set_session(sess)

architecture = 'my_net'

demand = 1.0
vehicle_rate = 5.0

env = sumoEnv(gui=False, folder="monash_intersection",
              maxEnvSteps=3600, demand = demand,
              randomizeRoutes=False, constant_demand = False, vehicle_rate = vehicle_rate)
agents = MA2C(env, 13, 3, 5, architecture=architecture)
env.create_env_connection()
env.restart_env()
env.close_env_connection()
t1, wait_time1, average_wait_time1, veh_num1, emergency_wait_time1, average_emergency_wait_time1, emergency_veh_num1 = agents.test(env=env, path='./RL_agent/models/my_net_MA2C_monash_tp_5_blind', ep=610)
t2, wait_time2, average_wait_time2, veh_num2, emergency_wait_time2, average_emergency_wait_time2, emergency_veh_num2 = agents.fixed_timer_test(env, 60)
t3, wait_time3, average_wait_time3, veh_num3, emergency_wait_time3, average_emergency_wait_time3, emergency_veh_num3 = agents.fixed_timer_test(env, 45)
t4, wait_time4, average_wait_time4, veh_num4, emergency_wait_time4, average_emergency_wait_time4, emergency_veh_num4 = agents.fixed_timer_test(env, 30)
t5, wait_time5, average_wait_time5, veh_num5, emergency_wait_time5, average_emergency_wait_time5, emergency_veh_num5 = agents.test(env=env, path='./RL_agent/models/my_net_MA2C_monash_tp_5_blind', ep=1010)
t6, wait_time6, average_wait_time6, veh_num6, emergency_wait_time6, average_emergency_wait_time6, emergency_veh_num6 = agents.test(env=env, path='./RL_agent/models/my_net_MA2C_monash_tp_5_blind_reward', ep=530)

d = {'t': t1, 'Waiting Time': wait_time1, 'Average Waiting Time': average_wait_time1, 'No. of Vehicles Stopped': veh_num1, 'Emergency Waiting Time': emergency_wait_time1, 'Average Emergency Waiting Time': average_emergency_wait_time1, 'No. of Emergency Vehicles Stopped': emergency_veh_num1}
df = pd.DataFrame(data=d)
df.to_csv(r'./csv_files/blind_MA2C_610.csv', index = False, header = True)
d = {'t': t2, 'Waiting Time': wait_time2, 'Average Waiting Time': average_wait_time2,  'No. of Vehicles Stopped': veh_num2, 'Emergency Waiting Time': emergency_wait_time2, 'Average Emergency Waiting Time': average_emergency_wait_time2, 'No. of Emergency Vehicles Stopped': emergency_veh_num2}
df = pd.DataFrame(data=d)
df.to_csv(r'./csv_files/fixed_timer_60.csv', index = False, header = True)
d = {'t': t3, 'Waiting Time': wait_time3, 'Average Waiting Time': average_wait_time3,  'No. of Vehicles Stopped': veh_num3, 'Emergency Waiting Time': emergency_wait_time3, 'Average Emergency Waiting Time': average_emergency_wait_time3, 'No. of Emergency Vehicles Stopped': emergency_veh_num3}
df = pd.DataFrame(data=d)
df.to_csv(r'./csv_files/fixed_timer_45.csv', index = False, header = True)
d = {'t': t4, 'Waiting Time': wait_time4, 'Average Waiting Time': average_wait_time4,  'No. of Vehicles Stopped': veh_num4, 'Emergency Waiting Time': emergency_wait_time4, 'Average Emergency Waiting Time': average_emergency_wait_time4, 'No. of Emergency Vehicles Stopped': emergency_veh_num4}
df = pd.DataFrame(data=d)
df.to_csv(r'./csv_files/fixed_timer_30.csv', index = False, header = True)
d = {'t': t5, 'Waiting Time': wait_time5, 'Average Waiting Time': average_wait_time5,  'No. of Vehicles Stopped': veh_num5, 'Emergency Waiting Time': emergency_wait_time5, 'Average Emergency Waiting Time': average_emergency_wait_time5, 'No. of Emergency Vehicles Stopped': emergency_veh_num5}
df = pd.DataFrame(data=d)
df.to_csv(r'./csv_files/blind_MA2C_1010.csv', index = False, header = True)
d = {'t': t6, 'Waiting Time': wait_time6, 'Average Waiting Time': average_wait_time6,  'No. of Vehicles Stopped': veh_num6, 'Emergency Waiting Time': emergency_wait_time6, 'Average Emergency Waiting Time': average_emergency_wait_time6, 'No. of Emergency Vehicles Stopped': emergency_veh_num6}
df = pd.DataFrame(data=d)
df.to_csv(r'./csv_files/blind_MA2C_1010.csv', index = False, header = True)