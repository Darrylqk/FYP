from  RL_agent.algorithms.A2C_simple.a2c_my_net import A2C

from Interface.sumo_interface import sumoEnv

import os
import sys

import numpy as np
from numpy import mean
import pandas as pd
import tensorflow as tf
import keras

from keras.backend.tensorflow_backend import set_session
from keras import backend as K

import matplotlib.pyplot as plt

K.clear_session()

config = tf.ConfigProto()

sess = tf.Session(config=config)

print(keras.__version__)


set_session(sess)
summary_writer = tf.summary.FileWriter("./tensorboard_logs")

#architecture='deep_trafficnet_PO4'
architecture = 'my_net'

demand = 1.0
seed = 1

env = sumoEnv(gui=True, folder="single_intersection", maxEnvSteps=3600, demand=demand, seed = seed, constant_demand = True, vehicle_rate = 10)
intersection = env.intersections[0]
agent = A2C(17, 4, 5, env.networkDict[intersection], intersection)
#env.create_env_connection()
#env.restart_env()
#env.close_env_connection()
t1, wait_time1, average_wait_time1, veh_num1, emergency_wait_time1, average_emergency_wait_time1, emergency_veh_num1 = agent.test(env,seed,1)
t5, wait_time5, average_wait_time5, veh_num5, emergency_wait_time5, average_emergency_wait_time5, emergency_veh_num5 = agent.test(env,seed,2)
t2, wait_time2, average_wait_time2, veh_num2, emergency_wait_time2, average_emergency_wait_time2, emergency_veh_num2 = agent.fixed_timer_test(env,60,seed)
t3, wait_time3, average_wait_time3, veh_num3, emergency_wait_time3, average_emergency_wait_time3, emergency_veh_num3 = agent.fixed_timer_test(env,45,seed)
t4, wait_time4, average_wait_time4, veh_num4, emergency_wait_time4, average_emergency_wait_time4, emergency_veh_num4 = agent.fixed_timer_test(env,30,seed)

d = {'t': t1, 'Waiting Time': wait_time1, 'Average Waiting Time': average_wait_time1, 'No. of Vehicles Stopped': veh_num1, 'Emergency Waiting Time': emergency_wait_time1, 'Average Emergency Waiting Time': average_emergency_wait_time1, 'No. of Emergency Vehicles Stopped': emergency_veh_num1}
df = pd.DataFrame(data=d)
df.to_csv(r'./csv_files/A2C_1.8_120.csv', index = False, header = True)
d = {'t': t2, 'Waiting Time': wait_time2, 'Average Waiting Time': average_wait_time2,  'No. of Vehicles Stopped': veh_num2, 'Emergency Waiting Time': emergency_wait_time2, 'Average Emergency Waiting Time': average_emergency_wait_time2, 'No. of Emergency Vehicles Stopped': emergency_veh_num2}
df = pd.DataFrame(data=d)
df.to_csv(r'./csv_files/fixed_timer_60.csv', index = False, header = True)
d = {'t': t3, 'Waiting Time': wait_time3, 'Average Waiting Time': average_wait_time3,  'No. of Vehicles Stopped': veh_num3, 'Emergency Waiting Time': emergency_wait_time3, 'Average Emergency Waiting Time': average_emergency_wait_time3, 'No. of Emergency Vehicles Stopped': emergency_veh_num3}
df = pd.DataFrame(data=d)
df.to_csv(r'./csv_files/fixed_timer_45.csv', index = False, header = True)
d = {'t': t4, 'Waiting Time': wait_time4, 'Average Waiting Time': average_wait_time4,  'No. of Vehicles Stopped': veh_num4, 'Emergency Waiting Time': emergency_wait_time4, 'Average Emergency Waiting Time': average_emergency_wait_time4, 'No. of Emergency Vehicles Stopped': emergency_veh_num4}
df = pd.DataFrame(data=d)
df.to_csv(r'./csv_files/fixed_timer_30.csv', index = False, header = True)#
d = {'t': t5, 'Waiting Time': wait_time5, 'Average Waiting Time': average_wait_time5,  'No. of Vehicles Stopped': veh_num5, 'Emergency Waiting Time': emergency_wait_time5, 'Average Emergency Waiting Time': average_emergency_wait_time5, 'No. of Emergency Vehicles Stopped': emergency_veh_num5}
df = pd.DataFrame(data=d)
df.to_csv(r'./csv_files/A2C_0.4-1.4_400.csv', index = False, header = True)




