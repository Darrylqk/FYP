from  RL_agent.algorithms.A2C_simple.a2c_my_net import A2C

from Interface.sumo_interface import sumoEnv

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from keras.backend.tensorflow_backend import set_session
from keras import backend as K

import wandb

K.clear_session()

config = tf.ConfigProto()

sess = tf.Session(config=config)

print(keras.__version__)


set_session(sess)
summary_writer = tf.summary.FileWriter("./tensorboard_logs")

#architecture='deep_trafficnet_PO4'
architecture = 'my_net'

demand = 1

wandb.init(name=architecture + '_single_3_tp_' + str(0.8), project='fyp')

env = sumoEnv(gui=False, folder="single_intersection", maxEnvSteps=3600, demand=demand , randomizeRoutes=False, constant_demand = True, vehicle_rate = 0.8)
intersection = env.intersections[0]

agent = A2C(17, 4, 5, env.networkDict[intersection], intersection, architecture=architecture)
agent.train(env, summary_writer, log=True, test=False, episodes=10000)
