from  RL_agent.algorithms.MA2C_simple.ma2c import MA2C
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

architecture = 'my_net'

demand = 1.0
veh_rate = 5
wandb.init(name=architecture + '_MA2C_monash_tp_' + str(veh_rate) + '_blind_reward', project='fyp')

env = sumoEnv(gui=True, folder="monash_intersection",
              maxEnvSteps=3600, demand = demand,
              randomizeRoutes=False, constant_demand = True, vehicle_rate = veh_rate)

agents = MA2C(env, 13, 3, 5, architecture=architecture)
agents.train(summary_writer, log=True, test=False, episodes=10000)

