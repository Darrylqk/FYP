import numpy as np
from numpy import mean
import scipy
import random

from tqdm import tqdm

import os
import keras
import tensorflow as tf
from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Dropout, Flatten, Multiply
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda, LSTM, Concatenate, Add, Average
from keras.optimizers import Adam, Adamax, RMSprop
from keras import backend as K
from keras import regularizers

import wandb
import traci

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
set_session(tf.Session(config=config))

random.seed(2)
np.random.seed(2)
tf.set_random_seed(2)

def categorical_crossentropy(target, output):
    _epsilon =  tf.convert_to_tensor(10e-8, dtype=output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    return (- target * tf.log(output))

class A2C:
    def __init__(self, state_dims, action_dims, frames, lanes, intersection, lr = 1e-4):

        self.filename = "A2C_my_net"
        self.actions = action_dims
        self.states = state_dims
        self.frames = frames

        self.lanes = lanes
        self.intersection = intersection
        
        self.actor_learning_rate  = 1e-4
        self.critic_learning_rate = 2e-4
        self.gamma = 0.5

        self.l2_reg = regularizers.l2(1e-80)

        self.dummy_act_picked = np.zeros((1,self.actions)) #just a placeholder

        # Actor
        input_layer = Input(shape=(self.frames, self.states))
        act_picked = Input(shape=(self.actions,))
        x = Dense(256, activation='relu', kernel_regularizer = self.l2_reg)(input_layer)
        x = LSTM(64, return_sequences=True, kernel_regularizer = self.l2_reg)(x)
        y = Flatten()(x)
        act_prob = Dense(self.actions,activation='softmax')(y)
        selected_act_prob = Multiply()([act_prob,act_picked])
        selected_act_prob = Lambda(lambda x:K.sum(x, axis=-1, keepdims=True),output_shape=(1,))(selected_act_prob)
        actorModel = Model(inputs=[input_layer,act_picked], outputs=[act_prob, selected_act_prob])
        opt = Adam(lr=self.actor_learning_rate, amsgrad=True)
        actorModel.compile(loss=['mse',categorical_crossentropy], loss_weights=[0.0,1.0],optimizer=opt) # dont care about mse loss, it's just a placeholder
        self.actor = actorModel

        # Critic
        input_layer = Input(shape=(self.frames, self.states))
        x = Dense(256, activation='relu', kernel_regularizer = self.l2_reg)(input_layer)
        x = LSTM(64, return_sequences=True, kernel_regularizer = self.l2_reg)(x)
        y = Flatten()(x)
        output_critic = Dense(1, activation='linear')(y)

        criticModel =  Model(inputs=input_layer , outputs=output_critic)

        opt = Adam(lr=self.critic_learning_rate, amsgrad=True)
        criticModel.compile(loss='mse', optimizer=opt)
        self.critic = criticModel

    def policy_action(self, states, benchmark=False):
        input_list = states[:]
        input_list.append(self.dummy_act_picked.copy())
        action_probability = self.actor.predict(input_list)[0].flatten()

        if benchmark:
            act = np.argmax(action_probability)
        else:
            act = np.random.choice(np.arange(self.actions),p=action_probability)

        act_one_hot = np.zeros((1,self.actions))
        act_one_hot[0,act] = 1.0

        return act, act_one_hot

    def train(self, env, episodes = 10000):
        env.create_env_connection()

        tqdm_e = tqdm(range(episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            # Init
            env.restart_env()
            mean_reward = []
            wait_time = []
            veh_num = []
            emergency_wait_time = []
            emergency_veh_num = []
            
            done = False
            old_state, _, _ = env.step(self.lanes, frames = self.frames)
            
            # Training progress
            while not done:
                act, act_one_hot = self.policy_action(old_state)
                env.perform_actions(act, self.intersection, self.lanes)

                next_state, reward, done = env.step(self.lanes, frames = self.frames)
                mean_reward.append(reward)
                
                wait_time_temp = 0
                veh_num_temp = 0
                emergency_wait_time_temp = 0
                emergency_veh_num_temp = 0
                for lane in self.lanes:
                    veh_num_temp += traci.edge.getLastStepHaltingNumber(lane)
                    veh_list = traci.edge.getLastStepVehicleIDs(lane)
                    for veh in veh_list:
                        veh_type = traci.vehicle.getTypeID(veh)
                        veh_lane = traci.vehicle.getLaneID(veh)
                        veh_lane = veh_lane.split("_",1)[0]
                        acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                        if veh not in env.vehicles:
                            env.vehicles[veh] = {veh_lane: acc}
                        else:
                            env.vehicles[veh][veh_lane] = acc - sum([env.vehicles[veh][lane] for lane in env.vehicles[veh].keys() if lane != veh_lane])
                            
                        wait_time_temp += env.vehicles[veh][veh_lane]
                              
                        if veh_type == 'emergency':
                            emergency_wait_time_temp += env.vehicles[veh][veh_lane]
                            if traci.vehicle.getSpeed(veh) <= 0.1:
                                emergency_veh_num_temp += 1
                wait_time.append(wait_time_temp)
                veh_num.append(veh_num_temp)
                emergency_wait_time.append(emergency_wait_time_temp)
                emergency_veh_num.append(emergency_veh_num_temp)
                
                predict_reward = self.critic.predict(old_state)
                predict_next_reward = self.critic.predict(next_state)

                td_target = np.array([reward]) + self.gamma*predict_next_reward
                td_error = td_target - predict_reward

                
                self.critic.train_on_batch(old_state, td_target)

                input_list = old_state[:]
                input_list.append(act_one_hot.copy())
                self.actor.train_on_batch(input_list,[self.dummy_act_picked,td_error])

                old_state = next_state

            tqdm_e.set_description("Score: " + str(mean(mean_reward)))
            tqdm_e.refresh()

            if e % 10 == 0:
                self.save_weights(ep = e)
            wandb.log({'Episode Reward': mean(mean_reward), 'Waiting Time': mean(wait_time), 'No. of Vehicles Stopped': mean(veh_num), 'Emergency Waiting Time': mean(emergency_wait_time), 'No. of Emergency Vehicles Stopped': mean(emergency_veh_num)})

        env.close_env_connection()

    def save_weights(self, path="./RL_agent/models", ep=0):
        file_path = os.path.join(path, self.filename);
        self.actor.save_weights(file_path+"_actor_ep" + str(ep) + '.h5')
        self.critic.save_weights(file_path+"_critic_ep" + str(ep) + '.h5')

    def load_weights(self, path="./RL_agent/models", ep=0):
        file_path = os.path.join(path, self.filename)
        self.actor.load_weights(file_path+"_actor_ep"+ str(ep) + '.h5')
        self.critic.load_weights(file_path+"_critic_ep"+ str(ep) + '.h5')

    def test(self, env, seed, model):
        env.create_env_connection()
        if model == 1:
            self.load_weights(path='./RL_agent/models/my_net_single_3_tp_0.8', ep = 120)
        elif model == 2:
            self.load_weights(path='./RL_agent/models/my_net_single_3_tp_1.0_scale 0.4-1.4', ep=400)
        wait_time = []
        average_wait_time = []
        veh_num = []
        emergency_wait_time = []
        average_emergency_wait_time = []
        emergency_veh_num = []
        t = []
        
        done = False
        old_state, _, _ = env.step(self.lanes, frames = self.frames)

        while not done:
            act, act_one_hot = self.policy_action(old_state, benchmark = True)
            env.perform_actions(act, self.intersection, self.lanes)
            
            next_state, reward, done = env.step(self.lanes, frames = self.frames)
            t.append(traci.simulation.getTime())
            
            wait_time_temp = 0
            average_wait_time_temp = []
            veh_num_temp = 0
            emergency_wait_time_temp = 0
            average_emergency_wait_time_temp = []
            emergency_veh_num_temp = 0
            
            for lane in self.lanes:
                veh_num_temp += traci.edge.getLastStepHaltingNumber(lane)
                veh_list = traci.edge.getLastStepVehicleIDs(lane)
                for veh in veh_list:
                    veh_type = traci.vehicle.getTypeID(veh)
                    veh_lane = traci.vehicle.getLaneID(veh)
                    veh_lane = veh_lane.split("_",1)[0]
                    acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                    if veh not in env.vehicles:
                        env.vehicles[veh] = {veh_lane: acc}
                    else:
                        env.vehicles[veh][veh_lane] = acc - sum([env.vehicles[veh][lane] for lane in env.vehicles[veh].keys() if lane != veh_lane])
                        
                    wait_time_temp += env.vehicles[veh][veh_lane]
                    average_wait_time_temp.append(env.vehicles[veh][veh_lane])
                          
                    if veh_type == 'emergency':
                        emergency_wait_time_temp += env.vehicles[veh][veh_lane]
                        average_emergency_wait_time_temp.append(env.vehicles[veh][veh_lane])
                        if traci.vehicle.getSpeed(veh) <= 0.1:
                            emergency_veh_num_temp += 1
            wait_time.append(wait_time_temp)
            if average_wait_time_temp != []:
                average_wait_time.append(mean(average_wait_time_temp))
            else:
                average_wait_time.append(0)
            veh_num.append(veh_num_temp)
            emergency_wait_time.append(emergency_wait_time_temp)
            if average_emergency_wait_time_temp != []:
                average_emergency_wait_time.append(mean(average_emergency_wait_time_temp))
            else:
                average_emergency_wait_time.append(0)
            emergency_veh_num.append(emergency_veh_num_temp)
            
            old_state = next_state
        
        env.close_env_connection()
        return t, wait_time, average_wait_time, veh_num, emergency_wait_time, average_emergency_wait_time, emergency_veh_num
        
    def fixed_timer_test(self, env, duration, seed):
        env.create_env_connection()

        wait_time = []
        average_wait_time = []
        veh_num = []
        emergency_wait_time = []
        average_emergency_wait_time = []
        emergency_veh_num = []
        t = []
        
        done = False
        env.step(self.lanes, frames = self.frames)
        time_since_last_action = 0
        act = 1
        while not done:
            if traci.simulation.getTime() - time_since_last_action == duration:
                currentPhase = traci.trafficlight.getPhase(self.intersection)
                traci.trafficlight.setPhase(self.intersection, (currentPhase + 1) % 8)
                for _ in range(env.yellowPhaseTime):
                    env.sumo_step()
                traci.trafficlight.setPhase(self.intersection, (currentPhase + 2) % 8)
                time_since_last_action = traci.simulation.getTime()
            else:
                currentPhase = traci.trafficlight.getPhase(self.intersection)
                traci.trafficlight.setPhase(self.intersection, currentPhase)
            
            next_state, _, done = env.step(self.lanes, frames = self.frames)
            t.append(traci.simulation.getTime())
            
            wait_time_temp = 0
            average_wait_time_temp = []
            veh_num_temp = 0
            emergency_wait_time_temp = 0
            average_emergency_wait_time_temp = []
            emergency_veh_num_temp = 0 
            for lane in self.lanes:
                veh_num_temp += traci.edge.getLastStepHaltingNumber(lane)
                veh_list = traci.edge.getLastStepVehicleIDs(lane)
                for veh in veh_list:
                    veh_type = traci.vehicle.getTypeID(veh)
                    veh_lane = traci.vehicle.getLaneID(veh)
                    veh_lane = veh_lane.split("_",1)[0]
                    acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                    if veh not in env.vehicles:
                        env.vehicles[veh] = {veh_lane: acc}
                    else:
                        env.vehicles[veh][veh_lane] = acc - sum([env.vehicles[veh][lane] for lane in env.vehicles[veh].keys() if lane != veh_lane])
                        
                    wait_time_temp += env.vehicles[veh][veh_lane]
                    average_wait_time_temp.append(env.vehicles[veh][veh_lane])
                          
                    if veh_type == 'emergency':
                        emergency_wait_time_temp += env.vehicles[veh][veh_lane]
                        average_emergency_wait_time_temp.append(env.vehicles[veh][veh_lane])
                        if traci.vehicle.getSpeed(veh) <= 0.1:
                            emergency_veh_num_temp += 1
            wait_time.append(wait_time_temp)
            if average_wait_time_temp != []:
                average_wait_time.append(mean(average_wait_time_temp))
            else:
                average_wait_time.append(0)
            veh_num.append(veh_num_temp)
            emergency_wait_time.append(emergency_wait_time_temp)
            if average_emergency_wait_time_temp != []:
                average_emergency_wait_time.append(mean(average_emergency_wait_time_temp))
            else:
                average_emergency_wait_time.append(0)
            emergency_veh_num.append(emergency_veh_num_temp)
            
            old_state = next_state
        
        env.close_env_connection()
        return t, wait_time, average_wait_time, veh_num, emergency_wait_time, average_emergency_wait_time, emergency_veh_num
  


