# Adapted from: https://github.com/veryprofessionalusername/FYP-trafficDensityDetectionandManagement

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

import traci
import wandb

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

random.seed(2)
np.random.seed(2)
tf.set_random_seed(2)

# ref : https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/

def categorical_crossentropy(target, output):
    _epsilon =  tf.convert_to_tensor(10e-8, dtype=output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    return (- target * tf.log(output))


class MA2C:
    def __init__(self, env, state_dims, action_dims, frames, lr = 1e-4):

        self.actions = action_dims
        self.states = state_dims
        self.frames = frames
        self.env = env
        
        self.actor_learning_rate  = 1e-4
        self.critic_learning_rate = 2e-4
        self.gamma = 0.5

        self.l2_reg = regularizers.l2(1e-80)

        self.dummy_act_picked = np.zeros((1,self.actions)) #just a placeholder

        self.agents = {i: self.create_actor_critic_model() for i in self.env.networkDict}

    def create_actor_critic_model(self):
        # Actor
        input_layer_1 = Input(shape=(self.frames, self.states))
        act_picked = Input(shape=(self.actions,))

        self.filename = "my_net"
        x = Dense(256, activation='relu', kernel_regularizer = self.l2_reg)(input_layer_1)
        x = LSTM(64, return_sequences=True, kernel_regularizer = self.l2_reg)(x)
        y = Flatten()(x)

        act_prob = Dense(self.actions,activation='softmax')(y)

        selected_act_prob = Multiply()([act_prob,act_picked])
        selected_act_prob = Lambda(lambda x:K.sum(x, axis=-1, keepdims=True),output_shape=(1,))(selected_act_prob)
            
        actorModel = Model(inputs=[input_layer_1,act_picked], outputs=[act_prob, selected_act_prob])
           
        opt = Adam(lr=self.actor_learning_rate, amsgrad=True)
        actorModel.compile(loss=['mse',categorical_crossentropy], loss_weights=[0.0,1.0],optimizer=opt) #we dont care about mse loss, it's just a placeholder
        actor = actorModel

        # Critic
        input_layer_1 = Input(shape=(self.frames, self.states))
        x = Dense(256, activation='relu', kernel_regularizer = self.l2_reg)(input_layer_1)
        x = LSTM(64, return_sequences=True, kernel_regularizer = self.l2_reg)(x)
        y = Flatten()(x)
        output_critic = Dense(1, activation='linear')(y)
        criticModel = Model(inputs=input_layer_1 , outputs=output_critic)

        opt = Adam(lr=self.critic_learning_rate, amsgrad=True)
        criticModel.compile(loss='mse', optimizer=opt)
        critic = criticModel

        return {'actor': actor, 'critic': critic}

    def policy_action(self, agent_actor, states, benchmark=False):
        input_list = states[:]
        input_list.append(self.dummy_act_picked.copy())
        action_probability = agent_actor.predict(input_list)[0].flatten()

        if benchmark:
            act = np.argmax(action_probability)
        else:
            act = np.random.choice(np.arange(self.actions),p=action_probability)

        act_one_hot = np.zeros((1,self.actions))
        act_one_hot[0,act] = 1.0

        return act, act_one_hot

    def train(self, episodes = 10000):
        self.env.create_env_connection()
        lanes = traci.edge.getIDList()

        try:
            self.load_weights(ep=150)
            print("Loaded weights\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        except:
            print("Creating new model\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

        tqdm_e = tqdm(range(episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
        # Init
            self.env.restart_env()
            mean_reward = {x: [] for x in self.env.networkDict}
            wait_time = []
            veh_num = []
            emergency_wait_time = []
            emergency_veh_num = []
            done = False

            old_state, _, _ = self.env.step(frames = self.frames)
            act_dict = {i:-1 for i in self.env.networkDict}
            act_one_hot_dict = {i:-1 for i in self.env.networkDict}
            # Training progress
            while not done:
                for intersection in old_state:
                    act_dict[intersection], act_one_hot_dict[intersection] = self.policy_action(self.agents[intersection]['actor'], old_state[intersection])

                self.env.perform_actions(act_dict)

                next_state, reward, done = self.env.step(frames = self.frames)
                for (x, y) in zip(reward, reward.values()):
                    mean_reward[x].append(y)
                    
                wait_time_temp = 0
                veh_num_temp = 0
                emergency_wait_time_temp = 0
                emergency_veh_num_temp = 0
                for lane in lanes:
                    veh_num_temp += traci.edge.getLastStepHaltingNumber(lane)
                    veh_list = traci.edge.getLastStepVehicleIDs(lane)
                    for veh in veh_list:
                        veh_type = traci.vehicle.getTypeID(veh)
                        veh_lane = traci.vehicle.getLaneID(veh)
                        veh_lane = veh_lane.split("_",1)[0]
                        acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                        if veh not in self.env.vehicles:
                            self.env.vehicles[veh] = {veh_lane: acc}
                        else:
                            self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                            
                        wait_time_temp += self.env.vehicles[veh][veh_lane]
                              
                        if veh_type == 'emergency':
                            emergency_wait_time_temp += self.env.vehicles[veh][veh_lane]
                            if traci.vehicle.getSpeed(veh) <= 0.1:
                                emergency_veh_num_temp += 1
                wait_time.append(wait_time_temp)
                veh_num.append(veh_num_temp)
                emergency_wait_time.append(emergency_wait_time_temp)
                emergency_veh_num.append(emergency_veh_num_temp)

                predict_reward = {i: self.agents[i]['critic'].predict(x) for i, x in zip(old_state, old_state.values())}
                predict_next_reward = {i: self.agents[i]['critic'].predict(x) for i, x in zip(next_state, next_state.values())}

                td_target = {i: np.array([reward[i]]) + self.gamma*PRN for i, PRN in zip(predict_next_reward, predict_next_reward.values())}
                td_error = {i: td_target[i] - predict_reward[i] for i in predict_reward.keys()}

                for i in self.env.networkDict:
                    self.agents[i]['critic'].train_on_batch(old_state[i], td_target[i])

                    input_list = old_state[i][:]
                    input_list.append(act_one_hot_dict[i].copy())
                    self.agents[i]['actor'].train_on_batch(input_list,[self.dummy_act_picked,td_error[i]])

                old_state = next_state
            avg_reward = mean((np.array(mean_reward[list(reward.keys())[0]]) + np.array(mean_reward[list(reward.keys())[1]]))/2)
            
            tqdm_e.set_description("Score: " + str(avg_reward))
            tqdm_e.refresh()
            if e % 10 == 0:
                self.save_weights(ep = e + 150)
            wandb.log({'Episode Reward': avg_reward,
                'Agent 1 Reward': mean(mean_reward[list(reward.keys())[0]]), 
                'Agent 2 Reward': mean(mean_reward[list(reward.keys())[1]]), 
                'Waiting Time': mean(wait_time), 
                'No. of Vehicles Stopped': mean(veh_num), 
                'Emergency Waiting Time': mean(emergency_wait_time), 
                'No. of Emergency Vehicles Stopped': mean(emergency_veh_num)})
            
        self.env.close_env_connection()


    def save_weights(self, path="./RL_agent/models", ep=0):
        file_path = os.path.join(path, self.filename);
        for i in self.env.networkDict:
            self.agents[i]['actor'].save_weights(file_path +"_"+ i + "_agent_actor_ep" + str(ep) + '.h5')
            self.agents[i]['critic'].save_weights(file_path +"_"+ i + "_agent_critic_ep" + str(ep) + '.h5')


    def load_weights(self, path="./RL_agent/models", ep=0):
        file_path = os.path.join(path, self.filename);
        for i in self.env.networkDict:
            file_path = os.path.join(path, self.filename)
            self.agents[i]['actor'].load_weights(file_path +"_"+ i + "_agent_actor_ep" + str(ep) + '.h5')
            self.agents[i]['critic'].load_weights(file_path +"_"+ i + "_agent_critic_ep" + str(ep) + '.h5')

    def test(self, env, path, ep):
        env.create_env_connection()
        lanes = traci.edge.getIDList()
        self.load_weights(path=path, ep=ep)

        wait_time = []
        average_wait_time = []
        veh_num = []
        emergency_wait_time = []
        average_emergency_wait_time = []
        emergency_veh_num = []
        t = []
        done = False
        
        
        old_state, _, _ = self.env.step(frames = self.frames)
        act_dict = {i:-1 for i in self.env.networkDict}
        act_one_hot_dict = {i:-1 for i in self.env.networkDict}
        
        while not done:
            for intersection in old_state:
                act_dict[intersection], act_one_hot_dict[intersection] = self.policy_action(self.agents[intersection]['actor'], old_state[intersection], benchmark=True)

            self.env.perform_actions(act_dict)

            next_state, _, done = self.env.step(frames = self.frames)
            t.append(traci.simulation.getTime())
                
            wait_time_temp = 0
            average_wait_time_temp = []
            veh_num_temp = 0
            emergency_wait_time_temp = 0
            emergency_veh_num_temp = 0
            average_emergency_wait_time_temp = []
            for lane in lanes:
                veh_num_temp += traci.edge.getLastStepHaltingNumber(lane)
                veh_list = traci.edge.getLastStepVehicleIDs(lane)
                for veh in veh_list:
                    veh_type = traci.vehicle.getTypeID(veh)
                    veh_lane = traci.vehicle.getLaneID(veh)
                    veh_lane = veh_lane.split("_",1)[0]
                    acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                    if veh not in self.env.vehicles:
                        self.env.vehicles[veh] = {veh_lane: acc}
                    else:
                        self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
          
                    wait_time_temp += self.env.vehicles[veh][veh_lane]
                    average_wait_time_temp.append(self.env.vehicles[veh][veh_lane])
                          
                    if veh_type == 'emergency':
                        emergency_wait_time_temp += self.env.vehicles[veh][veh_lane]
                        average_emergency_wait_time_temp.append(self.env.vehicles[veh][veh_lane])
                        if traci.vehicle.getSpeed(veh) <= 0.1:
                            emergency_veh_num_temp += 1
            wait_time.append(wait_time_temp)
            if average_wait_time_temp != []:
                average_wait_time.append(mean(average_wait_time_temp))
            else:
                average_wait_time.append(0)
            veh_num.append(veh_num_temp)
            emergency_wait_time.append(emergency_wait_time_temp)
            emergency_veh_num.append(emergency_veh_num_temp)
            if average_emergency_wait_time_temp != []:
                average_emergency_wait_time.append(mean(average_emergency_wait_time_temp))
            else:
                average_emergency_wait_time.append(0)

            old_state = next_state
            
        self.env.close_env_connection()
        return t, wait_time, average_wait_time, veh_num, emergency_wait_time, average_emergency_wait_time, emergency_veh_num
              
    def fixed_timer_test(self, env, duration):
        env.create_env_connection()
        lanes = traci.edge.getIDList()
        done = False

        old_state, _, _ = self.env.step(frames = self.frames)
        time_since_last_action = 0
        wait_time = []
        average_wait_time = []
        veh_num = []
        emergency_wait_time = []
        average_emergency_wait_time = []
        emergency_veh_num = []
        t = []
        traci.trafficlight.setPhase('jt2', 3)
        while not done:
            if traci.simulation.getTime() - time_since_last_action == duration:
                for intersection in old_state:
                    currentPhase = traci.trafficlight.getPhase(intersection)
                    if currentPhase == 0:
                        traci.trafficlight.setPhase(intersection, currentPhase + 2)
                    else:
                        traci.trafficlight.setPhase(intersection, currentPhase + 1)
                for _ in range(env.yellowPhaseTime):
                    env.sumo_step()
                for intersection in old_state:
                    currentPhase = traci.trafficlight.getPhase(intersection)
                    if currentPhase == 4:
                        traci.trafficlight.setPhase(intersection, currentPhase + 2)
                    else:
                        traci.trafficlight.setPhase(intersection, (currentPhase + 1) % 8)
                time_since_last_action = traci.simulation.getTime()
            else:
                for intersection in old_state:
                    currentPhase = traci.trafficlight.getPhase(intersection)
                    traci.trafficlight.setPhase(intersection, currentPhase)

            next_state, _, done = self.env.step(frames = self.frames)
            t.append(traci.simulation.getTime())
                
            wait_time_temp = 0
            average_wait_time_temp = []
            veh_num_temp = 0
            emergency_wait_time_temp = 0
            emergency_veh_num_temp = 0
            average_emergency_wait_time_temp = []
            for lane in lanes:
                veh_num_temp += traci.edge.getLastStepHaltingNumber(lane)
                veh_list = traci.edge.getLastStepVehicleIDs(lane)
                for veh in veh_list:
                    veh_type = traci.vehicle.getTypeID(veh)
                    veh_lane = traci.vehicle.getLaneID(veh)
                    veh_lane = veh_lane.split("_",1)[0]
                    acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                    if veh not in self.env.vehicles:
                        self.env.vehicles[veh] = {veh_lane: acc}
                    else:
                        self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                    
                    wait_time_temp += self.env.vehicles[veh][veh_lane]
                    average_wait_time_temp.append(self.env.vehicles[veh][veh_lane])
                          
                    if veh_type == 'emergency':
                        emergency_wait_time_temp += self.env.vehicles[veh][veh_lane]
                        average_emergency_wait_time_temp.append(self.env.vehicles[veh][veh_lane])
                        if traci.vehicle.getSpeed(veh) <= 0.1:
                            emergency_veh_num_temp += 1
            wait_time.append(wait_time_temp)
            if average_wait_time_temp != []:
                average_wait_time.append(mean(average_wait_time_temp))
            else:
                average_wait_time.append(0)
            veh_num.append(veh_num_temp)
            emergency_wait_time.append(emergency_wait_time_temp)
            emergency_veh_num.append(emergency_veh_num_temp)
            if average_emergency_wait_time_temp != []:
                average_emergency_wait_time.append(mean(average_emergency_wait_time_temp))
            else:
                average_emergency_wait_time.append(0)

            old_state = next_state
            
        self.env.close_env_connection()
        return t, wait_time, average_wait_time, veh_num, emergency_wait_time, average_emergency_wait_time, emergency_veh_num

