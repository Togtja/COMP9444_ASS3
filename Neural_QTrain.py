# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import gym
import tensorflow as tf
import numpy as np
import random
# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =  0.9 # discount factor
INITIAL_EPSILON = 0.7 # starting value of epsilon
FINAL_EPSILON =  0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# Define Network Graph

hiddensize = 100
#Layer1
w1 = tf.Variable(tf.random_normal([STATE_DIM, hiddensize]))
b1 = tf.Variable(tf.random_normal([hiddensize]))
logits_1 = tf.matmul(state_in, w1) + b1
preds_1 = tf.nn.relu(logits_1) # tf.sigmoid() or tf.tanh() or tf.relu() or anyother
    
#Layer2
w2 = tf.Variable(tf.random_normal([hiddensize, hiddensize]))
b2 = tf.Variable(tf.random_normal([hiddensize]))
logits_2 = tf.matmul(preds_1, w2) + b2
preds_2 = tf.nn.relu(logits_2) # tf.sigmoid() or tf.tanh() or tf.relu() or anyother
####Bassed on some runs I didn don't use tf.nn.softmax()####

#Layer3
w3 = tf.Variable(tf.random_normal([hiddensize, ACTION_DIM]))
b3 = tf.Variable(tf.random_normal([ACTION_DIM]))

    

#Network outputs
q_values = tf.matmul(preds_2, w3) + b3 #Logtis of the neural value
q_action = \
         tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

#Loss/Optimizer Definition
loss = tf.reduce_sum(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer().minimize(loss)


# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

# take second element for sort
def takeThird(elem):
    return elem[2]

replay_buffer = []
BATCH_SIZE = 200
BUFFER_SIZE = 4* BATCH_SIZE
# Main learning loop
for episode in range(EPISODE):
    # initialize task
    state = env.reset()
    # Update epsilon once per episode
    epsilon -= (epsilon - FINAL_EPSILON) / EPSILON_DECAY_STEPS
    
    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
         
        replay_buffer.append((state, action, reward, next_state, done))
        """
        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })
        """
        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        #q_values = q_values + ( reward + GAMMA * np.max(nextstate_q_values) - q_values)
        """
        if step == 0:
            target = reward
        else:
            target = reward + GAMMA * np.argmax(nextstate_q_values)
        """
        # Do one training step

        if(len(replay_buffer) > BATCH_SIZE):
            #Instead of random, take a sample of BATCH_SIZE of the best rewards
            batch = random.sample(replay_buffer, BATCH_SIZE)
            """
            replay_buffer.sort(key=takeThird, reverse=True)
            batch = []
            
            for i in range (0, BATCH_SIZE):
                batch.append(replay_buffer[i])
            """
            state_batch = [data[0] for data in batch]
            action_batch = [data[1] for data in batch]
            reward_batch = [data[2] for data in batch]
            next_state_batch = [data[3] for data in batch]

            target_batch = []
            q_value_batch = q_values.eval(feed_dict={
                state_in: next_state_batch
            })
            for i in range(0, BATCH_SIZE):
                if batch[i][4]:
                    target_batch.append(reward_batch[i])
                else:
                    target_batch.append((reward_batch[i] + GAMMA * np.max(q_value_batch[i])))
            
            session.run([optimizer], feed_dict={
                target_in: target_batch,
                action_in: action_batch,
                state_in: state_batch
            })
        if(len(replay_buffer) > BUFFER_SIZE):
            #replay_buffer.pop(0)
            x = replay_buffer[0][2]
            pos = 0
            for i in range (1, len(replay_buffer)):
                if(replay_buffer[i][2] < x):
                    x = replay_buffer[i][2]
                    pos = i
            replay_buffer.pop(pos)
        
        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                #env.render() #Uncommet if you want to se the game
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
