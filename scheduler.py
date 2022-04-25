import os
import torch
import random
import torch.nn as nn
from torch.optim import Adam

from collections import namedtuple, deque

import numpy as np

import torch.nn as nn
import environment as env

class Action:

    def __init__(self, task, machine):
        self.task = task
        self.machine = machine
    
    def __repr__(self):
        return 'Action(task={0} -> machine={1})'.format(self.task.label, self.machine.label)

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class SJFScheduler:
    def __init__(self, environment):
        self.environment = environment
    
    def schedule(self):
        """Higher priority for higher utilization."""
        actions = []
        indices = []

        # sort nodes according to reversed utilization, schedule tasks from queue to nodes
        for task_ind in range(len(self.environment.queue)):
            pairs = [(machine_ind, self.environment.machines[machine_ind].utlization()) for machine_ind in range(len(self.environment.machines))]
            pairs = sorted(pairs, key= lambda pair: pair[1], reverse=True)
            for pair in pairs:
                if self.environment.machines[pair[0]].schedule(self.environment.queue[task_ind]):
                    actions.append(Action(self.environment.queue[task_ind], self.environment.machines[pair[0]]))
                    indice.append(task_ind)
                    break
        
        for i in sorted(indices, reverse=True):
            del self.environment.queue[i]

        self.environment.timestep()
        return actions

    def get_total_rewards(self):

        rewards = 0
        tot_utils = 0
        self.environment, _ = env.set_up_enviroment(load_environment=True, load_scheduler=False)

        while not self.environment.terminated():
            
            indices = []
            # sort nodes according to reversed utilization, schedule tasks from queue to nodes
            for task_ind in range(len(self.environment.queue)):
                pairs = [(machine_ind, self.environment.machines[machine_ind].utilization()) for machine_ind in range(len(self.environment.machines))]
                pairs = sorted(pairs, key= lambda pair: pair[1], reverse=True)
                for pair in pairs:
                    if self.environment.machines[pair[0]].schedule(self.environment.queue[task_ind]):
                        indices.append(task_ind)
                        break

            for i in sorted(indices, reverse=True):
                del self.environment.queue[i]

            reward = self.environment.rewards()
            utils = self.environment.machine_utilization()

            tot_utils = tot_utils + utils
            rewards = rewards + reward
            self.environment.timestep()
        
        return rewards, tot_utils
        
class PackerScheduler:

    def __init__(self, environment):
        self.environment = environment

    def schedule(self):
        """Higher priority for lower utilization."""
        actions = []
        indices = []

        # sort nodes according to utilization, schedule tasks from queue to nodes
        for i_task in range(len(self.environment.queue)):
            pairs = [(i_node, self.environment.machines[i_node].utilization()) for i_node in range(len(self.environment.machines))]
            pairs = sorted(pairs, key=lambda pair: pair[1])
            for pair in pairs:
                if self.environment.machines[pair[0]].schedule(self.environment.queue[i_task]):
                    actions.append(Action(self.environment.queue[i_task], self.environment.machines[pair[0]]))
                    indices.append(i_task)
                    break
        for i in sorted(indices, reverse=True):
            del self.environment.queue[i]

        # proceed to the next timestep
        self.environment.timestep()

        return actions

    def get_total_rewards(self):

        rewards = 0
        tot_utils = 0
        self.environment, _ = env.set_up_enviroment(load_environment=True, load_scheduler=False)

        while not self.environment.terminated():

            indices = []
            # sort nodes according to utilization, schedule tasks from queue to nodes
            for i_task in range(len(self.environment.queue)):
                pairs = [(i_node, self.environment.machines[i_node].utilization()) for i_node in range(len(self.environment.machines))]
                pairs = sorted(pairs, key=lambda pair: pair[1])
                for pair in pairs:
                    if self.environment.machines[pair[0]].schedule(self.environment.queue[i_task]):
                        indices.append(i_task)
                        break
            for i in sorted(indices, reverse=True):
                del self.environment.queue[i]

            reward = self.environment.rewards()
            utils = self.environment.machine_utilization()

            rewards = rewards + reward
            tot_utils += utils
            # proceed to the next timestep
            self.environment.timestep()
        
        return rewards, tot_utils


class ExperienceReplay:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add_exp(self, *args):
        self.memory.append(Experience(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def pop_last(self):
        self.memory.pop()

    def __len__(self):
        return len(self.memory)


class DQN:

    def __init__(self, input_shape, output_shape, model_type='train'):
        self.lr = 0.001
        self.gamma = 0.99
        self.batch_size = 32
        self.min_experiences = 100
        self.max_experiences = 10000
        self.num_actions = output_shape
        self.save_dir = './models'
        self.model_path = self.save_dir + '/deep-rl-scheduler-1.pt'
        c,h,w = input_shape

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(12736, output_shape),
            nn.Softmax(dim=1)
        )
        # self.model = CNNModel(input_shape, output_shape)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.isfile(self.model_path):
            saved_dict = torch.load(self.model_path)
            self.model.load_state_dict(saved_dict['model'])
            self.optimizer.load_state_dict(saved_dict['optimizer'])

        self.exp = ExperienceReplay(self.max_experiences)
        self.loss_fn = nn.MSELoss()
        self.model.to(self.device)

    # prediction uitlity which calls the CNN model for the action as output
    def predict(self, input_data):
        input_data = input_data.type(torch.cuda.FloatTensor).to(self.device)
        return self.model(input_data)
    
    # Actions as returned by the CNN model. We do exploration vs exploitation depending on value of epsilon
    def get_action(self, states, epsilon):
        torch.no_grad()
        self.model.eval()
        if np.random.random() < epsilon:
            rand_ind = np.random.choice(self.num_actions)
            return torch.tensor([rand_ind]).to(self.device)
        else:
            x = self.predict(torch.unsqueeze(states, 0))
            return torch.argmax(x, dim=1)
    
    def train(self, dqn_target):
        if len(self.exp) < self.min_experiences:
            return

        experiences = self.exp.sample(self.batch_size)
        batches = Experience(*zip(*experiences))

        states = torch.cat(batches.state).to(self.device)
        actions = torch.cat(batches.action).to(self.device)
        rewards = torch.cat(batches.reward).to(self.device)
        next_states = torch.cat(batches.next_state).to(self.device)
        dones = torch.cat(batches.done).to(self.device)

        torch.enable_grad()
        self.model.train()
        self.optimizer.zero_grad()

        values_next = torch.max(dqn_target.predict(torch.unsqueeze(next_states, 1)), axis=1)
        actual_values = torch.where(dones == 0, rewards, rewards + self.gamma * values_next.values)

        pred = torch.sum(self.predict(torch.unsqueeze(states, 1)) * torch.eye(self.num_actions).to(self.device)[actions.type(torch.cuda.LongTensor)], dim=1)

        loss = self.loss_fn(actual_values, pred)

        loss.backward()
        self.optimizer.step()


    def add_experience(self, *args):
        if len(self.exp) >= self.max_experiences:
            self.exp.pop_last()
        self.exp.add_exp(*args)
        

    def copy_weights(self, dqn_src):
        self.model.load_state_dict(dqn_src.model.state_dict())

    def save_weights(self, step):
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        model_path = self.save_dir + '/deep-rl-scheduler-1.pt'
        torch.save(
            dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict()),
            model_path
        )
        # print(f"DeepRmScheduler saved to {model_path}")


class DeepRMTrainer:

    def __init__(self, environment):
        self.episodes = 500
        self.copy_steps = 32
        self.save_steps = 32

        self.epsilon = 0.99
        self.decay = 0.99
        self.min_epsilon = 0.1

        env_sum = environment.summary()
        input_shape = (1, env_sum.shape[0], env_sum.shape[1])
        output_shape = environment.queue_size * len(environment.machines) + 1

        self.dqn_train = DQN(input_shape, output_shape)
        self.dqn_target = DQN(input_shape, output_shape)

        self.total_rewards = torch.empty(self.episodes)
        self.total_utilization = torch.empty(self.episodes)
        self.environment = environment

    def train(self):
        torch.cuda.empty_cache()
        with open('training_results.txt', 'w+') as file:
            file.write('Iterations,JobSlowdown\n')
            for i in range(self.episodes):
                self.epsilon = max(self.min_epsilon, self.decay*self.epsilon)
                self.total_rewards[i], self.total_utilization[i] = self.train_episodes()
                print(f'\nEpisode {i} Job slowdown is {-self.total_rewards[i]}\n')
                file.write(f'{str(i+1)},{str(-self.total_rewards[i]),{str(self.total_utilization[i])}}\n')

    # Train episodes trains the dqn model until termination
    def train_episodes(self):
        rewards = 0
        total_util = 0
        step = 0
        
        self.environment, _ = env.set_up_enviroment(load_environment=True, load_scheduler=False)
        while not self.environment.terminated():
            # Get action index from current obs and retrieve task_index and machine_index for scheduling
            observation = self.environment.summary()
            action_index = self.dqn_train.get_action(observation, self.epsilon)
            task_index, machine_index = self.explain(action_index)

            #invalid action, proceed to next time step
            if task_index < 0 or machine_index < 0:
                self.environment.timestep()
                continue
            
            # Get the task and machine and start the scheduling process.
            scheduled_task = self.environment.queue[task_index]
            scheduled_machine = self.environment.machines[machine_index]
            scheduled = scheduled_machine.schedule(scheduled_task)

            # Continue to next timestep if not scheduled 
            if not scheduled:
                self.environment.timestep()
                continue

            # Get rewards and save the experience into the replay buffer
            del self.environment.queue[task_index]
            prev_observation = observation
            reward = self.environment.rewards()
            util = self.environment.machine_utilization()

            observation = self.environment.summary()
            rewards = rewards + reward
            total_util = total_util + util

            self.dqn_train.add_experience(prev_observation, action_index, torch.tensor([rewards]), observation, torch.tensor([self.environment.terminated()]))
            self.dqn_train.train(self.dqn_target)

            step += 1

            # Periodically copy the train_dqn into the target_dqn
            if step != 0 and step % self.copy_steps == 0:
                self.dqn_target.copy_weights(self.dqn_train)

            # Periodically save the train dqn
            if step != 0 and step % self.save_steps == 0:
                self.dqn_target.save_weights(step)

        return rewards, total_util

    # The action space is defined by i*q + j, where i is the machine index, q is the queue size and j is the task index. 
    def explain(self, action_index):
        task_limit = self.environment.queue_size
        machine_limit = len(self.environment.machines)

        # The void action is chosen
        if action_index == task_limit * machine_limit:
            task_index = torch.tensor([-1])
            machine_index = torch.tensor([-1])

        else:
            task_index = action_index % task_limit
            machine_index = torch.div(action_index, task_limit, rounding_mode='floor')

        # The job doesnt fit
        if task_index >= len(self.environment.queue):
            task_index = torch.tensor([-1])
            machine_index = torch.tensor([-1])
        return (task_index, machine_index)

class DeepRMScheduler:

    def __init__(self, env, train=True):

        if train:
            DeepRMTrainer(env).train()

        input_shape = (1, env.summary().shape[0], env.summary().shape[1])
        output_shape = env.queue_size * len(env.machines) + 1
        self.dqn_train = DQN(input_shape, output_shape)
        self.environment = env

    # Scheduling action using the trained model. Same as scheduling while training
    def schedule(self):

        actions = []

        while True:
            observation = self.environment.summary()
            action_index = self.dqn_train.get_action(observation, 0)
            task_index, machine_index = self.explain(action_index)

            if task_index < 0 or machine_index < 0:
                break

            scheduled_task = self.environment.queue[task_index]
            scheduled_machine = self.environment.machines[machine_index]
            scheduled = scheduled_machine.schedule(scheduled_task)
            if not scheduled:
                break

            del self.environment.queue[task_index]
            actions.append(Action(scheduled_task, scheduled_machine))

        self.environment.timestep()
        return actions

    # The action space is defined by i*q + j, where i is the machine index, q is the queue size and j is the task index. 
    def explain(self, action_index):
        task_limit = self.environment.queue_size
        machine_limit = len(self.environment.machines)

        # The void action is chosen
        if action_index == task_limit * machine_limit:
            task_index = torch.tensor([-1])
            machine_index = torch.tensor([-1])

        else:
            task_index = action_index % task_limit
            machine_index = torch.div(action_index, task_limit, rounding_mode='floor')

        # The job doesnt fit
        if task_index >= len(self.environment.queue):
            task_index = torch.tensor([-1])
            machine_index = torch.tensor([-1])
        return (task_index, machine_index)

    def get_total_rewards(self):

        rewards = 0
        self.environment, _ = env.set_up_enviroment(load_environment=True, load_scheduler=False)
        while not self.environment.terminated():
            observation = self.environment.summary()
            action_index = self.dqn_train.get_action(observation, 0)
            task_index, machine_index = self.explain(action_index)

            # Invalid action, proceed to next time step
            if task_index < 0 or machine_index < 0:
                self.environment.timestep()
                continue
            
            # Get the task and machine and start the scheduling process.
            scheduled_task = self.environment.queue[task_index]
            scheduled_machine = self.environment.machines[machine_index]
            scheduled = scheduled_machine.schedule(scheduled_task)

            # Continue to next timestep if not scheduled 
            if not scheduled:
                self.environment.timestep()
                continue

            del self.environment.queue[task_index]
            reward = self.environment.rewards()
            rewards = rewards + reward
        
        return rewards


        

class ComparisonClass:

    def __init__(self, environment):
        # self.deeprl_s = DeepRMScheduler(environment, train=False)
        self.sjf_s = SJFScheduler(environment)
        self.packer_s = PackerScheduler(environment)
        self.iterations = 500

    def comparison(self):
        with open('./comparison_results.txt', 'w+') as file:
            file.write('Iterations,ShortestJobFirst, Packer\n')
            for i in range(self.iterations):
                sjf_reward, sjf_tot_utils = self.sjf_s.get_total_rewards()
                packer_reward, packer_tot_utils = self.packer_s.get_total_rewards()
                file.write(f'{str(i+1)},{str(-sjf_reward)},{str(sjf_tot_utils)},{str(-packer_reward)},{str(packer_tot_utils)}\n')
                print(f'\nIteration {i+1} of {self.iterations} done\n')
