import os
import numpy as np
import torch
import json
from PIL import Image

from task import Task
from machine import Machine
from scheduler import DeepRMScheduler


class Environment:

    def __init__(self, machines, queue_size, backlog_size, task_gen):
        self.machines = machines
        self.queue_size = queue_size
        self.backlog_size = backlog_size
        self.task_gen = task_gen

        self.queue = []
        self.backlog = []
        self.timestep_counter = 0
        self.task_gen_end = False

    # This functons steps the enviroment by one time point
    def timestep(self):
        self.timestep_counter += 1

        # Step each machine by one timestep
        for machine in self.machines:
            machine.timestep()

        # keep adding tasks in backlog to main queue
        queue_ind = len(self.queue)
        backlog_ind = 0
        indices = []

        while queue_ind < self.queue_size and backlog_ind < len(self.backlog):
            self.queue.append(self.backlog[backlog_ind])
            indices.append(backlog_ind)
            backlog_ind += 1
            queue_ind += 1

        for i in sorted(indices, reverse=True):
            del self.backlog[i]

        # Keep adding new tasks into backlog
        backlog_ind = len(self.backlog)
        while backlog_ind <= self.backlog_size:
            new_task = next(self.task_gen, None)

            if new_task is None:
                self.task_gen_end = True
                break
            else:
                self.backlog.append(new_task)
                backlog_ind += 1

    # Reward is the average job slowdown
    # This is the inverse of the sum of duration of tasks in the machines, queue and backlog
    def rewards(self):
        r = 0
        for machine in self.machines:
            if machine.scheduled_tasks:
                r += 1/sum([task[0].duration for task in machine.scheduled_tasks])
        if self.queue:
            r += 2/sum([task.duration for task in self.queue])
        if self.backlog:
            r += 1/sum([task.duration for task in self.backlog])
        return -r

    def machine_utilization(self):
        tot_ut = 0
        for machine in self.machines:
            mac_ut = machine.utilization()
            tot_ut += mac_ut
        return tot_ut 

    def summary(self, bg_shape=None):

        # bg_shape
        # row: is the max duration capacity of all machines.
        # col: is the max resource capacity across all resources in all machines
        if bg_shape is None:
            bg_col = max([max(machine.resources) for machine in self.machines])
            bg_row = max([machine.duration for machine in self.machines])
            bg_shape = (bg_row, bg_col)

        if len(self.machines) > 0:
            dimension = self.machines[0].dimension

            # Getting state summary of each machine amd adding it to temp
            temp = self.machines[0].summary(bg_shape)
            for i in range(1, len(self.machines)):
                temp = np.concatenate((temp, self.machines[i].summary(bg_shape)), axis=1)
            
            # Getting state summary of the queue
            for i in range(len(self.queue)):
                temp = np.concatenate((temp, self.queue[i].summary(bg_shape)), axis=1)

            # Getting state summary of the empty portions of the queue
            blank_summary = Task([0]*dimension, 0, 'empty_task').summary(bg_shape)
            for i in range(len(self.queue), self.queue_size):
                temp = np.concatenate((temp, blank_summary), axis=1)

            #Getting state summary of the backlog
            backlog_summary = Task([0], 0, 'empty_task').summary(bg_shape)
            p_backlog = 0
            p_row = 0
            p_col = 0
            while p_row < bg_shape[0] and p_col < bg_shape[1] and p_backlog < len(self.backlog):
                backlog_summary[p_row, p_col] = 0
                p_row += 1
                if p_row == bg_shape[0]:
                    p_row = 0
                    p_col += 1
                p_backlog += 1
            temp = np.concatenate((temp, backlog_summary), axis=1)
            return (torch.unsqueeze(torch.tensor(temp), 0))
        
        else:
            return None

    def __repr__(self):
                return 'Environment(timestep_counter={0}, machines={1}, queue={2}, backlog={3})'.format(self.timestep_counter, self.machines, self.queue, self.backlog)

    # Determines when the environment needs to be terminated.
    # It does not get terminated when the machines are utilizied at the moment
    # If does not terminate if the queue or backlog is not empty
    # It terminates when there are no more tasks to handled.
    def terminated(self):

        for machine in self.machines:
            if machine.utilization() > 0:
                return 0

        if self.queue or self.backlog or not self.task_gen_end:
            return 0
        
        return 1

# Create task.txt file by randomizing values from task.conf.json
# task.conf.json specifies patterns that each task must follow.
# It has information such as batch_size of each task type, resource range(no. of resources and limits of instances on each type of resource).
# It also contains limits on durations of each task.
def generate_tasks():
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    if os.path.isfile('./data/task.txt'):
        return
    with open('./data/task.txt', 'w+') as file, open('conf/task.conf.json') as inp:
        task_conf = json.load(inp)
        if len(task_conf) > 0:

            for i in range(len(task_conf[0]['resource_range'])):
                file.write('resource' + str(i) + ',')
            file.write('duration,label\n')
            label = 0
            for task_type in task_conf:
                for i in range(task_type["batch_size"]):
                    label += 1
                    duration = np.random.randint(task_type['duration_range']['lowerLimit'], task_type['duration_range']['upperLimit'])
                    resources = []
                    for j in range(len(task_type['resource_range'])):
                        resources.append(str(np.random.randint(task_type['resource_range'][j]['lowerLimit'], task_type['resource_range'][j]['upperLimit'])))
                    file.write(','.join(resources) + ',' + str(duration) + ',' + 'task' + str(label) + '\n')

# Function which reads the tasks.txt file and create tasks objects fed into the enironment
def load_tasks():

    generate_tasks()
    label_index = None
    duration_index = None
    resource_index = []

    with open('./data/task.txt', 'r') as file:
        lines = file.readlines()
    
    splits = lines[0].split(',')
    for i in range(len(splits)):
        if splits[i].startswith('resource'):
            resource_index.append(i)
        elif splits[i].startswith('duration'):
            duration_index = i
        else:
            label_index = i
    tasks = []
    for i in range(1, len(lines)):
        line = lines[i]
        splits = line.split(',')
        resources = []
        duration = None
        label = None

        for ind in resource_index:
            resources.append(int(splits[ind]))
        duration = splits[duration_index]
        label = splits[label_index]
        tasks.append(Task(resources, int(duration), label))
    
    return tasks


# Set up environment by creating Machines, queues, backlog accroding to env.conf.json
def set_up_enviroment(load_environment, load_scheduler):
    tasks = load_tasks()
    task_gen = (task for task in tasks)
    env = None
    scheduler = None
    with open('./conf/env.conf.json', 'r') as file:
        env_data = json.load(file)
        machines = []
        label = 0

        # For every node create a machine
        for node_json in env_data['nodes']:
            label+=1
            machines.append(Machine(node_json['resource_capacity'], node_json['duration_capacity'], 'node' + str(label))) 

        if load_environment:
            env = Environment(machines, env_data['queue_size'], env_data['backlog_size'], task_gen)
            env.timestep()

        if load_scheduler:
            scheduler = DeepRMScheduler(env, env_data['train'])

    return (env, scheduler)

        

if __name__ == '__main__':
    set_up_enviroment(load_environment=True, load_scheduler=False)
