import numpy as np
import collections
import torch

class Machine:

    def __init__(self, resources, duration, label):
        self.resources = resources
        self.duration = duration
        self.label = label

        self.dimension = len(self.resources)
        self.timestep_counter = 0
        self.scheduled_tasks = []

        self.state_matrices = [np.full((duration, resource), 255) for resource in self.resources]
        self.state_matrices_capacity = [[resource]*duration for resource in self.resources]

    # Move machine by one timestep
    def timestep(self):
        
        self.timestep_counter += 1

        for i in range(self.dimension):
            # Delete the first row (The current timestep)
            temp = np.delete(self.state_matrices[i], (0), axis=0)
            temp = np.append(temp, np.array([[255 for x in range(temp.shape[1])]]), axis = 0)
            self.state_matrices[i] = temp
        
        for i in range(self.dimension):
            # Delete the fitst row (The current timestep)
            self.state_matrices_capacity[i].pop(0)
            self.state_matrices_capacity[i].append(self.resources[i])

        # Find jobs whose required time is over in this timestep and remove them.
        indices = []
        for i in range(len(self.scheduled_tasks)):
            if self.timestep_counter >= self.scheduled_tasks[i][1]:
                indices.append(i)
        
        for i in sorted(indices, reverse=True):
            del self.scheduled_tasks[i]

    # Occupy the resource slot in the machine by updating the resource capacity matrix.
    # Update the state matrix with value zero to mark the utlization of the resource. 
    def occupy(self, state_matrix, state_matrix_capacity, required_resource, required_duration, start_time):
        for i in range(start_time, start_time + required_duration):
            for j in range(required_resource):
                state_matrix[i, len(state_matrix[i]) - state_matrix_capacity[i] + j] = 0
            state_matrix_capacity[i] = state_matrix_capacity[i] - required_resource

    def schedule(self, task):
        start_time = self.satisfy(task.resources, task.duration)
        
        if start_time == -1:
            return False

        else:
            for i in range(task.dimension):
                self.occupy(self.state_matrices[i], self.state_matrices_capacity[i], task.resources[i], task.duration, start_time) 
            self.scheduled_tasks.append((task, self.timestep_counter+task.duration))
            return True

    # Set two pointers one to iterate till the end of the resource requirements.
    # one to see if the duration is met.
    def satisfy(self, resources, duration):
        p1 = 0
        p2 = 0
        duration_bound = min([len(capacity) for capacity in self.state_matrices_capacity])
        while p1 < duration_bound and p2 < duration:
            if False in [self.state_matrices_capacity[i][p1] >= resources[i] for i in range(len(resources))]:
                p1 += 1
                p2 = 0
            else:
                p1 += 1
                p2 += 1
        
        if p2 == duration:
            return p1 - duration
        else:
            return -1

    # Create a state representation for each machine.
    # A state representation involes the current state of the machine (duration, resource_capacity) for each resource.
    # State rep is expanded to the bg_shape which is of shape (max_resource, max_duration) of any machine.
    def summary(self, bg_shape=None):
        if self.dimension > 0:

            temp = self.expand(self.state_matrices[0], bg_shape)
            for i in range(1, self.dimension):
                temp = np.concatenate((temp, self.expand(self.state_matrices[i], bg_shape)), axis=1)
            return temp
        else:
            return None

    # Expand a matrix to the given bg_shape
    def expand(self, matrix, bg_shape=None):
        if bg_shape is not None and bg_shape[0] >= matrix.shape[0] and bg_shape[0] >= matrix.shape[1]:
            temp = matrix

            if bg_shape[0] > matrix.shape[0]:
                temp = np.concatenate((temp, np.full((bg_shape[0] - matrix.shape[0], matrix.shape[1]), 255)), axis=0)
            if bg_shape[1] > matrix.shape[0]:
                temp = np.concatenate((temp, np.full((matrix.shape[0], bg_shape[1] - matrix.shape[1]), 255)), axis=1)
            return temp

        else:
            return temp

    # Utilization is the ratio of utilization of resources of a machine over the duration
    def utilization(self):
        return sum([collections.Counter(matrix.flatten()).get(0,0) for matrix in self.state_matrices])/sum(self.resources)/self.duration

    def __repr__(self):
        return 'Machine(state_matrices={0}, label={1})'.format(self.state_matrices, self.label)
