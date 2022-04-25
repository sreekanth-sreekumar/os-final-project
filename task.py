import numpy as np

class Task:
    def __init__(self, resources, duration, label):
        self.resources = resources
        self.duration = duration
        self.label = label
        self.dimension = len(self.resources)

    def summary(self, bg_shape = None):
        
        if bg_shape == None:
            bg_shape = (self.duration, max(self.resources))

        if self.dimension > 0:
            state_matrices = [np.full(bg_shape, 255) for i in range(self.dimension)]
            for i in range(self.dimension):
                for row in range(self.duration):
                    for col in range(self.resources[i]):
                        state_matrices[i][row, col] = 0

            temp = state_matrices[0]
            for i in range(1, self.dimension):
                temp = np.concatenate((temp, state_matrices[i]), axis=1)
            return temp
        else:
            return None