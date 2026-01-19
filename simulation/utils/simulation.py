import numpy as np
from numba import jit, prange
from utils.randomBuffer import RandomBuffer

@jit(nopython=True)
def calculate_neighbors_vectorized(x, y, matrix):
    """Vectorized neighbor sum calculation with periodic boundaries"""
    n_height, n_width = matrix.shape
    neighbors_sum = (
        matrix[(x+1) % n_height, y] + matrix[(x-1) % n_height, y] +
        matrix[x, (y+1) % n_width] + matrix[x, (y-1) % n_width]
    )
    return neighbors_sum

@jit(nopython=True)  # NOT parallel=True - would cause race conditions!
def monte_carlo_batch(matrix, temp, positions, floats):
    """Process batch of Monte Carlo steps - TAKES ARRAYS FROM BUFFER"""
    n_steps = len(floats)
    
    for i in range(n_steps):
        x1 = positions[i, 0]
        y1 = positions[i, 1]
        x2 = positions[i, 2]
        y2 = positions[i, 3]
        dice = floats[i]
        
        state1 = matrix[x1, y1]
        state2 = matrix[x2, y2]
        
        if state1 == state2:
            continue
        
        # Calculate neighbors inline (faster than function call in loop)
        n_height, n_width = matrix.shape
        n1 = (matrix[(x1+1) % n_height, y1] + matrix[(x1-1) % n_height, y1] +
              matrix[x1, (y1+1) % n_width] + matrix[x1, (y1-1) % n_width])
        n2 = (matrix[(x2+1) % n_height, y2] + matrix[(x2-1) % n_height, y2] +
              matrix[x2, (y2+1) % n_width] + matrix[x2, (y2-1) % n_width])
        
        delta_energy = n1 * state1 + n2 * state2 - (n1 * state2 + n2 * state1)
        prob_relation = np.exp(-delta_energy / temp)
        prob_change = prob_relation / (1 + prob_relation)
        
        if dice < prob_change:
            matrix[x1, y1] = state2
            matrix[x2, y2] = state1

@jit(nopython=True, parallel=True)
def monte_carlo_batch2(matrix, temp, cQuemPot, floats):
    """Process batch of Monte Carlo steps - CHECKERBOARD PATTERN"""
    n_steps = len(floats)
    n_height, n_width = matrix.shape
    
    # FIRST PASS: Update "white" squares (even sum: x+y is even)
    for x1 in prange(n_height):
        # Start at y=0 if x1 is even, y=1 if x1 is odd (to get even sum)
        start_y = x1 % 2
        for y1 in range(start_y, n_width, 2):
            dice = floats[(x1 * (y1 + 1)) % n_steps]
            
            # Calculate neighbors with boundary conditions
            energy = 0
            if x1 + 1 < n_height:
                energy -= matrix[x1 + 1, y1]
            if x1 - 1 >= 0:
                energy -= matrix[x1 - 1, y1]
            if y1 + 1 < n_width:
                energy -= matrix[x1, y1 + 1]
            if y1 - 1 >= 0:
                energy -= matrix[x1, y1 - 1]
            
            delta_energy = (energy - cQuemPot * (-1))
            prob_relation = np.exp(-delta_energy / temp)
            prob = 1 / (1 + prob_relation)
            
            if prob < dice:
                matrix[x1, y1] = 1
            else:
                matrix[x1, y1] = 0
    
    # SECOND PASS: Update "black" squares (odd sum: x+y is odd)
    for x1 in prange(n_height):
        # Start at y=1 if x1 is even, y=0 if x1 is odd (to get odd sum)
        start_y = 1 - (x1 % 2)
        for y1 in range(start_y, n_width, 2):
            dice = floats[(x1 * (y1 + 1)) % n_steps]
            
            # Calculate neighbors with boundary conditions
            energy = 0
            if x1 + 1 < n_height:
                energy -= matrix[x1 + 1, y1]
            if x1 - 1 >= 0:
                energy -= matrix[x1 - 1, y1]
            if y1 + 1 < n_width:
                energy -= matrix[x1, y1 + 1]
            if y1 - 1 >= 0:
                energy -= matrix[x1, y1 - 1]
            
            delta_energy = (energy - cQuemPot * (-1))
            prob_relation = np.exp(-delta_energy / temp)
            prob = 1 / (1 + prob_relation)
            
            if prob < dice:
                matrix[x1, y1] = 1
            else:
                matrix[x1, y1] = 0

            

class Simulator:
    def __init__(self, width, height, type = 'phase_change_T', p=0.8, rand_buffer_size = 10000):

        ## essential variables 
        self.width = width
        self.height = height
        self.type = type
        self.matrix = np.random.choice([0, 1], size=(height, width), p=[p, 1.0-p])
        # optional (simulation)
        self.temp = 1.0
        self.quemicalPotential = 1.0

        #random variable
        self.buffer = RandomBuffer(width, height,buffer_size=rand_buffer_size)

        self._defStepFuntion()


    def _defStepFuntion(self):

        if self.type == 'phase_change_T':
            self.StepFuntion = monte_carlo_batch
            self.parameters = {
                'matrix' : self.matrix,
                'temp' : self.temp,
                'positions' : [],
                'floats' : []
                }
            
        elif self.type == 'phase_change_TC':
            self.StepFuntion = monte_carlo_batch2
            self.parameters = {
                'matrix' : self.matrix,
                'temp' : self.temp,
                'cQuemPot' : self.quemicalPotential,
                'floats' : []
                }

    
    def get_matrix(self):
        return self.matrix.copy()
    
    def update_temp(self, temp):
        self.temp = temp
    
    def update_quemicalPotential(self, c):
        self.quemicalPotential = c
    
    def simulate(self, steps=10):
        """Run simulation for 'steps' batches"""

        if self.type == 'phase_change_T':

            self.parameters['temp'] = self.temp
            self.parameters['positions'], self.parameters['floats'] = self.buffer.get_batch(steps)

        elif self.type == 'phase_change_TC':

            self.parameters['temp'] = self.temp
            self.parameters['cQuemPot'] = self.quemicalPotential
            __, self.parameters['floats'] = self.buffer.get_batch(7019)
            
        self.StepFuntion(**self.parameters)
        
