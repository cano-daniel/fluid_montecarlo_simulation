import numpy as np
from numba import jit

# ============================================================================
# SOLUTION 1: Pass data arrays instead of class instance (RECOMMENDED)
# ============================================================================

class RandomBuffer:
    """Regular Python class - easy to maintain"""
    def __init__(self, width, height, buffer_size=10000):
        self.buffer_size = buffer_size
        self.position = 0
        self.n_height = height
        self.n_width = width
        self.refill_buffer()
    
    def refill_buffer(self):
        """Generate fresh random numbers in vectorized batches"""
        self.rand_positions = np.column_stack([
            np.random.randint(0, self.n_height, self.buffer_size),
            np.random.randint(0, self.n_width, self.buffer_size),
            np.random.randint(0, self.n_height, self.buffer_size),
            np.random.randint(0, self.n_width, self.buffer_size)
        ])
        self.rand_floats = np.random.random(self.buffer_size)
        self.position = 0
    
    def get_batch(self, n):
        """Get n random number sets at once"""
        if self.position + n >= self.buffer_size:
            self.refill_buffer()
        
        start = self.position
        end = start + n
        positions = self.rand_positions[start:end]
        floats = self.rand_floats[start:end]
        self.position = end
        
        return positions, floats


@jit(nopython=True)
def calculate_neighbors_vectorized(x, y, matrix):
    """Vectorized neighbor sum calculation with periodic boundaries"""
    n_height, n_width = matrix.shape
    neighbors_sum = (
        matrix[(x+1) % n_height, y] + 
        matrix[(x-1) % n_height, y] +
        matrix[x, (y+1) % n_width] + 
        matrix[x, (y-1) % n_width]
    )
    return neighbors_sum


@jit(nopython=True)
def optimized_monte_carlo_step(matrix, temp, x1, y1, x2, y2, dice):
    """Perform single Monte Carlo step - NOW TAKES INDIVIDUAL VALUES"""
    # Early termination: skip if states are identical
    state1 = matrix[x1, y1]
    state2 = matrix[x2, y2]
    
    if state1 == state2:
        return  # No change possible, skip all calculations
    
    # Calculate neighbor sums
    n1 = calculate_neighbors_vectorized(x1, y1, matrix)
    n2 = calculate_neighbors_vectorized(x2, y2, matrix)
    
    # Energy change calculation
    delta_energy = n1 * state1 + n2 * state2 - (n1 * state2 + n2 * state1)
    prob_relation = np.exp(-delta_energy / temp)
    prob_change = prob_relation / (1 + prob_relation)
    
    if dice < prob_change:
        matrix[x1, y1] = state2
        matrix[x2, y2] = state1


@jit(nopython=True)
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


class Simulator:
    def __init__(self, width, height, step_function, p=0.8, batch_size=100):
        self.width = width
        self.height = height
        self.step_function = step_function
        self.temp = 1.0
        self.matrix = np.random.choice([0, 1], size=(height, width), p=[p, 1.0-p])
        self.buffer = RandomBuffer(width, height)
        self.batch_size = batch_size
    
    def get_matrix(self):
        return self.matrix.copy()
    
    def update_temp(self, temp):
        self.temp = temp
    
    def step(self):
        """Single step - extracts data from buffer and passes to numba function"""
        if self.step_function == optimized_monte_carlo_step:
            # For single-step function: get one set of randoms
            positions, floats = self.buffer.get_batch(1)
            x1, y1, x2, y2 = positions[0]
            dice = floats[0]
            self.step_function(self.matrix, self.temp, x1, y1, x2, y2, dice)
        elif self.step_function == monte_carlo_batch:
            # For batch function: get multiple randoms at once
            positions, floats = self.buffer.get_batch(self.batch_size)
            self.step_function(self.matrix, self.temp, positions, floats)
    
    def simulate(self, steps=10):
        """Run multiple steps"""
        if self.step_function == monte_carlo_batch:
            # Batch mode: each step processes batch_size iterations
            for i in range(steps):
                self.step()
        else:
            # Single mode: each step is one iteration
            for i in range(steps):
                self.step()


# ============================================================================
# ALTERNATIVE: Wrapper function approach
# ============================================================================

def create_step_function_with_buffer(buffer, batch_size=100):
    """
    Factory function that creates a step function with buffer access
    This encapsulates the buffer interaction
    """
    def step_wrapper(matrix, temp):
        positions, floats = buffer.get_batch(batch_size)
        monte_carlo_batch(matrix, temp, positions, floats)
    
    return step_wrapper


class SimulatorWithWrapper:
    """Alternative approach using wrapper functions"""
    def __init__(self, width, height, p=0.8, batch_size=100):
        self.width = width
        self.height = height
        self.temp = 1.0
        self.matrix = np.random.choice([0, 1], size=(height, width), p=[p, 1.0-p])
        self.buffer = RandomBuffer(width, height)
        self.batch_size = batch_size
        # Create the step function with buffer access built-in
        self.step_function = create_step_function_with_buffer(self.buffer, batch_size)
    
    def get_matrix(self):
        return self.matrix.copy()
    
    def update_temp(self, temp):
        self.temp = temp
    
    def step(self):
        """Step function automatically handles buffer interaction"""
        self.step_function(self.matrix, self.temp)
    
    def simulate(self, steps=10):
        for i in range(steps):
            self.step()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Single-step mode (original interface)")
    print("=" * 60)
    sim1 = Simulator(100, 100, optimized_monte_carlo_step, p=0.8)
    sim1.simulate(steps=10000)
    print(f"Matrix sum after 1000 steps: {sim1.matrix.sum()}")
    
    print("\n" + "=" * 60)
    print("Example 2: Batch mode (much faster)")
    print("=" * 60)
    sim2 = Simulator(100, 100, monte_carlo_batch, p=0.8, batch_size=10000)
    sim2.simulate(steps=10)  # 10 batches × 100 = 1000 iterations
    print(f"Matrix sum after 1000 steps: {sim2.matrix.sum()}")
    
    print("\n" + "=" * 60)
    print("Example 3: Wrapper approach (cleanest)")
    print("=" * 60)
    sim3 = SimulatorWithWrapper(100, 100, p=0.8, batch_size=10000)
    sim3.simulate(steps=10)  # 10 batches × 100 = 1000 iterations
    print(f"Matrix sum after 1000 steps: {sim3.matrix.sum()}")
    
    print("\n" + "=" * 60)
    print("Performance tip: Larger batch_size = better performance!")
    print("Try batch_size=1000 or even 10000 for best results")
    print("=" * 60)