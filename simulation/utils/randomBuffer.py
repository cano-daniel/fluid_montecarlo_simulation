import numpy as np
# NO NUMBA IMPORTS - keep this as pure Python!

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
