# Simulation Utilities Documentation

This directory contains the core components for the Fluid Monte Carlo Simulation. Here is a breakdown of each file and its purpose.

## 1. `simulation.py`
**Purpose**: The main engine of the simulation using the Ising Model.
- **`Simulator` Class**: The central controller.
    - `__init__`: Sets up the lattice matrix (0s and 1s), temperature, and random number buffer.
    - `_defStepFuntion`: Selects the physics algorithm based on `type` ('phase_change_T' or 'phase_change_TC').
    - `simulate(steps)`: Runs the simulation for a number of batches.
- **`monte_carlo_batch`**: (Numba Optimized) Performs the standard Metropolis-Hastings algorithm in high-performance batches. It uses pre-generated random numbers to fly through calculations.
- **`monte_carlo_batch2`**: (Numba Parallel) A parallelized "Checkerboard" update scheme. This allows updating "white" and "black" squares of the grid independently in parallel, which is much faster but requires a specific update order.
- **`calculate_neighbors_vectorized`**: Helper to count neighbors (up/down/left/right) with periodic boundary conditions (wrapping around edges).

## 2. `ising_plotter.py`
**Purpose**: Real-time visualization of the simulation.
- **`IsingPlotter` Class**: A PyQtGraph-based window.
    - Fast rendering of the matrix using `pg.ImageItem`.
    - Supports interactive sliders for Temperature, Field, etc.
    - `update(matrix)`: Refreshes the display.
- **`FastIsingRecorder` Class**: A highly optimized recorder that saves frames directly as PNGs during the simulation without slowing it down significantly. Can verify with `ffmpeg` to make videos.

## 3. `recorder.py`
**Purpose**: High-quality post-processing and recording.
- **`MatplotlibRecorder` Class**:
    - **Recording Phase**: Saves raw matrix data (lightweight) to a `.pkl` file instead of generating images immediately.
    - **Generation Phase**: Reads the saved data later to generate high-quality Matplotlib images or videos. This decouples the heavy rendering from the simulation loop.
- **`nearest_neighbor_upscale`**: Manually scales up the pixel art look of the simulation for cleaner videos.

## 4. `randomBuffer.py`
**Purpose**: Optimization helper.
- **`RandomBuffer` Class**: Python is slow at generating random numbers one by one. This class generates them in massive batches (e.g., 10,000 at a time) using Numpy, which `simulation.py` then consumes. This drastically improves performance.

## 5. `test.py`
**Purpose**: A playground/prototyping file to test different optimization strategies.
- Contains earlier versions of `Simulator` and comparison logic.
- **`optimized_monte_carlo_step`**: A function testing single-step updates.
- **`monte_carlo_batch`**: Testing the batch update logic.
- **`__main__`**: Runs a performance comparison between Single-step, Batch, and Wrapper approaches to prove that batching is faster.
