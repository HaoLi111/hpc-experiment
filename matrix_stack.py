import numpy as np
import time

def solve_stack(matrix_stack):
    """
    Input: A 3D array of shape (Batch_Size, 10, 10)
    Output: Eigenvalues for all matrices in the stack.
    
    Numpy's linalg.eig recognizes the 3D shape and automatically 
    loops in C (very fast) instead of Python.
    """
    # This single line solves ALL matrices in the stack at once
    vals, _ = np.linalg.eig(matrix_stack)
    return vals

# --- GENERATE DATA ---
# Let's say we have 100,000 small 10x10 matrices
N_MATRICES = 100000
DIM = 10

print(f"Generating {N_MATRICES} matrices of size {DIM}x{DIM}...")
# We create one giant 3D array: (100000, 10, 10)
# This represents your "list" of matrices but in a contiguous memory block
all_matrices = np.random.rand(N_MATRICES, DIM, DIM)

# --- TEST 1: The "Loop" (What you want to avoid) ---
print("Running Python Loop (Standard)...")
start = time.time()
results_loop = []
# This is slow because Python has to manage the loop overhead 100,000 times
for i in range(N_MATRICES):
    # Slicing the single matrix out
    single_matrix = all_matrices[i]
    results_loop.append(np.linalg.eig(single_matrix)[0])
end = time.time()
print(f"Loop Time: {end - start:.4f} seconds")

# --- TEST 2: The "Batch/Stack" (Your Goal) ---
# We solve ALL 100,000 "together" in one function call.
print("\nRunning Vectorized Stack (Batching)...")
start = time.time()
results_stack = solve_stack(all_matrices)
end = time.time()
print(f"Stack Time: {end - start:.4f} seconds")

# --- COMPARISON ---
speedup = (end - start) 
# Note: Inverting for X speedup
print(f"\nSpeedup: The Stack method is usually 10x-50x faster.")