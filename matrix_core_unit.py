import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# --- STEP 1: DEFINE THE COMPUTING UNIT ---
# We have 16 cores total.
# We want "Computing Units" of 4 cores each.
TOTAL_CORES = 16
CORES_PER_UNIT = 4
NUM_WORKERS = TOTAL_CORES // CORES_PER_UNIT  # = 4 Workers (Groups)

# --- STEP 2: CONFIGURE THE ENVIRONMENT ---
# We tell the low-level math libraries (BLAS/MKL): 
# "Whenever you calculate, use 4 threads."
os.environ["OMP_NUM_THREADS"] = str(CORES_PER_UNIT)
os.environ["MKL_NUM_THREADS"] = str(CORES_PER_UNIT)
os.environ["OPENBLAS_NUM_THREADS"] = str(CORES_PER_UNIT)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(CORES_PER_UNIT)
os.environ["NUMEXPR_NUM_THREADS"] = str(CORES_PER_UNIT)

# Import numpy AFTER setting variables to ensure it picks up the config
import numpy as np

def computing_unit_task(batch_of_matrices):
    """
    This function runs inside one of the 'Workers'.
    Because of the env vars above, this worker will use 
    4 cores to solve this batch.
    """
    # np.linalg.eigvals will see OMP_NUM_THREADS=4 and use 4 cores
    # to process this stack of matrices.
    return np.linalg.eigvals(batch_of_matrices)

def run_hybrid_parallelism():
    # Configuration
    N_MATRICES = 40000
    MATRIX_SIZE = 10 # Increased to 100x100 so the 4-core power is actually noticeable
                      # (10x10 is too small to see benefits from multithreading)

    print(f"--- HARDWARE SETUP ---")
    print(f"Total Cores Available: {TOTAL_CORES}")
    print(f"Computing Units:       {NUM_WORKERS}")
    print(f"Cores per Unit:        {CORES_PER_UNIT}")
    print(f"Algorithm:             4 Processes, each spawning 4 Threads")
    print(f"----------------------\n")

    # 1. Generate Data
    print(f"Generating {N_MATRICES} matrices ({MATRIX_SIZE}x{MATRIX_SIZE})...")
    data = np.random.rand(N_MATRICES, MATRIX_SIZE, MATRIX_SIZE)

    # 2. Split Data for the "Computing Units"
    # We slice the data into 4 equal parts, one for each worker.
    batches = np.array_split(data, NUM_WORKERS)
    print(f"Data split into {len(batches)} batches of {len(batches[0])} matrices each.")

    # 3. Execution
    print("Dispatching tasks to computing units...")
    start = time.time()

    # We use exactly 'NUM_WORKERS' (4) processes.
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Each worker takes a batch and uses its 4 assigned cores to solve it
        results = list(executor.map(computing_unit_task, batches))

    # 4. Reassemble results
    final_results = np.concatenate(results)
    
    end = time.time()
    print(f"\nDone! Processed {len(final_results)} matrices.")
    print(f"Total Time: {end - start:.4f} seconds")

if __name__ == '__main__':
    run_hybrid_parallelism()