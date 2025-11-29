import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# --- HARDWARE CONFIGURATION ---
TOTAL_CORES = 16
CORES_PER_UNIT = 4
NUM_WORKERS = TOTAL_CORES // CORES_PER_UNIT  # = 4 Workers (Groups)

# --- CONFIGURATION ---
N_MATRICES = 400000
MATRIX_SIZE = 10


# ============================================================================
# METHOD 1: VANILLA (Implicit Parallel - Let NumPy use all cores)
# ============================================================================
def method1_vanilla():
    """
    Vanilla approach: Let NumPy's underlying BLAS/LAPACK use all available cores.
    This is the default behavior - no explicit parallelization control.
    """
    print("\n" + "="*70)
    print("METHOD 1: VANILLA (Implicit Parallel - All Cores)")
    print("="*70)
    print("Strategy: Single process, NumPy uses all available cores internally")
    print(f"Expected cores used: ~{TOTAL_CORES} (automatic)")
    
    # Generate data
    print(f"\nGenerating {N_MATRICES} matrices ({MATRIX_SIZE}x{MATRIX_SIZE})...")
    data = np.random.rand(N_MATRICES, MATRIX_SIZE, MATRIX_SIZE)
    
    # Process all at once
    print("Processing all matrices in one call...")
    start = time.time()
    results = np.linalg.eigvals(data)
    end = time.time()
    
    elapsed = end - start
    print(f"✓ Processed {len(results)} matrices")
    print(f"⏱  Total Time: {elapsed:.4f} seconds")
    print(f"📊 Throughput: {N_MATRICES/elapsed:.2f} matrices/second")
    
    return elapsed


# ============================================================================
# METHOD 2: EMBARRASSINGLY PARALLEL (1 core per task, many tasks)
# ============================================================================
def single_core_task(matrix):
    """
    Process a single matrix using only 1 core.
    This function runs in a worker process.
    """
    return np.linalg.eigvals(matrix)


def method2_embarrassingly_parallel():
    """
    Embarrassingly parallel: Each worker gets 1 core and processes 1 matrix.
    Maximum parallelism at the task level, no parallelism at the operation level.
    """
    print("\n" + "="*70)
    print("METHOD 2: EMBARRASSINGLY PARALLEL (1 core per task)")
    print("="*70)
    print(f"Strategy: {TOTAL_CORES} workers, each with 1 core, processing individual matrices")
    print(f"Expected cores used: {TOTAL_CORES}")
    
    # Set each worker to use only 1 thread
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # Generate data
    print(f"\nGenerating {N_MATRICES} matrices ({MATRIX_SIZE}x{MATRIX_SIZE})...")
    data = np.random.rand(N_MATRICES, MATRIX_SIZE, MATRIX_SIZE)
    
    print(f"Dispatching {N_MATRICES} tasks to {TOTAL_CORES} workers...")
    start = time.time()
    
    # Use all cores as separate workers, each processing one matrix at a time
    with ProcessPoolExecutor(max_workers=TOTAL_CORES) as executor:
        results = list(executor.map(single_core_task, data))
    
    end = time.time()
    
    elapsed = end - start
    print(f"✓ Processed {len(results)} matrices")
    print(f"⏱  Total Time: {elapsed:.4f} seconds")
    print(f"📊 Throughput: {N_MATRICES/elapsed:.2f} matrices/second")
    
    return elapsed


# ============================================================================
# METHOD 3: HYBRID (Computing Units - 4 cores per unit)
# ============================================================================
def computing_unit_task(batch_of_matrices):
    """
    This function runs inside one of the 'Computing Units'.
    Each unit uses 4 cores to process a batch of matrices.
    """
    return np.linalg.eigvals(batch_of_matrices)


def method3_hybrid_computing_units():
    """
    Hybrid approach: Create 'computing units' of 4 cores each.
    Process parallelism (4 units) + Thread parallelism (4 threads per unit).
    """
    print("\n" + "="*70)
    print("METHOD 3: HYBRID (Computing Units - 4 cores per unit)")
    print("="*70)
    print(f"Strategy: {NUM_WORKERS} processes × {CORES_PER_UNIT} threads = {TOTAL_CORES} cores")
    print(f"Expected cores used: {TOTAL_CORES}")
    
    # Configure each worker to use 4 threads
    os.environ["OMP_NUM_THREADS"] = str(CORES_PER_UNIT)
    os.environ["MKL_NUM_THREADS"] = str(CORES_PER_UNIT)
    os.environ["OPENBLAS_NUM_THREADS"] = str(CORES_PER_UNIT)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(CORES_PER_UNIT)
    os.environ["NUMEXPR_NUM_THREADS"] = str(CORES_PER_UNIT)
    
    # Generate data
    print(f"\nGenerating {N_MATRICES} matrices ({MATRIX_SIZE}x{MATRIX_SIZE})...")
    data = np.random.rand(N_MATRICES, MATRIX_SIZE, MATRIX_SIZE)
    
    # Split data into batches for computing units
    batches = np.array_split(data, NUM_WORKERS)
    print(f"Data split into {len(batches)} batches of ~{len(batches[0])} matrices each")
    
    print(f"Dispatching {NUM_WORKERS} batches to {NUM_WORKERS} computing units...")
    start = time.time()
    
    # Use NUM_WORKERS processes, each using CORES_PER_UNIT threads
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(computing_unit_task, batches))
    
    # Reassemble results
    final_results = np.concatenate(results)
    end = time.time()
    
    elapsed = end - start
    print(f"✓ Processed {len(final_results)} matrices")
    print(f"⏱  Total Time: {elapsed:.4f} seconds")
    print(f"📊 Throughput: {N_MATRICES/elapsed:.2f} matrices/second")
    
    return elapsed


# ============================================================================
# MAIN COMPARISON
# ============================================================================
def main():
    print("\n" + "="*70)
    print("MATRIX EIGENVALUE COMPUTATION - PARALLELIZATION COMPARISON")
    print("="*70)
    print(f"Hardware: {TOTAL_CORES} cores")
    print(f"Task: Computing eigenvalues for {N_MATRICES} matrices ({MATRIX_SIZE}×{MATRIX_SIZE})")
    print("="*70)
    
    # Store results
    results = {}
    
    # Run Method 1: Vanilla
    time1 = method1_vanilla()
    results['Vanilla (All Cores)'] = time1
    
    # Run Method 2: Embarrassingly Parallel
    time2 = method2_embarrassingly_parallel()
    results['Embarrassingly Parallel (1 core/task)'] = time2
    
    # Run Method 3: Hybrid Computing Units
    time3 = method3_hybrid_computing_units()
    results['Hybrid (4 cores/unit)'] = time3
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - PERFORMANCE COMPARISON")
    print("="*70)
    
    # Find fastest
    fastest = min(results, key=results.get)
    fastest_time = results[fastest]
    
    print(f"\n{'Method':<40} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 70)
    
    for method, elapsed in results.items():
        speedup = fastest_time / elapsed
        marker = " ⭐ FASTEST" if method == fastest else ""
        print(f"{method:<40} {elapsed:>10.4f}   {speedup:>8.2f}x{marker}")
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("""
Method 1 (Vanilla): 
  - Pros: Simple, no code complexity
  - Cons: May have overhead with automatic thread management
  - Best for: Quick prototyping, unknown workload characteristics

Method 2 (Embarrassingly Parallel):
  - Pros: Maximum task-level parallelism, good for small matrices
  - Cons: No multi-threading per task, high process creation overhead
  - Best for: Many small independent tasks, I/O bound operations

Method 3 (Hybrid Computing Units):
  - Pros: Balanced task + thread parallelism, efficient resource use
  - Cons: More complex setup, requires tuning cores-per-unit
  - Best for: Large batches of medium-sized compute tasks
    """)
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
