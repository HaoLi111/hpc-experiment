import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# We try to import Plotly for high-end WebGL rendering
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: Plotly not found. Installing 'plotly' is recommended for 3D isosurfaces.")

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mandelbulb_sdf_batch(points, power=8.0, max_iter=5):
    """
    Computes the Signed Distance Function (SDF) for a batch of 3D points.
    Input: points (N, 3) tensor
    Output: dists (N,) tensor
    """
    z = points.clone()
    dr = torch.ones(z.shape[0], dtype=torch.float32, device=z.device)
    r = torch.zeros_like(dr)
    
    # Iterate to find the shape of the fractal
    for i in range(max_iter):
        r = torch.norm(z, dim=1)
        
        # Convert to polar coordinates
        # We use a clamp here to avoid division by zero in atan2 for the very center
        theta = torch.atan2(torch.sqrt(z[:,0]**2 + z[:,1]**2), z[:,2])
        phi = torch.atan2(z[:,1], z[:,0])
        
        # Calculate derivative (dr) for distance estimation
        dr = torch.pow(r, power - 1.0) * power * dr + 1.0
        
        # Power scaling
        zr = torch.pow(r, power)
        theta = theta * power
        phi = phi * power
        
        # Convert back to Cartesian
        z[:,0] = zr * torch.sin(theta) * torch.cos(phi)
        z[:,1] = zr * torch.sin(theta) * torch.sin(phi)
        z[:,2] = zr * torch.cos(theta)
        z = z + points
        
        # Optimization: Cap values to prevent infinity/NaNs in escaped regions
        over = r > 2.0
        z[over] = 2.0 
        
    r = torch.norm(z, dim=1)
    # Distance Estimator formula: 0.5 * log(r) * r / dr
    dist = 0.5 * torch.log(r) * r / dr
    
    # Fix NaN/Inf
    dist[torch.isnan(dist)] = 0.0
    return dist

def generate_volume_streamed(res=64, n_streams=4):
    """
    Generates a 3D density volume (res x res x res) using CUDA Streams.
    """
    print(f"Generating {res}x{res}x{res} volume on {device}...")
    
    # 1. Setup Coordinate Grid
    # We range from -1.2 to 1.2 to capture the full bulb
    min_bound, max_bound = -1.2, 1.2
    x = torch.linspace(min_bound, max_bound, res)
    y = torch.linspace(min_bound, max_bound, res)
    z = torch.linspace(min_bound, max_bound, res)
    
    chunk_size = res // n_streams
    streams = [torch.cuda.Stream() for _ in range(n_streams)]
    
    volume = torch.zeros((res, res, res), dtype=torch.float32)
    
    start_time = time.time()
    
    # 2. Execution Loop
    for i in range(n_streams):
        stream = streams[i]
        z_start = i * chunk_size
        z_end = res if i == n_streams - 1 else (i + 1) * chunk_size
        
        z_chunk = z[z_start:z_end]
        
        with torch.cuda.stream(stream):
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z_chunk, indexing='ij')
            
            points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
            points = points.to(device, non_blocking=True)
            
            sdf_values = mandelbulb_sdf_batch(points)
            
            slab_shape = (res, res, z_end - z_start)
            sdf_slab = sdf_values.reshape(slab_shape)
            
            # Copy back to CPU volume
            volume[:, :, z_start:z_end] = sdf_slab.cpu()

    torch.cuda.synchronize()
    print(f"Volume calculation time: {time.time() - start_time:.4f}s")
    
    return volume.numpy(), (min_bound, max_bound)

def visualize_plotly(volume, bounds, level=0.01):
    """
    Uses Plotly (WebGL) to extract and render the isosurface.
    This replaces skimage's Marching Cubes.
    """
    print("Rendering interactive isosurface with Plotly...")
    
    # Create the grid coordinates matching the volume
    res = volume.shape[0]
    x, y, z = np.mgrid[
        bounds[0]:bounds[1]:complex(res), 
        bounds[0]:bounds[1]:complex(res), 
        bounds[0]:bounds[1]:complex(res)
    ]
    
    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=volume.flatten(),
        isomin=0.0, # The 'inside' of the fractal (dist approx 0)
        isomax=level, # The 'surface' threshold
        surface_count=2, # How many layers to draw (low for sharp surface)
        caps=dict(x_show=False, y_show=False),
        colorscale='Plasma',
    ))
    
    fig.update_layout(
        title="Mandelbulb (Plotly WebGL)",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    fig.show()

def visualize_point_cloud_fallback(volume, bounds, level=0.02):
    """
    Fallback: Visualizes the surface as a Point Cloud using standard Matplotlib.
    Does not require skimage or plotly.
    """
    print("Using Matplotlib Point Cloud fallback...")
    
    # Filter points that are close to the surface (shell)
    # The fractal surface is where SDF is approx 0.
    # We take a thin slice: 0 < dist < level
    mask = (volume > 0) & (volume < level)
    
    # Get indices
    ix, iy, iz = np.where(mask)
    
    # Convert indices to world coordinates
    res = volume.shape[0]
    step = (bounds[1] - bounds[0]) / res
    
    px = bounds[0] + ix * step
    py = bounds[0] + iy * step
    pz = bounds[0] + iz * step
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    # s=1 makes them small dots
    ax.scatter(px, py, pz, s=1, c=pz, cmap='magma', alpha=0.6)
    
    ax.set_title("Mandelbulb Point Cloud (No External Libs)")
    plt.show()

if __name__ == "__main__":
    # Resolution: 64 is safe/fast. 100+ looks great but requires more VRAM/RAM.
    RES = 64 
    
    # 1. Compute
    vol, bounds = generate_volume_streamed(res=RES, n_streams=4)
    
    # 2. Visualize
    if HAS_PLOTLY:
        # Best visual, interactive, uses GPU
        visualize_plotly(vol, bounds, level=0.02)
    else:
        # Robust fallback, just shows points
        visualize_point_cloud_fallback(vol, bounds, level=0.02)