"""
Benchmark: Naive runtime Terrain-RGB gradient computation
vs. precomputed encoded tile lookup.

Naive approach:
  For each agent at each step:
    1. Read R,G,B of pixel (x,y)        -> compute E = R*65536 + G*256 + B
    2. Apply elevation formula           -> z = -10000 + 0.1*E
    3. Do same for 4 neighbors
    4. Apply central difference          -> g_NS, g_EW
    5. Dot product with heading          -> g_parallel

Encoded approach:
  Preprocessing (offline, not timed):
    Decode all elevations, compute all gradients, encode to 8-bit tile.
  Per agent per step:
    1. Read R,G of gradient tile at (x,y)
    2. Decode g_NS = (R-128)/127 * g_max
    3. Decode g_EW = (G-128)/127 * g_max
    4. Dot product with heading          -> g_parallel
"""

import numpy as np
import time

RNG = np.random.default_rng(42)
TILE_W, TILE_H = 2048, 2048   # large synthetic tile
G_MAX = 0.5
STEPS = 10_000

# ----------------------------------------------------------------
# Build synthetic Terrain-RGB tile (W x H x 3, uint8)
# ----------------------------------------------------------------
def make_terrain_rgb(W, H):
    # Synthetic elevation: smooth hills via sine waves
    x = np.linspace(0, 4*np.pi, W)
    y = np.linspace(0, 4*np.pi, H)
    XX, YY = np.meshgrid(x, y)
    elev_m = 200 + 150*np.sin(XX)*np.cos(YY*0.7) + 80*np.cos(XX*1.3 + YY)
    # Convert to Terrain-RGB encoding
    E = np.round((elev_m + 10000) / 0.1).astype(np.int32)
    E = np.clip(E, 0, 2**24 - 1)
    R = ((E >> 16) & 0xFF).astype(np.uint8)
    G = ((E >> 8)  & 0xFF).astype(np.uint8)
    B = ( E        & 0xFF).astype(np.uint8)
    tile = np.stack([R, G, B], axis=2)   # shape (H, W, 3)
    return tile

# ----------------------------------------------------------------
# PREPROCESSING: Build encoded gradient tile (offline)
# ----------------------------------------------------------------
def build_gradient_tile(terrain_rgb, lat_deg=43.81, zoom=14, g_max=G_MAX):
    H, W, _ = terrain_rgb.shape
    R = terrain_rgb[:,:,0].astype(np.float32)
    G = terrain_rgb[:,:,1].astype(np.float32)
    B = terrain_rgb[:,:,2].astype(np.float32)
    E = R * 65536 + G * 256 + B
    z = -10000.0 + 0.1 * E                # elevation in metres, shape (H,W)

    # Ground resolution (Web Mercator)
    d = 156543.034 * np.cos(np.radians(lat_deg)) / (2 ** zoom)

    # Central differences (interior pixels; border clamped)
    g_EW = np.zeros_like(z)
    g_NS = np.zeros_like(z)
    g_EW[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2 * d)
    g_NS[1:-1, :] = (z[:-2, :] - z[2:, :]) / (2 * d)
    # clamp borders
    g_EW[:, 0]  = g_EW[:, 1]
    g_EW[:, -1] = g_EW[:, -2]
    g_NS[0, :]  = g_NS[1, :]
    g_NS[-1, :] = g_NS[-2, :]

    # Encode to uint8
    def enc(g):
        g_clip = np.clip(g / g_max, -1.0, 1.0)
        b = np.round(128 + 127 * g_clip).astype(np.uint8)
        return b

    grad_tile = np.stack([enc(g_NS), enc(g_EW),
                          np.zeros((H, W), dtype=np.uint8)], axis=2)
    return grad_tile, d

# ----------------------------------------------------------------
# Generate random agent positions and headings
# ----------------------------------------------------------------
def make_agents(n, W, H):
    # pixel positions (interior so central diff always valid)
    px = RNG.integers(1, W-1, size=n)
    py = RNG.integers(1, H-1, size=n)
    # random unit headings (hx=east component, hy=north component)
    theta = RNG.uniform(0, 2*np.pi, size=n)
    hx = np.cos(theta).astype(np.float32)
    hy = np.sin(theta).astype(np.float32)
    return px, py, hx, hy

# ----------------------------------------------------------------
# NAIVE: decode elevation + central diff at runtime, vectorised
#        over all agents simultaneously (one step at a time)
# ----------------------------------------------------------------
def naive_step(terrain_rgb, px, py, hx, hy):
    """Compute heading-aligned grade for all agents in one step."""
    H, W, _ = terrain_rgb.shape
    # Neighbor indices (clamp to bounds)
    px_m = np.clip(px - 1, 0, W-1)
    px_p = np.clip(px + 1, 0, W-1)
    py_m = np.clip(py - 1, 0, H-1)
    py_p = np.clip(py + 1, 0, H-1)

    # Decode elevation at 5 positions per agent
    def decode(ix, iy):
        r = terrain_rgb[iy, ix, 0].astype(np.float32)
        g = terrain_rgb[iy, ix, 1].astype(np.float32)
        b = terrain_rgb[iy, ix, 2].astype(np.float32)
        E = r * 65536.0 + g * 256.0 + b
        return -10000.0 + 0.1 * E

    z_c  = decode(px,   py)
    z_xp = decode(px_p, py)
    z_xm = decode(px_m, py)
    z_yp = decode(px,   py_p)
    z_ym = decode(px,   py_m)

    # Precomputed constant d (same tile/zoom for all)
    d = 19.1          # metres/pixel at zoom 14, lat 43.81
    g_EW = (z_xp - z_xm) / (2.0 * d)
    g_NS = (z_ym - z_yp) / (2.0 * d)

    return hx * g_EW + hy * g_NS

# ----------------------------------------------------------------
# ENCODED: pixel read + linear decode + dot product, vectorised
# ----------------------------------------------------------------
def encoded_step(grad_tile, px, py, hx, hy, g_max=G_MAX):
    R = grad_tile[py, px, 0].astype(np.float32)
    G = grad_tile[py, px, 1].astype(np.float32)
    g_NS = ((R - 128.0) / 127.0) * g_max
    g_EW = ((G - 128.0) / 127.0) * g_max
    return hx * g_EW + hy * g_NS

# ----------------------------------------------------------------
# Full benchmark
# ----------------------------------------------------------------
def benchmark(n_agents, n_steps, terrain_rgb, grad_tile):
    px, py, hx, hy = make_agents(n_agents, TILE_W, TILE_H)

    # --- Naive ---
    t0 = time.perf_counter()
    for _ in range(n_steps):
        _ = naive_step(terrain_rgb, px, py, hx, hy)
    t_naive = time.perf_counter() - t0

    # --- Encoded ---
    t0 = time.perf_counter()
    for _ in range(n_steps):
        _ = encoded_step(grad_tile, px, py, hx, hy)
    t_enc = time.perf_counter() - t0

    speedup = t_naive / t_enc
    return t_naive, t_enc, speedup

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("Building synthetic terrain tile ...", flush=True)
    terrain_rgb = make_terrain_rgb(TILE_W, TILE_H)

    print("Building encoded gradient tile ...", flush=True)
    grad_tile, d = build_gradient_tile(terrain_rgb)

    configs = [1_000, 5_000, 10_000]

    print(f"\n{'Agents':>8} {'Steps':>8} {'Naive(s)':>10} {'Encoded(s)':>11} {'Speedup':>9}")
    print("-" * 52)

    results = []
    for n_agents in configs:
        t_naive, t_enc, speedup = benchmark(n_agents, STEPS, terrain_rgb, grad_tile)
        results.append((n_agents, STEPS, t_naive, t_enc, speedup))
        print(f"{n_agents:>8,} {STEPS:>8,} {t_naive:>10.3f} {t_enc:>11.3f} {speedup:>8.1f}x")

    # Print LaTeX table row data
    print("\n--- LaTeX table rows ---")
    for n_agents, n_steps, t_naive, t_enc, speedup in results:
        print(f"{n_agents:>6,} & {n_steps:>6,} & {t_naive:.2f} & {t_enc:.2f} & {speedup:.1f}$\\times$ \\\\")

    speedups = [r[4] for r in results]
    print(f"\nSpeedup range: {min(speedups):.1f}x -- {max(speedups):.1f}x")
