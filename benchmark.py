# Terrain Gradient Encoding — Benchmark

Companion code for the paper:

> **Two-Component Terrain Gradient Encoding for Constant-Time Slope Lookup in Grid-Based Simulation**
> Ehsan Sobhani, Geoffrey Pond
> Interservice/Industry Training, Simulation and Education Conference (I/ITSEC), 2025

## Overview

This repository contains the benchmark script used to empirically evaluate the
runtime performance of the proposed terrain gradient encoding scheme against a
naïve baseline in a synthetic grid-based agent simulation.

The proposed method precomputes signed north–south and east–west slope components
from Terrain-RGB elevation tiles and encodes them into the red and green channels
of an output raster tile. During simulation, the heading-aligned terrain grade is
retrieved in O(1) time per agent per step via a single pixel read and a dot product,
eliminating all elevation decoding and finite-difference computation from the
runtime inner loop.

## Repository Structure

```
terrain-gradient-encoding/
├── benchmark.py       # Full benchmark: tile generation, encoding, and runtime comparison
└── README.md
```

## What `benchmark.py` Does

The script implements and times two approaches end-to-end:

**Naïve approach** — performed per agent per step at runtime:
1. Read R, G, B channels of the center pixel and its 4 neighbors
2. Reconstruct 24-bit integer: `E = R×65536 + G×256 + B`
3. Decode elevation: `z = −10000 + 0.1 × E` (×5 pixels)
4. Apply central differences to estimate `g_NS` and `g_EW`
5. Dot product with agent heading → directional grade

**Encoded approach** — preprocessing done once offline, then per agent per step:
1. Read R, G channels from the precomputed gradient tile
2. Decode: `g_NS = ((R − 128) / 127) × g_max`
3. Decode: `g_EW = ((G − 128) / 127) × g_max`
4. Dot product with agent heading → directional grade

The script also contains the full preprocessing pipeline internally
(`build_gradient_tile`) which is run once before timing begins and is
excluded from all runtime measurements.

## Requirements

```
numpy
```

Install with:

```bash
pip install numpy
```

## Usage

```bash
python benchmark.py
```

Expected output (results will vary by machine):

```
Building synthetic terrain tile ...
Building encoded gradient tile ...

  Agents    Steps   Naive(s)  Encoded(s)   Speedup
----------------------------------------------------
   1,000   10,000      3.28       0.33      10.0x
   5,000   10,000      9.60       1.30       7.4x
  10,000   10,000     32.52       3.35       9.7x

Speedup range: 7.4x -- 10.0x
```

## Configuration

Key parameters are defined at the top of `benchmark.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TILE_W`, `TILE_H` | `2048, 2048` | Synthetic tile dimensions in pixels |
| `G_MAX` | `0.5` | Saturation grade (max representable slope, 50%) |
| `STEPS` | `10,000` | Number of simulation steps per benchmark run |

Agent population sizes are set in the `configs` list in `__main__`:
```python
configs = [1_000, 5_000, 10_000]
```

## Synthetic Terrain

The benchmark generates a synthetic `2048×2048` Terrain-RGB tile from a
smooth multi-frequency sinusoidal elevation field:

```python
elev_m = 200 + 150*sin(x)*cos(0.7y) + 80*cos(1.3x + y)
```

This produces terrain spanning approximately 50–530 m elevation, representative
of gently to moderately rolling terrain. The tile is encoded into the
Terrain-RGB format (24-bit elevation distributed across R, G, B channels)
before benchmarking begins.

## Encoding Details

Elevation decoding from Terrain-RGB:
```
E = R×2¹⁶ + G×2⁸ + B
z = −10000 + 0.1 × E   (metres, 0.1 m resolution)
```

Gradient encoding into output tile:
```
b = round(128 + 127 × clip(g / g_max, −1, +1))
```

- Red channel   → NS gradient component (`g_NS`)
- Green channel → EW gradient component (`g_EW`)
- Blue channel  → reserved (set to 0)

Runtime grade retrieval for agent heading `(hx, hy)`:
```
g_parallel = hx × g_EW + hy × g_NS
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sobhani2025terraingrad,
  author       = {Sobhani, Ehsan and Pond, Geoffrey},
  title        = {Terrain Gradient Encoding for Grid-Based Simulation},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://github.com/YOUR_USERNAME/terrain-gradient-encoding}
}
```

## License

MIT License

Copyright (c) 2025 Ehsan Sobhani, Geoffrey Pond

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
