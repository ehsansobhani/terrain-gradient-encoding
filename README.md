# Terrain Gradient Encoding

Companion code for the paper:

> **Two-Component Terrain Gradient Encoding for Constant-Time Slope Lookup
> in Grid-Based Simulation**
> Ehsan Sobhani, Geoffrey Pond
>    ... conference should be added

## Overview

This repository provides a lightweight preprocessing pipeline that converts
Terrain-RGB elevation tiles into a two-component signed gradient tile.
During simulation, terrain slope along any agent heading is retrieved in
O(1) time via a single pixel read and a dot product — no elevation decoding
or finite-difference computation required at runtime.

## How It Works
```
Terrain-RGB tile  →  Decode elevation  →  Central differences
→  Encode NS/EW grade into R/G channels  →  Gradient PNG tile
```

At runtime:
```python
g_NS = ((R - 128) / 127) * g_max
g_EW = ((G - 128) / 127) * g_max
g_parallel = hx * g_EW + hy * g_NS   # O(1) per agent per step
```

## Repository Structure
```
terrain-gradient-encoding/
├── benchmark.py          # Runtime comparison: naive vs encoded
├── encode.py             # Terrain-RGB → gradient tile pipeline
├── decode.py             # Runtime gradient retrieval utilities
├── requirements.txt      # Python dependencies
└── README.md
```

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/terrain-gradient-encoding
cd terrain-gradient-encoding
pip install -r requirements.txt
```

## Usage

### Encode a Terrain-RGB tile
```python
from encode import build_gradient_tile
import numpy as np
from PIL import Image

# Load your Terrain-RGB tile (H x W x 3, uint8)
terrain_rgb = np.array(Image.open("terrain_rgb.png"))

# Encode gradient tile
grad_tile, d = build_gradient_tile(
    terrain_rgb,
    lat_deg=43.81,   # tile center latitude
    zoom=14,         # zoom level
    g_max=0.5        # saturation grade (50% slope)
)

Image.fromarray(grad_tile).save("gradient_tile.png")
```

### Retrieve slope at runtime
```python
from decode import get_grade
import numpy as np

grad_tile = np.array(Image.open("gradient_tile.png"))

# Agent at pixel (x=500, y=300) heading northeast
g = get_grade(
    grad_tile,
    x=500, y=300,
    heading=(0.707, 0.707),   # (hx, hy) unit vector
    g_max=0.5
)
print(f"Directional grade: {g:.4f}")  # e.g. 0.0142
```

### Run the benchmark
```bash
python benchmark.py
```

Expected output:
```
  Agents    Steps   Naive(s)  Encoded(s)   Speedup
   1,000   10,000      3.28       0.33      10.0x
   5,000   10,000      9.60       1.30       7.4x
  10,000   10,000     32.52       3.35       9.7x
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `g_max` | Saturation grade (max representable slope) | `0.5` |
| `lat_deg` | Tile center latitude in degrees | required |
| `zoom` | Web Mercator zoom level | required |

## Encoding Details

Elevation is decoded from Terrain-RGB as:
```
E = R×2¹⁶ + G×2⁸ + B
z = −10000 + 0.1 × E   (metres)
```

Gradients are encoded as signed 8-bit values centered at 128:
```
b = round(128 + 127 × clip(g / g_max, −1, +1))
```

Red channel → NS component, Green channel → EW component,
Blue channel → reserved.

## Citation

If you use this code in your research, please cite:
```bibtex
@software{sobhani2025terraingrad,
  author    = {Sobhani, Ehsan and Pond, Geoffrey},
  title     = {Terrain Gradient Encoding for Grid-Based Simulation},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/YOUR_USERNAME/terrain-gradient-encoding}
}
```

## License

MIT License — see `LICENSE` for details.
```

---

**Topics/tags to add on GitHub** (improves discoverability):
```
terrain-analysis  gis  elevation  simulation  agent-based-modeling
slope  raster  web-mercator  geospatial  python
