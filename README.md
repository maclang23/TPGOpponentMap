# TPG Voronoi Map Generator

A Streamlit app that renders geographic Voronoi-style maps showing which TPG player has the closest (Win Mode) or furthest (Loss Mode) submission from any location on Earth.

---

## Features

- **Win Mode** — each map region is colored by the player whose nearest submission is closest to that point.
- **Loss Mode** — each map region is colored by the player whose nearest submission is *furthest* from that point.
- **Live player search** — type a name or partial name to find players from the TPG API.
- **Multi-player selection** — compare any combination of players simultaneously.
- **Adjustable grid resolution** — balance between sharpness and computation speed.
- **Download** the rendered map as a PNG.
- **Interactive scatter** — inspect each player's individual submission points on an interactive globe.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on Cartopy**: Cartopy requires some system libraries. On Ubuntu/Debian:
> ```bash
> sudo apt-get install libgeos-dev libproj-dev
> ```
> On macOS with Homebrew:
> ```bash
> brew install geos proj
> ```

### 2. Run locally

```bash
streamlit run app.py
```

### 3. Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set `app.py` as the main file.
4. Add a `packages.txt` file (see below) for system dependencies.

**`packages.txt`** (required for Streamlit Cloud):
```
libgeos-dev
libproj-dev
proj-data
proj-bin
```

---

## How it works

1. Player list is fetched from `https://tpg.marsmathis.com/api/players` (cached for 5 minutes).
2. When you click **Calculate**, submissions are fetched from `https://tpg.marsmathis.com/api/submissions/{discord_id}` for each selected player.
3. A global grid is computed at the chosen resolution (default 0.5°).
4. For each grid cell, the minimum Haversine distance to each player's submissions is computed using a **scipy cKDTree** for fast nearest-neighbor queries.
5. Each cell is assigned to the player with the **minimum** distance (Win) or **maximum** minimum distance (Loss).
6. The result is rendered with **Matplotlib + Cartopy** and displayed as a high-resolution PNG.

---

## Performance

| Resolution | Grid Size    | Approx. Time |
|------------|--------------|--------------|
| 1.0°       | ~65k cells   | ~5s          |
| 0.5°       | ~260k cells  | ~15s         |
| 0.35°      | ~530k cells  | ~30s         |
| 0.25°      | ~1M cells    | ~60s         |

Times depend on number of players and their submission counts.
