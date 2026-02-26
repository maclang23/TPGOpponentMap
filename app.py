import streamlit as st
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree
import io
import time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_BASE = "https://tpg.marsmathis.com/api"
GRID_STEP = 0.5          # degrees – trade quality vs speed
R_EARTH   = 6371.0       # km
MAX_COLORS = 20

st.set_page_config(
    page_title="TPG Voronoi Map",
    page_icon="🗺️",
    layout="wide",
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert lat/lon (degrees) to unit-sphere XYZ for cKDTree."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x, y, z])

def chord_to_km(chord: float) -> float:
    """Convert unit-sphere chord distance to great-circle km."""
    chord = np.clip(chord, 0, 2)
    return 2 * R_EARTH * np.arcsin(chord / 2)

# ─────────────────────────────────────────────
# API CALLS (cached)
# ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_players() -> list[dict]:
    """
    Returns a list of dicts: {discord_id, display_label, names: [...]}
    Deduplicates by discord_id; builds display as 'name (canonical_name)'.
    """
    resp = requests.get(f"{API_BASE}/players", timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    seen: dict[str, dict] = {}
    for entry in raw:
        did  = str(entry.get("discord_id", ""))
        name = entry.get("name", "").strip()
        canon = entry.get("canonical_name") or ""
        canon = canon.strip()

        if not did:
            continue

        if did not in seen:
            label = f"{name} ({canon})" if canon and canon != name else name
            seen[did] = {"discord_id": did, "display_label": label, "search_terms": set()}

        seen[did]["search_terms"].add(name.lower())
        if canon:
            seen[did]["search_terms"].add(canon.lower())

    return list(seen.values())


@st.cache_data(ttl=120, show_spinner=False)
def fetch_submissions(discord_id: str) -> np.ndarray | None:
    """Returns Nx2 array of (lat, lon) for a player, or None."""
    try:
        resp = requests.get(f"{API_BASE}/submissions/{discord_id}", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pts = []
        for entry in data:
            try:
                lat = float(entry["lat"])
                lon = float(entry["lon"])
                pts.append((lat, lon))
            except (KeyError, TypeError, ValueError):
                continue
        if not pts:
            return None
        return np.array(pts)
    except Exception as e:
        st.warning(f"Could not fetch submissions for {discord_id}: {e}")
        return None

# ─────────────────────────────────────────────
# COMPUTATION
# ─────────────────────────────────────────────

def compute_map(
    player_names: list[str],
    player_points: list[np.ndarray],
    mode: str,
    grid_step: float = GRID_STEP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (winner_grid, LON_mesh, LAT_mesh) where each cell contains the
    index of the winning (mode='Win') or losing (mode='Loss') player.
    """
    lat_grid = np.arange(-90,  90  + grid_step, grid_step)
    lon_grid = np.arange(-180, 180 + grid_step, grid_step)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    grid_xyz = latlon_to_xyz(LAT.ravel(), LON.ravel())   # (M, 3)

    # Build KDTree per player and query min distance for every grid point
    n_players = len(player_names)
    min_dists = np.empty((n_players, grid_xyz.shape[0]))

    for i, pts in enumerate(player_points):
        tree = cKDTree(latlon_to_xyz(pts[:, 0], pts[:, 1]))
        chord, _ = tree.query(grid_xyz, workers=-1)
        min_dists[i] = chord_to_km(chord)

    # For each grid cell select winner (min dist) or loser (max dist)
    if mode == "Win":
        result = np.argmin(min_dists, axis=0)
    else:
        result = np.argmax(min_dists, axis=0)

    return result.reshape(LAT.shape), LON, LAT

# ─────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────

TAB20 = plt.cm.get_cmap("tab20")

def render_map(
    result_grid: np.ndarray,
    LON: np.ndarray,
    LAT: np.ndarray,
    player_names: list[str],
    mode: str,
) -> bytes:
    n = len(player_names)
    colors = [TAB20(i / max(n - 1, 1)) for i in range(n)]
    cmap = ListedColormap(colors[:n])

    fig = plt.figure(figsize=(18, 9), facecolor="#0e1117")
    ax = plt.axes(projection=ccrs.Robinson(), facecolor="#0e1117")
    ax.set_global()

    ax.pcolormesh(
        LON, LAT, result_grid,
        cmap=cmap,
        vmin=-0.5, vmax=n - 0.5,
        alpha=0.75,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="white", zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.3, edgecolor="#aaaaaa", zorder=2)
    ax.gridlines(color="#444444", linewidth=0.3, zorder=2)

    legend_elements = [
        mpatches.Patch(facecolor=colors[i], edgecolor="white", linewidth=0.5, label=player_names[i])
        for i in range(n)
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        fontsize=9,
        framealpha=0.85,
        facecolor="#1e2130",
        edgecolor="#555555",
        labelcolor="white",
        title=f"{'Winner' if mode == 'Win' else 'Loser'} (closest {'to' if mode == 'Win' else 'from'} location)",
        title_fontsize=9,
    )
    legend.get_title().set_color("white")

    title_color = "#4CAF50" if mode == "Win" else "#f44336"
    plt.title(
        f"{'🏆 Win Regions' if mode == 'Win' else '💀 Loss Regions'} — Voronoi Map",
        color=title_color,
        fontsize=15,
        pad=12,
        fontweight="bold",
    )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMultiSelect [data-baseweb="tag"] { background-color: #2d4a6e; }
    .mode-win  { color: #4CAF50; font-weight: bold; }
    .mode-loss { color: #f44336; font-weight: bold; }
    div[data-testid="stRadio"] > label { font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

st.title("🗺️ TPG Voronoi Map Generator")
st.caption("See which player is geographically closest (Win) or furthest (Loss) from any location on Earth.")

# ── Sidebar Controls ──────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Map Mode",
        options=["Win", "Loss"],
        help=(
            "**Win Mode** — each region is colored by the player with the closest submission.\n\n"
            "**Loss Mode** — each region is colored by the player with the furthest submission."
        ),
    )
    if mode == "Win":
        st.success("🏆 Win Area Mode: closest player wins each region")
    else:
        st.error("💀 Loss Area Mode: furthest player loses each region")

    resolution = st.select_slider(
        "Grid Resolution",
        options=[1.0, 0.75, 0.5, 0.35, 0.25],
        value=0.5,
        help="Finer resolution = sharper map but longer computation.",
        format_func=lambda x: f"{x}° (~{'fast' if x >= 0.75 else ('medium' if x >= 0.5 else 'slow')})",
    )

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Each colored region shows the player whose nearest submission is closest (Win) "
        "or furthest (Loss) from that location — a geographic Voronoi diagram."
    )

# ── Player Selection ──────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    with st.spinner("Loading player list…"):
        try:
            players_data = fetch_players()
        except Exception as e:
            st.error(f"Failed to load players: {e}")
            st.stop()

    if not players_data:
        st.error("No players found in the API.")
        st.stop()

    options        = [p["display_label"] for p in players_data]
    discord_id_map = {p["display_label"]: p["discord_id"] for p in players_data}

    selected_labels = st.multiselect(
        "Select Players",
        options=options,
        placeholder="Type a name to search, or click to select…",
        help="Select 2 or more players to generate the map.",
    )

with col2:
    st.metric("Players Available", len(players_data))
    st.metric("Players Selected", len(selected_labels))

# ── Calculate Button ──────────────────────────────
if len(selected_labels) < 2:
    st.info("👆 Select **at least 2 players** from the dropdown, then hit Calculate.")
    st.stop()

if st.button("🔄 Calculate Map", type="primary", use_container_width=True):
    # --- Fetch submissions ---
    player_names  = []
    player_points = []
    fetch_errors  = []

    progress = st.progress(0, text="Fetching player submissions…")
    for i, label in enumerate(selected_labels):
        did = discord_id_map[label]
        pts = fetch_submissions(did)
        if pts is not None and len(pts) > 0:
            player_names.append(label)
            player_points.append(pts)
        else:
            fetch_errors.append(label)
        progress.progress((i + 1) / len(selected_labels), text=f"Fetching {label}…")

    progress.empty()

    if fetch_errors:
        st.warning(f"No submission data found for: {', '.join(fetch_errors)}")

    if len(player_names) < 2:
        st.error("Need at least 2 players with valid submissions to generate a map.")
        st.stop()

    # Show submission counts
    with st.expander("📋 Submission counts", expanded=False):
        for name, pts in zip(player_names, player_points):
            st.write(f"**{name}**: {len(pts)} submission(s)")

    # --- Compute grid ---
    with st.spinner(f"Computing {mode} regions across the globe… (this may take ~10–30 seconds)"):
        t0 = time.time()
        result_grid, LON, LAT = compute_map(player_names, player_points, mode, grid_step=resolution)
        elapsed = time.time() - t0

    st.success(f"✅ Computation complete in {elapsed:.1f}s")

    # --- Render map ---
    with st.spinner("Rendering map…"):
        img_bytes = render_map(result_grid, LON, LAT, player_names, mode)

    mode_label = "Win" if mode == "Win" else "Loss"
    st.subheader(f"{'🏆' if mode == 'Win' else '💀'} {mode_label} Regions Map")
    st.image(img_bytes, use_column_width=True)

    st.download_button(
        label="⬇️ Download Map (PNG)",
        data=img_bytes,
        file_name=f"voronoi_{mode.lower()}_map.png",
        mime="image/png",
    )

    # --- Show player submission scatter ---
    with st.expander("🔍 View player submission points on a map", expanded=False):
        import plotly.graph_objects as go
        fig = go.Figure()
        colors_hex = [matplotlib.colors.to_hex(TAB20(i / max(len(player_names) - 1, 1))) for i in range(len(player_names))]
        for name, pts, color in zip(player_names, player_points, colors_hex):
            fig.add_trace(go.Scattergeo(
                lat=pts[:, 0], lon=pts[:, 1],
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.8),
                name=name,
            ))
        fig.update_layout(
            geo=dict(showland=True, landcolor="#2a2a2a", showocean=True, oceancolor="#1a2a3a",
                     showcountries=True, countrycolor="#555555", bgcolor="#0e1117"),
            legend=dict(bgcolor="#1e2130", font=dict(color="white")),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            title=dict(text="Player Submission Points", font=dict(color="white")),
        )
        st.plotly_chart(fig, use_container_width=True)
