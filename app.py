import streamlit as st
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, to_hex
import plotly.graph_objects as go
from scipy.spatial import cKDTree
import io
import time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_BASE   = "https://tpg.marsmathis.com/api"
R_EARTH    = 6371.0   # km

# Grid steps: interactive uses coarser grid so Plotly stays snappy;
# static PNG uses finer grid for crisp output.
INTERACTIVE_GRID_STEP = 1.0    # ~65k cells
STATIC_GRID_STEP      = 0.5    # ~260k cells – better PNG quality

st.set_page_config(
    page_title="TPG Voronoi Map",
    page_icon="🗺️",
    layout="wide",
)

# ─────────────────────────────────────────────
# SHARED PALETTE  (up to 20 distinct colours)
# ─────────────────────────────────────────────
TAB20 = plt.cm.get_cmap("tab20")

def player_colors(n: int) -> list[str]:
    return [to_hex(TAB20(i / max(n - 1, 1))) for i in range(n)]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    return np.column_stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ])

def chord_to_km(chord: np.ndarray) -> np.ndarray:
    return 2 * R_EARTH * np.arcsin(np.clip(chord, 0, 2) / 2)

# ─────────────────────────────────────────────
# API CALLS  (cached)
# ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_players() -> list[dict]:
    resp = requests.get(f"{API_BASE}/players", timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    seen: dict[str, dict] = {}
    for entry in raw:
        did   = str(entry.get("discord_id", ""))
        name  = (entry.get("name") or "").strip()
        canon = (entry.get("canonical_name") or "").strip()
        if not did:
            continue
        if did not in seen:
            label = f"{name} ({canon})" if canon and canon != name else name
            seen[did] = {"discord_id": did, "display_label": label}
        seen[did].setdefault("search_terms", set()).add(name.lower())
        if canon:
            seen[did]["search_terms"].add(canon.lower())

    return list(seen.values())


@st.cache_data(ttl=120, show_spinner=False)
def fetch_submissions(discord_id: str) -> np.ndarray | None:
    try:
        resp = requests.get(f"{API_BASE}/submissions/{discord_id}", timeout=15)
        resp.raise_for_status()
        pts = [
            (float(e["lat"]), float(e["lon"]))
            for e in resp.json()
            if "lat" in e and "lon" in e
        ]
        return np.array(pts) if pts else None
    except Exception as exc:
        st.warning(f"Could not fetch submissions for {discord_id}: {exc}")
        return None

# ─────────────────────────────────────────────
# CORE COMPUTATION
# ─────────────────────────────────────────────

def compute_voronoi(
    player_points: list[np.ndarray],
    mode: str,
    grid_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (assignment_grid, LON_mesh, LAT_mesh).
    assignment_grid[i,j] = index of the nearest (Win) or furthest (Loss) player.
    """
    lat_arr = np.arange(-90,  90  + grid_step, grid_step)
    lon_arr = np.arange(-180, 180 + grid_step, grid_step)
    LON, LAT = np.meshgrid(lon_arr, lat_arr)
    grid_xyz = latlon_to_xyz(LAT.ravel(), LON.ravel())

    n = len(player_points)
    min_dists = np.empty((n, grid_xyz.shape[0]))
    for i, pts in enumerate(player_points):
        tree = cKDTree(latlon_to_xyz(pts[:, 0], pts[:, 1]))
        chord, _ = tree.query(grid_xyz, workers=-1)
        min_dists[i] = chord_to_km(chord)

    fn = np.argmin if mode == "Win" else np.argmax
    return fn(min_dists, axis=0).reshape(LAT.shape), LON, LAT

# ─────────────────────────────────────────────
# INTERACTIVE PLOTLY MAP
# ─────────────────────────────────────────────

def render_interactive(
    result_grid: np.ndarray,
    LON: np.ndarray,
    LAT: np.ndarray,
    player_names: list[str],
    player_points: list[np.ndarray],
    mode: str,
    show_submissions: bool,
) -> go.Figure:
    n      = len(player_names)
    colors = player_colors(n)
    flat   = result_grid.ravel()
    lats   = LAT.ravel()
    lons   = LON.ravel()

    fig = go.Figure()

    # One Voronoi region trace per player
    for i, (name, color) in enumerate(zip(player_names, colors)):
        mask = flat == i
        fig.add_trace(go.Scattergeo(
            lat=lats[mask],
            lon=lons[mask],
            mode="markers",
            marker=dict(
                symbol="square",
                size=7,
                color=color,
                opacity=0.82,
                line=dict(width=0),
            ),
            name=name,
            legendgroup=f"region_{i}",
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Lat: %{lat:.1f}°  Lon: %{lon:.1f}°<br>"
                f"<i>{'Nearest' if mode == 'Win' else 'Furthest'} player</i>"
                "<extra></extra>"
            ),
        ))

    # Optional submission-point overlay (hidden in legend by default)
    if show_submissions:
        for i, (name, pts, color) in enumerate(zip(player_names, player_points, colors)):
            fig.add_trace(go.Scattergeo(
                lat=pts[:, 0],
                lon=pts[:, 1],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=5,
                    color="white",
                    line=dict(color=color, width=1.5),
                ),
                name=f"{name} – submissions",
                legendgroup=f"sub_{i}",
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "Submission: %{lat:.4f}°, %{lon:.4f}°"
                    "<extra></extra>"
                ),
                visible="legendonly",   # click legend entry to reveal
            ))

    title_color = "#4CAF50" if mode == "Win" else "#f44336"
    fig.update_layout(
        title=dict(
            text=f"{'🏆 Win Regions' if mode == 'Win' else '💀 Loss Regions'} — Voronoi Map",
            font=dict(color=title_color, size=17),
            x=0.5, xanchor="center",
        ),
        geo=dict(
            projection_type="natural earth",
            showland=True,       landcolor="#2c2c2c",
            showocean=True,      oceancolor="#1a2a3a",
            showlakes=True,      lakecolor="#1a2a3a",
            showcountries=True,  countrycolor="#555555",
            showcoastlines=True, coastlinecolor="#888888",
            showframe=False,
            bgcolor="#0e1117",
            lonaxis=dict(range=[-180, 180]),
            lataxis=dict(range=[-90,   90]),
        ),
        legend=dict(
            bgcolor="#1e2130",
            bordercolor="#444444",
            borderwidth=1,
            font=dict(color="white", size=11),
            itemsizing="constant",
            title=dict(
                text=(
                    f"{'Nearest' if mode == 'Win' else 'Furthest'} player<br>"
                    "<sup>click = toggle · dbl-click = isolate</sup>"
                ),
                font=dict(color="#aaaaaa", size=10),
            ),
        ),
        paper_bgcolor="#0e1117",
        margin=dict(l=0, r=0, t=50, b=0),
        height=650,
        hoverlabel=dict(bgcolor="#1e2130", font_color="white", bordercolor="#555"),
        uirevision="voronoi",
    )
    return fig

# ─────────────────────────────────────────────
# STATIC MATPLOTLIB MAP  (high-res download)
# ─────────────────────────────────────────────

def render_static_png(
    result_grid: np.ndarray,
    LON: np.ndarray,
    LAT: np.ndarray,
    player_names: list[str],
    mode: str,
) -> bytes:
    n      = len(player_names)
    colors = player_colors(n)
    cmap   = ListedColormap(colors)

    fig = plt.figure(figsize=(18, 9), facecolor="#0e1117")
    ax  = plt.axes(projection=ccrs.Robinson(), facecolor="#0e1117")
    ax.set_global()
    ax.pcolormesh(
        LON, LAT, result_grid,
        cmap=cmap, vmin=-0.5, vmax=n - 0.5,
        alpha=0.80, shading="auto",
        transform=ccrs.PlateCarree(), zorder=1,
    )
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.5, edgecolor="white",   zorder=2)
    ax.add_feature(cfeature.BORDERS,    linestyle=":", linewidth=0.3,
                   edgecolor="#aaaaaa", zorder=2)
    ax.gridlines(color="#444444", linewidth=0.3, zorder=2)

    patches = [
        mpatches.Patch(facecolor=colors[i], edgecolor="white",
                       linewidth=0.4, label=player_names[i])
        for i in range(n)
    ]
    leg = ax.legend(
        handles=patches, loc="lower left", bbox_to_anchor=(1.01, 0.0),
        fontsize=9, framealpha=0.85, facecolor="#1e2130",
        edgecolor="#555555", labelcolor="white",
        title=f"{'Nearest' if mode == 'Win' else 'Furthest'} player",
        title_fontsize=9,
    )
    leg.get_title().set_color("white")

    tc = "#4CAF50" if mode == "Win" else "#f44336"
    ax.set_title(
        f"{'🏆 Win Regions' if mode == 'Win' else '💀 Loss Regions'} — Voronoi Map",
        color=tc, fontsize=15, pad=12, fontweight="bold",
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ═══════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMultiSelect [data-baseweb="tag"] { background-color: #2d4a6e; }
    div[data-testid="stRadio"] > label  { font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

st.title("🗺️ TPG Voronoi Map Generator")
st.caption(
    "Each region is coloured by the player with the **closest** (Win) or "
    "**furthest** (Loss) submission from that point on Earth. "
    "Zoom, pan, and hover to explore."
)

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Map Mode",
        ["Win", "Loss"],
        help=(
            "**Win** — closest player owns each region.\n\n"
            "**Loss** — furthest player owns each region."
        ),
    )
    if mode == "Win":
        st.success("🏆 Win Area Mode")
    else:
        st.error("💀 Loss Area Mode")

    st.divider()
    show_submissions = st.checkbox(
        "Add submission-point traces",
        value=False,
        help=(
            "Adds a per-player trace (hidden by default). "
            "Click a '– submissions' entry in the legend to reveal that player's points."
        ),
    )

    st.divider()
    st.markdown("**Map controls**")
    st.markdown(
        "- **Scroll** to zoom, **drag** to pan\n"
        "- **Hover** any region → see player + coordinates\n"
        "- **Click** a legend entry → hide/show that player\n"
        "- **Double-click** a legend entry → isolate it\n"
        "- High-res PNG available after computing (see expander below the map)"
    )

    st.divider()
    st.caption(
        "Interactive map uses a 1° grid for browser performance. "
        "The downloadable PNG is rendered at 0.5° for sharper detail."
    )

# ── Player Selection ──────────────────────────
col_sel, col_stats = st.columns([3, 1])

with col_sel:
    with st.spinner("Loading player list…"):
        try:
            players_data = fetch_players()
        except Exception as exc:
            st.error(f"Failed to load players: {exc}")
            st.stop()

    if not players_data:
        st.error("No players returned from the API.")
        st.stop()

    options        = [p["display_label"] for p in players_data]
    discord_id_map = {p["display_label"]: p["discord_id"] for p in players_data}

    selected_labels = st.multiselect(
        "Select Players",
        options=options,
        placeholder="Type a name or partial name to search…",
        help="Select 2 or more players, then click Calculate.",
    )

with col_stats:
    st.metric("Players Available", len(players_data))
    st.metric("Players Selected",  len(selected_labels))

# ── Calculate ─────────────────────────────────
if len(selected_labels) < 2:
    st.info("👆 Select **at least 2 players** then hit Calculate.")
    st.stop()

if st.button("🔄 Calculate Map", type="primary", use_container_width=True):

    # Fetch submissions
    player_names, player_points, fetch_errors = [], [], []
    prog = st.progress(0, text="Fetching player submissions…")
    for i, label in enumerate(selected_labels):
        pts = fetch_submissions(discord_id_map[label])
        if pts is not None and len(pts) > 0:
            player_names.append(label)
            player_points.append(pts)
        else:
            fetch_errors.append(label)
        prog.progress((i + 1) / len(selected_labels), text=f"Fetching {label}…")
    prog.empty()

    if fetch_errors:
        st.warning(f"No submission data for: {', '.join(fetch_errors)}")
    if len(player_names) < 2:
        st.error("Need at least 2 players with valid submissions.")
        st.stop()

    with st.expander("📋 Submission counts", expanded=False):
        for name, pts in zip(player_names, player_points):
            st.write(f"**{name}**: {len(pts):,} submission(s)")

    # Compute interactive grid
    with st.spinner("Computing Voronoi regions…"):
        t0 = time.time()
        grid_i, LON_i, LAT_i = compute_voronoi(player_points, mode, INTERACTIVE_GRID_STEP)
        elapsed = time.time() - t0

    st.success(f"✅ Computed in {elapsed:.1f}s")

    # Render & display interactive map
    st.subheader(f"{'🏆' if mode == 'Win' else '💀'} {mode} Regions")
    fig = render_interactive(
        grid_i, LON_i, LAT_i,
        player_names, player_points,
        mode, show_submissions,
    )
    st.plotly_chart(fig, use_container_width=True)

    # High-res PNG (lazy – only computed when expanded)
    with st.expander("⬇️ Download high-resolution PNG (0.5° grid)", expanded=False):
        with st.spinner("Rendering high-res static map…"):
            grid_s, LON_s, LAT_s = compute_voronoi(player_points, mode, STATIC_GRID_STEP)
            png_bytes = render_static_png(grid_s, LON_s, LAT_s, player_names, mode)
        st.image(png_bytes, use_column_width=True)
        st.download_button(
            label="⬇️ Save PNG",
            data=png_bytes,
            file_name=f"voronoi_{mode.lower()}_map.png",
            mime="image/png",
        )
