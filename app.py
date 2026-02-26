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
API_BASE              = "https://tpg.marsmathis.com/api"
R_EARTH               = 6371.0
INTERACTIVE_GRID_STEP = 1.0
STATIC_GRID_STEP      = 0.5

st.set_page_config(page_title="TPG Voronoi Map", page_icon="🗺️", layout="wide")

# ─────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────
TAB20 = plt.cm.get_cmap("tab20")

def player_colors(n: int) -> list[str]:
    return [to_hex(TAB20(i / max(n - 1, 1))) for i in range(n)]

# ─────────────────────────────────────────────
# MATH HELPERS
# ─────────────────────────────────────────────

def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lr, lor = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(lr)*np.cos(lor), np.cos(lr)*np.sin(lor), np.sin(lr)])

def chord_to_km(chord: np.ndarray) -> np.ndarray:
    return 2 * R_EARTH * np.arcsin(np.clip(chord, 0, 2) / 2)

def min_dist_to_player(query_lat: float, query_lon: float, pts: np.ndarray) -> float:
    """Great-circle distance (km) from a point to the nearest submission in pts."""
    # Squeeze to shape (3,) so cKDTree.query returns a plain scalar, not a (1,) array
    query_xyz = latlon_to_xyz(np.array([query_lat]), np.array([query_lon]))[0]
    tree = cKDTree(latlon_to_xyz(pts[:, 0], pts[:, 1]))
    chord, _ = tree.query(query_xyz)
    return float(2 * R_EARTH * np.arcsin(np.clip(float(chord), 0, 2) / 2))

# ─────────────────────────────────────────────
# CONTINENT / COASTLINE GEOMETRY  (cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_coastline_latlons(scale: str = "110m") -> tuple[list, list]:
    """
    Extract Natural Earth coastline + country borders as flat lat/lon lists
    with None separators so Plotly draws disconnected line segments correctly.
    """
    from cartopy.feature import NaturalEarthFeature

    def geom_to_coords(geom):
        lats, lons = [], []
        if geom.geom_type == "LineString":
            xs, ys = geom.xy
            lons += list(xs) + [None]
            lats += list(ys) + [None]
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                xs, ys = part.xy
                lons += list(xs) + [None]
                lats += list(ys) + [None]
        elif geom.geom_type == "Polygon":
            xs, ys = geom.exterior.xy
            lons += list(xs) + [None]
            lats += list(ys) + [None]
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                xs, ys = poly.exterior.xy
                lons += list(xs) + [None]
                lats += list(ys) + [None]
        return lats, lons

    all_lats, all_lons = [], []

    for feature_name in ("coastline", "land"):
        feat = NaturalEarthFeature("physical", feature_name, scale)
        for geom in feat.geometries():
            la, lo = geom_to_coords(geom)
            all_lats += la
            all_lons += lo

    for feature_name in ("admin_0_countries",):
        feat = NaturalEarthFeature("cultural", feature_name, scale)
        for geom in feat.geometries():
            la, lo = geom_to_coords(geom)
            all_lats += la
            all_lons += lo

    return all_lats, all_lons

# ─────────────────────────────────────────────
# API CALLS
# ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_players() -> list[dict]:
    resp = requests.get(f"{API_BASE}/players", timeout=15)
    resp.raise_for_status()
    seen: dict[str, dict] = {}
    for entry in resp.json():
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
        pts = [(float(e["lat"]), float(e["lon"])) for e in resp.json() if "lat" in e and "lon" in e]
        return np.array(pts) if pts else None
    except Exception as exc:
        st.warning(f"Could not fetch submissions for {discord_id}: {exc}")
        return None

# ─────────────────────────────────────────────
# VORONOI COMPUTATION
# ─────────────────────────────────────────────

def compute_voronoi(
    player_points: list[np.ndarray],
    mode: str,
    grid_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_arr = np.arange(-90,  90  + grid_step, grid_step)
    lon_arr = np.arange(-180, 180 + grid_step, grid_step)
    LON, LAT = np.meshgrid(lon_arr, lat_arr)
    grid_xyz = latlon_to_xyz(LAT.ravel(), LON.ravel())

    min_dists = np.empty((len(player_points), grid_xyz.shape[0]))
    for i, pts in enumerate(player_points):
        tree = cKDTree(latlon_to_xyz(pts[:, 0], pts[:, 1]))
        chord, _ = tree.query(grid_xyz, workers=-1)
        min_dists[i] = chord_to_km(chord)

    fn = np.argmin if mode == "Win" else np.argmax
    return fn(min_dists, axis=0).reshape(LAT.shape), LON, LAT


def query_point(
    query_lat: float,
    query_lon: float,
    player_names: list[str],
    player_points: list[np.ndarray],
    mode: str,
) -> dict:
    """
    Return full per-player distance breakdown and the winner/loser at a point.
    """
    dists = {
        name: min_dist_to_player(query_lat, query_lon, pts)
        for name, pts in zip(player_names, player_points)
    }
    ranked = sorted(dists.items(), key=lambda x: x[1])   # nearest first
    winner = ranked[0][0]   # closest  → wins
    loser  = ranked[-1][0]  # furthest → loses
    result = winner if mode == "Win" else loser
    return {"result": result, "ranked": ranked, "mode": mode}

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
    query_result: dict | None = None,
    query_lat: float | None   = None,
    query_lon: float | None   = None,
) -> go.Figure:
    n      = len(player_names)
    colors = player_colors(n)
    flat   = result_grid.ravel()
    lats_g = LAT.ravel()
    lons_g = LON.ravel()

    fig = go.Figure()

    # ── 1. Voronoi region tiles (one trace per player) ───────────
    for i, (name, color) in enumerate(zip(player_names, colors)):
        mask = flat == i
        fig.add_trace(go.Scattergeo(
            lat=lats_g[mask],
            lon=lons_g[mask],
            mode="markers",
            marker=dict(symbol="square", size=7, color=color, opacity=0.78, line=dict(width=0)),
            name=name,
            legendgroup=f"region_{i}",
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Lat: %{lat:.1f}°  Lon: %{lon:.1f}°<br>"
                f"<i>{'Nearest' if mode == 'Win' else 'Furthest'} player</i>"
                "<extra></extra>"
            ),
        ))

    # ── 2. Continent outlines (rendered ON TOP of tiles) ─────────
    coast_lats, coast_lons = get_coastline_latlons()
    fig.add_trace(go.Scattergeo(
        lat=coast_lats,
        lon=coast_lons,
        mode="lines",
        line=dict(color="white", width=0.7),
        hoverinfo="skip",
        showlegend=False,
        name="coastlines",
    ))

    # ── 3. Optional raw submission points ────────────────────────
    if show_submissions:
        for i, (name, pts, color) in enumerate(zip(player_names, player_points, colors)):
            fig.add_trace(go.Scattergeo(
                lat=pts[:, 0], lon=pts[:, 1],
                mode="markers",
                marker=dict(symbol="circle", size=5, color="white",
                            line=dict(color=color, width=1.5)),
                name=f"{name} – submissions",
                legendgroup=f"sub_{i}",
                hovertemplate=(
                    f"<b>{name}</b> – submission<br>"
                    "Lat: %{lat:.4f}°  Lon: %{lon:.4f}°<extra></extra>"
                ),
                visible="legendonly",
            ))

    # ── 4. Queried point marker ───────────────────────────────────
    if query_result is not None and query_lat is not None and query_lon is not None:
        winner     = query_result["result"]
        win_idx    = player_names.index(winner)
        win_color  = colors[win_idx]
        label_verb = "Winner" if mode == "Win" else "Loser"

        # Outer glow ring
        fig.add_trace(go.Scattergeo(
            lat=[query_lat], lon=[query_lon],
            mode="markers",
            marker=dict(symbol="circle", size=22, color="rgba(0,0,0,0)",
                        line=dict(color="white", width=2)),
            hoverinfo="skip",
            showlegend=False,
        ))
        # Filled pin
        hover_lines = "<br>".join(
            f"{'→ ' if r[0] == winner else '   '}<b>{r[0]}</b>: {r[1]:,.0f} km"
            for r in query_result["ranked"]
        )
        fig.add_trace(go.Scattergeo(
            lat=[query_lat], lon=[query_lon],
            mode="markers+text",
            marker=dict(symbol="circle", size=14, color=win_color,
                        line=dict(color="white", width=2)),
            text=[f"📍 {label_verb}: {winner}"],
            textposition="top center",
            textfont=dict(color="white", size=11),
            name=f"📍 Queried point",
            hovertemplate=(
                f"<b>Queried point</b><br>"
                f"Lat: {query_lat:.4f}°  Lon: {query_lon:.4f}°<br>"
                f"<b>{label_verb}: {winner}</b><br><br>"
                f"All distances (nearest → furthest):<br>"
                f"{hover_lines}"
                "<extra></extra>"
            ),
            showlegend=True,
        ))

    # ── Layout ────────────────────────────────────────────────────
    tc = "#4CAF50" if mode == "Win" else "#f44336"
    fig.update_layout(
        title=dict(
            text=f"{'🏆 Win Regions' if mode == 'Win' else '💀 Loss Regions'} — Voronoi Map",
            font=dict(color=tc, size=17), x=0.5, xanchor="center",
        ),
        geo=dict(
            projection_type="natural earth",
            showland=False,
            showocean=True,      oceancolor="#1a2a3a",
            showlakes=True,      lakecolor="#1a2a3a",
            showcountries=False,
            showcoastlines=False,   # we draw our own on top
            showframe=False,
            bgcolor="#0e1117",
            lonaxis=dict(range=[-180, 180]),
            lataxis=dict(range=[-90, 90]),
        ),
        legend=dict(
            bgcolor="#1e2130", bordercolor="#444444", borderwidth=1,
            font=dict(color="white", size=11), itemsizing="constant",
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
        height=660,
        hoverlabel=dict(bgcolor="#1e2130", font_color="white", bordercolor="#555"),
        uirevision="voronoi",
    )
    return fig

# ─────────────────────────────────────────────
# STATIC MATPLOTLIB PNG
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

    fig = plt.figure(figsize=(18, 9), facecolor="#0e1117")
    ax  = plt.axes(projection=ccrs.Robinson(), facecolor="#0e1117")
    ax.set_global()
    ax.pcolormesh(LON, LAT, result_grid,
                  cmap=ListedColormap(colors), vmin=-0.5, vmax=n - 0.5,
                  alpha=0.80, shading="auto", transform=ccrs.PlateCarree(), zorder=1)
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.6, edgecolor="white",   zorder=2)
    ax.add_feature(cfeature.BORDERS,    linestyle=":", linewidth=0.35, edgecolor="#aaaaaa", zorder=2)
    ax.gridlines(color="#444444", linewidth=0.3, zorder=2)

    patches = [mpatches.Patch(facecolor=colors[i], edgecolor="white",
                               linewidth=0.4, label=player_names[i]) for i in range(n)]
    leg = ax.legend(handles=patches, loc="lower left", bbox_to_anchor=(1.01, 0.0),
                    fontsize=9, framealpha=0.85, facecolor="#1e2130",
                    edgecolor="#555555", labelcolor="white",
                    title=f"{'Nearest' if mode=='Win' else 'Furthest'} player",
                    title_fontsize=9)
    leg.get_title().set_color("white")
    tc = "#4CAF50" if mode == "Win" else "#f44336"
    ax.set_title(f"{'🏆 Win Regions' if mode == 'Win' else '💀 Loss Regions'} — Voronoi Map",
                 color=tc, fontsize=15, pad=12, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
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
    .query-result-box {
        background: #1e2130; border: 1px solid #444;
        border-radius: 8px; padding: 1rem 1.25rem; margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🗺️ TPG Voronoi Map Generator")
st.caption(
    "Each region is coloured by the player with the **closest** (Win) or "
    "**furthest** (Loss) submission from that point on Earth. "
    "Zoom, pan, hover — and query any location below the map."
)

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio("Map Mode", ["Win", "Loss"],
                    help="**Win** — closest player owns each region.\n\n**Loss** — furthest player owns each region.")
    if mode == "Win":
        st.success("🏆 Win Area Mode")
    else:
        st.error("💀 Loss Area Mode")

    st.divider()
    show_submissions = st.checkbox(
        "Add submission-point traces",
        value=False,
        help="Adds hidden per-player traces. Click a '– submissions' entry in the legend to show them.",
    )

    st.divider()
    st.markdown("**Map controls**")
    st.markdown(
        "- **Scroll** to zoom, **drag** to pan\n"
        "- **Hover** any region → player + coordinates\n"
        "- **Click** legend entry → toggle\n"
        "- **Double-click** legend entry → isolate\n"
        "- Use the **Point Query** below the map to check any coordinate"
    )
    st.divider()
    st.caption("Interactive map: 1° grid for browser speed. PNG download: 0.5° for detail.")

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
        "Select Players", options=options,
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

    with st.spinner("Computing Voronoi regions…"):
        t0 = time.time()
        grid_i, LON_i, LAT_i = compute_voronoi(player_points, mode, INTERACTIVE_GRID_STEP)
        elapsed = time.time() - t0

    # Persist in session state so point-query rerenders don't recompute
    st.session_state["voronoi"] = dict(
        grid=grid_i, LON=LON_i, LAT=LAT_i,
        player_names=player_names, player_points=player_points,
        mode=mode,
    )
    st.session_state.pop("query_result", None)   # clear stale query on recalculate
    st.success(f"✅ Computed in {elapsed:.1f}s")

# ── Render map (from session state so point queries don't re-trigger compute) ──
if "voronoi" not in st.session_state:
    st.stop()

v             = st.session_state["voronoi"]
player_names  = v["player_names"]
player_points = v["player_points"]
stored_mode   = v["mode"]

# Warn if mode changed since last calculation
if stored_mode != mode:
    st.warning(f"ℹ️ The map below was computed in **{stored_mode}** mode. Hit Calculate again to update.")

qr        = st.session_state.get("query_result")
q_lat     = st.session_state.get("query_lat")
q_lon     = st.session_state.get("query_lon")

with st.expander("📋 Submission counts", expanded=False):
    for name, pts in zip(player_names, player_points):
        st.write(f"**{name}**: {len(pts):,} submission(s)")

st.subheader(f"{'🏆' if stored_mode == 'Win' else '💀'} {stored_mode} Regions")
fig = render_interactive(
    v["grid"], v["LON"], v["LAT"],
    player_names, player_points,
    stored_mode, show_submissions,
    query_result=qr, query_lat=q_lat, query_lon=q_lon,
)
st.plotly_chart(fig, use_container_width=True, key="main_map")

# ── Point Query ───────────────────────────────
st.divider()
st.subheader("📍 Point Query")
st.caption("Enter any coordinates to find out which player wins or loses at that exact location.")

qcol1, qcol2, qcol3 = st.columns([2, 2, 1])
with qcol1:
    input_lat = st.number_input("Latitude",  min_value=-90.0,  max_value=90.0,  value=0.0, step=0.0001, format="%.4f")
with qcol2:
    input_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.0001, format="%.4f")
with qcol3:
    st.write("")   # vertical alignment spacer
    st.write("")
    query_clicked = st.button("🔍 Query", type="primary", use_container_width=True)

if query_clicked:
    with st.spinner("Calculating distances…"):
        result = query_point(input_lat, input_lon, player_names, player_points, stored_mode)
    st.session_state["query_result"] = result
    st.session_state["query_lat"]    = input_lat
    st.session_state["query_lon"]    = input_lon
    st.rerun()

if qr is not None:
    winner     = qr["result"]
    label_verb = "Winner 🏆" if stored_mode == "Win" else "Loser 💀"
    win_color  = player_colors(len(player_names))[player_names.index(winner)]

    res_cols = st.columns([2, 3])
    with res_cols[0]:
        st.markdown(
            f"<div class='query-result-box'>"
            f"<div style='color:#aaa;font-size:0.8rem;margin-bottom:4px'>"
            f"Point: {q_lat:.4f}°, {q_lon:.4f}°</div>"
            f"<div style='font-size:0.9rem;margin-bottom:2px'>{label_verb}</div>"
            f"<div style='font-size:1.5rem;font-weight:700;color:{win_color}'>{winner}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with res_cols[1]:
        st.markdown("**All players — distance to nearest submission (closest → furthest)**")
        n_players = len(qr["ranked"])
        colors_all = player_colors(n_players)
        color_map  = {name: colors_all[player_names.index(name)] for name, _ in qr["ranked"]}

        for rank, (name, dist) in enumerate(qr["ranked"]):
            is_result = name == winner
            bar_pct   = dist / qr["ranked"][-1][1] if qr["ranked"][-1][1] > 0 else 0
            badge     = (f" &nbsp;<span style='background:{win_color};color:#000;"
                         f"border-radius:4px;padding:1px 6px;font-size:0.75rem'>"
                         f"{'WINNER' if stored_mode=='Win' else 'LOSER'}</span>") if is_result else ""
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
                f"<div style='width:12px;height:12px;border-radius:2px;"
                f"background:{color_map[name]};flex-shrink:0'></div>"
                f"<div style='flex:1'>"
                f"<div style='font-size:0.9rem'><b>{name}</b>{badge}</div>"
                f"<div style='background:#333;border-radius:3px;height:6px;margin-top:3px'>"
                f"<div style='background:{color_map[name]};width:{bar_pct*100:.1f}%;"
                f"height:6px;border-radius:3px'></div></div>"
                f"</div>"
                f"<div style='font-size:0.85rem;color:#ccc;min-width:80px;text-align:right'>"
                f"{dist:,.0f} km</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ── High-res PNG Download ─────────────────────
st.divider()
with st.expander("⬇️ Download high-resolution PNG (0.5° grid)", expanded=False):
    with st.spinner("Rendering high-res map…"):
        grid_s, LON_s, LAT_s = compute_voronoi(player_points, stored_mode, STATIC_GRID_STEP)
        png_bytes = render_static_png(grid_s, LON_s, LAT_s, player_names, stored_mode)
    st.image(png_bytes, use_column_width=True)
    st.download_button(
        label="⬇️ Save PNG",
        data=png_bytes,
        file_name=f"voronoi_{stored_mode.lower()}_map.png",
        mime="image/png",
    )
