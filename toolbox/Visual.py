from typing import Optional, Dict, List, Any
import yaml
from pathlib import Path
import os
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.patches as mpatches
import networkx as nx
import geopandas as gpd
import mapclassify
import rasterio
from rasterio.plot import show as rioshow
from rasterio.features import rasterize
from shapely import wkt
from shapely.geometry import Point
import pandas as pd
import pickle
import csv
# from fastmcp import FastMCP


cfg_path = Path(__file__).parent.parent / "config.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
OUTPUT_DIR = cfg["output_dir"]

def _get_out_path(name: str, base_dir: str = OUTPUT_DIR) -> str:
    p = os.path.join(base_dir, name)
    if os.path.exists(p):
        raise FileExistsError(f"Output file already exists: {p}")
    return p

# mcp = FastMCP()

# @mcp.tool()
def create_travel_time_network_map(
    layers: List[Dict[str, Any]],
    output_name: str,
    title: str = None,
) -> Dict[str, Any]:
    """
    Create a styled travel-time network map from GraphML layers and save as PNG.

    Args:
        layers: List of dicts, each containing "data" and "style".
            data can be: file path (.graphml) or a networkx graph.
            style keys: {attr, cmap, node_size, edge_linewidth, legend, legend_kwds}
        output_name: Output filename (must end with .png).
        title: Optional map title.

    Returns:
        output_path: The path to the output image file.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for layer in layers:
        data = layer.get("data")
        style = layer.get("style", {})

        graph = None
        if isinstance(data, str):
            p = os.path.abspath(data)
            if not os.path.isfile(p):
                continue
            graph = ox.load_graphml(p)
        elif isinstance(data, nx.Graph):
            graph = data
        else:
            continue

        attr = style.get("attr", "time2nhb")
        cmap_name = style.get("cmap", "RdYlBu")
        node_size = int(style.get("node_size", 20))
        edge_linewidth = float(style.get("edge_linewidth", 0.5))
        legend = bool(style.get("legend", False))
        legend_kwds = style.get("legend_kwds", {}) or {}

        for _, nd in graph.nodes(data=True):
            v = nd.get(attr, None)
            try:
                nd[attr] = float(v) if v is not None else np.nan
            except Exception:
                nd[attr] = np.nan

        node_colors = ox.plot.get_node_colors_by_attr(graph, attr, cmap=cmap_name)
        fig, ax = ox.plot_graph(
            graph,
            node_size=node_size,
            node_color=node_colors,
            edge_linewidth=edge_linewidth,
            show=False,
            close=False,
            ax=ax,
        )

        if legend:
            vals = np.array([d.get(attr, np.nan) for _, d in graph.nodes(data=True)], dtype=float)
            finite = vals[~np.isnan(vals)]
            if finite.size > 0:
                vmin = float(finite.min())
                vmax = float(finite.max())
            else:
                vmin = 0.0
                vmax = 1.0
            cm_obj = cm.get_cmap(cmap_name)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            sm = cm.ScalarMappable(cmap=cm_obj, norm=norm)
            orientation = legend_kwds.get("orientation", "vertical")
            cbar = fig.colorbar(sm, ax=ax, orientation=orientation)
            label_text = legend_kwds.get("label")
            if label_text:
                cbar.set_label(label_text)

    if title:
        ax.set_title(title, fontsize=14)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.abspath(_get_out_path(output_name))
    ax.ticklabel_format(style="plain")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"output_path": output_path}


# @mcp.tool()
def visualize_vector(
    input_path: str,
    output_name: str,
    style: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    legend: bool = True,
    legend_kwds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Advanced visualization for a single vector layer.

    Args:
        input_path: Path to the vector file (.shp, .geojson).
        output_name: Output filename (e.g. "map.png").
        style: Visualization parameters:
            - column: Column to visualize (choropleth).
            - scheme: Classification scheme (e.g. 'quantiles', 'natural_breaks', 'equal_interval').
            - k: Number of classes (default 5).
            - cmap: Colormap name (e.g. 'viridis', 'Reds').
            - color: Fixed color if not using column.
            - hatch: Texture pattern (e.g. '///', '...').
            - alpha: Transparency (0.0-1.0).
            - categorical: True for categorical data.
        title: Map title.
        legend: Whether to show legend.
        legend_kwds: Legend parameters (loc, title, etc.).

    Returns:
        output_path: Path to the saved image.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(_get_out_path(output_name))

    gdf = gpd.read_file(input_path)
    style = style or {}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Handle hatching separately for better compatibility
    hatch = style.pop("hatch", None)
    scheme = style.pop("scheme", None)
    k = style.pop("k", 5)
    categorical = style.pop("categorical", False)
    column = style.get("column")
    
    plot_kwds = style.copy()
    if legend:
        plot_kwds["legend"] = True
        if legend_kwds:
            plot_kwds["legend_kwds"] = legend_kwds

    if column:
        if categorical:
            plot_kwds["categorical"] = True
        elif scheme:
            plot_kwds["scheme"] = scheme
            plot_kwds["k"] = k
        
        gdf.plot(ax=ax, **plot_kwds)
    else:
        gdf.plot(ax=ax, **plot_kwds)
        
    # Overlay hatch if specified
    if hatch:
        # Create a new style for hatch only (no fill)
        hatch_style = {
            "facecolor": "none",
            "edgecolor": plot_kwds.get("edgecolor", "black"),
            "linewidth": plot_kwds.get("linewidth", 0.5),
            "hatch": hatch
        }
        gdf.plot(ax=ax, **hatch_style)

    if title:
        ax.set_title(title, fontsize=14)
    
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"output_path": out_path}


# @mcp.tool()
def visualize_raster(
    input_path: str,
    output_name: str,
    bands: Optional[List[int]] = None,
    cmap: Optional[str] = None,
    stretch: bool = True,
    title: Optional[str] = None,
    legend: bool = False,
    legend_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Advanced visualization for raster data.

    Args:
        input_path: Path to the raster file (.tif).
        output_name: Output filename (e.g. "image.png").
        bands: List of band indices to visualize (1-based).
               - If 1 index: grayscale/pseudocolor.
               - If 3 indices: RGB composite (e.g. [3, 2, 1]).
        cmap: Colormap for single-band visualization.
        stretch: Whether to apply histogram stretching (2%-98%) for contrast.
        title: Map title.
        legend: Whether to show colorbar (only for single band).
        legend_label: Label for the colorbar.
        vmin: Minimum value for colormap scaling.
        vmax: Maximum value for colormap scaling.

    Returns:
        output_path: Path to the saved image.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(_get_out_path(output_name))

    with rasterio.open(input_path) as src:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        use_bands = bands if bands else [1]
        
        if len(use_bands) == 1:
            data = src.read(use_bands[0])
            if stretch and vmin is None and vmax is None:
                # Robustly handle nodata and NaNs for percentile calculation
                if src.nodata is not None:
                    # Filter out nodata AND NaNs
                    valid_mask = (data != src.nodata) & (~np.isnan(data))
                    valid_data = data[valid_mask]
                else:
                    # Just filter NaNs
                    valid_data = data[~np.isnan(data)]
                
                if valid_data.size > 0:
                    p2, p98 = np.percentile(valid_data, (2, 98))
                    data = np.clip(data, p2, p98)
            
            rioshow(data, transform=src.transform, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
            
            if legend:
                 im = ax.images[-1] if ax.images else None
                 if im:
                     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                     if legend_label:
                         cbar.set_label(legend_label)
            
        elif len(use_bands) == 3:
            data = src.read(use_bands)
            # Normalize for RGB display
            data_norm = np.zeros_like(data, dtype=float)
            for i in range(3):
                band = data[i]
                if stretch:
                    valid = band[band != src.nodata]
                    if valid.size > 0:
                        p2, p98 = np.percentile(valid, (2, 98))
                        # Avoid division by zero
                        if p98 > p2:
                            band_norm = (band - p2) / (p98 - p2)
                            data_norm[i] = np.clip(band_norm, 0, 1)
                else:
                    data_norm[i] = band / np.max(band)
            
            # Transpose to (H, W, 3) for imshow
            rioshow(data_norm, transform=src.transform, ax=ax)
            
        else:
            raise ValueError("Bands must contain 1 or 3 indices.")

        if title:
            ax.set_title(title, fontsize=14)
        
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {"output_path": out_path}


# @mcp.tool()
def create_multilayer_map(
    layers: List[Dict[str, Any]],
    output_name: str,
    title: Optional[str] = None,
    legend_loc: str = "best",
) -> Dict[str, Any]:
    """
    Create a complex map with multiple overlaid layers (vector & raster).
    Supports mixed visualization styles including hatching and transparency.

    Args:
        layers: List of dicts, drawn in order (bottom to top). Each dict:
            - data: Path to file (.shp, .geojson, .tif).
            - type: 'vector' or 'raster'.
            - style: Dict of parameters.
                For vector: {column, scheme, k, cmap, color, facecolor, hatch, alpha, label, linewidth, edgecolor}
                For raster: {bands, cmap, alpha, stretch, vmin, vmax, label}
        output_name: Output filename.
        title: Map title.
        legend_loc: Location of the legend (e.g. 'upper right').

    Returns:
        output_path: Path to the saved image.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(_get_out_path(output_name))

    fig, ax = plt.subplots(figsize=(12, 10))
    legend_handles = []

    for layer in layers:
        data_path = layer.get("data")
        ltype = layer.get("type", "vector")
        style = layer.get("style", {})
        label = style.get("label")

        if ltype == "vector":
            gdf = gpd.read_file(data_path)
            
            column = style.get("column")
            if column:
                 try:
                     import pandas as pd
                     converted = pd.to_numeric(gdf[column], errors='coerce')
                     # Only update if we didn't turn valid data into all NaNs (implies it was text)
                     if not (converted.isna().all() and not gdf[column].isna().all()):
                         gdf[column] = converted
                 except Exception:
                     pass

            hatch = style.pop("hatch", None)
            scheme = style.pop("scheme", None)
            k = style.pop("k", 5)
            
            # Prepare plot arguments
            plot_kwds = style.copy()
            if column:
                if scheme:
                    plot_kwds["scheme"] = scheme
                    plot_kwds["k"] = k
                    plot_kwds["legend"] = True 
                elif plot_kwds.get("categorical"):
                    plot_kwds["legend"] = True # Explicitly enable legend for categorical
            
            # Primary plot (fill/color)
            gdf.plot(ax=ax, **plot_kwds)
            
            # If classification scheme or categorical was used, capture the generated legend handles
            if column and (scheme or plot_kwds.get("categorical")) and plot_kwds.get("legend"):
                leg = ax.get_legend()
                if leg:
                    # If there is a label for this layer, add it as a header
                    if label:
                        # Create an invisible patch to act as a title/header in the legend
                        header = mpatches.Patch(facecolor='none', edgecolor='none', label=label)
                        legend_handles.append(header)

                    # Extract handles and labels
                    # We want to preserve these instead of replacing them with a single label
                    for h, t in zip(leg.legend_handles, leg.get_texts()):
                        # Create a copy or use the handle directly? 
                        # Ideally, we just collect them.
                        # We can prepend the layer label to the first item or just list them.
                        h.set_label(t.get_text())
                        legend_handles.append(h)
                    
                    # Remove the automatic legend so we can create a combined one later
                    leg.remove()
                    # Prevent creating a single manual patch for this layer
                    label = None
            
            # Secondary plot (hatch)
            if hatch:
                hatch_style = {
                    "facecolor": "none",
                    "edgecolor": plot_kwds.get("edgecolor", "black"),
                    "linewidth": plot_kwds.get("linewidth", 0.0), # No border for hatch overlay usually
                    "hatch": hatch,
                    "alpha": plot_kwds.get("alpha", 1.0)
                }
                gdf.plot(ax=ax, **hatch_style)
                
            # Manual legend handle if label is provided
            if label:
                # Determine facecolor
                if column:
                    # Attempt to get representative color from cmap
                    cmap_name = style.get("cmap", "viridis")
                    try:
                        cmap = plt.get_cmap(cmap_name)
                        patch_face = cmap(0.6)
                    except:
                        patch_face = "blue"
                else:
                    # Prioritize facecolor, then color, then default
                    patch_face = style.get("facecolor")
                    if patch_face is None:
                        patch_face = style.get("color", "blue")
                
                patch_edge = style.get("edgecolor", "black")
                
                patch = mpatches.Patch(
                    facecolor=patch_face,
                    edgecolor=patch_edge,
                    hatch=hatch,
                    alpha=style.get("alpha", 1.0),
                    label=label
                )
                legend_handles.append(patch)

        elif ltype == "raster":
            with rasterio.open(data_path) as src:
                alpha = style.get("alpha", 1.0)
                cmap = style.get("cmap", "gray")
                data = src.read(1) # Default to band 1
                
                # Mask nodata
                if src.nodata is not None:
                    data = np.ma.masked_equal(data, src.nodata)
                
                # Determine vmin/vmax
                vmin = style.get("vmin")
                vmax = style.get("vmax")
                
                if vmin is None or vmax is None:
                    if isinstance(data, np.ma.MaskedArray):
                        valid_data = data.compressed()
                    else:
                        valid_data = data.flatten()
                        valid_data = valid_data[~np.isnan(valid_data)]

                    if valid_data.size > 0:
                        # Auto-stretch if not provided
                        p2, p98 = np.percentile(valid_data, (2, 98))
                        if vmin is None: vmin = p2
                        if vmax is None: vmax = p98
                        
                # Clip data for display if we have bounds (auto or manual)
                if vmin is not None and vmax is not None:
                     data = np.clip(data, vmin, vmax)
                
                rioshow(data, transform=src.transform, ax=ax, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)

                # Add colorbar if requested
                if style.get("colorbar"):
                     im = ax.images[-1]
                     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                     if label:
                         cbar.set_label(label)
                     # If colorbar is used, we might not want the manual patch in the main legend
                     # But we'll leave it up to the caller or logic below
                     # If label is consumed by colorbar, maybe don't add to legend_handles?
                     # Let's assume if colorbar is present, we skip the legend patch unless forced.
                     if not style.get("legend_patch", False):
                         label = None

            # Manual legend handle if label is provided for raster
            if label:
                # Attempt to get representative color from cmap
                try:
                    cm = plt.get_cmap(cmap)
                    # Pick a color from the middle-upper range to be visible
                    patch_face = cm(0.7)
                except:
                    patch_face = "red" # Default fallback for raster if cmap fails
                
                patch = mpatches.Patch(
                    facecolor=patch_face,
                    edgecolor="black",
                    alpha=alpha,
                    label=label
                )
                legend_handles.append(patch)

    if title:
        ax.set_title(title, fontsize=16)
    
    # Combine geopandas automatic handles with manual ones
    if legend_handles:
        # Force manual legend to ensure all layers are represented
        ax.legend(handles=legend_handles, loc=legend_loc)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"output_path": out_path}

# # @mcp.tool()
# def create_map(
#     layers: List[Dict[str, Any]],
#     output_name: Optional[str] = None,
#     title: str = None,
# ) -> Dict[str, Any]:
#     """
#     Create a styled map from multiple inputs (vectors, rasters, WKT, or coords).

#     Args:
#         layers: List of dicts, each containing "data" and "style".
#             data can be: file path (.shp, .geojson, .tif), WKT string, coords, or GeoDataFrame.
#             style keys: {color, column, cmap, alpha, linewidth, edgecolor, markersize, legend, legend_kwds}
#             NOTE: Layers are drawn in order. Put polygons first (bottom), then lines, then points (top).
#         output_name: File name only (e.g., "xxx.png").
#         title: Optional map title.

#     Returns:
#         output_path: The path to the output map file.
#     """
#     import os
#     import geopandas as gpd
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import rasterio
#     from rasterio.plot import show as rioshow
#     from shapely import wkt

#     fig, ax = plt.subplots(figsize=(10, 8))

#     for layer in layers:
#         data = layer.get("data")
#         style = layer.get("style", {})
#         label = style.pop("label", None)

#         gdf = None

#         if isinstance(data, str):
#             data = os.path.abspath(data)

#             if data.lower().endswith(".shp") or data.lower().endswith(".geojson"):
#                 gdf = gpd.read_file(data)
#             elif data.lower().endswith(".tif"):
#                 with rasterio.open(data) as src:
#                     cb_keys = ("legend", "legend_kwds")
#                     show_kwargs = {k: v for k, v in style.items() if k not in cb_keys}
#                     rioshow(src, ax=ax, **show_kwargs)
#                     if style.get("legend"):
#                         im = ax.get_images()[-1] if ax.get_images() else None
#                         if im is not None:
#                             kw = style.get("legend_kwds", {}) or {}
#                             orientation = kw.get("orientation", "vertical")
#                             cbar = fig.colorbar(im, ax=ax, orientation=orientation)
#                             label_text = kw.get("label")
#                             if label_text:
#                                 cbar.set_label(label_text)
#                     continue
#             else:
#                 geom = wkt.loads(data)
#                 gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

#         elif isinstance(data, gpd.GeoDataFrame):
#             gdf = data

#         elif isinstance(data, list):
#             from shapely.geometry import Polygon, LineString, Point
#             if len(data) > 2:
#                 geom = Polygon(data)
#             elif len(data) == 2:
#                 geom = LineString(data)
#             else:
#                 geom = Point(data)
#             gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

#         if gdf is not None:
#             if len(gdf) == 0:
#                 continue

#             # if specified column not in gdf, remove it to avoid error
#             if "column" in style and style["column"] not in gdf.columns:
#                 style.pop("column", None)

#             # if specified column exists but all values are NaN/None, remove it to avoid error
#             if "column" in style:
#                 col = style["column"]
#                 try:
#                     non_na = gdf[col].notna().sum()
#                 except Exception:
#                     non_na = 0
#                 if non_na == 0:
#                     style.pop("column", None)

#             if "column" in style:
#                 column = style.pop("column")
#                 # Ensure column is numeric so geopandas uses a colorbar (which accepts 'label' in legend_kwds)
#                 # rather than a categorical legend (which crashes with 'label' in legend_kwds).
#                 try:
#                     gdf[column] = pd.to_numeric(gdf[column], errors='coerce')
#                 except Exception:
#                     pass
#                 gdf.plot(ax=ax, column=column, **style, label=label)
#             else:
#                 gdf.plot(ax=ax, **style, label=label)

#     if title:
#         ax.set_title(title, fontsize=14)

#     output_name = output_name
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     output_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))
#     ax.ticklabel_format(style="plain")
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     return {
#         "output_path": output_path
#     }


# @mcp.tool()
def build_nodes_points_layer_from_mapping(
    graph_path: str,
    times_path: str,
    output_name: str,
    attr_name: str,
    fill_missing: Optional[float] = None,
    scale_from_max: Optional[float] = None,
    x_attr: str = "x",
    y_attr: str = "y",
    crs: Optional[str] = None,
    drop_missing_values: bool = False,
) -> Dict[str, Any]:
    """
    Build a node points GeoDataFrame using a time mapping for symbology.

    Args:
        graph_path: Path to GraphML containing node coordinates.
        times_path: Pickle file with node_id -> time mapping.
        output_name: GeoJSON filename to save in OUTPUT_DIR.
        attr_name: Column name to store time values (Required).
        fill_missing: Value to fill missing times; ignored if None.
        scale_from_max: If provided and missing, fill as max*scale_from_max.
        x_attr, y_attr: Node attribute names for x/y coordinates.
        crs: CRS string to use; falls back to graph CRS or EPSG:4326.
        drop_missing_values: If True, remove rows with missing time.

    Returns:
        A dict containing the output path of the saved GeoJSON.
    """
    gp = os.path.abspath(graph_path)
    tp = os.path.abspath(times_path)
    if not os.path.isfile(gp):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    if not os.path.isfile(tp):
        raise FileNotFoundError(f"Times file not found: {times_path}")
        
    G = ox.load_graphml(gp)
    with open(tp, "rb") as f:
        dists = pickle.load(f)
        
    xs, ys, vals = [], [], []
    for n, data in G.nodes(data=True):
        x = data.get(x_attr)
        y = data.get(y_attr)
        if x is None or y is None:
            continue
        v = dists.get(n, None)
        if v is None:
            val = None
        else:
            try:
                val = float(v)
            except Exception:
                try:
                    val = float(str(v).strip())
                except Exception:
                    val = None
        xs.append(x)
        ys.append(y)
        vals.append(val)
        
    finite = [v for v in vals if v is not None and np.isfinite(v)]
    if fill_missing is not None:
        if scale_from_max is not None and finite:
            mf = max(finite)
            vals = [mf * scale_from_max if v is None else v for v in vals]
        else:
            vals = [fill_missing if v is None else v for v in vals]
            
    if drop_missing_values:
        filtered = [(x, y, v) for x, y, v in zip(xs, ys, vals) if v is not None]
        xs, ys, vals = (list(t) for t in zip(*filtered)) if filtered else ([], [], [])
        
    use_crs = crs or G.graph.get("crs") or "EPSG:4326"
    gdf = gpd.GeoDataFrame({attr_name: vals}, geometry=[Point(x, y) for x, y in zip(xs, ys)], crs=use_crs)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(_get_out_path(output_name))
    gdf.to_file(out_path, driver="GeoJSON")
            
    return {"output_path": out_path}


# @mcp.tool()
def build_edges_lines_layer_from_graph(
    graph_path: str,
    output_name: str,
    attr_name: Optional[str] = None,
    drop_missing_values: bool = False,
) -> Dict[str, Any]:
    """
    Build an edges lines GeoDataFrame from a graph for mapping.

    Args:
        graph_path: Path to the GraphML file.
        output_name: GeoJSON filename to save in OUTPUT_DIR.
        attr_name: Optional edge attribute used for styling/filters.
        drop_missing_values: If True, drop edges with missing attr_name.

    Returns:
        A dict containing the output path of the saved GeoJSON.
    """
    gp = os.path.abspath(graph_path)
    if not os.path.isfile(gp):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
        
    G = ox.load_graphml(gp)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    
    if attr_name and attr_name not in edges_gdf.columns:
        attr_name = None
    if attr_name and drop_missing_values:
        edges_gdf = edges_gdf[edges_gdf[attr_name].notna()]
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(_get_out_path(output_name))
    edges_gdf.to_file(out_path, driver="GeoJSON")
            
    return {"output_path": out_path}


# @mcp.tool()
def create_chart(
        layers: List[Dict[str, Any]],
        output_name: str,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        invert_y: bool = False,
        secondary_xlabel: Optional[str] = None
) -> Dict[str, Any]:
    """
    Render line charts from CSV/data layers.

    Args:
        layers: List of dicts; each entry includes:
            - data: CSV path, or used with x_data/y_data
            - style: {x_col, y_col, label, color, linewidth, x_data?, y_data?}
        output_name: File name only (e.g., "xxx.png").
        title, xlabel, ylabel: Optional figure title and axis labels.
        invert_y: Whether to invert the y-axis (e.g., for depth).

    Returns:
        output_path: Path to the saved figure.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(_get_out_path(output_name))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax2 = None
    for layer in layers:
        data = layer.get("data")
        style = layer.get("style", {})
        label = style.get("label")
        color = style.get("color")
        linewidth = style.get("linewidth", 2)
        x_col = style.get("x_col")
        y_col = style.get("y_col")
        x = style.get("x_data") or layer.get("x_data")
        y = style.get("y_data") or layer.get("y_data")
        axis_sel = (style.get("axis") or style.get("x_axis") or "primary").lower()
        target = ax
        if axis_sel in ("secondary", "top"):
            if ax2 is None:
                ax2 = ax.twiny()
            target = ax2
        if x is None or y is None:
            csv_path = data if isinstance(data, str) else None
            if not csv_path or not x_col or not y_col:
                continue
            xs, ys = [], []
            with open(csv_path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        xs.append(float(row[x_col]))
                        ys.append(float(row[y_col]))
                    except Exception:
                        pass
            x = np.array(xs)
            y = np.array(ys)
        target.plot(x, y, label=label, color=color, linewidth=linewidth, alpha=0.85)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if secondary_xlabel and ax2 is not None:
        ax2.set_xlabel(secondary_xlabel)
    if invert_y:
        ax.invert_yaxis()
    handles = []
    labels = []
    h, l = ax.get_legend_handles_labels()
    handles += h; labels += l
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2; labels += l2
    if labels:
        ax.legend(handles, labels)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"output_path": out_path}

# @mcp.tool()
def plot_group_metrics_layers(
    analysis_results: List[Dict[str, Any]],
    group_field: str,
    layers: List[Dict[str, Any]],
    output_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Render a single figure with one subplot per layer.

    Args:
        analysis_results: List of records containing group and metric fields.
        group_field: Column name used as x-axis grouping.
        layers: List of dicts; each has 'key' (metric column) and optional 'style' dict.
        output_name: File name only (e.g., "xxx.png").

    Returns:
        output_path: Path to the generated chart image.
    """
    df = pd.DataFrame(analysis_results)
    if group_field not in df.columns:
        alt = f"{group_field}_"
        if alt in df.columns:
            group_field = alt
        else:
            raise ValueError(f"Group field '{group_field}' not in analysis results")
    if not layers or not isinstance(layers, list):
        raise ValueError("No layers provided")

    species = df[group_field].astype(str)
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]
    palette = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, layer in enumerate(layers):
        key = layer.get('key') or layer.get('metric_key')
        style = layer.get('style', {})
        if not key or key not in df.columns:
            raise ValueError(f"Metric key '{key}' not in analysis results")
        vals = pd.to_numeric(df[key], errors='coerce').fillna(0)
        color = style.get('color', palette[i % len(palette)])
        xlabel = style.get('xlabel', 'Group')
        ylabel = style.get('ylabel', key)
        title = style.get('title', key)
        rotation = style.get('rotation', 45)
        ax = axes[i]
        ax.bar(species, vals, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)
            label.set_horizontalalignment('right')

    output_name = output_name
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return {
        "output_path": output_path
    }

# @mcp.tool()
def rasterize_vector_labels(
    point_path: str,
    metadata_path: str,
    output_name: str,
    buffer_meters: float = 1000.0,
    label_value: int = 1,
    fill_value: int = 0,
) -> Dict[str, Any]:
    """
    Rasterize vector data (e.g., mineral occurrences) to create label data.
    
    Args:
        point_path: Path to the vector file (.shp/.geojson).
        metadata_path: Path to the metadata pickle file containing transform and shape.
        output_name: Output filename (must end with .npy).
        buffer_meters: Buffer distance in meters, default 1000.0.
        label_value: Value to burn into raster for features, default 1.
        fill_value: Background value, default 0.
        
    Returns:
        output_path: Path to the saved labels numpy array.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    transform = metadata["transform"]
    shape = metadata["shape"] # (n_bands, height, width)
    
    df = gpd.read_file(point_path)
    target_crs = metadata.get("crs")
    if target_crs is not None:
        try:
            df = df.to_crs(target_crs)
        except Exception:
            pass
    if buffer_meters and buffer_meters > 0:
        geoms = df.buffer(buffer_meters).geometry
    else:
        geoms = df.geometry
    geometry_generator = ((geom, label_value) for geom in geoms)
    
    # Shape for rasterize should be (height, width)
    out_shape = (shape[1], shape[2])
    
    labels = rasterize(
        shapes=geometry_generator,
        out_shape=out_shape,
        fill=fill_value,
        transform=transform
    ).astype('float32')
    
    output_path = _get_out_path(output_name)
    np.save(output_path, labels)
    
    return {"output_path": output_path}