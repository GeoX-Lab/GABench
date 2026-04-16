from typing import Optional, Dict, List, Any
import yaml
from pathlib import Path
import os
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy.ndimage import generic_filter
import geopandas as gpd
from shapely.geometry import shape, LineString
from shapely.ops import linemerge
from collections import deque
from scipy.interpolate import griddata
from rasterio.transform import from_origin
from rasterio.mask import mask
from numpy.lib.stride_tricks import sliding_window_view
from netCDF4 import Dataset
import pandas as pd
import heapq
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
def compute_slope(raster_path: str, output_name: str, band: int = 1) -> Dict[str, Any]:
    """
    Compute slope raster from a specified band of a raster file.

    Args:
        raster_path: Path to the input raster TIFF file.
        output_name: Output filename (must end with .tif).
        band: Band index to compute slope from (default: 1).

    Returns:
        output_path: Path to the output slope raster TIFF file.
    """
    if not os.path.isfile(raster_path):
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with rasterio.open(raster_path) as src:
        if not (1 <= band <= src.count):
            raise ValueError(f"Invalid band index: {band}. Raster has {src.count} band(s).")
        data = src.read(band, masked=True)
        imarray = np.asarray(data, dtype=np.float64)
        if np.ma.isMaskedArray(data):
            imarray = np.where(np.ma.getmaskarray(data), np.nan, np.ma.filled(data, np.nan))
        imarray = -imarray
        height, width = imarray.shape
        grad_y, grad_x = np.gradient(imarray, height, width)
        slope_deg = np.degrees(np.arctan(np.hypot(grad_x, grad_y)))

        profile = src.profile.copy()
        profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})
        slope_path = _get_out_path(output_name)
        with rasterio.open(slope_path, "w", **profile) as dst:
            dst.write(slope_deg.astype(np.float32), 1)

    return {
        "output_path": slope_path,
    }

# @mcp.tool()
def compute_aspect(raster_path: str, output_name: str, band: int = 1) -> Dict[str, Any]:
    """
    Compute aspect raster from a specified band of a raster file.

    Args:
        raster_path: Path to the input raster TIFF file.
        output_name: Output filename (must end with .tif).
        band: Band index to compute aspect from (default: 1).

    Returns:
        output_path: Path to the output aspect raster TIFF file.
    """
    if not os.path.isfile(raster_path):
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with rasterio.open(raster_path) as src:
        if not (1 <= band <= src.count):
            raise ValueError(f"Invalid band index: {band}. Raster has {src.count} band(s).")
        data = src.read(band, masked=True)
        imarray = np.asarray(data, dtype=np.float64)
        if np.ma.isMaskedArray(data):
            imarray = np.where(np.ma.getmaskarray(data), np.nan, np.ma.filled(data, np.nan))
        imarray = -imarray
        height, width = imarray.shape
        grad_y, grad_x = np.gradient(imarray, height, width)
        aspect_rad = np.arctan2(-grad_y, grad_x)
        aspect_deg = np.degrees(aspect_rad) % 360.0

        profile = src.profile.copy()
        profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})
        aspect_path = _get_out_path(output_name)
        with rasterio.open(aspect_path, "w", **profile) as dst:
            dst.write(aspect_deg.astype(np.float32), 1)

    return {
        "output_path": aspect_path,
    }

# @mcp.tool()
def compute_ruggedness(
        input_path: str,
        output_name: str,
        window_size: int = 3
) -> Dict[str, Any]:
    """
    Compute ruggedness (local range) from a single-band raster and write a GeoTIFF.

    Args:
        input_path: Path to the input raster (GeoTIFF).
        output_name: Output filename (must end with .tif).
        window_size: Odd kernel size for the local range filter.

    Returns:
        output_path: Path to the output ruggedness GeoTIFF.
    """
    if not os.path.isfile(input_path):
        return {"status": "error", "message": f"File not found: {input_path}"}

    with rasterio.open(input_path) as src:
        elevation = src.read(1)
        profile = dict(src.profile)

    elevation_np = np.array(elevation, dtype=np.float64)
    def range_filter(values):
        valid_values = values[~np.isnan(values)]
        if valid_values.size == 0:
            return np.nan
        return float(valid_values.max() - valid_values.min())
    ruggedness = generic_filter(elevation_np, range_filter, size=(window_size, window_size))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tif_path = _get_out_path(output_name)

    tif_profile = profile.copy()
    tif_profile.update({"dtype": "float32", "count": 1})
    with rasterio.open(tif_path, "w", **tif_profile) as dst:
        dst.write(ruggedness.astype(np.float32), 1)

    return {"output_path": tif_path}

# @mcp.tool()
def compute_flood_depth(elevation_raster_path: str, output_name: str, threshold: float) -> Dict[str, Any]:
    """
    Read elevation raster, compute flood depth based on threshold, and save as single-band raster.

    Args:
        elevation_raster_path: Path to elevation raster.
        output_name: Output filename (must end with .tif).
        threshold: Elevation threshold for flooding.

    Returns:
        output_path: Path to the output flood depth raster file.
    """
    with rasterio.open(elevation_raster_path) as src:
        elevation = src.read(1)
        profile = src.profile

    flood_depth = np.where(elevation <= threshold, (elevation - threshold) * -1, 0).astype(np.float32)

    out_tif = _get_out_path(output_name)
    new_profile = profile.copy()
    new_profile.update({"dtype": "float32", "count": 1, "nodata": 0})
    with rasterio.open(out_tif, "w", **new_profile) as dst:
        dst.write(flood_depth, 1)

    return {"output_path": out_tif}

# @mcp.tool()
def weighted_sum_rasters(
    raster_paths: List[str],
    output_name: str,
    bands: Optional[List[int]] = None,
    weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute weighted sum across multiple rasters. Assumes all inputs are aligned (same size, transform, CRS).

    Args:
        raster_paths: List of raster paths to include in the sum.
        output_name: Output filename (must end with .tif).
        bands: Optional list of band indices (default: [1, 1, ...]).
        weights: Optional list of weights (default: [1.0, 1.0, ...]).
    Returns:
        output_path: Path to the weighted sum raster.
    """
    if not raster_paths:
        raise ValueError("raster_paths must not be empty")
    n = len(raster_paths)
    bands = bands or [1] * n
    weights = weights or [1.0] * n
    if not (len(bands) == n and len(weights) == n):
        raise ValueError("bands and weights must match raster_paths length")

    base_profile = None
    def _is_int_weight(w):
        try:
            return float(w).is_integer()
        except Exception:
            return False
    all_int_weights = all(_is_int_weight(w) for w in weights)
    acc_dtype = np.int64 if all_int_weights else np.float64
    acc = None
    valid_mask = None

    for idx, (rp, b, w) in enumerate(zip(raster_paths, bands, weights)):
        with rasterio.open(os.path.abspath(rp)) as src:
            if not (1 <= int(b) <= src.count):
                raise ValueError(f"Invalid band index {b}; raster has {src.count} band(s) for {rp}")
            arr = src.read(int(b))
            nd = src.nodata if getattr(src, "nodata", None) is not None else src.profile.get("nodata")
            if nd is not None:
                inv = np.isnan(arr) if isinstance(nd, float) and np.isnan(nd) else (arr == nd)
                arr = np.where(inv, 0, arr)
                vm = ~inv
            else:
                vm = np.ones_like(arr, dtype=bool)
            if base_profile is None:
                base_profile = src.profile.copy()
                acc = np.zeros_like(arr, dtype=acc_dtype)
                valid_mask = vm.copy()
            else:
                if (src.width != base_profile["width"] or src.height != base_profile["height"] or src.transform != base_profile["transform"] or src.crs != base_profile["crs"]):
                    raise ValueError("Input rasters are not aligned (size/transform/CRS mismatch)")
                valid_mask = valid_mask | vm
            if all_int_weights:
                acc += arr.astype(np.int64) * int(round(w))
            else:
                acc += arr.astype(np.float64) * float(w)

    dtype_out = rasterio.float32 if not all_int_weights else rasterio.int32
    out_arr = acc.astype(np.float32 if not all_int_weights else np.int32)
    if valid_mask is not None:
        out_arr = np.where(valid_mask, out_arr, 0)

    base_profile.update({"dtype": dtype_out, "count": 1, "nodata": 0})
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = _get_out_path(output_name)
    with rasterio.open(out_path, "w", **base_profile) as dst:
        dst.write(out_arr, 1)
    return {"output_path": out_path}

# @mcp.tool()
def reclassify_raster(
    raster_path: str,
    output_name: str,
    reclass_dict: Dict[int, int],
    band: int = 1,
) -> Dict[str, Any]:
    """
    Reclassify discrete pixel values using an explicit value->class mapping.

    This function targets categorical rasters where specific source values need to
    be remapped to new class codes. It applies an exact-value lookup (no ranges),
    preserves NoData (or sets it to 255 when absent), and writes a single-band
    uint8 GeoTIFF. Use this when your inputs are discrete labels rather than continuous measurements.

    Args:
        raster_path: Path to the input raster (TIFF).
        output_name: Output filename (must end with .tif).
        reclass_dict: Mapping from original pixel values to target classes.
        band: Band index to process (default: 1).

    Returns:
        output_path: Path to the reclassified TIFF written.
    """
    src_path = Path(raster_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"Raster not found: {src_path}")

    with rasterio.open(src_path) as src:
        if not (1 <= band <= src.count):
            raise ValueError(f"Invalid band index {band}; raster has {src.count} band(s).")
        data = src.read(band)
        profile = src.profile.copy()
        nodata_value = profile.get("nodata")

    reclassified = np.copy(data)
    for original_value, target_value in reclass_dict.items():
        reclassified[data == original_value] = target_value

    if nodata_value is None:
        nodata_value = 255
    else:
        reclassified[data == nodata_value] = nodata_value

    reclassified = reclassified.astype(np.uint8)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)

    profile.update(dtype=rasterio.uint8, count=1, nodata=nodata_value)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(reclassified, 1)

    return {"output_path": out_path}

# @mcp.tool()
def threshold_raster(
    raster_path: str,
    output_name: str,
    threshold: float = 100.0,
    band: int = 1,
    true_value: int = 1,
    false_value: int = 0,
) -> Dict[str, Any]:
    """
    Create a binary mask raster by applying a threshold to values.

    Values > threshold are set to true_value (default 1).
    Values <= threshold are set to false_value (default 0).

    Args:
        raster_path: Path to the input raster GeoTIFF.
        output_name: Output filename (must end with .tif).
        threshold: Threshold value.
        band: Band index to read (default: 1).
        true_value: Value for pixels above threshold (0-255).
        false_value: Value for pixels below or equal to threshold (0-255).

    Returns:
        output_path: Path to the output mask raster.
        pixel_count: Count of pixels satisfying the condition.
    """
    rp = os.path.abspath(raster_path)
    if not os.path.isfile(rp):
        raise FileNotFoundError(f"Raster file not found: {rp}")

    with rasterio.open(rp) as src:
        if not (1 <= int(band) <= src.count):
            raise ValueError(f"Invalid band index: {band}. Raster has {src.count} band(s).")
        arr = src.read(int(band)).astype(np.float64)
        nd = src.nodata if getattr(src, "nodata", None) is not None else src.profile.get("nodata")
        if nd is not None:
            mask = np.isnan(arr) if isinstance(nd, float) and np.isnan(nd) else (arr == nd)
        else:
            mask = np.isnan(arr)

        # General "high" condition means "above threshold"
        above_thresh = (arr > float(threshold)) & (~mask)
        out = np.where(mask, false_value, np.where(above_thresh, true_value, false_value)).astype("uint8")

        profile = src.profile.copy()
        profile.update({"dtype": "uint8", "count": 1, "nodata": false_value, "compress": "deflate"})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
    return {"output_path": out_path, "pixel_count": int(np.count_nonzero(above_thresh))}

# @mcp.tool()
def calculate_raster_difference(
    raster_a_path: str,
    raster_b_path: str,
    output_name: str
) -> Dict[str, Any]:
    '''
    Compute the difference between two rasters (A - B) and export.

    Args:
        raster_a_path: Path to raster A.
        raster_b_path: Path to raster B.
        output_name: Output filename (must end with .tif).

    Returns:
        output_path: Path to the output difference raster file.
    '''
    if not os.path.isfile(raster_a_path) or not os.path.isfile(raster_b_path):
        return {'status': 'error', 'message': 'missing inputs'}
    with rasterio.open(raster_a_path) as src_a, rasterio.open(raster_b_path) as src_b:
        a = src_a.read(1).astype(np.float32)
        b = src_b.read(1).astype(np.float32)
        if a.shape != b.shape:
            return {'status': 'error', 'message': f'shape mismatch {a.shape} vs {b.shape}'}
        diff = a - b
        profile = src_a.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(diff, 1)
    valid = diff[np.isfinite(diff)]
    return {
        'status': 'success',
        'output_path': output_path,
        'min_value': float(np.min(valid)) if valid.size else None,
        'max_value': float(np.max(valid)) if valid.size else None,
        'mean_value': float(np.mean(valid)) if valid.size else None,
        'pixel_count': int(valid.size),
    }

# @mcp.tool()
def compute_nbr(
    raster_path: str,
    output_name: str,
    nir_band: int,
    swir_band: int,
) -> Dict[str, Any]:
    """
    Compute the Normalized Burn Ratio (NBR) for a single raster and write the
    result as a GeoTIFF.

    Args:
        raster_path: Path to the input raster (GeoTIFF).
        output_name: Output filename (must end with .tif).
        nir_band: 1-based band index of the Near-Infrared band.
        swir_band: 1-based band index of the Shortwave-Infrared band.

    Returns:
        output_path: Absolute path to the written NBR GeoTIFF.
    """
    if not os.path.isfile(raster_path):
        return {"status": "error", "message": "Raster file not found"}

    with rasterio.open(raster_path) as src:
        nir = src.read(nir_band)
        swir = src.read(swir_band)
        denom = nir.astype(float) + swir.astype(float)
        num = nir.astype(float) - swir.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            nbr = num / denom
            nbr[denom == 0] = np.nan
        profile = src.profile.copy()
        profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(nbr.astype(np.float32), 1)

    return {"output_path": out_path}

# @mcp.tool()
def calculate_savi(
    input_raster_path: str,
    output_name: str,
    nir_band: int,
    red_band: int,
) -> Dict[str, Any]:
    '''
    Compute Soil Adjusted Vegetation Index (SAVI).

    Args:
        input_raster_path: Path to input raster.
        output_name: Output filename (must end with .tif).
        nir_band: Near-Infrared band index (1-based).
        red_band: Red band index (1-based).
    '''
    if not os.path.isfile(input_raster_path):
        return {'status': 'error', 'message': f'missing {input_raster_path}'}
    with rasterio.open(input_raster_path) as src:
        nir = src.read(nir_band).astype(np.float32)
        red = src.read(red_band).astype(np.float32)
        with np.errstate(invalid='ignore', divide='ignore'):
            savi = ((nir - red) / (nir + red + 0.5)) * 1.5
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(savi, 1)
    valid = savi[np.isfinite(savi)]
    return {
        'status': 'success',
        'output_path': output_path,
        'min_value': float(np.min(valid)) if valid.size else None,
        'max_value': float(np.max(valid)) if valid.size else None,
        'mean_value': float(np.mean(valid)) if valid.size else None,
        'pixel_count': int(valid.size),
    }

# @mcp.tool()
def convert_mask_to_polygons(
    mask_path: str,
    output_name: str,
    mask_value: int = 1,
) -> Dict[str, Any]:
    """
    Convert a raster mask to vector polygons (GeoJSON).

    Extracts polygons for pixels matching the specified mask_value.

    Args:
        mask_path: Path to the binary mask raster (GeoTIFF).
        output_name: Output filename (must end with .geojson).
        mask_value: Pixel value to vectorize (default: 1).

    Returns:
        output_path: Path to the output GeoJSON file.
        feature_count: Number of polygons created.
    """
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask raster not found: {mask_path}")

    with rasterio.open(mask_path) as src:
        image = src.read(1)
        # Create a boolean mask for the shapes function
        # shapes(image, mask=mask) -> mask here means "valid data region"
        # We want to extract shapes where image == mask_value
        bool_mask = (image == mask_value)
        transform = src.transform
        crs = src.crs
    
    # Check if there are any valid pixels
    if not np.any(bool_mask):
        return {"status": "warning", "message": f"No pixels found with value {mask_value}", "output_path": None, "feature_count": 0}

    # Extract shapes
    # Note: shapes() yields (geometry, value) pairs
    # We pass the boolean mask to `mask` argument to only process True areas
    geom_iter = shapes(image, mask=bool_mask, transform=transform)
    
    geoms = []
    for geom, val in geom_iter:
        # Double check value matches (though mask ensures it mostly)
        if val == mask_value:
            geoms.append(shape(geom))

    if not geoms:
         return {"status": "warning", "message": "No valid geometries created", "output_path": None, "feature_count": 0}

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    gdf.to_file(out_path, driver="GeoJSON")
    return {"output_path": out_path, "feature_count": len(gdf)}


# @mcp.tool()
def extract_stream_network(
    flowacc_path: str,
    output_name: str,
    threshold: float
) -> Dict[str, Any]:
    """
    Extract stream network based on flow accumulation threshold.

    Definition:
        Cells with flow accumulation > threshold are marked as stream (1), others NoData.

    Args:
        flowacc_path: Path to the flow accumulation raster.
        output_name: Output filename (must end with .tif).
        threshold: Accumulation threshold to define streams.

    Returns:
        output_path: Path to the stream network raster.
    """
    if not os.path.isfile(flowacc_path):
        raise FileNotFoundError(f"Flow accumulation file not found: {flowacc_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with rasterio.open(flowacc_path) as src:
        profile = src.profile.copy()
        acc = src.read(1)
        nodata = src.nodata

    # Handle NoData
    if nodata is not None:
        mask_valid = acc != nodata
    else:
        mask_valid = ~np.isnan(acc)

    # Extract streams
    stream = np.where((mask_valid) & (acc > threshold), 1, 0).astype(np.uint8)
    
    # Update Profile, set NoData to 0 (background)
    profile.update(dtype="uint8", nodata=0)
    
    output_path = _get_out_path(output_name)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stream, 1)

    return {"output_path": output_path}

# @mcp.tool()
def compute_watershed_basins(
    flow_direction_path: str,
    output_name: str
) -> Dict[str, Any]:
    """
    Compute watershed basins based on flow direction raster.
    
    Principle:
        1. Identify Outlets: Cells flowing out of the boundary or into NoData.
        2. Trace Upstream: BFS from outlets against flow direction.
        3. Label Basins: Assign unique ID to each connected component.
    
    Args:
        flow_direction_path: Path to D8 flow direction raster.
        output_name: Output filename (must end with .tif).
        
    Returns:
        Dict with output_path and basin_count.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)

    if not os.path.isfile(flow_direction_path):
        raise FileNotFoundError(f"Flow direction file not found: {flow_direction_path}")

    # 1. Read Flow Direction
    with rasterio.open(flow_direction_path) as src:
        fdir = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else -1
        h, w = fdir.shape

    # D8 Encoding (East=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128)
    # (dy, dx)
    d8_map = {
        1: (0, 1),    # E
        2: (1, 1),    # SE
        4: (1, 0),    # S
        8: (1, -1),   # SW
        16: (0, -1),  # W
        32: (-1, -1), # NW
        64: (-1, 0),  # N
        128: (-1, 1)  # NE
    }

    # 2. Identify Outlets & Initialize
    basin = np.zeros((h, w), dtype=np.int32)
    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    
    basin_id_counter = 0

    # Scan for outlets (boundary or flow to NoData)
    for i in range(h):
        for j in range(w):
            code = fdir[i, j]
            
            if code == nodata:
                continue
                
            dy, dx = d8_map.get(code, (0, 0))
            if dy == 0 and dx == 0:
                is_outlet = True # Sink or invalid
            else:
                ni, nj = i + dy, j + dx
                # Check boundary
                if ni < 0 or ni >= h or nj < 0 or nj >= w:
                    is_outlet = True
                else:
                    # Check flow to NoData
                    if fdir[ni, nj] == nodata:
                        is_outlet = True
                    else:
                        is_outlet = False
            
            if is_outlet:
                basin_id_counter += 1
                basin[i, j] = basin_id_counter
                visited[i, j] = True
                q.append((i, j))

    # 3. Upstream BFS
    # Neighbor offsets for 8-connectivity
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    while q:
        curr_i, curr_j = q.popleft()
        curr_id = basin[curr_i, curr_j]
        
        for dy, dx in neighbor_offsets:
            ni, nj = curr_i + dy, curr_j + dx
            
            if 0 <= ni < h and 0 <= nj < w:
                if not visited[ni, nj]:
                    # Check if neighbor flows INTO current cell
                    n_code = fdir[ni, nj]
                    if n_code == nodata:
                        continue
                        
                    n_dy, n_dx = d8_map.get(n_code, (0, 0))
                    
                    # If (ni, nj) + flow == (curr_i, curr_j)
                    if (ni + n_dy == curr_i) and (nj + n_dx == curr_j):
                        basin[ni, nj] = curr_id
                        visited[ni, nj] = True
                        q.append((ni, nj))

    # 4. Save Result
    profile.update(dtype="int32", nodata=0, count=1)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(basin, 1)
        
    return {"output_path": output_path, "basin_count": basin_id_counter}


# @mcp.tool()
def vectorize_stream_network(
    stream_raster_path: str,
    flow_direction_path: str,
    output_name: str
) -> Dict[str, Any]:
    """
    Vectorize stream raster to polyline features (Stream to Feature).

    Definition:
        Uses D8 flow direction to trace connections between stream cells,
        connecting cell centers to generate vector lines, then merging them into features.

    Args:
        stream_raster_path: Path to stream raster (1 for stream, 0/NoData for background).
        flow_direction_path: Path to D8 flow direction raster.
        output_name: Output filename (must end with .shp or .geojson).

    Returns:
        output_path: Path to the output vector file.
    """
    if not os.path.isfile(stream_raster_path):
        raise FileNotFoundError(f"Stream raster not found: {stream_raster_path}")
    if not os.path.isfile(flow_direction_path):
        raise FileNotFoundError(f"Flow direction raster not found: {flow_direction_path}")

    # Read data
    with rasterio.open(stream_raster_path) as src_s:
        stream = src_s.read(1)
        transform = src_s.transform
        crs = src_s.crs

    with rasterio.open(flow_direction_path) as src_d:
        fdir = src_d.read(1)

    h, w = stream.shape
    lines = []

    # D8 Flow Direction Mapping (ArcGIS/Hydro style)
    # E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128
    dy_map = {1: 0, 2: 1, 4: 1, 8: 1, 16: 0, 32: -1, 64: -1, 128: -1}
    dx_map = {1: 1, 2: 1, 4: 0, 8: -1, 16: -1, 32: -1, 64: 0, 128: 1}

    # Trace streams
    for i in range(h):
        for j in range(w):
            if stream[i, j] != 1:
                continue

            code = fdir[i, j]
            if code == 0:
                continue

            ni = i + dy_map.get(code, 0)
            nj = j + dx_map.get(code, 0)

            if 0 <= ni < h and 0 <= nj < w:
                if stream[ni, nj] == 1:
                    # Connect centers
                    x1, y1 = transform * (j + 0.5, i + 0.5)
                    x2, y2 = transform * (nj + 0.5, ni + 0.5)
                    lines.append(LineString([(x1, y1), (x2, y2)]))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)

    if not lines:
        return {"output_path": output_path, "status": "no streams found"}

    # Merge lines
    merged = linemerge(lines)
    
    if merged.is_empty:
        geoms = []
    elif merged.geom_type == 'LineString':
        geoms = [merged]
    elif merged.geom_type == 'MultiLineString':
        geoms = list(merged.geoms)
    else:
        geoms = list(merged)

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    gdf.to_file(output_path)

    return {"output_path": output_path}


# @mcp.tool()
def compute_stream_link(
    stream_raster_path: str,
    flow_direction_path: str,
    output_name: str
) -> Dict[str, Any]:
    """
    Generate Stream Link raster.

    Definition:
        Assigns a unique ID to each stream segment (reach) between junctions or from source to junction.

    Args:
        stream_raster_path: Path to stream raster.
        flow_direction_path: Path to D8 flow direction raster.
        output_name: Output filename (must end with .tif).

    Returns:
        output_path: Path to the output Stream Link raster.
    """
    if not os.path.isfile(stream_raster_path) or not os.path.isfile(flow_direction_path):
        raise FileNotFoundError("Input files not found")

    with rasterio.open(stream_raster_path) as src_s:
        stream = src_s.read(1)
        profile = src_s.profile.copy()
        
    with rasterio.open(flow_direction_path) as src_d:
        fdir = src_d.read(1)

    h, w = stream.shape
    
    # 1. Calculate In-degree for stream cells
    indegree = np.zeros((h, w), dtype=np.int32)
    dy_map = {1: 0, 2: 1, 4: 1, 8: 1, 16: 0, 32: -1, 64: -1, 128: -1}
    dx_map = {1: 1, 2: 1, 4: 0, 8: -1, 16: -1, 32: -1, 64: 0, 128: 1}

    for i in range(h):
        for j in range(w):
            if stream[i, j] != 1: continue
            code = fdir[i, j]
            if code == 0: continue
            ni, nj = i + dy_map.get(code, 0), j + dx_map.get(code, 0)
            if 0 <= ni < h and 0 <= nj < w and stream[ni, nj] == 1:
                indegree[ni, nj] += 1

    # 2. Identify Segments
    link_raster = np.zeros((h, w), dtype=np.int32)
    current_link_id = 1
    
    active_indegree = indegree.copy()
    q = deque()
    
    # Initialize with sources (indegree == 0)
    for i in range(h):
        for j in range(w):
            if stream[i, j] == 1 and indegree[i, j] == 0:
                q.append((i, j))
                link_raster[i, j] = current_link_id
                current_link_id += 1

    while q:
        i, j = q.popleft()
        curr_id = link_raster[i, j]
        
        code = fdir[i, j]
        if code == 0: continue
        ni, nj = i + dy_map.get(code, 0), j + dx_map.get(code, 0)
        
        if 0 <= ni < h and 0 <= nj < w and stream[ni, nj] == 1:
            active_indegree[ni, nj] -= 1
            
            if indegree[ni, nj] > 1:
                # Junction: Ends current link. Starts new link when all upstreams arrive.
                if active_indegree[ni, nj] == 0:
                    link_raster[ni, nj] = current_link_id
                    current_link_id += 1
                    q.append((ni, nj))
            else:
                # Normal flow: continue current link
                link_raster[ni, nj] = curr_id
                q.append((ni, nj))

    profile.update(dtype="int32", nodata=0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(link_raster, 1)

    return {"output_path": output_path}


# @mcp.tool()
def compute_stream_order(
    stream_raster_path: str,
    flow_direction_path: str,
    output_name: str,
    method: str = "strahler"
) -> Dict[str, Any]:
    """
    Compute Stream Order (Strahler or Shreve).

    Args:
        stream_raster_path: Path to stream raster.
        flow_direction_path: Path to D8 flow direction raster.
        output_name: Output filename (must end with .tif).
        method: "strahler" or "shreve".

    Returns:
        output_path: Path to the output Stream Order raster.
    """
    if not os.path.isfile(stream_raster_path) or not os.path.isfile(flow_direction_path):
        raise FileNotFoundError("Input files not found")

    with rasterio.open(stream_raster_path) as src_s:
        stream = src_s.read(1)
        profile = src_s.profile.copy()

    with rasterio.open(flow_direction_path) as src_d:
        fdir = src_d.read(1)

    h, w = stream.shape
    
    # 1. Calculate In-degree
    indegree = np.zeros((h, w), dtype=np.int32)
    dy_map = {1: 0, 2: 1, 4: 1, 8: 1, 16: 0, 32: -1, 64: -1, 128: -1}
    dx_map = {1: 1, 2: 1, 4: 0, 8: -1, 16: -1, 32: -1, 64: 0, 128: 1}

    for i in range(h):
        for j in range(w):
            if stream[i, j] > 0:
                code = fdir[i, j]
                if code != 0:
                    ni, nj = i + dy_map.get(code, 0), j + dx_map.get(code, 0)
                    if 0 <= ni < h and 0 <= nj < w and stream[ni, nj] > 0:
                        indegree[ni, nj] += 1
    
    # 2. Propagate Orders
    upstream_orders = {} 
    q = deque()
    for i in range(h):
        for j in range(w):
            if stream[i, j] > 0 and indegree[i, j] == 0:
                q.append((i, j))
    
    order_raster = np.full((h, w), np.nan, dtype=np.float32)
    
    while q:
        i, j = q.popleft()
        orders = upstream_orders.get((i, j), [])
        
        if not orders:
            current_order = 1
        else:
            if method.lower() == "strahler":
                max_ord = max(orders)
                if orders.count(max_ord) > 1:
                    current_order = max_ord + 1
                else:
                    current_order = max_ord
            elif method.lower() == "shreve":
                current_order = sum(orders)
            else:
                current_order = 1
        
        order_raster[i, j] = float(current_order)
        
        code = fdir[i, j]
        if code == 0: continue
        ni, nj = i + dy_map.get(code, 0), j + dx_map.get(code, 0)
        
        if 0 <= ni < h and 0 <= nj < w and stream[ni, nj] > 0:
            if (ni, nj) not in upstream_orders:
                upstream_orders[(ni, nj)] = []
            upstream_orders[(ni, nj)].append(current_order)
            
            indegree[ni, nj] -= 1
            if indegree[ni, nj] == 0:
                q.append((ni, nj))

    profile.update(dtype="float32", nodata=np.nan)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(order_raster, 1)

    return {"output_path": output_path}

# @mcp.tool()
def apply_mask_to_raster(
    input_raster_path: str,
    mask_raster_path: str,
    output_name: str,
    mask_value: int = 1,
) -> Dict[str, Any]:
    """
    Apply a binary mask to a raster.

    Keeps pixels from the input raster where the mask equals mask_value.
    Other pixels are set to NoData (NaN).

    Args:
        input_raster_path: Path to the source raster (values to keep).
        mask_raster_path: Path to the mask raster.
        output_name: Output filename (must end with .tif).
        mask_value: Value in the mask raster that indicates "keep" (default: 1).

    Returns:
        output_path: Path to the masked raster.
    """
    if not os.path.isfile(input_raster_path) or not os.path.isfile(mask_raster_path):
        raise FileNotFoundError("Input or mask raster not found.")

    with rasterio.open(input_raster_path) as src_val, rasterio.open(mask_raster_path) as src_mask:
        # Basic alignment check
        if src_val.shape != src_mask.shape:
            raise ValueError(f"Shape mismatch: Input {src_val.shape} vs Mask {src_mask.shape}")
        
        val_arr = src_val.read(1)
        mask_arr = src_mask.read(1)
        
        # Prepare output array
        # Promote integer types to float if needed to support NaN for NoData
        if np.issubdtype(val_arr.dtype, np.integer):
             out_arr = val_arr.astype(np.float32)
             dtype = "float32"
             nodata = np.nan
        else:
             out_arr = val_arr.copy()
             dtype = val_arr.dtype
             nodata = np.nan

        # Apply mask: Set pixels where mask != mask_value to NoData
        out_arr[mask_arr != mask_value] = nodata
        
        profile = src_val.profile.copy()
        profile.update({"dtype": dtype, "count": 1, "nodata": nodata})
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = _get_out_path(output_name)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_arr, 1)

    return {"output_path": out_path}

# # @mcp.tool()
# def identify_burn_scar_from_nbr(
#     nbr_a_path: str,
#     nbr_b_path: str,
#     output_name: str,
#     threshold: float = 0.2,
# ) -> Dict[str, Any]:
#     """
#     Derive burn-scar regions from two NBR rasters by computing the delta
#     (A - B), thresholding, and exporting both a mask raster and polygon
#     features.

#     Args:
#         nbr_a_path: Path to NBR raster for dataset A.
#         nbr_b_path: Path to NBR raster for dataset B.
#         output_name: Output filename (must end with .tif).
#         threshold: Delta threshold above which pixels are considered burn scar.

#     Returns:
#         polygons_geojson_path: Absolute path to the burn-scar polygons GeoJSON.
#         mask_raster_path: Absolute path to the masked delta-NBR GeoTIFF.
#     """
#     import os
#     import numpy as np
#     import rasterio
#     from rasterio.features import shapes
#     import geopandas as gpd
#     from shapely.geometry import shape

#     if not (os.path.isfile(nbr_a_path) and os.path.isfile(nbr_b_path)):
#         return {"status": "error", "message": "NBR raster files not found"}

#     with rasterio.open(nbr_a_path) as src_a, rasterio.open(nbr_b_path) as src_b:
#         a = src_a.read(1)
#         b = src_b.read(1)
#         if a.shape != b.shape:
#             return {"status": "error", "message": "Raster dimensions do not match"}
#         delta = a - b
#         mask = delta > float(threshold)
#         os.makedirs(OUTPUT_DIR, exist_ok=True)
        
#         disp_profile = src_a.profile.copy()
#         disp_profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})
#         display = delta.astype(np.float32)
#         display[~mask] = np.nan
#         display_path = os.path.join(OUTPUT_DIR, output_name)
#         with rasterio.open(display_path, "w", **disp_profile) as dst:
#             dst.write(display, 1)

#         geom_iter = shapes(mask.astype(np.uint8), mask=mask, transform=src_a.transform)
#         geoms = [shape(geom) for geom, val in geom_iter if val == 1]
#         if not geoms:
#             return {"status": "error", "message": "No burn scar regions identified"}
#         gdf = gpd.GeoDataFrame(geometry=geoms, crs=src_a.crs)
#         base_out = os.path.splitext(output_name)[0]
#         poly_path = os.path.join(OUTPUT_DIR, f"{base_out}.geojson")
#         gdf.to_file(poly_path, driver="GeoJSON")

#     return {"polygons_geojson_path": poly_path, "mask_raster_path": display_path}

# @mcp.tool()
def reclassify_raster_values(
    raster_path: str,
    output_name: str,
    ranges: list,
    band: int = 1,
) -> Dict[str, Any]:
    """
    Bin continuous raster values into classes using inclusive ranges and save as int16 GeoTIFF.

    This function targets continuous rasters (e.g., elevation, distance, indices)
    where value intervals define classes. Each range is applied as an inclusive
    mask (min ≤ x ≤ max). NoData is preserved (defaults to 0), and the output is
    a single‑band int16 GeoTIFF. Use this when your inputs are measurements that
    need threshold‑based categorization rather than exact value remapping.

    Args:
        raster_path: Path to the input raster (TIFF).
        output_name: Output filename (must end with .tif).
        ranges: List of ranges, each item is [min, max, class_value].
        band: Band index to process (default: 1).

    Returns:
        output_path: Path to the written reclassified raster.
    """
    src_path = os.path.abspath(raster_path)
    with rasterio.open(src_path) as src:
        if not (1 <= band <= src.count):
            raise ValueError(f"Invalid band index {band}; raster has {src.count} band(s).")
        arr = src.read(band)
        profile = src.profile.copy()
        nodata_value = profile.get("nodata")

    out = np.zeros_like(arr, dtype=np.int16)
    for r in ranges:
        if not (isinstance(r, (list, tuple)) and len(r) == 3):
            continue
        rmin, rmax, cls = r
        mask = (arr >= float(rmin)) & (arr <= float(rmax))
        out[mask] = int(cls)

    if nodata_value is None:
        nodata_value = 0
    else:
        out[arr == nodata_value] = nodata_value

    profile.update({"dtype": rasterio.int16, "count": 1, "nodata": nodata_value})
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = _get_out_path(output_name)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out.astype(np.int16), 1)
    return {"output_path": out_path}

# @mcp.tool()
def overlay_weighted_rasters(
    raster_paths: List[str],
    output_name: str,
    weights: Optional[List[float]] = None,
    bands: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute a weighted overlay over aligned rasters.

    Args:
        raster_paths: List of input raster file paths.
        output_name: Output filename (must end with .tif).
        weights: Per-raster weights. Defaults to all 1.0.
        bands: Band indices (1-based) for each raster. Defaults to all 1.

    Returns:
        output_path: Path to the generated raster.
    """
    if not raster_paths:
        raise ValueError("raster_paths must not be empty")
    n = len(raster_paths)
    weights = weights or [1.0] * n
    bands = bands or [1] * n
    if len(weights) != n or len(bands) != n:
        raise ValueError("Length of weights and bands must match raster_paths.")

    with rasterio.open(os.path.abspath(raster_paths[0])) as src0:
        ref_profile = src0.profile.copy()
        ref_shape = (src0.height, src0.width)
        ref_crs = src0.crs
        ref_transform = src0.transform

    acc = np.zeros(ref_shape, dtype=np.float64)
    weight_accum = np.zeros(ref_shape, dtype=np.float64)

    for i, (path, band, w) in enumerate(zip(raster_paths, bands, weights)):
        with rasterio.open(os.path.abspath(path)) as src:
            if (src.width != ref_profile["width"] or
                src.height != ref_profile["height"] or
                src.transform != ref_transform or
                src.crs != ref_crs):
                raise ValueError(f"Raster {path} is not aligned with the first raster.")

            arr = src.read(band).astype(np.float64)
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
            valid_mask = ~np.isnan(arr)
            acc[valid_mask] += w * arr[valid_mask]
            weight_accum[valid_mask] += w

    denom = weight_accum
    result = np.where(denom > 0, acc / denom, np.nan)

    out_nodata = np.nan
    result = result.astype(np.float32)

    ref_profile.update(
        dtype="float32",
        count=1,
        nodata=out_nodata,
        compress="deflate"
    )

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = _get_out_path(output_name)

    with rasterio.open(out_path, "w", **ref_profile) as dst:
        dst.write(result, 1)
    return {"output_path": out_path}

# @mcp.tool()
def normalize_raster_values(
    input_raster_path: str,
    output_name: str,
    lower: float = 1.0,
    upper: float = 10.0
) -> Dict[str, Any]:
    """
    Normalize a raster band to a target numeric range and write GeoTIFF.

    Args:
        input_raster_path: Path to the input raster TIFF.
        output_name: Output filename (must end with .tif).
        lower: Lower bound of the target range (inclusive). Default: 1.0
        upper: Upper bound of the target range (inclusive). Default: 10.0

    Returns:
        output_path: Path to the normalized raster.
    """
    if float(upper) <= float(lower):
        raise ValueError("upper must be greater than lower")

    with rasterio.open(os.path.abspath(input_raster_path)) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = profile.get("nodata")

    if nodata is not None:
        arr = np.where(arr == float(nodata), np.nan, arr)

    amin = float(np.nanmin(arr))
    amax = float(np.nanmax(arr))
    if amax == amin:
        out = np.full_like(arr, float(lower), dtype=np.float32)
    else:
        scale = (float(upper) - float(lower)) / (amax - amin)
        out = float(lower) + (arr - amin) * scale

    out = np.nan_to_num(out, nan=0.0)

    profile.update({"dtype": "float32", "count": 1, "nodata": 0.0})
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_raster_path = _get_out_path(output_name)
    with rasterio.open(output_raster_path, "w", **profile) as dst:
        dst.write(out.astype(np.float32), 1)
    return {"output_path": output_raster_path}

# @mcp.tool()
def invert_dem_values(raster_path: str, output_name: str, band: int = 1) -> Dict[str, Any]:
    """
    Invert DEM values (make ocean depths positive) and write a single-band GeoTIFF.

    Args:
        raster_path: Path to the input DEM raster (.tif).
        output_name: Output filename (must end with .tif).
        band: Band index to invert (default: 1).

    Returns:
        output_path: Path to the output inverted DEM raster.
    """
    rp = os.path.abspath(raster_path)
    if not os.path.isfile(rp):
        raise FileNotFoundError(f"Raster file not found: {rp}")

    with rasterio.open(rp) as src:
        if not (1 <= int(band) <= src.count):
            raise ValueError(f"Invalid band index: {band}. Raster has {src.count} band(s).")
        arr = src.read(int(band)).astype(np.float64)
        nd = src.nodata if getattr(src, "nodata", None) is not None else src.profile.get("nodata")
        if nd is not None:
            inv = np.isnan(arr) if isinstance(nd, float) and np.isnan(nd) else (arr == nd)
            arr = np.where(inv, np.nan, arr)

        depth = -arr
        profile = src.profile.copy()
        profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(depth.astype(np.float32), 1)

    return {"output_path": out_path}

# @mcp.tool()
def extract_raster_classes_by_mask(
    lc_path: str,
    high_risk_mask_path: str,
    output_name: str,
    undeveloped_codes: Optional[List[int]] = None,
    lc_band: int = 1,
    mask_band: int = 1,
    mask_true_value: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Identify undeveloped areas within high-risk zones and write an output raster.

    Args:
        lc_path: Path to the land cover raster GeoTIFF.
        high_risk_mask_path: Path to the high-risk mask raster GeoTIFF.
        output_name: Output filename (must end with .tif).
        undeveloped_codes: Land cover codes considered undeveloped.
        lc_band: Land cover raster band index.
        mask_band: Mask raster band index.
        mask_true_value: Value indicating high risk; if None, any nonzero (and non-nodata) is high risk.

    Returns:
        output_path: Path to the output raster file.
    """
    if undeveloped_codes is None:
        undeveloped_codes = [31, 41, 43, 52, 71, 81, 90, 95]

    lp = os.path.abspath(lc_path)
    hp = os.path.abspath(high_risk_mask_path)
    if not os.path.isfile(lp):
        raise FileNotFoundError(f"Land cover raster not found: {lp}")
    if not os.path.isfile(hp):
        raise FileNotFoundError(f"High-risk mask raster not found: {hp}")

    with rasterio.open(lp) as lc_src, rasterio.open(hp) as hr_src:
        if (lc_src.width != hr_src.width) or (lc_src.height != hr_src.height) or (lc_src.transform != hr_src.transform):
            raise ValueError("Land cover and mask rasters must have identical grid (size/transform/CRS)")

        lc = lc_src.read(int(lc_band), masked=True)
        hr = hr_src.read(int(mask_band), masked=True)

        if mask_true_value is not None:
            high = (~hr.mask) & (hr.data == mask_true_value)
        else:
            high = (~hr.mask) & (hr.data > 0)

        u_codes = np.array(undeveloped_codes, dtype=lc.data.dtype)
        undeveloped = np.isin(lc.data, u_codes)
        keep = high & undeveloped & (~lc.mask)

        profile = lc_src.profile.copy()
        dtype = np.dtype(profile.get("dtype", lc.data.dtype))
        if lc_src.nodata is not None:
            out_nodata = lc_src.nodata
        else:
            out_nodata = (np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else np.nan)

        out_array = np.where(keep, lc.data, out_nodata).astype(dtype)
        profile.update(count=1, nodata=out_nodata, compress="deflate", dtype=str(dtype))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_array, 1)
    return {"output_path": out_path}

# @mcp.tool()
def interpolate_points_to_raster(
    points_path: str, 
    value_column: str, 
    mask_path: str, 
    output_name: str,
    cell_size: float, 
    target_crs: str = None
) -> Dict[str, Any]:
    """
    Interpolate a point attribute to a regular raster grid using linear
    interpolation with nearest-neighbor fill for gaps, constrained by a mask.

    Args:
        points_path: Path to the point dataset containing the attribute.
        value_column: Attribute column name to interpolate.
        mask_path: Polygon layer defining raster extent (mask).
        output_name: Output GeoTIFF file name (must end with .tif).
        cell_size: Cell size in units of the target CRS.
        target_crs: Optional target CRS; inputs are reprojected if needed.

    Returns:
        output_path: Path to the generated raster.
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    gdf_points = gpd.read_file(points_path)
    gdf_mask = gpd.read_file(mask_path)
    
    # Reproject if needed
    if target_crs:
        if gdf_points.crs != target_crs:
            gdf_points = gdf_points.to_crs(target_crs)
        if gdf_mask.crs != target_crs:
            gdf_mask = gdf_mask.to_crs(target_crs)
    
    # Determine bounds from mask
    minx, miny, maxx, maxy = gdf_mask.total_bounds
    
    # Grid Dimensions
    width = int((maxx - minx) / cell_size)
    height = int((maxy - miny) / cell_size)
    
    transform = from_origin(minx, maxy, cell_size, cell_size)
    
    # Generate Grid Coords
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
    grid_x = np.array(xs)
    grid_y = np.array(ys)
    
    # Source Data
    src_x = gdf_points.geometry.x.values
    src_y = gdf_points.geometry.y.values
    src_val = gdf_points[value_column].values
    
    # Interpolate
    xi = (grid_x.flatten(), grid_y.flatten())
    points_zip = list(zip(src_x, src_y))
    
    # Linear interpolation
    grid_z = griddata(points_zip, src_val, xi, method='linear')
    
    # Fill NaNs with nearest neighbor (extrapolation)
    if np.isnan(grid_z).any():
        grid_z_nearest = griddata(points_zip, src_val, xi, method='nearest')
        mask_nan = np.isnan(grid_z)
        grid_z[mask_nan] = grid_z_nearest[mask_nan]
        
    grid_z = grid_z.reshape(height, width)
    
    # Save
    output_path = _get_out_path(output_name)
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': rasterio.float32,
        'crs': gdf_points.crs,
        'transform': transform,
        'nodata': -9999
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(grid_z.astype(rasterio.float32), 1)
        
    return {
        "output_path": output_path
    }

# # @mcp.tool()
# def calculate_zonal_mean(
#     polygons_path: str,
#     raster_path: str,
#     output_name: str,
#     mean_field: str,
#     band: int = 1,
# ) -> Dict[str, Any]:
#     """
#     Calculate the mean value of a raster band within each vector polygon and save the result.

#     Args:
#         polygons_path (str): Path to the polygon vector file (.shp/.geojson).
#         raster_path (str): Path to the raster file (.tif).
#         output_name (str): Output filename (must end with .shp or .geojson).
#         mean_field (str): Field name for the calculated mean in the output.
#         band (int): Raster band index (1-based), default 1.

#     Returns:
#         output_path: Path to the output vector file.
#     """
#     import os
#     import geopandas as gpd
#     import rasterio
#     from rasterio.mask import mask
#     import numpy as np

#     pp = os.path.abspath(polygons_path)
#     rp = os.path.abspath(raster_path)
#     if not os.path.isfile(pp) or not os.path.isfile(rp):
#         return {"output_path": None}

#     gdf = gpd.read_file(pp)
#     with rasterio.open(rp) as src:
#         if gdf.crs and src.crs and gdf.crs != src.crs:
#             gdf = gdf.to_crs(src.crs)
#         nod = src.nodata

#         means = []
#         for geom in gdf.geometry:
#             data, _ = mask(src, [geom], crop=True, filled=True, indexes=band)
#             arr = data[0].astype(float)
#             if nod is not None:
#                 arr[arr == nod] = np.nan
#             val = float(np.nanmean(arr)) if np.isfinite(np.nanmean(arr)) else 0.0
#             means.append(val)

#     gdf[mean_field] = means
#     out_path = os.path.join(OUTPUT_DIR, output_name)
#     gdf.to_file(out_path)
#     return {"output_path": out_path}

# @mcp.tool()
def count_netcdf_spells(
        nc_path: str,
        var_name: str,
        output_name: str,
        threshold: float,
        spell_length: int
) -> Dict[str, Any]:
    """
    Count warm spells over time for a NetCDF variable and write a GeoTIFF.

    Args:
        nc_path: Path to the NetCDF file.
        var_name: Variable name to process.
        output_name: Output filename (must end with .tif).
        threshold: Value threshold; a spell occurs when values exceed this threshold.
        spell_length: Window length (in time steps) that must all exceed `threshold`.

    Returns:
        output_path: Path to the written single-band int32 GeoTIFF.
    """
    p = os.path.abspath(nc_path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"NetCDF file not found: {p}")
    with rasterio.open(f'NETCDF:"{p}":{var_name}') as src:
        stack = src.read(masked=True)
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
    if np.ma.isMaskedArray(stack):
        arr_rio = np.where(np.ma.getmaskarray(stack), np.nan, np.ma.filled(stack, np.nan)).astype(np.float64)
    else:
        arr_rio = stack.astype(np.float64)
    ds_nc = Dataset(p, mode="r")
    var_nc = ds_nc.variables[var_name]
    raw = var_nc[:]
    raw = np.ma.filled(raw, np.nan).astype(np.float64)
    names = list(var_nc.dimensions)
    lowers = [n.lower() for n in names]
    t_candidates = [i for i, n in enumerate(lowers) if ("time" in n) or ("forecast_period" in n) or ("valid_time" in n)]
    t_axis = t_candidates[0] if t_candidates else 0
    lat_candidates = [i for i, n in enumerate(lowers) if ("lat" in n) or ("latitude" in n) or (n == "y")]
    lon_candidates = [i for i, n in enumerate(lowers) if ("lon" in n) or ("longitude" in n) or (n == "x")]
    lat_axis = lat_candidates[0] if lat_candidates else (1 if raw.ndim > 1 else 0)
    lon_axis = lon_candidates[0] if lon_candidates else (2 if raw.ndim > 2 else (1 if raw.ndim > 1 else 0))
    arr_nc = np.moveaxis(raw, [t_axis, lat_axis, lon_axis], [0, 1, 2])
    if arr_nc.ndim > 3:
        extra_axes = tuple(range(3, arr_nc.ndim))
        slicer = (slice(None), slice(None), slice(None)) + tuple(0 for _ in extra_axes)
        arr_nc = arr_nc[slicer]
    sf = getattr(var_nc, "scale_factor", None)
    ao = getattr(var_nc, "add_offset", None)
    fill = getattr(var_nc, "_FillValue", None)
    miss = getattr(var_nc, "missing_value", None)
    vr = getattr(var_nc, "valid_range", None)
    arr = arr_nc
    if sf is not None:
        arr = arr * float(sf)
    if ao is not None:
        arr = arr + float(ao)
    if fill is not None:
        arr = np.where(arr == float(fill), np.nan, arr)
    if miss is not None:
        arr = np.where(arr == float(miss), np.nan, arr)
    try:
        if vr is not None and len(vr) == 2:
            vmin_v, vmax_v = float(vr[0]), float(vr[1])
            arr = np.where((arr < vmin_v) | (arr > vmax_v), np.nan, arr)
    except Exception:
        pass
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    n, h, w = arr.shape
    L = max(1, int(spell_length))
    L_eff = min(L, n)
    hits = arr > float(threshold)
    if L_eff == 1:
        out = np.sum(hits, axis=0).astype(np.int32)
    else:
        win = sliding_window_view(hits, window_shape=L_eff, axis=0)
        full = np.all(win, axis=3)
        out = np.sum(full, axis=0).astype(np.int32)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": rasterio.int32,
        "nodata": -9999,
        "transform": transform,
        "crs": crs,
    }
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = _get_out_path(output_name)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
    return {"output_path": out_path}

# @mcp.tool()
def zonal_statistics(
    raster_path: str,
    polygons_path: str,
    output_name: str,
    band: int = 1,
    stats: List[str] = ["mean", "count"]
) -> Dict[str, Any]:
    """
    Calculate zonal statistics for polygons over a raster.

    Args:
        raster_path: Path to the input raster (GeoTIFF).
        polygons_path: Path to the input polygons (GeoJSON, SHP, etc.).
        output_name: Output filename (e.g., 'result.geojson', 'result.shp').
                     The format is inferred from the file extension.
        band: Raster band to use (default 1).
        stats: List of statistics to calculate (e.g., 'mean', 'min', 'max', 'count').

    Returns:
        output_path: Path to the output vector file with statistics appended.
    """
    if not os.path.isfile(raster_path):
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    if not os.path.isfile(polygons_path):
        raise FileNotFoundError(f"Polygon file not found: {polygons_path}")

    gdf = gpd.read_file(polygons_path)
    
    with rasterio.open(raster_path) as src:
        # Reproject polygons to match raster CRS if needed
        if gdf.crs and src.crs and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        
        results = []
        nodata = src.nodata
        for geom in gdf.geometry:
            try:
                # Mask raster by polygon
                out_image, _ = mask(src, [geom], crop=True)
                data = out_image[band-1].flatten()
                
                # Identify valid pixels
                if nodata is not None:
                    if np.isnan(nodata):
                        valid_mask = ~np.isnan(data)
                    else:
                        valid_mask = data != nodata
                else:
                    valid_mask = np.ones_like(data, dtype=bool)
                
                # Also exclude any NaNs that might exist in data (e.g. float raster)
                valid_mask &= ~np.isnan(data)
                
                valid_data = data[valid_mask]
                
                # Calculate stats
                row_stats = {}
                if valid_data.size > 0:
                    if "mean" in stats:
                        row_stats["mean"] = float(np.mean(valid_data))
                    if "count" in stats:
                        row_stats["count"] = int(valid_data.size)
                    if "min" in stats:
                        row_stats["min"] = float(np.min(valid_data))
                    if "max" in stats:
                        row_stats["max"] = float(np.max(valid_data))
                else:
                    for s in stats:
                        row_stats[s] = None
                results.append(row_stats)
            except Exception:
                results.append({s: None for s in stats})

    # Append results
    stats_df = pd.DataFrame(results)
    # Rename columns to avoid collision and clarify
    stats_df.columns = [f"prediction_{c}" for c in stats_df.columns]
    
    gdf = pd.concat([gdf, stats_df], axis=1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)
    
    # Allow driver inference from file extension (e.g. .shp -> ESRI Shapefile)
    # Default to GeoJSON only if explicitly requested or ambiguous? 
    # Actually, to_file infers well. But let's be safe.
    if output_name.lower().endswith(('.json', '.geojson')):
        gdf.to_file(output_path, driver="GeoJSON")
    else:
        gdf.to_file(output_path)
    return {"output_path": output_path}


# @mcp.tool()
def compute_flow_length_upstream(flow_direction_path: str, output_name: str, weight_raster_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute upstream flow length.

    Definition:
        For each cell, calculate the longest weighted distance to its flow source (where in-degree is 0).

    Args:
        flow_direction_path: Path to the D8 flow direction raster file.
        output_name: Output filename (must end with .tif).
        weight_raster_path: Optional path to a weight raster file.

    Returns:
        output_path: Path to the upstream flow length raster file.
    """
    if not os.path.isfile(flow_direction_path):
        raise FileNotFoundError(f"Flow direction file not found: {flow_direction_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DX_MAP = {1: 1, 2: 1, 4: 0, 8: -1, 16: -1, 32: -1, 64: 0, 128: 1}
    DY_MAP = {1: 0, 2: 1, 4: 1, 8: 1, 16: 0, 32: -1, 64: -1, 128: -1}
    DIST_MAP = {
        1: 1.0, 2: 1.41421356, 4: 1.0, 8: 1.41421356,
        16: 1.0, 32: 1.41421356, 64: 1.0, 128: 1.41421356
    }

    with rasterio.open(flow_direction_path) as src:
        profile = src.profile.copy()
        d8 = src.read(1).astype(np.uint16)
        cell_size = src.transform[0]
        h, w = src.height, src.width
        
    if weight_raster_path:
        if not os.path.isfile(weight_raster_path):
             raise FileNotFoundError(f"Weight raster file not found: {weight_raster_path}")
        with rasterio.open(weight_raster_path) as wsrc:
            weight = wsrc.read(1).astype(np.float64)
    else:
        weight = np.ones((h, w), dtype=np.float64)

    length_arr = np.zeros((h, w), dtype=np.float32)
    indeg = np.zeros((h, w), dtype=np.int32)

    for i in range(h):
        for j in range(w):
            code = d8[i, j]
            if code == 0:
                continue
            ni = i + DY_MAP.get(code, 0)
            nj = j + DX_MAP.get(code, 0)
            if 0 <= ni < h and 0 <= nj < w:
                indeg[ni, nj] += 1

    q = deque()
    for i in range(h):
        for j in range(w):
            if indeg[i, j] == 0:
                q.append((i, j))

    while q:
        i, j = q.popleft()
        code = d8[i, j]
        if code == 0:
            continue
        ni = i + DY_MAP.get(code, 0)
        nj = j + DX_MAP.get(code, 0)
        if 0 <= ni < h and 0 <= nj < w:
            step_dist = weight[ni, nj] * DIST_MAP.get(code, 1.0) * cell_size
            if length_arr[i, j] + step_dist > length_arr[ni, nj]:
                length_arr[ni, nj] = length_arr[i, j] + step_dist
            indeg[ni, nj] -= 1
            if indeg[ni, nj] == 0:
                q.append((ni, nj))

    profile.update(dtype="float32", nodata=0.0)
    output_path = _get_out_path(output_name)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(length_arr.astype(np.float32), 1)

    return {"output_path": output_path}


# @mcp.tool()
def compute_flow_length_downstream(flow_direction_path: str, output_name: str, weight_raster_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute downstream flow length.

    Definition:
        For each cell, calculate the weighted distance to the outlet (sink or boundary).

    Args:
        flow_direction_path: Path to the D8 flow direction raster file.
        output_name: Output filename (must end with .tif).
        weight_raster_path: Optional path to a weight raster file.

    Returns:
        output_path: Path to the downstream flow length raster file.
    """
    if not os.path.isfile(flow_direction_path):
        raise FileNotFoundError(f"Flow direction file not found: {flow_direction_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DX_MAP = {1: 1, 2: 1, 4: 0, 8: -1, 16: -1, 32: -1, 64: 0, 128: 1}
    DY_MAP = {1: 0, 2: 1, 4: 1, 8: 1, 16: 0, 32: -1, 64: -1, 128: -1}
    DIST_MAP = {
        1: 1.0, 2: 1.41421356, 4: 1.0, 8: 1.41421356,
        16: 1.0, 32: 1.41421356, 64: 1.0, 128: 1.41421356
    }

    with rasterio.open(flow_direction_path) as src:
        profile = src.profile.copy()
        d8 = src.read(1).astype(np.uint16)
        cell_size = src.transform[0]
        h, w = src.height, src.width
        
    if weight_raster_path:
        if not os.path.isfile(weight_raster_path):
             raise FileNotFoundError(f"Weight raster file not found: {weight_raster_path}")
        with rasterio.open(weight_raster_path) as wsrc:
            weight = wsrc.read(1).astype(np.float64)
    else:
        weight = np.ones((h, w), dtype=np.float64)

    length_arr = np.zeros((h, w), dtype=np.float32)
    upstream_neighbors = [[[] for _ in range(w)] for _ in range(h)]

    for i in range(h):
        for j in range(w):
            code = d8[i, j]
            if code == 0:
                continue
            ni = i + DY_MAP.get(code, 0)
            nj = j + DX_MAP.get(code, 0)
            if 0 <= ni < h and 0 <= nj < w:
                upstream_neighbors[ni][nj].append((i, j))

    q = deque()
    for i in range(h):
        for j in range(w):
            code = d8[i, j]
            is_outlet = False
            if code == 0:
                is_outlet = True
            else:
                ni = i + DY_MAP.get(code, 0)
                nj = j + DX_MAP.get(code, 0)
                if not (0 <= ni < h and 0 <= nj < w):
                    is_outlet = True
            
            if is_outlet:
                q.append((i, j))

    while q:
        i, j = q.popleft()
        for ui, uj in upstream_neighbors[i][j]:
            ucode = d8[ui, uj]
            step_dist = weight[ui, uj] * DIST_MAP.get(ucode, 1.0) * cell_size
            length_arr[ui, uj] = length_arr[i, j] + step_dist
            q.append((ui, uj))

    profile.update(dtype="float32", nodata=0.0)
    output_path = _get_out_path(output_name)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(length_arr.astype(np.float32), 1)

    return {"output_path": output_path}


# @mcp.tool()
def compute_flow_accumulation(flow_direction_path: str, output_name: str, weight_raster_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute flow accumulation from a D8 flow direction raster.

    Args:
        flow_direction_path: Path to the D8 flow direction raster file.
        output_name: Output filename (must end with .tif).
        weight_raster_path: Optional path to a weight raster file.

    Returns:
        output_path: Path to the flow accumulation raster file.
    """
    if not os.path.isfile(flow_direction_path):
        raise FileNotFoundError(f"Flow direction file not found: {flow_direction_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with rasterio.open(flow_direction_path) as src:
        profile = src.profile.copy()
        d8 = src.read(1).astype(np.uint16)
        h, w = d8.shape
        
    if weight_raster_path:
        if not os.path.isfile(weight_raster_path):
             raise FileNotFoundError(f"Weight raster file not found: {weight_raster_path}")
        with rasterio.open(weight_raster_path) as wsrc:
            weight = wsrc.read(1).astype(np.float64)
    else:
        weight = np.ones((h, w), dtype=np.float64)

    acc = weight.copy()
    indeg = np.zeros((h, w), dtype=np.int32)

    # D8 Encoding: E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128
    dx_map = {1: 1, 2: 1, 4: 0, 8: -1, 16: -1, 32: -1, 64: 0, 128: 1}
    dy_map = {1: 0, 2: 1, 4: 1, 8: 1, 16: 0, 32: -1, 64: -1, 128: -1}

    # Calculate in-degrees
    for i in range(h):
        for j in range(w):
            code = d8[i, j]
            if code == 0:
                continue
            ni = i + dy_map.get(code, 0)
            nj = j + dx_map.get(code, 0)
            if 0 <= ni < h and 0 <= nj < w:
                indeg[ni, nj] += 1

    # Topological sort (queue-based)
    q = deque()
    for i in range(h):
        for j in range(w):
            if indeg[i, j] == 0:
                q.append((i, j))

    while q:
        i, j = q.popleft()
        code = d8[i, j]
        if code == 0:
            continue
        ni = i + dy_map.get(code, 0)
        nj = j + dx_map.get(code, 0)
        if 0 <= ni < h and 0 <= nj < w:
            acc[ni, nj] += acc[i, j]
            indeg[ni, nj] -= 1
            if indeg[ni, nj] == 0:
                q.append((ni, nj))

    profile.update(dtype="float32", nodata=0.0)
    output_path = _get_out_path(output_name)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(acc.astype(np.float32), 1)

    return {"output_path": output_path}

# @mcp.tool()
def fill_depressions(raster_path: str, output_name: str) -> Dict[str, Any]:
    """
    Fill depressions in a DEM using the priority-flood algorithm.

    Args:
        raster_path: Path to the input DEM raster file.
        output_name: Output filename (must end with .tif).

    Returns:
        output_path: Path to the filled DEM raster file.
    """
    if not os.path.isfile(raster_path):
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
        nodata = src.nodata
        dem = src.read(1)

    dem = dem.astype(np.float64)
    if nodata is None:
        nodata = np.nan

    # Create mask for valid data
    if np.isnan(nodata):
        mask = ~np.isnan(dem)
    else:
        mask = dem != nodata
        
    filled = dem.copy()

    processed = np.zeros_like(mask, dtype=bool)
    heap = []

    h, w = dem.shape
    
    # Push border cells to heap
    for i in range(h):
        for j in [0, w - 1]:
            if mask[i, j] and not processed[i, j]:
                heapq.heappush(heap, (filled[i, j], i, j))
                processed[i, j] = True
    for j in range(w):
        for i in [0, h - 1]:
            if mask[i, j] and not processed[i, j]:
                heapq.heappush(heap, (filled[i, j], i, j))
                processed[i, j] = True

    neigh = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]

    while heap:
        z, i, j = heapq.heappop(heap)
        for di, dj in neigh:
            ni, nj = i + di, j + dj
            if ni < 0 or nj < 0 or ni >= h or nj >= w:
                continue
            if not mask[ni, nj] or processed[ni, nj]:
                continue
            if filled[ni, nj] < z:
                filled[ni, nj] = z
            processed[ni, nj] = True
            heapq.heappush(heap, (filled[ni, nj], ni, nj))

    profile.update(dtype="float32", nodata=nodata)
    output_path = _get_out_path(output_name)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(filled.astype(np.float32), 1)
        
    return {"output_path": output_path}

# @mcp.tool()
def compute_flow_direction(raster_path: str, output_name: str) -> Dict[str, Any]:
    """
    Compute D8 flow direction from a filled DEM.

    Encoding:
        E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128.

    Args:
        raster_path: Path to the filled DEM raster file.
        output_name: Output filename (must end with .tif).

    Returns:
        output_path: Path to the flow direction raster file.
    """
    if not os.path.isfile(raster_path):
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fp = os.path.abspath(raster_path)

    with rasterio.open(fp) as src:
        profile = src.profile.copy()
        nodata = src.nodata
        z = src.read(1).astype(np.float64)

    h, w = z.shape
    dir_r = np.zeros((h, w), dtype=np.uint16)

    # D8 encoding and offsets
    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    code_arr = [1, 2, 4, 8, 16, 32, 64, 128]
    dist = [1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)]
    
    delta_to_code = {
        (0, 1): 1,    # E
        (1, 1): 2,    # SE
        (1, 0): 4,    # S
        (1, -1): 8,   # SW
        (0, -1): 16,  # W
        (-1, -1): 32, # NW
        (-1, 0): 64,  # N
        (-1, 1): 128  # NE
    }

    # 1. Initial flow direction (Steepest descent)
    for i in range(h):
        for j in range(w):
            zij = z[i, j]
            if nodata is not None and zij == nodata:
                continue
            if np.isnan(zij):
                continue
            
            best_slope = -np.inf
            best_code = 0
            
            for k in range(8):
                ni, nj = i + dy[k], j + dx[k]
                if ni < 0 or nj < 0 or ni >= h or nj >= w:
                    continue
                nz = z[ni, nj]
                if nodata is not None and nz == nodata:
                    continue
                if np.isnan(nz):
                    continue
                
                # Slope calculation: (high - low) / distance
                # Note: We only care about downward flow (slope > 0)
                slope = (zij - nz) / dist[k]
                
                if slope > best_slope:
                    best_slope = slope
                    best_code = code_arr[k]
            
            # Only assign if there is a downward slope, otherwise 0 (flat or sink)
            if best_slope > 0:
                dir_r[i, j] = best_code

    # 2. Resolve flats
    # Iteratively pull flow: if neighbor is flat and same height, and current cell has flow, 
    # then make neighbor flow to current.
    q = deque()
    
    # Add all cells with flow to queue
    for i in range(h):
        for j in range(w):
            if dir_r[i, j] != 0:
                q.append((i, j))
                
    while q:
        i, j = q.popleft()
        zij = z[i, j]
        
        for k in range(8):
            ni, nj = i + dy[k], j + dx[k]
            if ni < 0 or nj < 0 or ni >= h or nj >= w:
                continue
            
            # If neighbor has no flow direction, and height is "equal" to current point (float error)
            # Note: Reverse search, neighbor flows TO current point
            # Neighbor ni,nj has no flow (dir_r == 0)
            # And neighbor is not NoData
            nz = z[ni, nj]
            if nodata is not None and nz == nodata:
                continue
            if np.isnan(nz):
                continue

            if dir_r[ni, nj] == 0 and abs(nz - zij) < 1e-5:
                # Calculate code for neighbor flowing to current
                # Neighbor is (ni, nj), Current is (i, j)
                # Flow direction is (i-ni, j-nj)
                di, dj = i - ni, j - nj
                new_code = delta_to_code.get((di, dj), 0)
                
                if new_code != 0:
                    dir_r[ni, nj] = new_code
                    q.append((ni, nj))

    profile.update(dtype="uint16", nodata=0)
    output_path = _get_out_path(output_name)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dir_r, 1)
        
    return {"output_path": output_path}

# @mcp.tool()
def compute_depression_depth(dem_path: str, filled_dem_path: str, output_name: str) -> Dict[str, Any]:
    """
    Compute depression depth (sink depth) raster.

    Definition:
        depth = filled_dem - original_dem.

    Args:
        dem_path: Path to the original DEM raster file.
        filled_dem_path: Path to the filled DEM raster file.
        output_name: Output filename (must end with .tif).

    Returns:
        output_path: Path to the depression depth raster file.
    """
    if not os.path.isfile(dem_path):
        raise FileNotFoundError(f"DEM file not found: {dem_path}")
    if not os.path.isfile(filled_dem_path):
        raise FileNotFoundError(f"Filled DEM file not found: {filled_dem_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with rasterio.open(dem_path) as src_o, rasterio.open(filled_dem_path) as src_f:
        profile = src_o.profile.copy()
        o = src_o.read(1).astype(np.float64)
        f = src_f.read(1).astype(np.float64)

    depth = f - o
    depth[~np.isfinite(depth)] = 0.0
    
    profile.update(dtype="float32", nodata=0.0)
    output_path = _get_out_path(output_name)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(depth.astype(np.float32), 1)
        
    return {"output_path": output_path}