from typing import Optional, Dict, List, Any
import yaml
from pathlib import Path
import os
import geopandas as gpd
import rasterio
from rasterio.features import shapes, rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
import pickle
import re
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
def get_vector_info(vector_path: str) -> Dict[str, Any]:
    """
    Read a vector file (e.g., SHP, GeoJSON, GPKG) and return key metadata.

    Args:
        vector_path: Path to the vector file.

    Returns:
        feature_count: Number of features.
        geometry_types: Unique geometry types present.
        crs: Coordinate reference system as a string.
        crs_type: Type of CRS ("geographic" or "projected").
        bounds: Bounding box of all features.
        spatial_extent_km2: Approximate area covered by bounds in km².
        columns: Attribute columns (excluding geometry).
        null_geometries: Number of features with null/empty geometries.
        column_info: Dictionary with column statistics (type, null_count, unique_count).
    """
    if not os.path.isfile(vector_path):
        raise FileNotFoundError(f"Vector file not found: {vector_path}")

    # Use fiona engine to avoid WKB type 16 errors with pyogrio
    gdf = gpd.read_file(vector_path, engine="fiona")
    crs = gdf.crs.to_string() if gdf.crs else None
    bounds = list(gdf.total_bounds)
    feature_count = int(len(gdf))
    geometry_types = list(gdf.geom_type.unique())
    
    # Determine CRS type
    crs_type = "undefined"
    if gdf.crs:
        if gdf.crs.is_geographic:
            crs_type = "geographic"
        elif gdf.crs.is_projected:
            crs_type = "projected"
    
    # Calculate spatial extent in km²
    spatial_extent_km2 = None
    if gdf.crs and len(gdf) > 0:
        try:
            if gdf.crs.is_geographic:
                # Reproject to appropriate UTM zone for area calculation
                gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
                spatial_extent_km2 = float(gdf_proj.unary_union.envelope.area / 1e6)
            else:
                # Assume projected CRS is in meters
                spatial_extent_km2 = float(gdf.unary_union.envelope.area / 1e6)
        except Exception:
            pass
    
    # Count null geometries
    null_geometries = int(gdf.geometry.isna().sum() + gdf.geometry.is_empty.sum())
    
    # Exclude the active geometry column from the list of attributes
    geom_col = gdf.geometry.name if hasattr(gdf, "geometry") else "geometry"
    columns = [c for c in gdf.columns if c != geom_col]
    
    # Column statistics
    column_info = {}
    for col in columns:
        col_data = gdf[col]
        null_count = int(col_data.isna().sum())
        unique_count = int(col_data.nunique(dropna=True))
        
        # Infer type
        if pd.api.types.is_numeric_dtype(col_data):
            col_type = "numeric"
            # Add basic stats for numeric columns
            valid_data = col_data.dropna()
            if len(valid_data) > 0:
                column_info[col] = {
                    "type": col_type,
                    "null_count": null_count,
                    "unique_count": unique_count,
                    "min": float(valid_data.min()),
                    "max": float(valid_data.max()),
                    "mean": float(valid_data.mean()),
                }
            else:
                column_info[col] = {
                    "type": col_type,
                    "null_count": null_count,
                    "unique_count": unique_count,
                }
        else:
            col_type = "string"
            column_info[col] = {
                "type": col_type,
                "null_count": null_count,
                "unique_count": unique_count,
            }

    return {
        "feature_count": feature_count,
        "geometry_types": geometry_types,
        "crs": crs,
        "crs_type": crs_type,
        "bounds": bounds,
        "spatial_extent_km2": spatial_extent_km2,
        "null_geometries": null_geometries,
        "columns": columns,
        "column_info": column_info,
    }

# @mcp.tool()
def get_raster_info(raster_path: str) -> Dict[str, Any]:
    """
    Read a raster file (e.g., GeoTIFF) and return key metadata.

    Args:
        raster_path: Path to the raster file.

    Returns:
        shape: List of two integers, height and width of the raster.
        band_count: Number of bands in the raster.
        band_dtypes: List of data types for each band.
        crs: Coordinate reference system as a string.
        crs_type: Type of CRS ("geographic" or "projected").
        bounds: List of four floats, bounding box of the raster.
        pixel_size: [width, height] in CRS units.
        pixel_size_meters: [width, height] in meters (if possible).
        nodata_info: List of dictionaries with nodata value and percentage per band.
        value_ranges: List of [min, max] for each band (sampled).
    """
    if not os.path.isfile(raster_path):
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        shape = [src.height, src.width]
        band_count = src.count
        profile = dict(src.profile)
        band_dtypes = list(getattr(src, "dtypes", [])) or ([str(profile.get("dtype"))] if profile.get("dtype") else [])
        crs = src.crs.to_string() if src.crs else None
        bounds = list(src.bounds)
        
        # Determine CRS type
        crs_type = "undefined"
        if src.crs:
            if src.crs.is_geographic:
                crs_type = "geographic"
            elif src.crs.is_projected:
                crs_type = "projected"
        
        # Pixel size in CRS units
        transform = src.transform
        pixel_size = [abs(transform.a), abs(transform.e)]
        
        # Try to calculate pixel size in meters
        pixel_size_meters = None
        if src.crs:
            if src.crs.is_projected:
                # Assume units are meters for projected CRS
                pixel_size_meters = pixel_size
            elif src.crs.is_geographic:
                # Approximate conversion at center latitude (111km per degree)
                center_lat = (bounds[1] + bounds[3]) / 2
                meters_per_degree = 111320 * np.cos(np.radians(center_lat))
                pixel_size_meters = [
                    pixel_size[0] * meters_per_degree,
                    pixel_size[1] * 111320  # latitude degrees to meters
                ]
        
        # NoData and value range analysis
        nodata_info = []
        value_ranges = []
        
        for band_idx in range(1, band_count + 1):
            band_data = src.read(band_idx, masked=True)
            nodata_val = src.nodatavals[band_idx - 1] if src.nodatavals else None
            
            # Calculate nodata percentage
            if hasattr(band_data, 'mask'):
                nodata_pct = float(band_data.mask.sum() / band_data.size * 100)
            else:
                nodata_pct = 0.0
            
            nodata_info.append({
                "band": band_idx,
                "nodata_value": nodata_val,
                "nodata_percentage": round(nodata_pct, 2)
            })
            
            # Calculate value range (use sample if large)
            valid_data = band_data.compressed() if hasattr(band_data, 'compressed') else band_data.flatten()
            if len(valid_data) > 0:
                # Sample if too large
                if len(valid_data) > 1000000:
                    sample_idx = np.random.choice(len(valid_data), 1000000, replace=False)
                    valid_data = valid_data[sample_idx]
                
                value_ranges.append({
                    "band": band_idx,
                    "min": float(np.min(valid_data)),
                    "max": float(np.max(valid_data)),
                    "mean": float(np.mean(valid_data)),
                })
            else:
                value_ranges.append({
                    "band": band_idx,
                    "min": None,
                    "max": None,
                    "mean": None,
                })

    return {
        "shape": shape,
        "band_count": band_count,
        "band_dtypes": band_dtypes,
        "crs": crs if crs is not None else "undefined",
        "crs_type": crs_type,
        "bounds": bounds,
        "pixel_size": pixel_size,
        "pixel_size_meters": pixel_size_meters,
        "nodata_info": nodata_info,
        "value_ranges": value_ranges,
    }

# @mcp.tool()
def get_csv_info(csv_path: str) -> Dict[str, Any]:
    """
    Read a CSV file and return key metadata.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        row_count: Number of rows in the CSV.
        columns: List of column names.
        column_types: Dictionary mapping column names to inferred types.
        missing_values: Dictionary mapping column names to missing value counts.
        numeric_summary: Dictionary with basic statistics for numeric columns.
        has_coordinates: Boolean indicating if potential coordinate columns exist.
    """
    cp = os.path.abspath(csv_path)
    if not os.path.isfile(cp):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(cp, index_col=None, header=0)
    
    # Column type inference
    column_types = {}
    missing_values = {}
    numeric_summary = {}
    
    for col in df.columns:
        missing_values[col] = int(df[col].isna().sum())
        
        # Try to infer type
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                numeric_summary[col] = {
                    "min": float(valid_data.min()),
                    "max": float(valid_data.max()),
                    "mean": float(valid_data.mean()),
                    "std": float(valid_data.std()),
                }
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = "datetime"
        else:
            # Try to convert to numeric
            try:
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                if numeric_data.notna().sum() > len(df) * 0.8:  # >80% convertible
                    column_types[col] = "numeric_string"
                    valid_data = numeric_data.dropna()
                    if len(valid_data) > 0:
                        numeric_summary[col] = {
                            "min": float(valid_data.min()),
                            "max": float(valid_data.max()),
                            "mean": float(valid_data.mean()),
                            "std": float(valid_data.std()),
                        }
                else:
                    column_types[col] = "string"
            except Exception:
                column_types[col] = "string"
    
    # Check for coordinate columns
    coord_keywords = {
        'lon': ['longitude', 'lon', 'lng', 'x'],
        'lat': ['latitude', 'lat', 'y']
    }
    has_coordinates = False
    for col_lower in [c.lower() for c in df.columns]:
        if any(kw in col_lower for kw in coord_keywords['lon'] + coord_keywords['lat']):
            has_coordinates = True
            break
    
    return {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "column_types": column_types,
        "missing_values": missing_values,
        "numeric_summary": numeric_summary,
        "has_coordinates": has_coordinates,
    }

# @mcp.tool()
def get_nc_info(nc_path: str) -> Dict[str, Any]:
    """
    Read a NetCDF file and return key metadata.

    Args:
        nc_path: Path to the NetCDF file.

    Returns:
        variable_count: Number of variables in the NetCDF.
        variable_names: List of variable names.
        variable_info: Dictionary with details for each variable (dimensions, shape, dtype).
        global_attributes: Dictionary of global attributes from the NetCDF file.
    """
    p = os.path.abspath(nc_path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"NetCDF file not found: {p}")
    
    from netCDF4 import Dataset
    
    try:
        ds = Dataset(p, mode="r")
    except Exception as e:
        raise RuntimeError(f"Failed to open NetCDF: {str(e)}")
    
    # Get variable names (excluding coordinate variables)
    coord_vars = {"time", "lat", "latitude", "lon", "longitude", "x", "y", "z", "depth", "time_bnds", "bounds"}
    var_names = [v for v in ds.variables.keys() if v.lower() not in coord_vars]
    
    # Get detailed info for each variable
    variable_info = {}
    for var_name in ds.variables.keys():
        var = ds.variables[var_name]
        var_info = {
            "dimensions": list(var.dimensions),
            "shape": list(var.shape),
            "dtype": str(var.dtype),
        }
        
        # Add attributes if available
        if hasattr(var, 'units'):
            var_info["units"] = str(var.units)
        if hasattr(var, 'long_name'):
            var_info["long_name"] = str(var.long_name)
        if hasattr(var, '_FillValue'):
            var_info["fill_value"] = float(var._FillValue)
        
        # Sample statistics for non-coordinate variables
        if var_name.lower() not in coord_vars and var.size > 0:
            try:
                data = var[:]
                if hasattr(data, 'mask'):
                    valid_data = data.compressed()
                else:
                    valid_data = data.flatten()
                
                if len(valid_data) > 0:
                    # Sample if too large
                    if len(valid_data) > 100000:
                        sample_idx = np.random.choice(len(valid_data), 100000, replace=False)
                        valid_data = valid_data[sample_idx]
                    
                    var_info["value_range"] = {
                        "min": float(np.min(valid_data)),
                        "max": float(np.max(valid_data)),
                        "mean": float(np.mean(valid_data)),
                    }
            except Exception:
                pass
        
        variable_info[var_name] = var_info
    
    # Get global attributes
    global_attributes = {}
    for attr_name in ds.ncattrs():
        try:
            global_attributes[attr_name] = str(ds.getncattr(attr_name))
        except Exception:
            pass
    
    ds.close()
    
    return {
        "variable_count": len(var_names),
        "variable_names": var_names,
        "variable_info": variable_info,
        "global_attributes": global_attributes,
    }

# @mcp.tool()
def get_graphml_info(graphml_path: str) -> Dict[str, Any]:
    """
    Read a GraphML file and return key metadata.

    Args:
        graphml_path: Path to the GraphML file.

    Returns:
        num_nodes: Number of nodes in the graph.
        num_edges: Number of edges in the graph.
        is_directed: Whether the graph is directed.
        is_weighted: Whether edges have weights.
        crs: Coordinate reference system as a string.
        node_attributes: List of available node attribute names.
        edge_attributes: List of available edge attribute names.
        has_geometry: Whether nodes have x/y coordinates.
        bounds: Bounding box if geometry exists [minx, miny, maxx, maxy].
    """
    if not os.path.isfile(graphml_path):
        raise FileNotFoundError(f"GraphML file not found: {graphml_path}")

    G = ox.load_graphml(graphml_path)
    crs = G.graph.get("crs") if hasattr(G, "graph") else None

    # Get node attributes
    node_attributes = []
    if G.number_of_nodes() > 0:
        sample_node = list(G.nodes(data=True))[0][1]
        node_attributes = list(sample_node.keys())
    
    # Get edge attributes
    edge_attributes = []
    if G.number_of_edges() > 0:
        sample_edge = list(G.edges(data=True))[0][2]
        edge_attributes = list(sample_edge.keys())
    
    # Check for geometry
    has_geometry = False
    bounds = None
    if 'x' in node_attributes and 'y' in node_attributes:
        has_geometry = True
        xs = [data.get('x') for _, data in G.nodes(data=True) if data.get('x') is not None]
        ys = [data.get('y') for _, data in G.nodes(data=True) if data.get('y') is not None]
        if xs and ys:
            bounds = [min(xs), min(ys), max(xs), max(ys)]

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "is_directed": G.is_directed(),
        "is_weighted": nx.is_weighted(G),
        "crs": crs if crs is not None else "undefined",
        "node_attributes": node_attributes,
        "edge_attributes": edge_attributes,
        "has_geometry": has_geometry,
        "bounds": bounds,
    }

# @mcp.tool()
def get_pkl_info(pkl_path: str) -> Dict[str, Any]:
    """
    Read a pickle file and return key metadata.

    Args:
        pkl_path: Path to the pickle file.

    Returns:
        data_type: Type of the pickled object.
        num_items: Number of items (if applicable).
        keys: List of keys (if dict-like).
        crs: Coordinate reference system as a string (if available).
        structure_info: Additional structure information.
    """
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    data_type = type(data).__name__
    num_items = None
    keys = None
    structure_info = {}
    
    # Try to get length
    try:
        num_items = len(data)
    except Exception:
        pass
    
    # Check if dict-like
    if hasattr(data, 'keys'):
        try:
            keys = list(data.keys())
        except Exception:
            pass
    
    # Check for CRS
    crs = None
    if hasattr(data, 'crs'):
        crs = str(data.crs)
    elif isinstance(data, dict) and 'crs' in data:
        crs = str(data['crs'])
    elif hasattr(data, 'graph') and hasattr(data.graph, 'get'):
        crs = data.graph.get('crs')
    
    # Additional structure info based on type
    if isinstance(data, (list, tuple)):
        if len(data) > 0:
            structure_info["first_item_type"] = type(data[0]).__name__
    elif isinstance(data, dict):
        structure_info["key_count"] = len(data.keys())
        if keys and len(keys) > 0:
            structure_info["sample_value_types"] = {
                k: type(data[k]).__name__ for k in list(keys)[:5]
            }
    elif hasattr(data, '__dict__'):
        structure_info["attributes"] = list(vars(data).keys())[:10]  # First 10 attributes

    return {
        "data_type": data_type,
        "num_items": num_items,
        "keys": keys[:20] if keys else None,  # Limit to first 20 keys
        "crs": crs if crs is not None else "undefined",
        "structure_info": structure_info,
    }

# @mcp.tool()
def reproject_vector(vector_path: str, target_crs: str, output_name: str) -> Dict[str, Any]:
    """
    Reproject a vector file (e.g., SHP, GeoJSON) to a target CRS.

    Args:
        vector_path: Path to the input vector file.
        target_crs: Target CRS string.
        output_name: Output filename (supported extensions: .shp, .geojson).

    Returns:
        output_path: Path to the reprojected vector file.
    """
    if not os.path.isfile(vector_path):
        raise FileNotFoundError(f"Vector file not found: {vector_path}")

    gdf = gpd.read_file(vector_path)
    
    # Perform reprojection
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    else:
        gdf = gdf.to_crs(target_crs)

    # Determine output format and path
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Infer driver from extension
    ext = os.path.splitext(output_name)[1].lower()
    if ext == ".shp":
        driver = "ESRI Shapefile"
    else:
        driver = "GeoJSON"

    output_path = _get_out_path(output_name)
    gdf.to_file(output_path, driver=driver)

    return {"output_path": output_path}

# @mcp.tool()
def reproject_raster(raster_path: str, target_crs: str, output_name: str, band: int | None = None) -> Dict[str, Any]:
    """
    Reproject a raster file to a target CRS.

    Args:
        raster_path: Path to the input raster file.
        target_crs: Target CRS string.
        output_name: Output filename (must end with .tif).
        band: Optional band index to reproject. If None, reprojects all bands.

    Returns:
        output_path: Path to the reprojected raster file.
    """
    if not os.path.isfile(raster_path):
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    rp = os.path.abspath(raster_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    out_path = _get_out_path(output_name)

    with rasterio.open(rp) as src:
        transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
        profile = src.profile.copy()
        
        # Handle single band vs all bands
        if band is not None:
             out_count = 1
             dtype = src.dtypes[(band - 1) if band else 0]
        else:
             out_count = src.count
             dtype = src.dtypes[0] # Assuming same dtype for all bands

        profile.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "count": out_count,
            "dtype": dtype
        })

        with rasterio.open(out_path, "w", **profile) as dst:
            if band is None:
                for i in range(1, src.count + 1):
                    dest = np.empty((height, width), dtype=src.dtypes[i - 1])
                    reproject(
                        source=rasterio.band(src, i),
                        destination=dest,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest,
                    )
                    dst.write(dest, i)
            else:
                i = int(band)
                if not (1 <= i <= src.count):
                    raise ValueError(f"Invalid band index {i}; raster has {src.count} band(s).")
                dest = np.empty((height, width), dtype=src.dtypes[i - 1])
                reproject(
                    source=rasterio.band(src, i),
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )
                dst.write(dest, 1)
    return {"output_path": out_path}

# @mcp.tool()
def align_raster_to_reference(
    raster_path: str,
    reference_raster_path: str,
    output_name: str,
    resampling: str = "nearest",
    band: int | None = None,
) -> Dict[str, Any]:
    """
    Align a raster exactly to the grid of a reference raster.

    Args:
        raster_path: Path to the source raster file.
        reference_raster_path: Path to the reference raster file.
        output_name: Output filename (must end with .tif).
        resampling: Resampling method ("nearest", "bilinear", "cubic").
        band: Optional band index to reproject (None for all bands).

    Returns:
        output_path: Path to the aligned raster file.
    """
    src_path = os.path.abspath(raster_path)
    ref_path = os.path.abspath(reference_raster_path)
    
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Raster not found: {src_path}")
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(f"Reference raster not found: {ref_path}")

    with rasterio.open(ref_path) as ref:
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
        dst_crs = ref.crs

    with rasterio.open(src_path) as src:
        method_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }
        rs = method_map.get((resampling or "nearest").lower(), Resampling.nearest)
        
        out_count = src.count if band is None else 1
        # Determine output dtype
        out_dtype = src.dtypes[0] if band is None else src.dtypes[(int(band) - 1) if band else 0]
        
        # Handle nodata
        src_nodata = src.nodata if getattr(src, "nodata", None) is not None else src.profile.get("nodata")
        dst_nodata = src_nodata

        profile = src.profile.copy()
        profile.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "count": out_count,
            "dtype": out_dtype,
        })
        if dst_nodata is not None:
            profile.update({"nodata": dst_nodata})

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        out_path = _get_out_path(output_name)

        with rasterio.open(out_path, "w", **profile) as dst:
            if band is None:
                for i in range(1, src.count + 1):
                    dest = np.empty((dst_height, dst_width), dtype=out_dtype)
                    reproject(
                        source=rasterio.band(src, i),
                        destination=dest,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        src_nodata=src_nodata,
                        dst_nodata=dst_nodata,
                        resampling=rs,
                    )
                    dst.write(dest, i)
            else:
                i = int(band)
                if not (1 <= i <= src.count):
                    raise ValueError(f"Invalid band index {i}; raster has {src.count} band(s).")
                dest = np.empty((dst_height, dst_width), dtype=out_dtype)
                reproject(
                    source=rasterio.band(src, i),
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    src_nodata=src_nodata,
                    dst_nodata=dst_nodata,
                    resampling=rs,
                )
                dst.write(dest, 1)
    return {"output_path": out_path}

# @mcp.tool()
def add_vector_coords(vector_path: str, output_name: str) -> Dict[str, Any]:
    """
    Add Latitude and Longitude fields to a vector file (points or centroids).

    Args:
        vector_path: Path to the input vector file.
        output_name: Output filename (must end with .geojson).

    Returns:
        output_path: Path to the file with added coordinates.
    """
    if not os.path.isfile(vector_path):
        raise FileNotFoundError(f"Vector file not found: {vector_path}")
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gdf = gpd.read_file(vector_path)
    
    geom_series = gdf.geometry
    if not (gdf.geometry.geom_type == "Point").all():
         geom_series = gdf.geometry.centroid

    lats = []
    lons = []
    for geom in geom_series:
        if geom is None:
            lats.append(None)
            lons.append(None)
        else:
            lats.append(geom.y)
            lons.append(geom.x)

    gdf["Latitude"] = lats
    gdf["Longitude"] = lons

    output_path = _get_out_path(output_name)
    
    # Save as GeoJSON to ensure field support
    gdf.to_file(output_path, driver="GeoJSON")
    
    return {"output_path": output_path}

# @mcp.tool()
def raster_to_polygons(raster_path: str, output_name: str, band: int = 1) -> Dict[str, Any]:
    """
    Convert non-zero pixels in a specific band to polygon vectors.

    Args:
        raster_path: Input raster path.
        output_name: Output filename (must end with .geojson).
        band: Band index to sample (default 1).

    Returns:
        output_path: Path to the output polygon vector file.
    """
    raster_path = os.path.abspath(raster_path)
    with rasterio.open(raster_path) as src:
        arr = src.read(band)
        transform = src.transform
        crs = src.crs

    # 非零像元作为有效区域
    mask = arr != 0
    geoms = (s for _, (s, _) in enumerate(shapes(arr, mask=mask, transform=transform)))
    polygons = [{"type": "Feature", "properties": {}, "geometry": s} for s in geoms]
    gdf = gpd.GeoDataFrame.from_features(polygons, crs=crs)

    out_geojson = _get_out_path(output_name)
    gdf.to_file(out_geojson, driver="GeoJSON")
    return {"output_path": out_geojson}

# @mcp.tool()
def rasterize_polygons(
    polygons_path: str,
    reference_raster_path: str,
    value_field: str,
    output_name: str,
    boundary_path: str | None = None,
    value_map: Dict[str | int, int] | None = None,
    fill_value: int = 0,
) -> Dict[str, Any]:
    """
    Rasterize polygons by a given attribute field, aligned to a reference raster.

    Args:
        polygons_path: Path to polygons shapefile.
        reference_raster_path: Path to reference raster for grid alignment.
        value_field: Attribute field name to rasterize.
        output_name: Output filename (must end with .tif).
        boundary_path: Optional path to boundary shapefile to clip.
        value_map: Optional mapping for string or numeric values to integer classes.
        fill_value: Fill value for cells without data.

    Returns:
        output_path: Path to the output raster.
    """
    rp = os.path.abspath(reference_raster_path)
    with rasterio.open(rp) as ref:
        transform = ref.transform
        height, width = ref.height, ref.width
        ref_crs = ref.crs

    pp = os.path.abspath(polygons_path)
    from fiona.env import Env
    with Env(SHAPE_RESTORE_SHX="YES"):
        gdf_poly = gpd.read_file(pp, engine="fiona")

    if boundary_path:
        bp = os.path.abspath(boundary_path)
        with Env(SHAPE_RESTORE_SHX="YES"):
            gdf_bound = gpd.read_file(bp, engine="fiona")
    else:
        gdf_bound = None

    if gdf_poly.crs and ref_crs and gdf_poly.crs != ref_crs:
        gdf_poly = gdf_poly.to_crs(ref_crs)
    if gdf_bound is not None:
        if gdf_bound.crs and ref_crs and gdf_bound.crs != ref_crs:
            gdf_bound = gdf_bound.to_crs(ref_crs)
        clipped = gpd.overlay(gdf_poly, gdf_bound, how="intersection")
    else:
        clipped = gdf_poly

    def _norm(s):
        return str(s).strip().lower()

    str_map = {}
    num_map = {}
    if value_map:
        for k, v in value_map.items():
            if isinstance(k, str):
                str_map[_norm(k)] = int(v)
            else:
                try:
                    num_map[int(k)] = int(v)
                except Exception:
                    pass

    shapes = []
    for _, row in clipped.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        raw = row.get(value_field)
        if raw is None:
            continue
        val = None
        try:
            val = int(float(raw))
        except Exception:
            key = _norm(str(raw))
            if key in str_map:
                val = str_map[key]
        if val is None and isinstance(raw, (int, float)):
            ival = int(raw)
            if ival in num_map:
                val = num_map[ival]
        if val is None:
            continue
        shapes.append((geom, int(val)))

    arr = rasterize(shapes, out_shape=(height, width), transform=transform, fill=fill_value, dtype=np.int32)

    profile = {
        "driver": "GTiff",
        "dtype": rasterio.int32,
        "count": 1,
        "crs": ref_crs,
        "transform": transform,
        "height": height,
        "width": width,
        "nodata": int(fill_value),
    }

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = _get_out_path(output_name)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.int32), 1)
    return {"output_path": out_path}

# @mcp.tool()
def csv_to_points(
    csv_path: str,
    output_name: str,
    crs: Optional[str] = None,
    lon_field: Optional[str] = None,
    lat_field: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a point Shapefile from a CSV with longitude/latitude columns.

    Args:
        csv_path: Path to the input CSV file.
        output_name: Output filename (must end with .shp).
        crs: CRS of the input coordinates.
        lon_field: Optional longitude column name; auto-detected if None.
        lat_field: Optional latitude column name; auto-detected if None.

    Returns:
        output_path: Path to the written Shapefile.
        count: Number of points written.
    """
    cp = os.path.abspath(csv_path)
    if not os.path.isfile(cp):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pandas.read_csv(cp)
    lon_candidates = ["longitude", "lon", "x", "LONGITUDE", "Lng"]
    lat_candidates = ["latitude", "lat", "y", "LATITUDE", "Lat"]
    def _find_first(cols, candidates):
        for c in candidates:
            if c in cols:
                return c
        return None
    points_lon_field = lon_field or _find_first(df.columns, lon_candidates)
    points_lat_field = lat_field or _find_first(df.columns, lat_candidates)
    if points_lon_field is None or points_lat_field is None:
        raise ValueError("Unable to auto-detect lon/lat fields; provide via fields={'lon':..., 'lat':...}")
    # Cast to numeric and drop invalid rows
    df[points_lon_field] = pandas.to_numeric(df[points_lon_field], errors="coerce")
    df[points_lat_field] = pandas.to_numeric(df[points_lat_field], errors="coerce")
    df_valid = df.dropna(subset=[points_lon_field, points_lat_field]).copy()
    geometry = [Point(xy) for xy in zip(df_valid[points_lon_field], df_valid[points_lat_field])]
    in_crs = crs or "EPSG:4326"
    pts = gpd.GeoDataFrame(df_valid, crs=in_crs, geometry=geometry)

    # Determine output path in OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    pts.to_file(out_path, driver="ESRI Shapefile")
    return {"output_path": out_path, "count": int(len(pts))}

# @mcp.tool()
def export_nc_var(
        nc_path: str,
        var_name: str,
        output_name: str
) -> Dict[str, Any]:
    """
    Convert a NetCDF variable to a single-band GeoTIFF.

    Args:
        nc_path: Path to the NetCDF file.
        var_name: Variable name to export.
        output_name: Output filename (must end with .tif).

    Returns:
        output_path: Path to the written tif.
    """
    p = os.path.abspath(nc_path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"NetCDF file not found: {p}")
    try:
        with rasterio.open(f'NETCDF:"{p}":{var_name}') as src:
            data = src.read(1)
            h, w = src.height, src.width
            src_crs = src.crs
            src_transform = src.transform
            src_nodata = src.nodata
    except Exception as e:
        raise RuntimeError(f"Failed to read variable {var_name}: {str(e)}")
    
    # Handle nodata
    if src_nodata is not None:
        data = np.where(data == src_nodata, np.nan, data)
        
    arr = data.astype(np.float32)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "nodata": np.nan,
        "crs": src_crs,
        "transform": src_transform,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
    return {"output_path": out_path}

# @mcp.tool()
def join_table_to_vector(
    vector_path: str,
    table_path: str,
    vector_key: str,
    table_key: str,
    output_name: str
) -> Dict[str, Any]:
    """
    Join a tabular dataset to a vector layer by a key.

    Args:
        vector_path: Path to the vector dataset (Shapefile/GeoJSON etc.).
        table_path: Path to the tabular dataset (Excel or CSV).
        vector_key: Join key on the vector side.
        table_key: Join key on the table side.
        output_name: Output filename (must end with .geojson).

    Returns:
        output_path: Path to the joined output file.
    """
    join_how = "left"

    states = gpd.read_file(vector_path, engine="fiona")
    
    if table_path.lower().endswith(".csv"):
        homeless = pd.read_csv(table_path)
    else:
        homeless = pd.read_excel(table_path, sheet_name="Sheet1")

    lk = vector_key
    rk = table_key

    if lk not in states.columns:
        raise ValueError(f"Vector key '{lk}' not found in vector dataset.")
    if rk not in homeless.columns:
        raise ValueError(f"Table key '{rk}' not found in table dataset. Available columns: {list(homeless.columns)}")

    s = states[lk].astype(str)
    h = homeless[rk].astype(str)
    s = s.str.strip(); h = h.str.strip()
    s = s.str.upper(); h = h.str.upper()
    states[lk] = s
    homeless[rk] = h

    homeless = homeless.drop_duplicates(subset=[rk])

    joined = states.merge(homeless, left_on=lk, right_on=rk, how=join_how)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    joined.to_file(out_path, driver="GeoJSON")
    return {"output_path": out_path}

# @mcp.tool()
def calculate_vector_field(
    input_path: str,
    output_name: str,
    new_field: str,
    expression: str
) -> Dict[str, Any]:
    """
    Calculate a new field in a vector dataset using a Python expression.

    Args:
        input_path: Path to the input vector file.
        output_name: Output filename (must end with .geojson).
        new_field: Name of the new field to create.
        expression: Python expression string. Can use column names directly (e.g. "ColA / ColB") 
                   supported by pandas.DataFrame.eval().

    Returns:
        output_path: Path to the output vector file.
    """
    gdf = gpd.read_file(input_path)
    
    # Attempt to convert columns to numeric for calculation
    for col in gdf.columns:
        if col != "geometry":
            try:
                gdf[col] = pd.to_numeric(gdf[col])
            except Exception:
                pass
    
    try:
        # Use pandas eval for efficient calculation using column names
        gdf[new_field] = gdf.eval(expression)
    except Exception:
        # Fallback: simple eval with df context
        try:
            local_dict = {"df": gdf, "np": np, "pd": pd}
            gdf[new_field] = eval(expression, {}, local_dict)
        except Exception as e:
            raise ValueError(f"Calculation failed: {e}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output_name)
    gdf.to_file(out_path, driver="GeoJSON")
    return {"output_path": out_path}

# @mcp.tool()
def split_train_test(
    csv_path: str,
    output_name: str,
    label_column: str = "label",
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = True,
    index: bool = False,
) -> Dict[str, Any]:
    """
    Split a CSV dataset into training and testing sets.

    Args:
        csv_path: Path to the input CSV file.
        output_name: Base name for the output files (e.g., "my_data" -> "my_data_X_train.csv").
        label_column: Name of the column to be used as the target variable (y).
        test_size: Proportion of the dataset to include in the test split (default: 0.3).
        random_state: Seed used by the random number generator (default: 42).
        stratify: If True, data is split in a stratified fashion, using the label column as the class labels.
        index: Whether to write row names (index) to the output CSV files (default: False).

    Returns:
        X_train_path: Path to the saved training features CSV.
        X_test_path: Path to the saved testing features CSV.
        y_train_path: Path to the saved training target CSV.
        y_test_path: Path to the saved testing target CSV.
        train_size: Number of samples in the training set.
        test_size: Number of samples in the testing set.
    """
    DEFAULT_SEP = ","
    DEFAULT_ENCODING = None
    df = pd.read_csv(os.path.abspath(csv_path), sep=DEFAULT_SEP, encoding=DEFAULT_ENCODING)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in CSV: {csv_path}")
    y = df[label_column].values
    X = df.drop(columns=[label_column])

    SHUFFLE = True
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=SHUFFLE, stratify=(y if stratify else None)
    )

    base_dir = OUTPUT_DIR
    os.makedirs(base_dir, exist_ok=True)
    X_train_path = os.path.join(base_dir, f"{output_name}_X_train.csv")
    X_test_path = os.path.join(base_dir, f"{output_name}_X_test.csv")
    y_train_path = os.path.join(base_dir, f"{output_name}_y_train.csv")
    y_test_path = os.path.join(base_dir, f"{output_name}_y_test.csv")

    X_train.to_csv(X_train_path, index=index)
    X_test.to_csv(X_test_path, index=index)
    pd.DataFrame({label_column: y_train}).to_csv(y_train_path, index=index)
    pd.DataFrame({label_column: y_test}).to_csv(y_test_path, index=index)

    return {
        "X_train_path": X_train_path,
        "X_test_path": X_test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

# @mcp.tool()
def aggregate_od_flows(
    od_path: str,
    attributes_path: str,
    flow_field: str,
    output_name: str,
    origin_field: str,
    destination_field: str,
    join_key: str,
    undirected: bool = False,
) -> Dict[str, Any]:
    """
    Aggregate Origin-Destination (OD) flows by node pairs and merge with node attributes.

    Args:
        od_path: Path to the OD flow data CSV.
        attributes_path: Path to the attribute data CSV (e.g., socio-economic, environmental).
        flow_field: The column name in the OD data representing the flow volume to aggregate.
        output_name: Output filename (must end with .csv).
        origin_field: The column name for the origin node in the OD data.
        destination_field: The column name for the destination node in the OD data.
        join_key: The column name in the attribute data to join with origin/destination nodes.
        undirected: If True, treats OD pairs as undirected (A->B equals B->A) by sorting origin and destination (default: False).

    Returns:
        output_path: The absolute path to the saved merged CSV file.
        rows: The number of rows in the merged dataset.
        columns: A list of column names in the merged dataset.
        flow_field_used: The name of the flow column in the output.
    """
    op = os.path.abspath(od_path)
    ap = os.path.abspath(attributes_path)
    
    DEFAULT_HEADER = 0
    DEFAULT_INDEX_COL = None
    DEFAULT_SEP = ","
    DEFAULT_ENCODING = None
    
    attrs = pd.read_csv(ap, index_col=DEFAULT_INDEX_COL, header=DEFAULT_HEADER, sep=DEFAULT_SEP, encoding=DEFAULT_ENCODING)
    od = pd.read_csv(op, index_col=DEFAULT_INDEX_COL, header=DEFAULT_HEADER, sep=DEFAULT_SEP, encoding=DEFAULT_ENCODING)
    
    # Check required columns
    for req_col in [origin_field, destination_field, flow_field]:
        if req_col not in od.columns:
            raise ValueError(f"OD data missing required column '{req_col}'. Available: {list(od.columns)}")
    if join_key not in attrs.columns:
        raise ValueError(f"Attribute data missing join key '{join_key}'. Available: {list(attrs.columns)}")
    
    # Ensure flow column is numeric
    try:
        od[flow_field] = pd.to_numeric(od[flow_field], errors="coerce")
    except Exception:
        pass
        
    # Group by Origin-Destination
    by_fields = [origin_field, destination_field]
    if undirected:
        od["_O"] = od[[origin_field, destination_field]].astype(str).min(axis=1)
        od["_D"] = od[[origin_field, destination_field]].astype(str).max(axis=1)
        by_fields = ["_O", "_D"]
    
    DEFAULT_AGG = None
    DEFAULT_NUMERIC_ONLY = True
    if DEFAULT_AGG is not None:
        grouped = od.groupby(by_fields, as_index=False).agg(DEFAULT_AGG)
    else:
        grouped = od.groupby(by_fields, as_index=False).sum(numeric_only=DEFAULT_NUMERIC_ONLY)
    
    # Standardize flow column name
    found = flow_field
    if found not in grouped.columns:
        raise ValueError(f"Flow field '{found}' not found in grouped columns: {list(grouped.columns)}")
    DEFAULT_FLOW_OUTPUT_FIELD = "TotalFlow"
    if found != DEFAULT_FLOW_OUTPUT_FIELD:
        grouped = grouped.rename(columns={found: DEFAULT_FLOW_OUTPUT_FIELD})
        
    # Merge attributes to Origin
    DEFAULT_JOIN_HOW = "left"
    merged = pd.merge(grouped, attrs, left_on=by_fields[0], right_on=join_key, how=DEFAULT_JOIN_HOW)
    
    # Merge attributes to Destination (with suffixes)
    DEFAULT_SUFFIXES = ("_Origin", "_Destination")
    merged = pd.merge(merged, attrs, left_on=by_fields[1], right_on=join_key, suffixes=DEFAULT_SUFFIXES, how=DEFAULT_JOIN_HOW)
    
    # Drop redundant join keys
    DEFAULT_DROP_MERGE_KEYS = True
    if DEFAULT_DROP_MERGE_KEYS:
        drop_candidates = [f"{join_key}{DEFAULT_SUFFIXES[0]}", f"{join_key}{DEFAULT_SUFFIXES[1]}"]
        to_drop = [c for c in drop_candidates if c in merged.columns]
        if len(to_drop) > 0:
            merged = merged.drop(columns=to_drop)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output_name)
    merged.to_csv(out_path, index=False)

    return {
        "output_path": out_path,
        "rows": int(len(merged)),
        "columns": list(merged.columns),
        "flow_field_used": DEFAULT_FLOW_OUTPUT_FIELD,
    }

# @mcp.tool()
def set_geojson_crs(geojson_path: str, output_name: str, crs: str) -> Dict[str, Any]:
    """
    Set the CRS of a GeoJSON file.

    Args:
        geojson_path: The path to the input GeoJSON file.
        output_name: Output filename (must end with .geojson).
        crs: The target CRS to set.

    Returns:
        output_path: The path to the output GeoJSON file with the updated CRS.
    """
    import os
    import json
    if not os.path.isfile(geojson_path):
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data.get("type") != "FeatureCollection":
        raise ValueError("Input must be a GeoJSON FeatureCollection.")
    data["crs"] = {"type": "name", "properties": {"name": crs}}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    return {"output_path": output_path}

# @mcp.tool()
def round_coordinate_values(
    points_path: str,
    output_name: str,
    lat_col: str,
    lon_col: str,
    decimals: int = 2,
) -> Dict[str, Any]:
    """
    Round coordinate columns to the specified decimals and write to new fields.

    Args:
        points_path: Path to the input vector file.
        output_name: Output filename (must end with .geojson).
        lat_col: Latitude column name to round (in-place).
        lon_col: Longitude column name to round (in-place).
        decimals: Number of decimal places to round to (default: 2).

    Returns:
        output_path: Path to the output file.
    """
    import os
    import geopandas as gpd
    import pandas as pd

    p_abs = os.path.abspath(points_path)
    gdf = gpd.read_file(p_abs, engine='fiona')

    if lat_col not in gdf.columns or lon_col not in gdf.columns:
        raise ValueError(f"Missing columns: {lat_col}, {lon_col}")

    lat_round = pd.to_numeric(gdf[lat_col], errors="coerce").round(decimals)
    lon_round = pd.to_numeric(gdf[lon_col], errors="coerce").round(decimals)
    gdf[lat_col + "Round"] = lat_round
    gdf[lon_col + "Round"] = lon_round

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_name)

    gdf.to_file(output_path, driver="GeoJSON")
    return {
        "output_path": output_path,
    }

# @mcp.tool()
def extract_netcdf_profile(
    nc_path: str,
    output_name: str,
    variables: List[str] = None,
    lon_value: float = 330.5,
    lat_min: float = -10.0,
    lat_max: float = -9.0,
    depth_max: float = 1000.0,
) -> Dict[str, Any]:
    """
    Extract a profile from NetCDF and save as CSV.

    Args:
        nc_path: Path to the NetCDF file.
        output_name: Output filename (must end with .csv).
        variables: List of variable names to extract as columns.
        lon_value: Target longitude (nearest).
        lat_min: Minimum latitude.
        lat_max: Maximum latitude.
        depth_max: Max depth included.

    Returns:
        output_path: Path to the saved CSV file.
    """
    import os
    import csv
    import numpy as np
    import rasterio
    p = os.path.abspath(nc_path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"NetCDF file not found: {p}")
    vars_list = list(variables) if variables else []
    lat_var_candidates = ["latitude", "lat", "y"]
    lon_var_candidates = ["longitude", "lon", "x"]
    depth_var_candidates = ["depth", "Depth", "DEPTH"]
    lat_arr = None
    for nm in lat_var_candidates:
        try:
            with rasterio.open(f'NETCDF:"{p}":{nm}') as la:
                lat_arr = la.read(1).squeeze().flatten()
                break
        except Exception:
            continue
    lon_arr = None
    for nm in lon_var_candidates:
        try:
            with rasterio.open(f'NETCDF:"{p}":{nm}') as lo:
                lon_arr = lo.read(1).squeeze().flatten()
                break
        except Exception:
            continue
    if lat_arr is None or lon_arr is None:
        raise RuntimeError("Latitude/longitude variables not found")
    lat_mask = (lat_arr > lat_min) & (lat_arr < lat_max)
    lat_cand = np.where(lat_mask)[0]
    target_lat = (lat_min + lat_max) / 2.0
    j = int(lat_cand[np.argmin(np.abs(lat_arr[lat_cand] - target_lat))]) if lat_cand.size > 0 else int(np.argmin(np.abs(lat_arr - target_lat)))
    i = int(np.argmin(np.abs(lon_arr - lon_value)))
    depth = None
    for nm in depth_var_candidates:
        try:
            with rasterio.open(f'NETCDF:"{p}":{nm}') as ds:
                depth = ds.read(1).squeeze().flatten()
                break
        except Exception:
            continue
    if depth is None:
        ref_n = None
        pick_from = list(vars_list) or []
        if not pick_from and 'available_vars' in locals():
            pick_from = [v for v in available_vars if v.lower() not in {"latitude","lat","y","longitude","lon","x","depth","DEPTH","Depth"}]
        for v in pick_from:
            try:
                with rasterio.open(f'NETCDF:"{p}":{v}') as vsrc:
                    stack = vsrc.read()
                    if stack.ndim == 2:
                        stack = stack[np.newaxis, ...]
                    ref_n = stack.shape[0]
                    break
            except Exception:
                continue
        if ref_n is None:
            raise RuntimeError("Depth variable not found")
        depth = np.arange(ref_n, dtype=float)
    profiles = {}
    for vnm in vars_list:
        try:
            with rasterio.open(f'NETCDF:"{p}":{vnm}') as vsrc:
                stack = vsrc.read()
                if stack.ndim == 2:
                    stack = stack[np.newaxis, ...]
                n, h, w = stack.shape
        except Exception:
            continue
        dc = min(depth.size, n)
        dmask = (depth[:dc] <= depth_max)
        if lat_arr.size == h and lon_arr.size == w:
            profiles[vnm] = stack[:dc, j, i][dmask]
        elif lat_arr.size == w and lon_arr.size == h:
            profiles[vnm] = stack[:dc, i, j][dmask]
        else:
            profiles[vnm] = stack[:dc, min(j, h - 1), min(i, w - 1)][dmask]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output_name)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["depth"] + vars_list
        w.writerow(header)
        dc = min(depth.size, max((len(vsrc_vals) for vsrc_vals in profiles.values()), default=depth.size))
        dmask_global = (depth[:dc] <= depth_max)
        rows = zip(depth[:dc][dmask_global].tolist(), *[profiles[v] for v in vars_list])
        for row in rows:
            w.writerow(list(row))
    return {"output_path": out_path}

# @mcp.tool()
def fit_netcdf_timeseries(
    nc_path: str,
    output_name: str,
    degree: int = 2,
    var_constraint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fit a polynomial to a NetCDF time series and export as CSV.

    Args:
        nc_path: Path to the NetCDF file.
        output_name: Output filename (must end with .csv).
        degree: Polynomial degree (e.g., 2 for quadratic).
        var_constraint: Optional variable name to select the variable.

    Returns:
        output_path: Path to the CSV with columns.
    """
    import os
    from pathlib import Path
    import numpy as np
    import csv
    import cftime
    from netCDF4 import Dataset

    p = os.path.abspath(nc_path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"NetCDF file not found: {p}")

    ds = Dataset(p, mode="r")
    # pick target variable
    if var_constraint and var_constraint in ds.variables:
        v = ds.variables[var_constraint]
    else:
        # choose first non-coordinate variable that has a time dimension
        def _is_coord(name: str) -> bool:
            ln = name.lower()
            return ln in {"time", "lat", "latitude", "lon", "longitude", "time_bnds", "bounds", "forecast_period"}
        v = None
        for name, var in ds.variables.items():
            if _is_coord(name):
                continue
            dims = [d.lower() for d in var.dimensions]
            if any(d in dims for d in ["time", "forecast_period", "valid_time"]):
                v = var
                break
        if v is None:
            # fallback to first non-coordinate variable
            for name, var in ds.variables.items():
                if not _is_coord(name):
                    v = var
                    break
            if v is None:
                raise RuntimeError("No suitable data variable found for time series fitting")

    # identify dimension axes
    dim_names = [d.lower() for d in v.dimensions]
    try:
        t_axis = dim_names.index("time")
    except ValueError:
        for candidate in ["forecast_period", "valid_time"]:
            if candidate in dim_names:
                t_axis = dim_names.index(candidate)
                break
        else:
            t_axis = 0

    # get time points
    time_name = v.dimensions[t_axis]
    tvar = ds.variables.get(time_name)
    x_points = np.array(tvar[:], dtype=float)
    units = getattr(tvar, "units", None)
    calendar = getattr(tvar, "calendar", "standard")

    # read data and collapse non-time dims by mean to get a single time series
    arr = np.ma.filled(v[:], np.nan).astype(np.float64)
    # move time axis to front
    if arr.ndim > 1:
        arr = np.moveaxis(arr, t_axis, 0)
        # average across the remaining axes
        y_points = np.nanmean(arr, axis=tuple(range(1, arr.ndim)))
    else:
        y_points = arr

    # polyfit with finite values only
    mask = np.isfinite(x_points) & np.isfinite(y_points)
    xs = x_points[mask]
    ys = y_points[mask]
    deg = max(1, int(degree))
    deg = min(deg, max(1, len(xs) - 1))
    coeffs = np.polyfit(xs, ys, deg)
    y_fitted = np.polyval(coeffs, x_points)

    # convert time to years for CSV
    if units is None:
        # fallback: treat numeric as Unix days since 1970
        units = "days since 1970-01-01"
    years = [cftime.num2date(x, units=units, calendar=calendar).strftime("%Y") for x in x_points]

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output_name)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Year", "Fitted_Temperature"])
        for yr, y_fit in zip(years, y_fitted):
            w.writerow([yr, float(y_fit) if np.isfinite(y_fit) else ""]) 
    return {"output_path": out_path}