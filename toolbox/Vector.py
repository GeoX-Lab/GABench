from typing import Optional, Dict, List, Any
import yaml
from pathlib import Path
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Polygon, Point
import math
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
def buffer_vector(input_path: str, distance: float, output_name: str) -> Dict[str, Any]: 
    """ 
    Perform buffer analysis on vector data.

    Args: 
        input_path: Path to input vector data.
        distance: Buffer radius in CRS units.
        output_name: Output filename (must end with .geojson).

    Returns: 
        output_path: Path to the buffered vector file. 
    """ 
    ip = os.path.abspath(input_path) 
    gdf = gpd.read_file(ip) 
    buff = gdf.buffer(float(distance)) 
    out_gdf = gpd.GeoDataFrame(geometry=buff, crs=gdf.crs) 

    out_path = _get_out_path(output_name) 
    out_gdf.to_file(out_path, driver="GeoJSON") 
    return {"output_path": out_path}

# @mcp.tool()
def dissolve_polygons(input_path: str, output_name: str) -> Dict[str, Any]:
    """
    Dissolve all polygons into a single multipart geometry.

    Args:
        input_path: Path to input polygon layer.
        output_name: Output filename (must end with .geojson).

    Returns:
        output_path: Path to the dissolved vector file.
        dissolved_count: Number of output features.
        crs: CRS string.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    gdf = gpd.read_file(input_path)
    gdf_dissolved = gdf.dissolve()
    
    output_path = _get_out_path(output_name)
    gdf_dissolved.to_file(output_path, driver="GeoJSON")
    
    return {
        "output_path": output_path,
        "dissolved_count": len(gdf_dissolved),
        "crs": str(gdf.crs)
    }

# @mcp.tool()
def create_spatial_join(
    left_path: str,
    right_path: str,
    output_name: str,
    predicate: str = "intersects",
    how: str = "left",
    max_distance: Optional[float] = None,
    distance_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform a general-purpose spatial join between two vector datasets.

    Args:
        left_path: Path to the left vector dataset (Shapefile/GeoJSON).
        right_path: Path to the right vector dataset.
        output_name: Output filename (must end with .geojson).
        predicate: Spatial relationship: one of 'intersects', 'within', 'contains', 'touches', 'overlaps', 'crosses', 'nearest'.
        how: Join type: one of 'left', 'right', 'inner'.
        max_distance: Only used when predicate='nearest'; maximum search distance in the dataset CRS units.
        distance_col: Only used when predicate='nearest'; column name to store computed distances (default 'distance').

    Returns:
        output_path: Path to the written GeoJSON.
        count: Number of features in the joined result.
    """
    lp = os.path.abspath(left_path)
    rp = os.path.abspath(right_path)
    if not os.path.isfile(lp):
        raise FileNotFoundError(f"Left vector file not found: {lp}")
    if not os.path.isfile(rp):
        raise FileNotFoundError(f"Right vector file not found: {rp}")

    left = gpd.read_file(lp)
    right = gpd.read_file(rp)
    if left.crs is None and right.crs is not None:
        left = left.set_crs(right.crs, allow_override=True)
    if right.crs is None and left.crs is not None:
        right = right.set_crs(left.crs, allow_override=True)
    if left.crs != right.crs:
        right = right.to_crs(left.crs)
    allowed_predicates = {"intersects", "within", "contains", "touches", "overlaps", "crosses", "nearest"}
    if predicate not in allowed_predicates:
        raise ValueError(f"Unsupported predicate: {predicate}")
    allowed_how = {"left", "right", "inner"}
    if how not in allowed_how:
        raise ValueError(f"Unsupported how: {how}")
    if predicate == "nearest":
        dcol = distance_col or "distance"
        joined = gpd.sjoin_nearest(left, right, how=how, max_distance=max_distance, distance_col=dcol)
    else:
        joined = gpd.sjoin(left, right, how=how, predicate=predicate, lsuffix="left", rsuffix="right")
    joined = joined.reset_index().rename(columns={"index": "index_left"})
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    joined.to_file(out_path, driver="GeoJSON")
    return {"output_path": out_path, "count": int(len(joined))}

# @mcp.tool()
def overlay_difference(a_path: str, b_path: str, output_name: str) -> dict:
    """
    Calculate the geometric difference between two polygon layers (A minus B).
    This operation removes the overlapping areas of layer B from layer A.

    Args:
        a_path: Path to the input polygon layer A.
        b_path: Path to the erase polygon layer B.
        output_name: Output filename (must end with .geojson).

    Returns:
        output_path: Path to the difference result vector file (.geojson).
    """
    ap = os.path.abspath(a_path)
    bp = os.path.abspath(b_path)
    a = gpd.read_file(ap)
    b = gpd.read_file(bp)
    if a.crs and b.crs and a.crs != b.crs:
        b = b.to_crs(a.crs)
    a = a.copy()
    b = b.copy()
    a["geometry"] = a["geometry"].buffer(0)
    b["geometry"] = b["geometry"].buffer(0)
    a_diss = a.dissolve()
    diff = gpd.overlay(a_diss, b, how="difference")
    
    out_path = _get_out_path(output_name)
    diff.to_file(out_path, driver="GeoJSON")
    return {"output_path": out_path}

# # @mcp.tool()
# def extract_intersecting_features(input_path: str, intersect_path: str, output_name: str) -> Dict[str, Any]:
#     """
#     Extract features from the input layer that spatially intersect with the intersect layer.
#     (Performs a spatial inner join, preserving input geometries).

#     Args:
#         input_path: Path to the input vector file (features to select).
#         intersect_path: Path to the intersecting vector file (filter mask).
#         output_name: Output filename (must end with .geojson).

#     Returns:
#         output_path: Path to the extracted vector file.
#     """
#     import os
#     import geopandas as gpd

#     input_gdf = gpd.read_file(input_path)
#     intersect_gdf = gpd.read_file(intersect_path)
    
#     if input_gdf.crs and intersect_gdf.crs and input_gdf.crs != intersect_gdf.crs:
#         intersect_gdf = intersect_gdf.to_crs(input_gdf.crs)

#     # Spatial join with inner method to keep only matching records
#     joined = gpd.sjoin(input_gdf, intersect_gdf, how="inner", predicate="intersects")
    
#     # Preserve original index
#     joined["original_index"] = joined.index
    
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     out_path = _get_out_path(output_name)
#     joined.to_file(out_path, driver="GeoJSON")
#     return {"output_path": out_path}

# @mcp.tool()
def clip_by_attribute(input_geojson_path: str, mask_geojson_path: str, output_name: str, attribute_field: str | None = "AREATYPE", attribute_value: str | None = "RURAL") -> Dict[str, Any]:
    """
    Clip input data to a specific area based on attribute values (general purpose).
    If attribute_field is None or not found in the mask fields, the entire mask is used for clipping.

    Args:
        input_geojson_path: Path to input vector data.
        mask_geojson_path: Path to area mask vector data.
        output_name: Output filename (must end with .geojson).
        attribute_field: Classification field name in mask data (optional).
        attribute_value: Classification value to retain (optional).

    Returns:
        Output path and basic statistics.
    """
    # Read data
    input_gdf = gpd.read_file(input_geojson_path)
    mask_gdf = gpd.read_file(mask_geojson_path)
    
    # Ensure CRS alignment
    if input_gdf.crs and mask_gdf.crs and input_gdf.crs != mask_gdf.crs:
        mask_gdf = mask_gdf.to_crs(input_gdf.crs)
        
    # Fix invalid geometries
    input_gdf["geometry"] = input_gdf["geometry"].buffer(0)
    mask_gdf["geometry"] = mask_gdf["geometry"].buffer(0)

    if attribute_field and attribute_field in mask_gdf.columns:
        selected = mask_gdf[mask_gdf[attribute_field] == attribute_value]
    else:
        selected = mask_gdf
        
    clipped = gpd.clip(input_gdf, selected)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)
    clipped.to_file(output_path, driver="GeoJSON")
    property_fields = [col for col in clipped.columns if col != "geometry"]
    
    return {
        "output_path": output_path,
        "statistics": {
            "original_features": int(len(input_gdf)),
            "clipped_features": int(len(clipped)),
            "property_fields": property_fields
        }
    }

# # @mcp.tool()
# def aggregate_points_to_polygons(
#     points_geojson_path: str,
#     polygons_geojson_path: str,
#     point_value_property: str,
#     output_name: str,
#     agg: str = "mean",
#     polygon_id_field: str | None = None,
#     predicate: str = "within"
# ) -> Dict[str, Any]:
#     """
#     Aggregate point values within polygon geometries.

#     Args:
#         points_geojson_path: The path to the input GeoJSON file containing point data.
#         polygons_geojson_path: The path to the input GeoJSON file containing polygon data.
#         point_value_property: The property name in the point features to use for aggregation.
#         output_name: Output filename (must end with .geojson).
#         agg: The aggregation method, e.g., 'mean', 'sum', 'median'.
#         polygon_id_field: The field name in polygon features to use as polygon ID.
#         predicate: The spatial predicate to use for intersection, e.g., 'within', 'intersects'.

#     Returns:
#         output_path: The path to the output GeoJSON file with aggregated point values.
#     """
#     import os
#     import geopandas as gpd
#     import pandas as pd

#     # Read data
#     gdf_polys = gpd.read_file(polygons_geojson_path)
#     gdf_points = gpd.read_file(points_geojson_path)

#     # Create temporary ID for polygons if not provided or to ensure uniqueness for join
#     temp_id_col = "__pid__"
#     if polygon_id_field and polygon_id_field in gdf_polys.columns:
#         gdf_polys[temp_id_col] = gdf_polys[polygon_id_field]
#     else:
#         gdf_polys[temp_id_col] = gdf_polys.index

#     # Enforce CRS alignment
#     if gdf_points.crs and gdf_polys.crs and gdf_points.crs != gdf_polys.crs:
#         gdf_points = gdf_points.to_crs(gdf_polys.crs)

#     # Filter invalid points
#     gdf_points = gdf_points[gdf_points[point_value_property].notna()]
#     # Ensure value column is numeric
#     gdf_points[point_value_property] = pd.to_numeric(gdf_points[point_value_property], errors='coerce')
#     gdf_points = gdf_points.dropna(subset=[point_value_property])

#     # Spatial join
#     if predicate not in ("intersects", "within", "contains", "touches", "overlaps"):
#         predicate = "intersects"
    
#     joined = gpd.sjoin(
#         gdf_points[[point_value_property, 'geometry']],
#         gdf_polys[[temp_id_col, 'geometry']],
#         how='inner',
#         predicate=predicate
#     )

#     # Group and aggregate
#     if agg == "mean":
#         grp = joined.groupby(temp_id_col)[point_value_property].agg(['count', 'mean']).reset_index()
#     elif agg == "sum":
#         grp = joined.groupby(temp_id_col)[point_value_property].agg(['count', 'sum']).reset_index()
#         grp.rename(columns={'sum': 'mean'}, inplace=True) 
#     elif agg == "median":
#         grp = joined.groupby(temp_id_col)[point_value_property].agg(['count', 'median']).reset_index()
#         grp.rename(columns={'median': 'mean'}, inplace=True)
#     else:
#         grp = joined.groupby(temp_id_col)[point_value_property].agg(['count', 'mean']).reset_index()

#     # Rename aggregate columns
#     agg_field = f"{point_value_property}_mean"
#     count_field = f"{point_value_property}_count"
#     grp.rename(columns={'count': count_field, 'mean': agg_field}, inplace=True)

#     # Merge aggregate back to original polygons
#     gdf_out = gdf_polys.merge(grp, on=temp_id_col, how='left')
    
#     # Fill NaN with 0 or None as appropriate (count=0, mean=NaN)
#     gdf_out[count_field] = gdf_out[count_field].fillna(0).astype(int)

#     # Clean up temp column
#     if temp_id_col in gdf_out.columns and temp_id_col != polygon_id_field:
#         gdf_out = gdf_out.drop(columns=[temp_id_col])

#     # Write output
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     output_path = _get_out_path(output_name)
#     gdf_out.to_file(output_path, driver="GeoJSON")

#     return {"output_path": output_path}

# @mcp.tool()
def count_points_in_polygons(polygons_path: str, points_path: str, output_name: str, count_field_name: str = 'point_count') -> Dict[str, Any]:
    """
    Count the number of point features inside each polygon geometry.

    Args:
        polygons_path: Path to the input polygon vector file.
        points_path: Path to the input point vector file.
        output_name: Output filename (must end with .geojson).
        count_field_name: Name of the new field to store the count (default: 'point_count').

    Returns:
        output_path: Path to the output vector file with the count field.
    """
    polygons_gdf = gpd.read_file(polygons_path)
    points_gdf = gpd.read_file(points_path)

    if polygons_gdf.crs is None:
        raise ValueError("Polygons CRS is undefined")
    if points_gdf.crs is None:
        points_gdf = points_gdf.set_crs(polygons_gdf.crs)
    elif points_gdf.crs != polygons_gdf.crs:
        points_gdf = points_gdf.to_crs(polygons_gdf.crs)

    joined = gpd.sjoin(polygons_gdf, points_gdf, how="left", predicate="intersects")
    
    # Count occurrences of valid right indices per left index
    counts = joined.groupby(joined.index)['index_right'].count()
    
    # Assign counts to original polygons
    polygons_gdf[count_field_name] = counts

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    polygons_gdf.to_file(out_path, driver="GeoJSON")
    return {"output_path": out_path}

# @mcp.tool()
def compute_area_sum(vector_path: str) -> float:
    """
    Calculate the union area of polygon data (fixes geometries and computes union), returns value, no file output.

    Args:
        vector_path: Path to input polygon vector data (.geojson/.shp).

    Returns:
        Total area as a float (area of the geometric union).
    """
    vp = os.path.abspath(vector_path)
    gdf = gpd.read_file(vp)
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].buffer(0)
    union_geom = gdf.geometry.unary_union
    total_area = float(union_geom.area)
    return total_area

# @mcp.tool()
def summarize_damage(
    buildings_in_flood_path: str,
    original_buildings_path: str,
    raster_path: str,
    output_name: str,
    group_field: str | None = None,
    sample_band: int = 1,
) -> Dict[str, Any]:
    """
    Perform damage assessment on buildings based on flood depth.
    
    Calculates damage costs using a specific formula:
    Cost = (0.298 * ln(0.01 * depth) + 1.4502) * 271 * area
    (Applied where mean depth > 1).

    Args:
        buildings_in_flood_path: Path to affected buildings (must include group_field referencing original buildings).
        original_buildings_path: Path to original buildings vector file.
        raster_path: Path to the flood depth raster.
        output_name: Output filename (must end with .geojson).
        group_field: Field used to link affected parts to original buildings. If None, uses index.
        sample_band: Raster band index to sample (default: 1).

    Returns:
        output_path: Path to the output vector file with 'mean_depth' and 'DamageCosts' fields.
    """
    # Read vectors
    bif = gpd.read_file(buildings_in_flood_path)
    buildings = gpd.read_file(original_buildings_path)

    # Sample raster values
    with rasterio.open(raster_path) as src:
        # Reproject temporary gdf for sampling if needed
        if bif.crs and src.crs and bif.crs != src.crs:
            bif_proj = bif.to_crs(src.crs)
        else:
            bif_proj = bif
            
        coords = [(geom.centroid.x, geom.centroid.y) for geom in bif_proj.geometry]
        
        # Handle band index
        band_idx = sample_band
        if band_idx < 1: band_idx = 1
        if band_idx > src.count: band_idx = src.count
        
        # Sample (rasterio.sample expects 1-based index in 'indexes' argument, or we pick from result)
        # src.sample(..., indexes=band_idx) returns iterator of arrays
        sampled_gen = src.sample(coords, indexes=band_idx)
        sampled_vals = [float(vals[0]) for vals in sampled_gen]
        
    bif["sampled_depth"] = sampled_vals

    # Aggregate statistics
    if group_field and group_field in bif.columns:
        group_keys = bif[group_field]
    else:
        group_keys = bif.index

    grp = bif.groupby(group_keys).agg({
        "sampled_depth": "mean", 
        "geometry": "size"
    }).rename(columns={
        "sampled_depth": "mean_depth", 
        "geometry": "intersections"
    })

    # Merge back to original buildings
    buildings["mean_depth"] = 0.0
    buildings["intersections"] = 0
    
    # Update records that exist in both (assuming group_field matches buildings index)
    common_ids = grp.index.intersection(buildings.index)
    if not common_ids.empty:
        buildings.loc[common_ids, "mean_depth"] = grp.loc[common_ids, "mean_depth"]
        buildings.loc[common_ids, "intersections"] = grp.loc[common_ids, "intersections"].astype(int)

    # Calculate damage costs
    def calculate_cost(row):
        depth = row["mean_depth"]
        if depth > 1.0:
            # Formula: (0.298 * ln(0.01 * depth) + 1.4502) * 271 * area
            return (0.298 * (math.log(0.01 * depth)) + 1.4502) * 271 * row["geometry"].area
        return 0.0

    buildings["DamageCosts"] = buildings.apply(calculate_cost, axis=1)

    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    buildings.to_file(out_path, driver="GeoJSON")
    
    return {"output_path": out_path}

# @mcp.tool()
def filter_features_by_expression(
    input_path: str,
    output_name: str,
    expression: str,
) -> Dict[str, Any]:
    """
    Filter geospatial features (GeoJSON, Shapefile, etc.) by a simple expression string and save the result.

    Supported:
      - Relational ops: >, >=, <, <=, ==, !=
      - Logical ops: and, or, not (auto-converted to &, |, ~)
      - Parentheses, numeric and string literals
      - Column names as variables; use prop("col name") for names with spaces/non-ASCII
      - Membership: isin(series, [values,...])  -> pandas.Series.isin(...)

    Args:
        input_path: Path to input geospatial file (GeoJSON, SHP, etc.)
        output_name: Output filename (must end with .geojson).
        expression: Expression string

    Returns:
        output_path: The path to the output file with filtered features.
    """
    # Load data
    gdf = gpd.read_file(input_path)

    # Preprocess logical operators for pandas bitwise operations
    expr = expression.strip()
    expr = re.sub(r'\band\b', '&', expr)
    expr = re.sub(r'\bor\b', '|', expr)
    expr = re.sub(r'\bnot\b', '~', expr)

    # Automatically brackets comparison clauses, avoiding & and | causes a parsing error
    def _wrap_comparisons(e: str) -> str:
        parts = re.split(r'(\&|\|)', e)
        def needs_wrap(token: str) -> bool:
            t = token.strip()
            if not t or (t.startswith("(") and t.endswith(")")):
                return False
            return bool(re.search(r'(==|!=|>=|<=|>|<)', t))
        out = []
        for p in parts:
            if p in ("&", "|"):
                out.append(p)
            else:
                out.append(f"({p.strip()})" if needs_wrap(p) else p)
        return " ".join(out)
    expr = _wrap_comparisons(expr)

    # Safe evaluation environment: expose only columns and helper funcs
    env = {c: gdf[c] for c in gdf.columns}    # columns accessible as variables
    env["prop"] = lambda name: gdf[name]      # access columns with spaces/non-ASCII
    env["isin"] = lambda s, vals: s.astype(object).isin(vals)  # membership check

    # Evaluate expression to boolean mask
    try:
        mask = eval(expr, {"__builtins__": {}}, env)
    except Exception as e:
        raise ValueError(f"Failed to parse/evaluate expression: {e}")

    # Normalize mask to a boolean Series aligned with gdf.index
    if isinstance(mask, np.ndarray):
        mask = pd.Series(mask, index=gdf.index)
    elif isinstance(mask, pd.Series):
        pass
    else:
        raise ValueError(f"Expression must yield a pandas Series or ndarray, got {type(mask)}")

    if mask.dtype != bool:
        try:
            mask = mask.astype(bool)
        except Exception:
            raise ValueError("Expression result is not boolean and cannot be converted to boolean.")

    filtered = gdf[mask]

    # Auto-generate concise output path under OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)

    # Save result
    ext = os.path.splitext(output_path)[1].lower()
    driver = "GeoJSON" if ext == ".geojson" else None
    
    if driver:
        filtered.to_file(output_path, driver=driver)
    else:
        filtered.to_file(output_path)

    # If both input and output are GeoJSON, try to preserve exact CRS name
    input_ext = os.path.splitext(input_path)[1].lower()
    if input_ext == ".geojson" and ext == ".geojson":
        try:
            input_crs = get_geojson_crs(input_path)
            if input_crs:
                import json
                with open(output_path, 'r', encoding='utf-8') as f:
                    out_gj = json.load(f)
                out_gj["crs"] = {"type": "name", "properties": {"name": input_crs}}
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(out_gj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Ignore CRS patching errors

    return {"output_path": output_path}

# @mcp.tool()
def compute_convex_hull(input_path: str, output_name: str) -> dict:
    """
    Compute the minimum bounding geometry (convex hull) and save as a vector file.

    Args:
        input_path: Path to input vector data (.geojson/.shp).
        output_name: Output filename for the convex hull (must end with .geojson).

    Returns:
        output_path: The path to the output file with the convex hull.
    """
    ip = os.path.abspath(input_path)
    gdf = gpd.read_file(ip)
    hull = (gdf.geometry.union_all().convex_hull if hasattr(gdf.geometry, "union_all") else gdf.unary_union.convex_hull)
    out_gdf = gpd.GeoDataFrame(geometry=[hull], crs=gdf.crs)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)
    out_gdf.to_file(output_path, driver="GeoJSON")
    return {"output_path": output_path}

# @mcp.tool()
def create_hexagonal_grid(
    ref_gdf_path: str,
    output_name: str,
    hex_size: float = 500
) -> Dict[str, Any]:
    """
    Create a hexagonal grid that covers the extent of a reference shapefile
    and save it as a Shapefile.

    Args:
        ref_gdf_path: Path to the reference shapefile used for extent and CRS.
        output_name: Output filename (must end with .shp).
        hex_size: Hexagon size measured as center-to-vertex distance
                  (in the same units as the shapefile CRS).

    Returns:
        output_path: Path to the output hexagonal grid shapefile.
    """
    ref_gdf = gpd.read_file(ref_gdf_path, engine="fiona")
    bounds = ref_gdf.total_bounds
    crs = ref_gdf.crs
    ref_union = ref_gdf.unary_union
    
    min_x, min_y, max_x, max_y = bounds
    hex_width = hex_size * 2
    hex_height = hex_size * np.sqrt(3)
    cols = int(np.ceil((max_x - min_x) / (hex_width * 0.75))) + 2
    rows = int(np.ceil((max_y - min_y) / hex_height)) + 2
    hexagons = []
    for row in range(rows):
        for col in range(cols):
            x = min_x + col * (hex_width * 0.75)
            y = min_y + row * hex_height
            if col % 2 == 1:
                y += hex_height / 2
            hexagon = Polygon([
                (x - hex_size, y), (x - hex_size / 2, y + hex_height / 2),
                (x + hex_size / 2, y + hex_height / 2), (x + hex_size, y),
                (x + hex_size / 2, y - hex_height / 2), (x - hex_size / 2, y - hex_height / 2),
            ])
            if hexagon.intersects(ref_union):
                hexagons.append(hexagon)
            
    hexagon_gdf = gpd.GeoDataFrame(geometry=hexagons, crs=crs)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    hexagon_gdf.to_file(out_path, driver="ESRI Shapefile")
    return {"output_path": out_path}

# @mcp.tool()
def generate_quadtree_cells(
    points_geojson_path: str,
    polygons_geojson_path: str,
    output_name: str,
    value_field: str,
    indicator_field: str = "nullity",
    nmax: int = 100,
    max_depth: int = 10,
    target_cells: Optional[int] = None,
    min_quadrant_points: int = 0,
) -> Dict[str, Any]:
    """
    Partition point features within a polygon extent using a quadtree and aggregate per cell. Computes the per-cell point count and the non-null ratio for a specified attribute.

    Args:
        points_geojson_path: Path to input point features (GeoJSON).
        polygons_geojson_path: Path to clipping/extent polygons (GeoJSON).
        output_name: Output filename (must end with .geojson).
        value_field: Attribute in points used to check non-null values.
        indicator_field: Name of the derived indicator field; 1 for non-null, 0 for null.
        nmax: Maximum number of points per leaf cell before stopping subdivision.
        max_depth: Maximum recursion depth for quadtree subdivision.
        target_cells: Desired number of cells; used to adaptively estimate the per-cell point threshold.
        min_quadrant_points: Minimum points required in each child quadrant; if any child falls below, stop subdividing at this node.

    Returns:
        output_path: Absolute path to the generated cell grid GeoJSON.
    """
    if not (os.path.isfile(points_geojson_path) and os.path.isfile(polygons_geojson_path)):
        return {"status": "error", "message": "Input files not found"}

    points = gpd.read_file(points_geojson_path)
    polygons = gpd.read_file(polygons_geojson_path)

    if value_field not in points.columns:
        return {"status": "error", "message": f"Property not found: {value_field}"}

    points = points.assign(**{indicator_field: points[value_field].notnull().astype(int)})

    union = polygons.unary_union
    minx, miny, maxx, maxy = union.bounds

    xs = points.geometry.x.values
    ys = points.geometry.y.values
    idx_all = list(range(len(points)))

    import math
    from shapely.geometry import box

    nmax_eff = nmax
    if target_cells and target_cells > 0:
        nmax_eff = max(nmax_eff, int(math.ceil(len(points) / float(target_cells))))

    records = []

    def rec(bounds, idxs, depth):
        if not idxs:
            return
        if len(idxs) <= nmax_eff or depth >= max_depth:
            poly = box(*bounds)
            inter = poly.intersection(union)
            if not inter.is_empty:
                cnt = len(idxs)
                null_cnt = int(points.iloc[idxs][indicator_field].sum())
                ratio = (null_cnt / cnt) if cnt > 0 else 0.0
                records.append({
                    "geometry": inter,
                    "count": int(cnt),
                    "non_null_count": int(null_cnt),
                    "non_null_ratio": float(ratio),
                })
            return
        lx, ly, rx, ry = bounds
        mx = (lx + rx) / 2.0
        my = (ly + ry) / 2.0
        q1 = [i for i in idxs if (xs[i] >= lx and xs[i] < mx and ys[i] >= ly and ys[i] < my)]
        q2 = [i for i in idxs if (xs[i] >= mx and xs[i] <= rx and ys[i] >= ly and ys[i] < my)]
        q3 = [i for i in idxs if (xs[i] >= lx and xs[i] < mx and ys[i] >= my and ys[i] <= ry)]
        q4 = [i for i in idxs if (xs[i] >= mx and xs[i] <= rx and ys[i] >= my and ys[i] <= ry)]
        child_counts = [len(q1), len(q2), len(q3), len(q4)]
        if min_quadrant_points > 0 and min(child_counts) < min_quadrant_points:
            poly = box(*bounds)
            inter = poly.intersection(union)
            if not inter.is_empty:
                cnt = len(idxs)
                null_cnt = int(points.iloc[idxs][indicator_field].sum())
                ratio = (null_cnt / cnt) if cnt > 0 else 0.0
                records.append({
                    "geometry": inter,
                    "count": int(cnt),
                    "non_null_count": int(null_cnt),
                    "non_null_ratio": float(ratio),
                })
            return
        rec((lx, ly, mx, my), q1, depth + 1)
        rec((mx, ly, rx, my), q2, depth + 1)
        rec((lx, my, mx, ry), q3, depth + 1)
        rec((mx, my, rx, ry), q4, depth + 1)

    rec((minx, miny, maxx, maxy), idx_all, 0)

    if not records:
        return {"status": "error", "message": "No cells generated"}

    cells_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)

    cells_gdf.to_file(output_path, driver="GeoJSON")

    return {"output_path": output_path}

# @mcp.tool()
def generate_random_points(
    reference_polygon_path: str, 
    num_points: int, 
    output_name: str,
    target_crs: str = None,
    label_field: Optional[str] = None,
    label_value: Any = 1
) -> Dict[str, Any]:
    """
    Overview:
        Generate uniformly distributed random points inside the unary union
        of the reference polygons. MultiPolygon areas are sampled with area-weighted
        allocation.

    Args:
        reference_polygon_path: Path to the polygon layer defining the sampling area.
        num_points: Number of points to generate.
        output_name: Output filename (must end with .geojson or .shp).
        target_crs: Optional target CRS; reference layer is reprojected if needed.
        label_field: Optional attribute name to add to the output points.
        label_value: Value to assign when label_field is provided.

    Returns:
        Dict with:
            - output_path: Absolute path to the output points file
            - generated_count: Number of points generated
            - crs: Output CRS string
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    gdf_ref = gpd.read_file(reference_polygon_path)
    
    # Ensure CRS match if provided
    if target_crs and gdf_ref.crs != target_crs:
        gdf_ref = gdf_ref.to_crs(target_crs)
    
    # Use unary_union to treat as one area
    geom = gdf_ref.geometry.unary_union
    
    points = []
    if geom is not None and not geom.is_empty:
        if geom.geom_type == 'MultiPolygon':
            parts = list(geom.geoms)
            areas = [p.area for p in parts]
            total_area = sum(areas)
            if total_area > 0:
                probs = [a/total_area for a in areas]
                counts = np.random.multinomial(num_points, probs)
                for part, count in zip(parts, counts):
                    if count > 0:
                        minx, miny, maxx, maxy = part.bounds
                        attempts = 0
                        max_attempts = count * 100
                        generated = 0
                        while generated < count and attempts < max_attempts:
                            x = np.random.uniform(minx, maxx)
                            y = np.random.uniform(miny, maxy)
                            p = Point(x, y)
                            if part.contains(p):
                                points.append(p)
                                generated += 1
                            attempts += 1
        elif geom.geom_type == 'Polygon':
            minx, miny, maxx, maxy = geom.bounds
            attempts = 0
            max_attempts = num_points * 100
            generated = 0
            while generated < num_points and attempts < max_attempts:
                x = np.random.uniform(minx, maxx)
                y = np.random.uniform(miny, maxy)
                p = Point(x, y)
                if geom.contains(p):
                    points.append(p)
                    generated += 1
                attempts += 1
    
    gdf_points = gpd.GeoDataFrame(geometry=points, crs=gdf_ref.crs)
    if label_field is not None:
        gdf_points[label_field] = label_value
    
    output_path = _get_out_path(output_name)
    gdf_points.to_file(output_path)
    
    return {
        "output_path": output_path,
        "generated_count": len(gdf_points),
        "crs": str(gdf_points.crs)
    }

# @mcp.tool()
def calculate_deforestation_percentage(
    output_name: str,
    buffer_area_csv: str | None = None,
    deforest_area_csv: str | None = None,
    buffer_area_field: str = "area_sum",
    deforest_area_field: str = "area_sum",
    buffer_area_value: float | None = None,
    deforest_area_value: float | None = None,
) -> dict:
    """
    Calculate the deforestation percentage based on total buffer area and deforested area, 
    and save the result as a CSV file.

    Args:
        output_name: Output filename (must end with .csv).
        buffer_area_csv: Path to the CSV file summarizing road buffer area (optional).
        deforest_area_csv: Path to the CSV file summarizing deforested area (optional).
        buffer_area_field: Field name for buffer area in the CSV (default "area_sum").
        deforest_area_field: Field name for deforested area in the CSV (default "area_sum").
        buffer_area_value: Direct value for buffer area (takes precedence if provided).
        deforest_area_value: Direct value for deforested area (takes precedence if provided).

    Returns:
        output_path: Path to the output CSV file containing the deforestation percentage.
    """
    if buffer_area_value is not None and deforest_area_value is not None:
        buffer_area = float(buffer_area_value)
        deforest_area = float(deforest_area_value)
    else:
        ba = pd.read_csv(os.path.abspath(buffer_area_csv))
        da = pd.read_csv(os.path.abspath(deforest_area_csv))
        buffer_area = float(ba[buffer_area_field].iloc[0]) if buffer_area_field in ba.columns else 0.0
        deforest_area = float(da[deforest_area_field].iloc[0]) if deforest_area_field in da.columns else 0.0
    pct = (deforest_area / buffer_area) if buffer_area != 0 else 0.0

    out_csv = _get_out_path(output_name)
    pd.DataFrame({"percentage_deforestation": [pct]}).to_csv(out_csv, index=False)
    return {"output_path": out_csv}

# @mcp.tool()
def calculate_flood_damage_costs(
    buildings_stats_path: str,
    output_name: str,
    mean_field: str = "MEAN_gridcode",
    cost_field: str = "DmgCost",
    area_field: str | None = None,
) -> Dict[str, Any]:
    """
    Calculate damage costs based on a specified mean field and building area, 
    and write the result to a file.

    Damage formula: if mean_field > 1, then
        DmgCost = (0.298 * (log(0.01 * mean_field)) + 1.4502) * 271 * Shape_Area_km2
        Otherwise 0.

    Args:
        buildings_stats_path: Path to building polygon data containing the mean field.
        output_name: Output filename (must end with .shp).
        mean_field: Name of the mean field, default "MEAN_gridcode".
        cost_field: Name of the output damage cost field, default "DmgCost".
        area_field: Name of the area field (if provided, its value is used, assumed to be in square kilometers; if not provided, geometric area in m² is converted to km²).

    Returns:
        output_path: Path to the output shapefile with damage costs.
    """
    sp = os.path.abspath(buildings_stats_path)
    if not os.path.isfile(sp):
        return {"output_path": None}

    gdf = gpd.read_file(sp)
    # Calculate area (assuming coordinate units are meters), convert to square kilometers
    if area_field and area_field in gdf.columns:
        area_km2 = gdf[area_field].astype(float)
    else:
        area_km2 = gdf.geometry.area.astype(float) / 1_000_000.0
    if mean_field not in gdf.columns:
        gdf[mean_field] = 0.0
    mean_depth = gdf[mean_field].astype(float)
    costs = np.zeros(len(gdf), dtype=float)
    mask = mean_depth > 1.0
    costs[mask] = (0.298 * (np.log(0.01 * mean_depth[mask])) + 1.4502) * 271.0 * area_km2[mask]
    gdf[cost_field] = costs

    out_path = _get_out_path(output_name)
    gdf.to_file(out_path)
    return {"output_path": out_path}

# @mcp.tool()
def extract_control_points(
    input_path: str,
    observer_id: int,
    target_id: int,
    id_field: str
) -> Dict[str, Any]:
    """
    Extract observer and target 3D coordinates from a point vector file.

    Args:
        input_path: Path to the 3D point shapefile/GeoJSON.
        observer_id: ID of the observer point.
        target_id: ID of the target point.
        id_field: Field name for IDs. If "FID" is used and the column doesn't exist, the feature index is used.

    Returns:
        Dictionary containing:
            - observer: [x, y, z] coordinates.
            - target: [x, y, z] coordinates.
    """
    import geopandas as gpd
    import os
    
    ip = os.path.abspath(input_path)
    gdf = gpd.read_file(ip)
    
    # Handle ID field case sensitivity
    cols = {c.upper(): c for c in gdf.columns}
    if id_field.upper() in cols:
        actual_id_field = cols[id_field.upper()]
        gdf["_id_"] = gdf[actual_id_field]
    elif id_field.upper() == "FID":
         # GeoPandas uses index as FID usually if not present
         gdf["_id_"] = gdf.index
    else:
        raise ValueError(f"ID field {id_field} not found")

    obs_row = gdf[gdf["_id_"] == observer_id]
    tgt_row = gdf[gdf["_id_"] == target_id]
    
    if obs_row.empty:
        raise ValueError(f"Observer {observer_id} not found")
    if tgt_row.empty:
        raise ValueError(f"Target {target_id} not found")
        
    def get_xyz(geom):
        if hasattr(geom, 'has_z') and geom.has_z:
            return [float(geom.x), float(geom.y), float(geom.z)]
        return [float(geom.x), float(geom.y), 0.0]

    observer = get_xyz(obs_row.iloc[0].geometry)
    target = get_xyz(tgt_row.iloc[0].geometry)
    
    return {"observer": observer, "target": target}

# @mcp.tool()
def extract_obstacle_vertices(
    input_path: str,
    output_name: str
) -> Dict[str, Any]:
    """
    Extract all vertices from 3D polygons as obstacle points.

    Args:
        input_path: Path to the 3D building/polygon vector file.
        output_name: Output JSON filename (must end with .json).

    Returns:
        Dictionary containing:
            - output_path: Path to the saved JSON file.
            - count: Number of points extracted.
    """
    import geopandas as gpd
    import json
    import os
    
    ip = os.path.abspath(input_path)
    # Use fiona engine to avoid WKB type 16 errors with pyogrio
    gdf = gpd.read_file(ip, engine="fiona")
    
    points = []
    
    for geom in gdf.geometry:
        if geom is None:
            continue
        # Support Polygon and MultiPolygon
        if geom.geom_type == 'MultiPolygon':
            geoms = geom.geoms
        else:
            geoms = [geom]
        
        for g in geoms:
            if g.geom_type == 'Polygon':
                # Exterior
                for coord in g.exterior.coords:
                    if len(coord) >= 3:
                        points.append([float(coord[0]), float(coord[1]), float(coord[2])])
                    else:
                        points.append([float(coord[0]), float(coord[1]), 0.0])
                # Interiors
                for interior in g.interiors:
                    for coord in interior.coords:
                        if len(coord) >= 3:
                            points.append([float(coord[0]), float(coord[1]), float(coord[2])])
                        else:
                            points.append([float(coord[0]), float(coord[1]), 0.0])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not output_name.lower().endswith(".json"):
        output_name += ".json"
        
    out_path = _get_out_path(output_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(points, f, indent=2)
        
    return {"output_path": out_path, "count": len(points)}

# @mcp.tool()
def flatten_3d_polygons(
    input_path: str,
    output_name: str
) -> Dict[str, Any]:
    """
    Convert 3D polygon shapefile to 2D by removing Z coordinates.

    Args:
        input_path: Path to 3D polygon shapefile.
        output_name: Output filename (must end with .shp).

    Returns:
        Dictionary containing output path and statistics.
    """
    import fiona
    from shapely.geometry import shape, mapping
    import os
    
    ip = os.path.abspath(input_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)

    def remove_z_from_geometry(geom):
        """Recursively remove Z coordinates from geometry"""
        if geom['type'] == 'Point':
            coords = geom['coordinates']
            if len(coords) >= 3:
                return {'type': 'Point', 'coordinates': (coords[0], coords[1])}
            return geom
        elif geom['type'] == 'LineString':
            coords = geom['coordinates']
            new_coords = [(pt[0], pt[1]) if len(pt) >= 3 else (pt[0], pt[1]) for pt in coords]
            return {'type': 'LineString', 'coordinates': new_coords}
        elif geom['type'] == 'Polygon':
            rings = geom['coordinates']
            # Only keep outer ring, remove holes and sides (if 3D specific logic)
            if rings:
                new_ring = [(pt[0], pt[1]) if len(pt) >= 3 else (pt[0], pt[1]) for pt in rings[0]]
                new_rings = [new_ring]
            else:
                new_rings = []
            return {'type': 'Polygon', 'coordinates': new_rings}
        elif geom['type'] == 'MultiPolygon':
            polys = geom['coordinates']
            new_polys = []
            for poly in polys:
                if poly and len(poly) > 0:
                    new_ring = [(pt[0], pt[1]) if len(pt) >= 3 else (pt[0], pt[1]) for pt in poly[0]]
                    new_polys.append([new_ring])
                else:
                    new_polys.append([])
            return {'type': 'MultiPolygon', 'coordinates': new_polys}
        elif geom['type'] == 'GeometryCollection':
            new_geoms = [remove_z_from_geometry(subgeom) for subgeom in geom['geometries']]
            # Flatten to MultiPolygon if possible
            polygons = []
            for geom_obj in new_geoms:
                if geom_obj['type'] == 'Polygon':
                    polygons.append(geom_obj['coordinates'])
                elif geom_obj['type'] == 'MultiPolygon':
                    polygons.extend(geom_obj['coordinates'])
            if len(polygons) == 1:
                return {'type': 'Polygon', 'coordinates': polygons[0]}
            else:
                return {'type': 'MultiPolygon', 'coordinates': polygons}
        else:
            return geom
    
    try:
        with fiona.open(ip, 'r') as src:
            schema = src.schema.copy()
            crs = src.crs
            driver = src.driver
            
            features = []
            
            for feat in src:
                geom = feat['geometry']
                props = feat['properties']
                
                if geom:
                    new_geom = remove_z_from_geometry(geom)
                    features.append({
                        'geometry': new_geom,
                        'properties': props
                    })
            
            if 'geometry' in schema:
                # Force 2D type if it was 3D
                if schema['geometry'] == '3D Polygon':
                    schema['geometry'] = 'Polygon'
            
            with fiona.open(
                out_path, 'w',
                driver=driver,
                crs=crs,
                schema=schema
            ) as dst:
                for feat in features:
                    dst.write(feat)
        
        return {
            "output_path": out_path,
            "feature_count": len(features)
        }
    except Exception as e:
        raise RuntimeError(f"Failed to convert 3D polygons to 2D: {e}")