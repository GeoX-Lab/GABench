from typing import Optional, Dict, List, Any
import yaml
from pathlib import Path
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import libpysal
import esda
import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.mask import mask
from pykrige.ok import OrdinaryKriging
from scipy.stats import gaussian_kde
from shapely.geometry import Point, Polygon, shape
import json
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
def density_clusters(
    input_path: str,
    min_cluster_size: int,
    output_name: str,
) -> Dict[str, Any]:
    """
    Perform density-based clustering (HDBSCAN) to assign cluster labels.

    Args:
        input_path: Path to input point data (.geojson/.shp).
        min_cluster_size: Minimum cluster size (HDBSCAN parameter).
        output_name: Output filename (must end with .geojson).

    Returns:
        output_path: Path to the clustered vector file with CLUSTER_ID.
    """
    ip = os.path.abspath(input_path)
    gdf = gpd.read_file(ip)
    # Extract coordinates and perform HDBSCAN clustering
    coords = np.vstack([gdf.geometry.x.values, gdf.geometry.y.values]).T
    clusterer = HDBSCAN(min_cluster_size=int(min_cluster_size))
    labels = clusterer.fit_predict(coords)
    # Write cluster labels to new GeoDataFrame and save
    gdf_out = gdf.copy()
    gdf_out["CLUSTER_ID"] = labels

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not output_name.lower().endswith(".geojson"):
        output_name += ".geojson"
    output_path = _get_out_path(output_name)

    gdf_out.to_file(output_path, driver="GeoJSON")
    return {"output_path": output_path}

# @mcp.tool()
def multivariate_clusters(
    candidates_path: str,
    risk_fields: List[str],
    output_name: str,
) -> Dict[str, Any]:
    """
    Perform KMeans clustering on risk fields and write cluster labels.

    Args:
        candidates_path: Path to candidate shapefile which includes risk fields.
        risk_fields: List of risk field names used for clustering.
        output_name: Output filename (must end with .geojson).

    Returns:
        output_path: Path to clustered output with CLUSTER_ID.
    """
    cp = os.path.abspath(candidates_path)
    gdf = gpd.read_file(cp, engine="fiona")
    feats = gdf[risk_fields].copy()
    feats = feats.apply(pd.to_numeric, errors="coerce")
    feats = feats.fillna(feats.median())

    seeds_column = "SEEDS"
    seeds = None
    k = None
    if seeds_column in gdf.columns:
        uniq = sorted([v for v in pd.unique(gdf[seeds_column]) if pd.notna(v)])
        if len(uniq) > 0:
            centers = []
            for v in uniq:
                centers.append(feats[gdf[seeds_column] == v].mean().values)
            seeds = np.vstack(centers)
            k = len(uniq)

    if seeds is not None:
        km = KMeans(n_clusters=k, init=seeds, n_init=1, random_state=42)
        clusters_used = k
    else:
        clusters_used = 5
        km = KMeans(n_clusters=clusters_used, init="k-means++", n_init=10, random_state=42)

    labels = km.fit_predict(feats)
    out = gdf.copy()
    out["CLUSTER_ID"] = labels.astype("int64")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = _get_out_path(output_name)
    gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs).to_file(out_path, driver="GeoJSON")
    return {"output_path": out_path}

# @mcp.tool()
def check_line_of_sight(
    observer: List[float],
    target: List[float],
    obstacles_path: str,
    output_name: str,
    distance_threshold: float = 2.0
) -> Dict[str, Any]:
    """
    Analyze 3D line-of-sight between observer and target, considering building obstacles.

    Args:
        observer: Observer coordinates [x, y, z].
        target: Target coordinates [x, y, z].
        obstacles_path: Path to the JSON file containing obstacle points.
        output_name: Output filename for results (must end with .json).
        distance_threshold: 2D distance threshold to consider an obstacle relevant (meters).

    Returns:
        Dictionary containing visibility status and path to detailed results.
    """
    # Load obstacles
    try:
        with open(obstacles_path, 'r', encoding='utf-8') as f:
            building_pts = json.load(f)
    except Exception as e:
        return {
            "status": "error",
            "visible": False,
            "message": f"Failed to read building points: {e}",
            "output_path": "",
            "statistics": {}
        }

    Ax, Ay, Az = observer
    Bx, By, Bz = target

    if np.hypot(Bx - Ax, By - Ay) < 1e-6:
        stats = {"visible": True, "message": "Observer and target coincide"}
        visible = True
    else:
        # Precompute segment constants
        abx, aby = Bx - Ax, By - Ay
        AB_sq = abx**2 + aby**2
        len_ab = np.sqrt(AB_sq)

        # LOS Check
        blocking_info = None
        for (Px, Py, Pz) in building_pts:
            if np.hypot(Px - Ax, Py - Ay) < 0.5 or np.hypot(Px - Bx, Py - By) < 0.5:
                continue

            # Calculate point_to_segment_distance_2d inline
            apx, apy = Px - Ax, Py - Ay
            bpx, bpy = Px - Bx, Py - By
            
            if abx * apx + aby * apy <= 0:
                dist_2d = np.hypot(apx, apy)
            elif abx * bpx + aby * bpy >= 0:
                dist_2d = np.hypot(bpx, bpy)
            else:
                cross = abs(abx * apy - aby * apx)
                dist_2d = cross / len_ab if len_ab > 0 else np.hypot(apx, apy)

            if dist_2d > distance_threshold:
                continue

            if AB_sq == 0:
                continue
            t = ((Px - Ax)*(Bx - Ax) + (Py - Ay)*(By - Ay)) / AB_sq
            t = np.clip(t, 0.0, 1.0)
            sight_z = Az + t * (Bz - Az)

            if Pz >= sight_z - 1e-3:
                blocking_info = {
                    "blocking_point": [float(Px), float(Py), float(Pz)],
                    "sight_z_at_block": float(sight_z),
                    "block_z": float(Pz),
                    "distance_to_line_m": float(dist_2d)
                }
                break

        visible = blocking_info is None
        stats = {"visible": visible}
        if not visible:
            stats.update(blocking_info)

    # Save result to OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not output_name.lower().endswith(".json"):
        output_name += ".json"
    out_path = _get_out_path(output_name)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
        
    return {
        "visible": visible,
        "output_path": out_path,
    }

# @mcp.tool()
def local_moran(
    input_path: str,
    value_field: str,
    output_name: str,
    p_threshold: float = 0.05,
    group_field: Optional[str] = None,
    polygons_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute Local Moran's I on polygons using a numeric field; optionally aggregate by a grouping key.

    Args:
        input_path: Path to the vector dataset (GeoJSON/Shapefile).
        value_field: Numeric field used for Local Moran's I.
        output_name: Output filename (must end with .geojson).
        p_threshold: p-value threshold for significance.
        group_field: Optional grouping key to aggregate values and geometries. When None, duplicates are collapsed by geometry.
        polygons_path: Optional path to a polygon dataset for aggregation. When None, no aggregation is performed.

    Returns:
        output_path: Path to the GeoJSON with classification labels.
        counts: Mapping of label to counts.
        column: Name of the classification column ('cl').
    """
    jp = os.path.abspath(input_path)
    if not os.path.isfile(jp):
        raise FileNotFoundError(f"Joined file not found: {jp}")
    try:
        j = gpd.read_file(jp)
    except Exception:
        from shapely.geometry import shape as _shape
        import json as _json
        if jp.lower().endswith(".geojson") or jp.lower().endswith(".json"):
            with open(jp, "r", encoding="utf-8") as f:
                gj = _json.load(f)
            feats = gj.get("features") or []
            props = [feat.get("properties") or {} for feat in feats]
            geoms = [_shape(feat.get("geometry")) for feat in feats]
            j = gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")
        else:
            raise
    if value_field not in j.columns:
        raise ValueError(f"Field not found: {value_field}")
    j[value_field] = pd.to_numeric(j[value_field], errors="coerce")
    if group_field and polygons_path:
        if group_field not in j.columns:
            raise ValueError(f"Group field not found: {group_field}")
        polys_abs = os.path.abspath(polygons_path)
        polys = gpd.read_file(polys_abs)
        if polys.crs is None and j.crs is not None:
            polys = polys.set_crs(j.crs, allow_override=True)
        if j.crs is None and polys.crs is not None:
            j = j.set_crs(polys.crs, allow_override=True)
        if polys.crs != j.crs:
            j = j.to_crs(polys.crs)
        gb = j[value_field].groupby(j[group_field]).mean()
        df = polys.copy()
        df["median_pri"] = df[group_field].map(gb)
        df["median_pri"] = df["median_pri"].fillna(df["median_pri"].mean())
    else:
        if "index_left" not in j.columns:
            j = j.reset_index().rename(columns={"index": "index_left"})
        gb = j[value_field].groupby(j["index_left"]).mean()
        left_unique = j.drop_duplicates("index_left")[ ["index_left", "geometry"] ].set_index("index_left")
        left_unique["median_pri"] = gb
        if group_field and group_field in j.columns:
            left_groups = j.drop_duplicates("index_left")[ ["index_left", group_field] ].set_index("index_left")
            g_by_group = j[value_field].groupby(j[group_field]).mean()
            left_unique["median_pri"] = left_groups[group_field].map(g_by_group)
        left_unique["median_pri"] = left_unique["median_pri"].fillna(left_unique["median_pri"].mean())
        df = left_unique
    wq = libpysal.weights.Queen.from_dataframe(df)
    wq.transform = 'r'
    y = df["median_pri"].values
    li = esda.moran.Moran_Local(y, wq)
    sig = 1 * (li.p_sim < p_threshold)
    labels = []
    nonsig_label = "0 ns"
    for i in range(len(df)):
        if not sig[i]:
            labels.append(nonsig_label)
        else:
            qi = int(li.q[i])
            if qi == 1:
                labels.append('1 hot spot')
            elif qi == 2:
                labels.append('2 doughnut')
            elif qi == 3:
                labels.append('3 cold spot')
            elif qi == 4:
                labels.append('4 diamond')
            else:
                labels.append(nonsig_label)
    df = df.copy()
    df["cl"] = labels
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    df[["geometry", "cl"]].to_file(out_path, driver="GeoJSON")
    counts = {k: int(v) for k, v in df["cl"].value_counts().to_dict().items()}
    return {"output_path": out_path, "counts": counts, "column": "cl"}

# @mcp.tool()
def classify_hotspots(
    input_vector_path: str,
    field: str,
    output_name: str,
) -> Dict[str, Any]:
    """
    Classify hotspots based on standard scores of a specified field and save as GeoJSON.

    Note: This is a simplified alternative implementation using global mean and standard deviation to calculate Z-scores.

    Args:
        input_vector_path: Path to input vector file.
        field: Name of the numeric field for classification.
        output_name: Output filename (must end with .geojson).

    Returns:
        output_path: Output file path.
        category_counts: Counts per category.
    """
    ip = os.path.abspath(input_vector_path)
    gdf = gpd.read_file(ip)
    x = gdf.get(field)
    if x is None:
        raise ValueError("field not found")
    x = gdf[field]
    x = x.astype(float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if sd == 0:
        z = np.zeros(len(x))
    else:
        z = (x - mu) / sd
    gdf["hspot_type"] = "Not Significant"
    gdf.loc[z >= 2.0, "hspot_type"] = "Very High Hot Spot"
    gdf.loc[(z >= 1.0) & (z < 2.0), "hspot_type"] = "Hot Spot"
    gdf.loc[z <= -1.0, "hspot_type"] = "Cold Spot"
    out_path = _get_out_path(output_name)
    gdf.to_file(out_path, driver="GeoJSON")
    vc = dict(gdf["hspot_type"].value_counts())
    return {"output_path": out_path, "category_counts": vc}

# @mcp.tool()
def calculate_bandwidth(input_path: str) -> Dict[str, Any]:
    """
    Calculate the bandwidth for Kernel Density Estimation using Scott's Rule.

    Args:
        input_path: Path to input point data (.geojson/.shp).

    Returns:
        dict: Dictionary containing `bandwidth` and `data_points`.
    """
    ip = os.path.abspath(input_path)
    gdf = gpd.read_file(ip)
    
    # Extract coordinates (N, 2) array
    coords = np.vstack([gdf.geometry.x.values, gdf.geometry.y.values]).T
    
    # Scott's Rule
    n = coords.shape[0]
    d = coords.shape[1]
    
    # Calculate standard deviation of coordinates
    sigma = np.std(coords)
    
    # Scott's Rule formula: n^(-1/(d+4)) * sigma
    scotts_bandwidth = n ** (-1.0 / (d + 4)) * sigma
    
    return {
        "bandwidth": float(scotts_bandwidth),
        "data_points": int(n)
    }


# @mcp.tool()
def compute_kernel_density(input_path: str, bandwidth: float, grid_res: int, output_name: str) -> dict:
    """
    Compute Kernel Density Estimation (KDE) on vector points and save as a raster.

    Args:
        input_path: Path to input point vector data (.geojson/.shp).
        bandwidth: KDE bandwidth.
        grid_res: Grid resolution (number of samples in x and y directions).
        output_name: Output filename (must end with .tif).

    Returns:
        dict: Dictionary containing `output_path`.
    """
    ip = os.path.abspath(input_path)
    gdf = gpd.read_file(ip)
    # 依据数据范围构造规则网格
    x_min, y_min, x_max, y_max = gdf.total_bounds
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res))
    # 组装点坐标用于 KDE 拟合
    coords = np.vstack([gdf.geometry.x.values, gdf.geometry.y.values]).T
    kde = KernelDensity(bandwidth=float(bandwidth), metric="euclidean")
    kde.fit(coords)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()]).T
    # 计算概率密度并重塑为网格
    z = np.exp(kde.score_samples(grid_coords)).reshape(xx.shape).astype(np.float32)
    # 按边界生成仿射变换，使栅格空间定位与数据一致
    transform = from_bounds(x_min, y_min, x_max, y_max, z.shape[1], z.shape[0])
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "crs": gdf.crs,
        "transform": transform,
        "height": z.shape[0],
        "width": z.shape[1],
        "nodata": np.nan,
    }
    
    if not output_name.lower().endswith(".tif"):
        output_name += ".tif"
        
    output_path = _get_out_path(output_name)
    
    # 写出密度栅格供后续叠加展示
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(z, 1)
    return {"output_path": output_path}

# @mcp.tool()
def local_k_function(flow_path: str, output_name: str) -> Dict[str, Any]:
    """
    Compute the local K-function matrix for geographical flows.

    Args:
        flow_path: Path to the flow CSV with columns `x_o,y_o,x_d,y_d`.
        output_name: Output filename (must end with .csv).

    Returns:
        output_path: Path to the written local K matrix CSV.
    """
    df = pd.read_csv(flow_path)
    flow = df[["x_o", "y_o", "x_d", "y_d"]].values
    scale = np.linspace(0.01, 1.2, 100)
    n = flow.shape[0]
    localK = np.zeros((n, len(scale)))
    for i in range(n):
        ref = flow[i]
        ox_all, oy_all = flow[:, 0], flow[:, 1]
        dx_all, dy_all = flow[:, 2], flow[:, 3]
        dist_o = np.sqrt((ref[0] - ox_all) ** 2 + (ref[1] - oy_all) ** 2)
        dist_d = np.sqrt((ref[2] - dx_all) ** 2 + (ref[3] - dy_all) ** 2)
        dist = np.maximum(dist_o, dist_d)
        sorted_indices = np.argsort(dist)
        kvalue = 0.0
        local_k_row = np.zeros(len(scale))
        for j, s in enumerate(scale):
            if j == 0:
                indices = np.where(dist[sorted_indices] <= s)[0]
            else:
                indices = np.where((dist[sorted_indices] > scale[j - 1]) & (dist[sorted_indices] <= s))[0]
            if len(indices) >= 1:
                kvalue += float(len(indices))
            local_k_row[j] = kvalue
        localK[i, :] = local_k_row
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    localK_path = _get_out_path(output_name)
    np.savetxt(localK_path, localK, delimiter=",")
    return {
        "output_path": localK_path,
    }

# @mcp.tool()
def ordinary_kriging(
    points_geojson_path: str,
    value_property: str,
    grid_bounds: list[float],
    nx: int,
    ny: int,
    variogram_model: str,
    output_name: str
) -> Dict[str, Any]:
    """
    Perform ordinary kriging interpolation on point data from a GeoJSON file for a specified attribute,
    generating a regular grid with prediction and variance within given spatial bounds and resolution.
    The result is saved to a GeoTIFF file.

    Args:
        points_geojson_path: The path to the input GeoJSON file containing point data.
        value_property: The property name in the point features to use for interpolation.
        grid_bounds: A list [minx, miny, maxx, maxy] defining the spatial bounds for the grid.
        nx, ny: Number of grid points in the x and y directions, respectively.
        variogram_model: The variogram model to use for kriging interpolation, e.g., 'spherical', 'gaussian', etc.
        output_name: Output filename (should end with .tif).

    Returns:
        output_path: The path to the output GeoTIFF file.
    """
    if not os.path.isfile(points_geojson_path):
        raise FileNotFoundError(f"Point GeoJSON file not found: {points_geojson_path}")

    # Load point data
    gdf = gpd.read_file(points_geojson_path)
    
    # Extract coordinates and values
    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    z = gdf[value_property].values

    # Generate regular grid
    minx, miny, maxx, maxy = grid_bounds
    x_grid = np.linspace(minx, maxx, nx)
    y_grid = np.linspace(miny, maxy, ny)

    # Perform ordinary kriging
    ok = OrdinaryKriging(
        x, y, z,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False
    )
    # execute('grid') returns z, ss. z is (y, x)
    # y coordinates in y_grid are increasing (min to max).
    # so z_pred[0] corresponds to miny (bottom).
    z_pred, z_var = ok.execute('grid', x_grid, y_grid)

    # Rasterio expects data to be top-down (row 0 is maxy).
    # So we need to flip the array vertically.
    z_pred = np.flipud(z_pred)
    z_var = np.flipud(z_var)

    # Define transform
    # from_bounds(west, south, east, north, width, height)
    transform = from_bounds(minx, miny, maxx, maxy, nx, ny)

    # Write to GeoTIFF
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not output_name.lower().endswith('.tif'):
        output_name = os.path.splitext(output_name)[0] + '.tif'
    output_path = _get_out_path(output_name)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=ny,
        width=nx,
        count=2,
        dtype=z_pred.dtype,
        crs=gdf.crs,
        transform=transform,
    ) as dst:
        dst.write(z_pred, 1)
        dst.write(z_var, 2)
        dst.set_band_description(1, 'prediction')
        dst.set_band_description(2, 'variance')

    return {"output_path": output_path}


# @mcp.tool()
def kde_interpolate(
    points_path: str,
    polygons_path: str,
    output_name: str,
    value_field: str = "MeasureValue",
) -> Dict[str, Any]:
    """
    Perform weighted KDE interpolation within polygon bounds and write a raster.

    Args:
        points_path: Path to point features.
        polygons_path: Path to polygon features that define the interpolation extent and mask.
        output_name: Output filename (must end with .tif).
        value_field: Name of the numeric column used as KDE weights.

    Returns:
        output_path: Path to the generated GeoTIFF raster.
    """
    if not (os.path.isfile(points_path) and os.path.isfile(polygons_path)):
        raise FileNotFoundError("Input files not found")

    points_gdf = gpd.read_file(points_path)
    polygons_gdf = gpd.read_file(polygons_path)

    if value_field not in points_gdf.columns:
        raise ValueError(f"Value field not found: {value_field}")

    coords = np.array([(geom.x, geom.y) for geom in points_gdf.geometry])
    values = points_gdf[value_field].to_numpy()

    bw = 0.6
    resolution = 150
    kde = gaussian_kde(coords.T, weights=values, bw_method=bw)

    minx, miny, maxx, maxy = polygons_gdf.total_bounds
    x, y = np.mgrid[minx:maxx:complex(resolution), miny:maxy:complex(resolution)]
    positions = np.vstack([x.ravel(), y.ravel()])
    z = kde(positions).reshape(x.shape)

    union = polygons_gdf.unary_union
    mask_flat = np.array([union.contains(Point(px, py)) for px, py in zip(x.ravel(), y.ravel())])
    mask = mask_flat.reshape(x.shape)
    z_masked = np.where(mask, z, np.nan)

    nodata = -9999.0
    data = np.where(np.isnan(z_masked.T), nodata, z_masked.T).astype(np.float32)
    data = np.flipud(data)
    transform = from_bounds(minx, miny, maxx, maxy, data.shape[1], data.shape[0])

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    raster_path = _get_out_path(output_name)
    profile = {
        "driver": "GTiff",
        "height": int(data.shape[0]),
        "width": int(data.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": polygons_gdf.crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(data, 1)

    return {"output_path": raster_path}

# @mcp.tool()
def ols_regression(input_vector_path: str, y_field: str, x_field: str, output_name: str) -> Dict[str, Any]:
    """
    Perform OLS linear regression.

    Note: This is a simplified non-ArcPy alternative to GWR (Geographically Weighted Regression), using global OLS fitting.

    Args:
        input_vector_path: Path to input vector file.
        y_field: Dependent variable field name.
        x_field: Independent variable field name.
        output_name: Output filename (must end with .geojson).

    Returns:
        output_path: Path to the generated GeoJSON file.
        beta: Regression coefficients.
    """
    ip = os.path.abspath(input_vector_path)
    gdf = gpd.read_file(ip)
    y = gdf.get(y_field)
    x = gdf.get(x_field)
    if y is None or x is None:
        raise ValueError("field not found")
    y = np.asarray(gdf[y_field], dtype=float)
    X = np.asarray(gdf[x_field], dtype=float)
    X = np.column_stack([np.ones_like(X), X])
    XtX = X.T @ X
    XtY = X.T @ y
    beta = np.linalg.pinv(XtX) @ XtY
    yhat = X @ beta
    resid = y - yhat
    gdf["predicted_y"] = yhat
    gdf["residual"] = resid
    out_path = _get_out_path(output_name)
    gdf.to_file(out_path, driver="GeoJSON")
    return {"output_path": out_path, "beta": [float(b) for b in beta]}

# @mcp.tool()
def search_similarity(
    candidates_path: str,
    targets_path: str,
    attrs: List[str],
    output_name: str,
    id_field: str = "ID",
) -> Dict[str, Any]:
    """
    Compute similarity rank of candidates to targets based on selected attributes.

    Args:
        candidates_path: Path to candidate shapefile.
        targets_path: Path to target shapefile.
        attrs: List of attribute names to use for similarity.
        output_name: Output filename (must end with .geojson).
        id_field: Identifier field for candidates (default: "ID").

    Returns:
        output_path: Path to similarity result file.
    """
    cp = os.path.abspath(candidates_path)
    tp = os.path.abspath(targets_path)
    candidates = gpd.read_file(cp, engine="fiona")
    targets = gpd.read_file(tp, engine="fiona")

    missing = [c for c in attrs if c not in candidates.columns or c not in targets.columns]
    if missing:
        raise ValueError(f"Attributes not found in both datasets: {missing}")

    Xc = candidates[attrs].apply(pd.to_numeric, errors="coerce")
    Xt = targets[attrs].apply(pd.to_numeric, errors="coerce")
    Xall = pd.concat([Xc, Xt], axis=0)
    med = Xall.median()
    Xc = Xc.fillna(med)
    Xt = Xt.fillna(med)

    sc = StandardScaler()

    if sc is not None:
        sc.fit(pd.concat([Xc, Xt], axis=0))
        C = sc.transform(Xc)
        T = sc.transform(Xt)
    else:
        C = Xc.values
        T = Xt.values

    centroid = T.mean(axis=0)
    dists = np.linalg.norm(C - centroid, axis=1)
    ranks = pd.Series(dists).rank(method="first", ascending=True).astype(int)

    res = candidates[[id_field, "geometry"]].copy()
    res["SIMRANK"] = ranks.values
    gres = gpd.GeoDataFrame(res, geometry="geometry", crs=candidates.crs)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = _get_out_path(output_name)
    gres.to_file(out_path, driver="GeoJSON")
    return {"output_path": out_path}

# @mcp.tool()
def sample_slope_aspect_at_points(
    points_geojson_path: str,
    slope_raster_path: str,
    aspect_raster_path: str,
    output_name: str,
    group_field_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sample slope and aspect raster values at point locations and save to a CSV.

    Args:
        points_geojson_path: Path to the input points file (GeoJSON, Shapefile, etc.).
        slope_raster_path: Path to the slope raster (GeoTIFF).
        aspect_raster_path: Path to the aspect raster (GeoTIFF).
        output_name: Output filename (must end with .csv).
        group_field_name: Optional attribute name from the points to include as a grouping column in the output table.

    Returns:
        output_path: Path to the CSV written.
    """
    if not os.path.isfile(points_geojson_path):
        raise FileNotFoundError(f"Points GeoJSON not found: {points_geojson_path}")
    if not os.path.isfile(slope_raster_path):
        raise FileNotFoundError(f"Slope raster not found: {slope_raster_path}")
    if not os.path.isfile(aspect_raster_path):
        raise FileNotFoundError(f"Aspect raster not found: {aspect_raster_path}")

    gdf = gpd.read_file(points_geojson_path)

    with rasterio.open(slope_raster_path) as slope_src, rasterio.open(aspect_raster_path) as aspect_src:
        gdf_crs = getattr(gdf, "crs", None)
        if gdf_crs and gdf_crs != slope_src.crs:
            raise ValueError(f"CRS mismatch between points and slope raster")
        if gdf_crs and gdf_crs != aspect_src.crs:
            raise ValueError(f"CRS mismatch between points and aspect raster")

        if slope_src.count != 1:
            raise ValueError(f"Slope raster must be single-band; got {slope_src.count}")
        if aspect_src.count != 1:
            raise ValueError(f"Aspect raster must be single-band; got {aspect_src.count}")
        slope_arr = slope_src.read(1)
        aspect_arr = aspect_src.read(1)
        height, width = slope_arr.shape
        xmin, ymin, xmax, ymax = gdf.total_bounds
        xres = (xmax - xmin) / width
        yres = (ymax - ymin) / height
        transform = from_origin(xmin, ymax, xres, yres)
        slope_vals, aspect_vals = [], []
        for geom in gdf.geometry:
            x, y = geom.x, geom.y
            col = int((x - transform.c) / transform.a)
            row = int((y - transform.f) / transform.e)
            if 0 <= row < height and 0 <= col < width:
                slope_vals.append(float(slope_arr[row, col]))
                aspect_vals.append(float(aspect_arr[row, col]))
            else:
                slope_vals.append(float('nan'))
                aspect_vals.append(float('nan'))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)

    df = pd.DataFrame({
        "point_index": list(range(len(slope_vals))),
        "slope": slope_vals,
        "aspect": aspect_vals
    })
    if group_field_name and group_field_name in gdf.columns:
        df[group_field_name] = gdf[group_field_name].astype(str).values

    df.to_csv(out_path, index=False)

    return {
        "output_path": out_path,
    }

# @mcp.tool()
def calculate_group_statistics(
    csv_path: str,
    output_name: str,
    group_field_name: str,
    metric: str = "mean"
) -> Dict[str, Any]:
    """
    Compute grouped statistics from a sampled table and save to CSV.

    Args:
        csv_path: Path to a CSV file containing numeric measurement columns and an optional grouping column.
        output_name: Output filename (must end with .csv).
        group_field_name: Column name used to group rows.
        metric: Single statistic to compute; one of "mean", "min", "max", "median", "std", "count".

    Returns:
        output_path: Path to the output CSV file.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if group_field_name not in df.columns:
        raise ValueError(f"Group field '{group_field_name}' not found in CSV file")

    allowed = {"mean", "min", "max", "median", "std", "count"}
    metric = (metric or "mean").strip().lower()
    if metric not in allowed:
        raise ValueError(f"Unsupported metric '{metric}'. Allowed: {sorted(allowed)}")

    agg = {"slope": [metric], "aspect": [metric]}
    stats = df.groupby(group_field_name).agg(agg).reset_index()

    flat_cols = []
    for c in stats.columns:
        if isinstance(c, tuple):
            flat_cols.append(f"{c[0]}_{c[1]}")
        else:
            flat_cols.append(c)
    stats.columns = flat_cols

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = _get_out_path(output_name)
    stats.to_csv(out_path, index=False)

    return {
        "output_path": out_path,
    }

# @mcp.tool()
def global_l_function(
    localK_path: str,
    o_area_path: str,
    d_area_path: str,
    output_name: str
) -> Dict[str, Any]:
    """
    Compute the global K/L functions from a precomputed local K matrix.

    Args:
        localK_path: Path to the local K matrix CSV.
        o_area_path: Path to origin area CSV with columns `x,y`.
        d_area_path: Path to destination area CSV with columns `x,y`.
        output_name: Output filename (must end with .csv).

    Returns:
        output_path: Path to the written global L CSV.
    """
    out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    localK = np.loadtxt(localK_path, delimiter=",")
    o_df = pd.read_csv(o_area_path)
    d_df = pd.read_csv(d_area_path)
    o_area_val = float(Polygon(o_df[["x", "y"]].values).area)
    d_area_val = float(Polygon(d_df[["x", "y"]].values).area)
    n = int(localK.shape[0])
    lambda_value = n / (o_area_val * d_area_val)
    num_scales = int(localK.shape[1])
    scale = np.linspace(0.01, 1.2, num_scales)
    localK = localK / lambda_value
    K = np.column_stack((scale, np.mean(localK, axis=0)))
    L = K.copy()
    L[:, 1] = (L[:, 1] / (np.pi ** 2)) ** 0.25 - scale
    L_path = _get_out_path(output_name, out_dir)
    np.savetxt(L_path, L, delimiter=",", header="r,L", comments="")
    return {
        "output_path": L_path,
    }

# @mcp.tool()
def join_similarity_attributes(
    candidates_path: str,
    simrank_path: str,
    output_name: str,
    id_field: str = "ID",
    label: str = "risk",
) -> Dict[str, Any]:
    """
    Join similarity result back to candidates and compute a risk score.

    Args:
        candidates_path: Path to candidate shapefile.
        simrank_path: Path to similarity result shapefile containing SIMRANK.
        output_name: Output filename (must end with .geojson).
        id_field: Identifier field used for join.
        label: Label used to name risk column (e.g., "trans", "susc").

    Returns:
        output_path: Path to joined candidate file with <label>_risk.
    """
    cp = os.path.abspath(candidates_path)
    sp = os.path.abspath(simrank_path)
    candidates = gpd.read_file(cp, engine="fiona")
    simrank = gpd.read_file(sp, engine="fiona")[[id_field, "SIMRANK"]]
    merged = candidates.merge(simrank.rename(columns={"SIMRANK": f"SIMRANK_{label}"}), on=id_field, how="left")
    n = len(candidates)
    base = int(n)
    merged[f"{label}_risk"] = (base - merged[f"SIMRANK_{label}"]).astype("int64")
    merged = merged.drop(columns=[f"SIMRANK_{label}"])

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = _get_out_path(output_name)
    gpd.GeoDataFrame(merged, geometry="geometry", crs=candidates.crs).to_file(out_path, driver="GeoJSON")
    return {"output_path": out_path}

# @mcp.tool()
def calculate_area_above_threshold(
    parcels_path: str,
    raster_path: str,
    output_name: str,
    parcel_id_field: str = "Parcel_ID",
    threshold: float = 0.0,
) -> dict:
    """
    Retain pixels with values > threshold within parcel boundaries, sum their area per parcel, and output a CSV.

    Args:
        parcels_path (str): Path to parcel vector file (.shp/.geojson), must match raster CRS.
        raster_path (str): Path to raster file (.tif).
        output_name (str): Output CSV filename (must end with .csv).
        parcel_id_field (str): Parcel unique identifier field, default "Parcel_ID".
        threshold (float): Pixel retention threshold (retain if value > threshold).

    Returns:
        output_path: Path to the output CSV file.
    """
    pp = os.path.abspath(parcels_path)
    rp = os.path.abspath(raster_path)

    gdf = gpd.read_file(pp)
    gdf = gdf.copy()
    ds = rasterio.open(rp)
    transform = ds.transform
    pixel_area = abs(transform.a) * abs(transform.e)
    rb = ds.bounds
    rb_box = box(rb.left, rb.bottom, rb.right, rb.top)

    records = []
    from shapely import make_valid
    for idx, row in gdf.iterrows():
        geom0 = row.geometry
        if geom0 is None or geom0.is_empty:
            continue
        geom0 = make_valid(geom0)
        if geom0 is None or geom0.is_empty:
            continue
        gt = getattr(geom0, "geom_type", None)
        if gt not in ("Polygon", "MultiPolygon"):
            continue
        if not geom0.intersects(rb_box):
            continue
        geom = [geom0]
        # 裁剪出该地块对应的栅格窗口
        data, _ = mask(ds, geom, crop=True)
        arr = data[0]
        # 有效像元
        valid = np.isfinite(arr)
        # 判定不透水像元（值 > 阈值）
        imperv = (arr > threshold) & valid
        count = int(imperv.sum())
        area = float(count) * float(pixel_area)
        pid = row.get(parcel_id_field)
        records.append({parcel_id_field: pid, "impervious_area": area})

    ds.close()

    out_csv = _get_out_path(output_name)
    import pandas as pd
    df_out = pd.DataFrame(records, columns=[parcel_id_field, "impervious_area"])
    df_out.to_csv(out_csv, index=False)
    return {"output_path": out_csv}