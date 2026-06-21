from typing import Optional, Dict, List, Any
import yaml
from pathlib import Path
import os
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio import features
from shapely.geometry import shape, LineString, MultiLineString, Point
import osmnx as ox
import networkx as nx
from pyproj import Geod
import pickle
import math
import json
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
def calculate_travel_time(
    speed_path: str,
    densified_paths_path: str,
    output_name: str,
) -> Dict[str, Any]:
    """
    Derive travel times along densified paths by intersecting with a speed raster.

    Args:
        speed_path: Path to speed raster GeoTIFF.
        densified_paths_path: Path to densified path lines (Shapefile/GeoJSON).
        output_name: Output filename (must end with .shp).

    Returns:
        output_path: Path to the segment shapefile with travel time.
        segments_count: Number of segments written.
        routes_count: Number of routes summarized.
    """
    sp = os.path.abspath(speed_path)
    dp = os.path.abspath(densified_paths_path)
    if not os.path.isfile(sp):
        raise FileNotFoundError(f"Speed raster not found: {sp}")
    if not os.path.isfile(dp):
        raise FileNotFoundError(f"Paths file not found: {dp}")

    gdf_paths = gpd.read_file(dp, engine="fiona")

    with rasterio.open(sp) as src:
        # Ensure CRS alignment: reproject paths to raster CRS if needed
        if gdf_paths.crs != src.crs:
            gdf_paths = gdf_paths.to_crs(src.crs)

        shapes = [geom.__geo_interface__ for geom in gdf_paths.geometry]
        out_image, out_transform = rasterio.mask.mask(
            src, shapes, crop=True, nodata=0, filled=True, all_touched=True
        )

    # Quantize speeds to integer m/s to reduce polygon variety
    arr = np.floor(out_image[0]).astype(np.int32)
    polys, vals = [], []
    for geom, val in features.shapes(arr, mask=arr > 0, transform=out_transform):
        iv = int(val)
        if iv > 0:
            polys.append(shape(geom))
            vals.append(iv)

    speed_polys = gpd.GeoDataFrame({"gridcode": vals}, geometry=polys, crs=gdf_paths.crs)

    segments = []
    if not speed_polys.empty:
        sindex = speed_polys.sindex
        for idx, row in gdf_paths.iterrows():
            route_val = row.get("Route", idx)
            line = row.geometry
            possible_matches_index = list(sindex.intersection(line.bounds))
            possible_matches = speed_polys.iloc[possible_matches_index]
            possible_matches = possible_matches[possible_matches.intersects(line)]

            for _, poly_row in possible_matches.iterrows():
                gridcode = int(poly_row.gridcode)
                inter = line.intersection(poly_row.geometry)
                if inter.is_empty:
                    continue
                geoms = (
                    [inter]
                    if isinstance(inter, LineString)
                    else list(inter.geoms)
                    if isinstance(inter, MultiLineString)
                    else []
                )
                for part in geoms:
                    if part.length > 0 and gridcode > 0:
                        # part.length assumed meters (paths reprojected to raster CRS)
                        time_h = (part.length / gridcode) / 3600.0
                        segments.append(
                            {
                                "Route": route_val,
                                "gridcode": gridcode,
                                "TravelTimeHours": time_h,
                                "geometry": part,
                            }
                        )

    gdf_segments = gpd.GeoDataFrame(segments, crs=gdf_paths.crs)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    

    routes_count = 0
    if not gdf_segments.empty:
        gdf_save = gdf_segments.rename(columns={"TravelTimeHours": "TravelTime"})
        gdf_save.to_file(out_path)
        try:
            routes_count = int(gdf_segments["Route"].nunique())
        except Exception:
            routes_count = 0

    return {
        "output_path": out_path,
        "segments_count": int(len(gdf_segments)),
        "routes_count": routes_count,
    }

# @mcp.tool()
def calculate_speed(
    depth_path: str,
    output_name: str,
    band: int = 1,
) -> Dict[str, Any]:
    """
    Compute speed raster as sqrt(g * depth) from a depth GeoTIFF and save.

    Args:
        depth_path: Path to the input depth raster (.tif).
        output_name: Output filename (must end with .tif).
        band: Band index to compute from (default: 1).

    Returns:
        output_path: Path to the output speed raster TIFF file.
    """
    dp = os.path.abspath(depth_path)
    if not os.path.isfile(dp):
        raise FileNotFoundError(f"Raster file not found: {dp}")

    with rasterio.open(dp) as src:
        if not (1 <= int(band) <= src.count):
            raise ValueError(f"Invalid band index: {band}. Raster has {src.count} band(s).")
        depth = src.read(int(band)).astype(np.float64)
        nd = src.nodata if getattr(src, "nodata", None) is not None else src.profile.get("nodata")
        if nd is not None:
            inv = np.isnan(depth) if isinstance(nd, float) and np.isnan(nd) else (depth == nd)
            depth = np.where(inv, np.nan, depth)

        speed = np.full_like(depth, np.nan, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            speed = np.sqrt(9.80665 * depth)

        profile = src.profile.copy()
        profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(speed.astype(np.float32), 1)

    return {"output_path": output_path}

# @mcp.tool()
def find_nearest_node(graph_path: str, lat: float, lon: float) -> Dict[str, Any]:
    """
    Find the nearest graph node to a given latitude/longitude.

    Args:
        graph_path: Path to the GraphML file.
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        node_id: Nearest node identifier.
        lon: Node longitude.
        lat: Node latitude.
    """
    gp = os.path.abspath(graph_path)
    if not os.path.isfile(gp):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
        
    G = ox.load_graphml(gp)
    node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
    
    x = G.nodes[node].get("x")
    y = G.nodes[node].get("y")
    
    return {
        "node_id": int(node),
        "lon": float(x) if x is not None else None,
        "lat": float(y) if y is not None else None
    }

# @mcp.tool()
def annotate_edges_with_attributes(
    graph_path: str,
    output_name: str,
    default_speed_kmh: float,
    extra_annotations: Optional[Dict[str, Any]] = None,
    expressions: Optional[Dict[str, str]] = None,
    highway_speed_defaults: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Annotate edges with speed and travel time, plus optional custom attributes.

    Args:
        graph_path: Path to the input GraphML.
        output_name: Output GraphML filename (must end with .graphml).
        default_speed_kmh: Fallback speed (km/h) when maxspeed/highway is missing.
        extra_annotations: Key-value pairs to add to every edge.
        expressions: Mapping of field name to safe expression evaluated per edge.
        highway_speed_defaults: Optional mapping of highway types to default speeds (km/h).
            If None, defaults to:
            motorway: 100, trunk: 80, primary: 60, secondary: 50,
            tertiary: 40, residential: 30, living_street: 10, service: 20

    Returns:
        output_path: Path to the saved GraphML.
    """
    if not output_name.lower().endswith(".graphml"):
        raise ValueError("output_name must end with .graphml")

    gp = os.path.abspath(graph_path)
    if not os.path.isfile(gp):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)
    
    G = ox.load_graphml(gp)
    
    def parse_maxspeed(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, list):
            for v in val:
                s = parse_maxspeed(v)
                if s is not None:
                    return s
            return None
        s = str(val).lower()
        num = ""
        for ch in s:
            if ch.isdigit() or ch == ".":
                num += ch
        if not num:
            return None
        spd = float(num)
        if "mph" in s:
            spd = spd * 1.60934
        return spd
        
    hw_speed = highway_speed_defaults or {
        "motorway": 100.0,
        "trunk": 80.0,
        "primary": 60.0,
        "secondary": 50.0,
        "tertiary": 40.0,
        "residential": 30.0,
        "living_street": 10.0,
        "service": 20.0,
    }
    
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get("length")
        maxspeed = data.get("maxspeed")
        highway = data.get("highway")
        spd_kmh = parse_maxspeed(maxspeed)
        if spd_kmh is None and isinstance(highway, list) and highway:
            spd_kmh = hw_speed.get(str(highway[0]).lower())
        if spd_kmh is None and isinstance(highway, str):
            spd_kmh = hw_speed.get(highway.lower())
        if spd_kmh is None:
            spd_kmh = default_speed_kmh
        data["speed_kmh"] = float(spd_kmh)
        
        if length is None:
            data["travel_time_s"] = None
        else:
            spd_mps = float(spd_kmh) / 3.6
            if spd_mps <= 0:
                data["travel_time_s"] = None
            else:
                data["travel_time_s"] = float(length) / spd_mps
        
        if extra_annotations:
            for key, val in extra_annotations.items():
                data[key] = val
        
        if expressions:
            # build locals from edge attrs, flatten simple lists
            locals_env = {}
            for kk, vv in data.items():
                locals_env[kk] = vv[0] if isinstance(vv, list) and vv else vv
            for key, expr in expressions.items():
                try:
                    val = eval(expr, {"__builtins__": {}, "math": math}, locals_env)
                except Exception:
                    val = None
                data[key] = val
                
    ox.save_graphml(G, output_path)
    
    return {"output_path": output_path}

# @mcp.tool()
def save_network_graph(graph_path: str, output_name: str) -> Dict[str, Any]:
    """
    Save a network graph to GraphML.

    Args:
        graph_path: Path to the input GraphML file.
        output_name: Output filename (must end with .graphml).

    Returns:
        output_path: Absolute path to the saved GraphML.
    """
    if not output_name.lower().endswith(".graphml"):
        raise ValueError("output_name must end with .graphml")

    gp = os.path.abspath(graph_path)
    if not os.path.isfile(gp):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)
    
    G = ox.load_graphml(gp)
    ox.save_graphml(G, output_path)
    
    return {"output_path": output_path}

# @mcp.tool()
def densify_paths(
    paths_path: str,
    output_name: str,
    spacing_m: int = 5000,
) -> Dict[str, Any]:
    """
    Densify path lines using WGS84 geodesic spacing.

    Args:
        paths_path: Path to the input paths file (Shapefile/GeoJSON).
        output_name: Output filename (must end with .shp).
        spacing_m: Target spacing (meters) for inserted geodesic points.

    Returns:
        output_path: Path to the densified paths written (Shapefile).
        count: Number of features written.
    """
    pp = os.path.abspath(paths_path)
    if not os.path.isfile(pp):
        raise FileNotFoundError(f"Paths file not found: {pp}")

    gdf = gpd.read_file(pp, engine="fiona")
    gdf_wgs84 = gdf.to_crs(4326)
    geod = Geod(ellps="WGS84")

    def densify_geom(geom):
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            out = []
            for i in range(len(coords) - 1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i + 1]
                _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
                npts = int(max(0, dist // spacing_m))
                out.append((lon1, lat1))
                if npts > 0:
                    pts = geod.npts(lon1, lat1, lon2, lat2, npts)
                    out.extend(pts)
            out.append(coords[-1])
            return LineString(out)
        elif isinstance(geom, MultiLineString):
            parts = []
            for ls in geom.geoms:
                parts.append(densify_geom(ls))
            return MultiLineString(parts)
        else:
            return geom

    densified_geoms = gdf_wgs84.geometry.apply(densify_geom)
    gdf_densified = gpd.GeoDataFrame(
        gdf_wgs84.drop(columns="geometry"),
        geometry=densified_geoms,
        crs="EPSG:4326",
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _get_out_path(output_name)
    gdf_densified.to_file(out_path, driver="ESRI Shapefile")
    return {"output_path": out_path, "count": int(len(gdf_densified))}

# @mcp.tool()
def extract_route_origins(
    paths_path: str,
    output_name: str,
    route_field: str
) -> Dict[str, Any]:
    """
    Extract origin points from the start of each route feature and save.

    Args:
        paths_path: Path to the input routes file (.shp/.geojson).
        output_name: Output filename (must end with .geojson or .shp).
        route_field: Field name for route identifier.

    Returns:
        output_path: Path to the output origins file.
    """
    pp = os.path.abspath(paths_path)
    if not os.path.isfile(pp):
        raise FileNotFoundError(f"Paths file not found: {pp}")

    gdf = gpd.read_file(pp, engine="fiona")
    origins = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            pt = Point(geom.coords[0])
            origins.append({"geometry": pt, "Route": row.get(route_field, idx)})
        elif isinstance(geom, MultiLineString) and len(geom.geoms) > 0:
            pt = Point(geom.geoms[0].coords[0])
            origins.append({"geometry": pt, "Route": row.get(route_field, idx)})

    gdf_origins = gpd.GeoDataFrame(origins, crs=gdf.crs)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)

    if not gdf_origins.empty:
        driver = "GeoJSON" if output_name.lower().endswith(".geojson") else "ESRI Shapefile"
        gdf_origins.to_file(output_path, driver=driver)

    return {"output_path": output_path}


# @mcp.tool()
def calculate_population_access_proportion(
    roads_rural_path: str,
    population_rural_path: str,
    output_name: str,
    population_field: str
) -> Dict[str, Any]:
    """
    Calculate the proportion of population with access to roads in rural areas.

    Args:
        roads_rural_path: Path to rural road buffer file (.shp/.geojson).
        population_rural_path: Path to rural population file (.shp/.geojson).
        output_name: Output filename (must end with .geojson).
        population_field: Population count field name.

    Returns:
        output_path: Path to the output access proportion file.
    """
    # Read data
    roads_rural = gpd.read_file(roads_rural_path)
    population_rural = gpd.read_file(population_rural_path)
    
    # Calculate intersecting road buffer area for each rural population zone
    population_rural["roads_area"] = 0.0
    
    for idx, pop_geom in population_rural.geometry.items():
        # Find all road buffers intersecting with current population zone
        intersecting_roads = roads_rural[roads_rural.geometry.intersects(pop_geom)]
        
        if not intersecting_roads.empty:
            # Sum intersection areas
            intersection_area = intersecting_roads.geometry.intersection(pop_geom).area.sum()
            population_rural.loc[idx, "roads_area"] = intersection_area
    
    # Calculate area proportion
    population_rural["area"] = population_rural.geometry.area
    population_rural["proportion"] = population_rural["roads_area"] / population_rural["area"]
    
    # Calculate population weighted proportion
    population_rural["population_proportion"] = population_rural[population_field] * population_rural["proportion"]
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)
    population_rural.to_file(output_path, driver="GeoJSON")
    
    return {"output_path": output_path}

# @mcp.tool()
def compute_shortest_path_lengths(
    graph_path: str,
    source_node: int,
    output_name: str
) -> Dict[str, Any]:
    """
    Compute Dijkstra shortest travel times (seconds) from a source node and save to a pickle file.

    Args:
        graph_path: Path to the GraphML file with edge travel_time_s.
        source_node: Node id to start from.
        output_name: Output filename (must end with .pkl).

    Returns:
        A dict containing:
        - output_path: Absolute path to the saved pickle.
        - node_count: Number of reachable nodes.
        - distances: Mapping of node_id to shortest travel time in seconds.
    """
    gp = os.path.abspath(graph_path)
    if not os.path.isfile(gp):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
        
    G = ox.load_graphml(gp)
    
    for u, v, k, data in G.edges(keys=True, data=True):
        w = data.get("travel_time_s")
        if w is None:
            data["__w__"] = float("inf")
            continue
        try:
            data["__w__"] = float(w)
        except Exception:
            try:
                data["__w__"] = float(str(w).strip())
            except Exception:
                data["__w__"] = float("inf")
                
    dists = nx.single_source_dijkstra_path_length(G, source=source_node, weight="__w__")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _get_out_path(output_name)
    with open(output_path, "wb") as f:
        pickle.dump(dists, f)
        
    return {
        "output_path": output_path,
        "node_count": len(dists),
        "distances": dists
    }

# @mcp.tool()
def summarize_distances(
    graph_path: str,
    distances: Dict[int, float],
    output_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Summarize shortest travel times and optionally write a human-readable report.

    Args:
        graph_path: Path to the GraphML file.
        distances: Mapping of node_id to travel time (seconds).
        output_name: Optional text filename to write summary (saved in OUTPUT_DIR).

    Returns:
        A dict with totals and statistics:
        - total_nodes, total_edges, reachable_nodes, unreachable_nodes
        - avg_time_s, max_time_s, min_time_s
        - stats_path (if output_name provided)
    """
    gp = os.path.abspath(graph_path)
    if not os.path.isfile(gp):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
        
    G = ox.load_graphml(gp)
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    reachable = len(distances)
    unreachable = total_nodes - reachable
    times = list(distances.values())
    avg_time = float(np.mean(times)) if times else 0.0
    max_time = float(np.max(times)) if times else 0.0
    min_time = float(np.min(times)) if times else 0.0
    
    ret = {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "reachable_nodes": reachable,
        "unreachable_nodes": unreachable,
        "avg_time_s": avg_time,
        "max_time_s": max_time,
        "min_time_s": min_time,
    }
    
    if output_name:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stats_path = _get_out_path(output_name)
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("Street Network Statistics\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Nodes: {total_nodes}\n")
            f.write(f"Total Edges: {total_edges}\n")
            f.write(f"Reachable Nodes: {reachable}\n")
            f.write(f"Unreachable Nodes: {unreachable}\n")
            f.write(f"\nReachable Nodes Statistics:\n")
            f.write(f"  Average Travel Time: {avg_time:.2f} seconds\n")
            f.write(f"  Max Travel Time: {max_time:.2f} seconds\n")
            f.write(f"  Min Travel Time: {min_time:.2f} seconds\n")
        ret["stats_path"] = stats_path
        
    return ret

# @mcp.tool()
def build_graph_from_road_shp(
    roads_path: str,
    output_name: str,
    snap_tolerance: float = 1.0,
    length_field: str = "METERS",
    oneway_field: str = "ONEWAY",
    road_id_field: Optional[str] = None,
    encoding: str = "gbk",
) -> Dict[str, Any]:
    """
    Build a NetworkX MultiDiGraph from a polyline road Shapefile and save as GraphML.

    Args:
        roads_path: Path to the road network shapefile.
        output_name: Output GraphML filename.
        snap_tolerance: Distance threshold to merge nearby endpoints.
        length_field: Field storing road segment length.
        oneway_field: Field storing one-way direction information.
        road_id_field: Optional field used as stable road segment identifier.
        encoding: Optional file encoding used when reading the shapefile.

    Returns:
        status: str, "success"
        output_path: str, absolute path to GraphML
        node_count: int, number of graph nodes
        edge_count: int, number of directed edges
        weakly_connected_components: int, number of weakly connected components
        crs: str, CRS string of the source roads
    """
    import geopandas as gpd
    import networkx as nx
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree

    rp = os.path.abspath(roads_path)
    if not os.path.isfile(rp):
        raise FileNotFoundError(f"Roads file not found: {roads_path}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))

    try:
        roads = gpd.read_file(rp, encoding=encoding)
    except Exception:
        roads = gpd.read_file(rp)

    G = nx.MultiDiGraph(crs=str(roads.crs))
    endpoints = []
    keep_idx = []
    for idx, row in roads.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        endpoints.append(coords[0])
        endpoints.append(coords[-1])
        keep_idx.append(idx)
    pts = np.asarray(endpoints, dtype=float)
    tree = cKDTree(pts)
    node_of_ep = -np.ones(len(pts), dtype=int)
    coord_of_node = {}
    next_id = 0
    for i in range(len(pts)):
        if node_of_ep[i] != -1:
            continue
        nbrs = tree.query_ball_point(pts[i], r=snap_tolerance)
        for j in nbrs:
            if node_of_ep[j] == -1:
                node_of_ep[j] = next_id
        coord_of_node[next_id] = tuple(pts[i])
        next_id += 1
    for nid, (x, y) in coord_of_node.items():
        G.add_node(int(nid), x=float(x), y=float(y))
    for k_road, idx in enumerate(keep_idx):
        row = roads.loc[idx]
        u = int(node_of_ep[2 * k_road])
        v = int(node_of_ep[2 * k_road + 1])
        if u == v:
            continue
        if length_field in roads.columns and pd.notna(row.get(length_field)):
            length_m = float(row[length_field])
        else:
            length_m = float(row.geometry.length)
        road_id = row[road_id_field] if road_id_field and road_id_field in roads.columns else int(idx)
        oneway = row.get(oneway_field) if oneway_field in roads.columns else None
        oneway_norm = str(oneway).strip().upper() if (oneway is not None and pd.notna(oneway)) else "NONE"
        if oneway_norm == "FT":
            G.add_edge(u, v, length_m=length_m, oneway=oneway_norm, road_idx=road_id)
        elif oneway_norm == "TF":
            G.add_edge(v, u, length_m=length_m, oneway=oneway_norm, road_idx=road_id)
        else:
            G.add_edge(u, v, length_m=length_m, oneway=oneway_norm, road_idx=road_id)
            G.add_edge(v, u, length_m=length_m, oneway=oneway_norm, road_idx=road_id)

    nx.write_graphml(G, out_path)
    return {
        "status": "success",
        "output_path": out_path,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "weakly_connected_components": nx.number_weakly_connected_components(G),
        "crs": str(roads.crs),
    }

# @mcp.tool()
def snap_points_to_network_batch(
    graph_path: str,
    points_path: str,
    output_name: str,
    id_field: Optional[str] = None,
    encoding: str = "gbk",
    point_id_col: str = "point_id",
    node_id_col: str = "node_id",
    snap_distance_col: str = "snap_distance_m",
    x_col: str = "x",
    y_col: str = "y",
) -> Dict[str, Any]:
    """
    Snap every point in a vector layer to its nearest graph node (batch).

    Args:
        graph_path: Path to the GraphML network.
        points_path: Path to the point vector layer.
        output_name: Output CSV filename.
        id_field: Optional field used as point identifier.
        encoding: Optional file encoding used when reading the point layer.
        point_id_col: Output column name for point ids.
        node_id_col: Output column name for snapped graph node ids.
        snap_distance_col: Output column name for snap distance.
        x_col: Output column name for x coordinate.
        y_col: Output column name for y coordinate.

    Returns:
        status: str, "success"
        output_path: str, absolute path to snapped CSV
        n_snapped: int, number of snapped points
        mean_snap_distance_m: float, average snap distance
        max_snap_distance_m: float, maximum snap distance
    """
    import geopandas as gpd
    import networkx as nx
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree

    gp = os.path.abspath(graph_path)
    pp = os.path.abspath(points_path)
    if not os.path.isfile(gp):
        raise FileNotFoundError(f"Graph not found: {graph_path}")
    if not os.path.isfile(pp):
        raise FileNotFoundError(f"Points file not found: {points_path}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))

    G = nx.read_graphml(gp)
    node_ids = []
    node_xy = []
    for n, d in G.nodes(data=True):
        if d.get("x") is None or d.get("y") is None:
            continue
        node_ids.append(int(n))
        node_xy.append((float(d["x"]), float(d["y"])))
    coords = np.array(node_xy)
    tree = cKDTree(coords)

    try:
        pts = gpd.read_file(pp, encoding=encoding)
    except Exception:
        pts = gpd.read_file(pp)
    pcoords = np.array([[p.x, p.y] for p in pts.geometry])
    dists, idxs = tree.query(pcoords, k=1)
    if id_field and id_field in pts.columns:
        ids = pts[id_field].values
    else:
        ids = np.arange(len(pts))
    df = pd.DataFrame({
        point_id_col: ids,
        node_id_col: [int(node_ids[i]) for i in idxs],
        snap_distance_col: dists.astype(float),
        x_col: pcoords[:, 0].astype(float),
        y_col: pcoords[:, 1].astype(float),
    })
    df.to_csv(out_path, index=False)
    return {
        "status": "success",
        "output_path": out_path,
        "n_snapped": len(df),
        "mean_snap_distance_m": float(dists.mean()),
        "max_snap_distance_m": float(dists.max()),
    }


# @mcp.tool()
def compute_od_cost_matrix(
    graph_path: str,
    origins_snapped_csv: str,
    destinations_snapped_csv: str,
    output_name: str,
    weight_attr: str = "length_m",
    origin_id_col: str = "point_id",
    origin_node_col: str = "node_id",
    dest_id_col: str = "point_id",
    dest_node_col: str = "node_id",
    output_origin_col: str = "origin_id",
    output_dest_col: str = "dest_id",
    output_cost_col: str = "cost",
    output_reachable_col: str = "reachable",
) -> Dict[str, Any]:
    """
    Compute the N origins × M destinations OD shortest-path cost matrix on the graph.

    Args:
        graph_path: Path to the GraphML network.
        origins_snapped_csv: CSV of snapped origins.
        destinations_snapped_csv: CSV of snapped destinations.
        output_name: Output OD matrix CSV filename.
        weight_attr: Edge attribute used as travel cost.
        origin_id_col: Input column name for origin ids.
        origin_node_col: Input column name for origin node ids.
        dest_id_col: Input column name for destination ids.
        dest_node_col: Input column name for destination node ids.
        output_origin_col: Output column name for origin ids.
        output_dest_col: Output column name for destination ids.
        output_cost_col: Output column name for path cost.
        output_reachable_col: Output column name for reachability flag.

    Returns:
        status: str, "success"
        output_path: str, absolute path to OD matrix CSV
        n_pairs: int, total number of OD pairs
        n_reachable: int, number of reachable pairs
        n_unreachable: int, number of unreachable pairs
        mean_cost: float, average reachable cost
        median_cost: float, median reachable cost
        max_cost: float, maximum reachable cost
    """
    import networkx as nx
    import numpy as np
    import pandas as pd

    gp = os.path.abspath(graph_path)
    op = os.path.abspath(origins_snapped_csv)
    dp = os.path.abspath(destinations_snapped_csv)
    for p in [gp, op, dp]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))

    G = nx.read_graphml(gp)
    G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})
    for u, v, data in G.edges(data=True):
        if weight_attr in data:
            try:
                data[weight_attr] = float(data[weight_attr])
            except Exception:
                data[weight_attr] = float("inf")

    origins = pd.read_csv(op)
    dests = pd.read_csv(dp)
    dest_nodes = dests[dest_node_col].astype(int).tolist()
    dest_ids = dests[dest_id_col].tolist()
    rows = []
    for _, orow in origins.iterrows():
        try:
            costs = nx.single_source_dijkstra_path_length(G, source=int(orow[origin_node_col]), weight=weight_attr)
        except Exception:
            costs = {}
        for dnode, did in zip(dest_nodes, dest_ids):
            c = costs.get(dnode, float("inf"))
            rows.append({
                output_origin_col: orow[origin_id_col],
                output_dest_col: did,
                output_cost_col: float(c),
                output_reachable_col: bool(np.isfinite(c)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    reach_costs = df.loc[df[output_reachable_col], output_cost_col]
    return {
        "status": "success",
        "output_path": out_path,
        "n_pairs": len(df),
        "n_reachable": int(df[output_reachable_col].sum()),
        "n_unreachable": int((~df[output_reachable_col]).sum()),
        "mean_cost": float(reach_costs.mean()) if len(reach_costs) else 0.0,
        "median_cost": float(reach_costs.median()) if len(reach_costs) else 0.0,
        "max_cost": float(reach_costs.max()) if len(reach_costs) else 0.0,
    }


# @mcp.tool()
def apply_gravity_model(
    od_matrix_csv: str,
    output_name: str,
    beta: float = 2.0,
    constraint: str = "production",
    origin_id_col: str = "origin_id",
    dest_id_col: str = "dest_id",
    cost_col: str = "cost",
    reachable_col: str = "reachable",
    output_cost_col: str = "d_ij",
    output_flow_col: str = "flow",
    min_cost: float = 1e-6,
    balance_iterations: int = 30,
) -> Dict[str, Any]:
    """
    Apply Wilson spatial-interaction (gravity) model on an OD cost matrix.

    Args:
        od_matrix_csv: Path to the OD matrix CSV.
        output_name: Output predicted-flow CSV filename.
        beta: Distance-decay parameter.
        constraint: production, attraction, unconstrained, or doubly.
        origin_id_col: Input column name for origin ids.
        dest_id_col: Input column name for destination ids.
        cost_col: Input column name for travel cost.
        reachable_col: Input column name for reachability flag.
        output_cost_col: Output column name for adjusted travel cost.
        output_flow_col: Output column name for predicted flow.
        min_cost: Minimum cost floor used to avoid divide-by-zero.
        balance_iterations: Iteration count used for doubly constrained balancing.

    Returns:
        status: str, "success"
        output_path: str, absolute path to predicted-flow CSV
        beta: float, decay parameter used
        constraint: str, constraint type used
        n_pairs: int, number of predicted OD pairs
        total_flow: float, sum of all predicted flows
    """
    import numpy as np
    import pandas as pd

    mp = os.path.abspath(od_matrix_csv)
    if not os.path.isfile(mp):
        raise FileNotFoundError(f"OD matrix not found: {od_matrix_csv}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))

    od = pd.read_csv(mp)
    pred = od[od[reachable_col]].copy()
    pred[output_cost_col] = pred[cost_col].clip(lower=min_cost)
    pred["raw_T"] = 1.0 / np.power(pred[output_cost_col], beta)
    if constraint == "production":
        pred[output_flow_col] = pred.groupby(origin_id_col)["raw_T"].transform(lambda s: s / s.sum())
    elif constraint == "attraction":
        pred[output_flow_col] = pred.groupby(dest_id_col)["raw_T"].transform(lambda s: s / s.sum())
    elif constraint == "unconstrained":
        pred[output_flow_col] = pred["raw_T"] / pred["raw_T"].sum()
    elif constraint == "doubly":
        pred[output_flow_col] = pred["raw_T"].copy()
        for _ in range(balance_iterations):
            pred[output_flow_col] = pred.groupby(origin_id_col)[output_flow_col].transform(lambda s: s / s.sum())
            pred[output_flow_col] = pred.groupby(dest_id_col)[output_flow_col].transform(lambda s: s / s.sum())
    else:
        raise ValueError(f"Unknown constraint: {constraint}")
    out = pred[[origin_id_col, dest_id_col, output_cost_col, output_flow_col]].reset_index(drop=True)
    out.to_csv(out_path, index=False)
    return {
        "status": "success",
        "output_path": out_path,
        "beta": float(beta),
        "constraint": constraint,
        "n_pairs": len(out),
        "total_flow": float(out[output_flow_col].sum()),
    }


# @mcp.tool()
def build_od_desire_lines(
    predicted_flow_csv: str,
    origins_snapped_csv: str,
    destinations_snapped_csv: str,
    output_name: str,
    top_k: Optional[int] = None,
    min_flow: Optional[float] = None,
    crs: Optional[str] = None,
    origin_id_col: str = "origin_id",
    dest_id_col: str = "dest_id",
    flow_col: str = "flow",
    origin_point_id_col: str = "point_id",
    dest_point_id_col: str = "point_id",
    x_col: str = "x",
    y_col: str = "y",
    driver: str = "GeoJSON",
) -> Dict[str, Any]:
    """
    Build LineString GeoJSON of OD desire lines from a predicted-flow CSV.

    Args:
        predicted_flow_csv: CSV containing predicted OD flows.
        origins_snapped_csv: CSV of snapped origins.
        destinations_snapped_csv: CSV of snapped destinations.
        output_name: Output GeoJSON filename.
        top_k: Keep only top-K largest flows if provided.
        min_flow: Keep only flows >= this threshold if provided.
        crs: CRS string for the output desire lines.
        origin_id_col: Flow-table column name for origin ids.
        dest_id_col: Flow-table column name for destination ids.
        flow_col: Flow-table column name for flow value.
        origin_point_id_col: Origin snap-table column name for point ids.
        dest_point_id_col: Destination snap-table column name for point ids.
        x_col: Column name for x coordinate in snap tables.
        y_col: Column name for y coordinate in snap tables.
        driver: Vector driver used to write output, e.g. GeoJSON.

    Returns:
        status: str, "success"
        output_path: str, absolute path to desire-line GeoJSON
        line_count: int, number of desire lines generated
        max_flow: float, maximum flow among output lines
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import LineString

    pp = os.path.abspath(predicted_flow_csv)
    op = os.path.abspath(origins_snapped_csv)
    dp = os.path.abspath(destinations_snapped_csv)
    for p in [pp, op, dp]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))

    pred = pd.read_csv(pp)
    if min_flow is not None:
        pred = pred[pred[flow_col] >= min_flow]
    if top_k is not None:
        pred = pred.nlargest(top_k, flow_col)
    o_xy = pd.read_csv(op).set_index(origin_point_id_col)[[x_col, y_col]].to_dict("index")
    d_xy = pd.read_csv(dp).set_index(dest_point_id_col)[[x_col, y_col]].to_dict("index")
    geoms = []
    keep_idx = []
    for i, r in pred.iterrows():
        if r[origin_id_col] in o_xy and r[dest_id_col] in d_xy:
            ox_, oy_ = o_xy[r[origin_id_col]][x_col], o_xy[r[origin_id_col]][y_col]
            dx_, dy_ = d_xy[r[dest_id_col]][x_col], d_xy[r[dest_id_col]][y_col]
            geoms.append(LineString([(ox_, oy_), (dx_, dy_)]))
            keep_idx.append(i)
    pred = pred.loc[keep_idx].reset_index(drop=True)
    gdf = gpd.GeoDataFrame(pred, geometry=geoms, crs=crs)
    gdf.to_file(out_path, driver=driver)
    return {
        "status": "success",
        "output_path": out_path,
        "line_count": len(gdf),
        "max_flow": float(gdf[flow_col].max()) if len(gdf) else 0.0,
    }


# @mcp.tool()
def summarize_od_mobility(
    od_matrix_csv: str,
    predicted_flow_csv: str,
    output_name: str,
    top_k: int = 20,
    reachable_col: str = "reachable",
    cost_col: str = "cost",
    flow_col: str = "flow",
    origin_id_col: str = "origin_id",
    dest_id_col: str = "dest_id",
    report_cost_col: str = "d_ij",
    report_title: str = "Pair-level OD Analysis Report (Task 51)",
) -> Dict[str, Any]:
    """
    Generate an OD flow / network mobility statistics report (Top-K, Pareto, marginals).

    Args:
        od_matrix_csv: Path to the OD matrix CSV.
        predicted_flow_csv: Path to the predicted-flow CSV.
        output_name: Output report filename.
        top_k: Number of top OD flows to include in report.
        reachable_col: OD-table column name for reachability flag.
        cost_col: OD-table column name for travel cost.
        flow_col: Flow-table column name for predicted flow.
        origin_id_col: Flow-table column name for origin ids.
        dest_id_col: Flow-table column name for destination ids.
        report_cost_col: Flow-table column name for reported travel cost.
        report_title: Report title written to the output text file.

    Returns:
        status: str, "success"
        output_path: str, absolute path to the report file
        total_pairs: int, total OD pairs
        reachable_pairs: int, reachable OD pairs
        unreachable_pairs: int, unreachable OD pairs
        mean_cost: float, average reachable cost
        median_cost: float, median reachable cost
        n_for_50_pct: int, number of pairs carrying 50% of flow
        n_for_80_pct: int, number of pairs carrying 80% of flow
        top_hub_dest_id: int, destination id with highest aggregated flow
    """
    import numpy as np
    import pandas as pd

    mp = os.path.abspath(od_matrix_csv)
    pp = os.path.abspath(predicted_flow_csv)
    for p in [mp, pp]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(_get_out_path(output_name))

    od = pd.read_csv(mp)
    pred = pd.read_csv(pp)
    total = len(od)
    reach = int(od[reachable_col].sum())
    reach_costs = od.loc[od[reachable_col], cost_col]
    sorted_flows = pred[flow_col].sort_values(ascending=False).values
    cum = np.cumsum(sorted_flows)
    n50 = int(np.searchsorted(cum, 0.5 * cum[-1])) + 1
    n80 = int(np.searchsorted(cum, 0.8 * cum[-1])) + 1
    D_j = pred.groupby(dest_id_col)[flow_col].sum().sort_values(ascending=False)
    top_flows = pred.nlargest(top_k, flow_col)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{report_title}\n")
        f.write("=" * 60 + "\n")
        f.write("1. OD Pair Statistics:\n")
        f.write(f"   total_pairs: {total}\n")
        f.write(f"   reachable_pairs: {reach}\n")
        f.write(f"   unreachable_pairs: {total - reach}\n")
        f.write(f"   mean_cost (m): {reach_costs.mean():.2f}\n")
        f.write(f"   median_cost (m): {reach_costs.median():.2f}\n")
        f.write(f"   max_cost (m): {reach_costs.max():.2f}\n")
        f.write("\n2. Pareto Flow Concentration:\n")
        f.write(f"   pairs carrying 50% of flow: {n50} ({100 * n50 / len(pred):.1f}%)\n")
        f.write(f"   pairs carrying 80% of flow: {n80} ({100 * n80 / len(pred):.1f}%)\n")
        f.write(f"\n3. Top {top_k} Dominant OD Flows:\n")
        for _, r in top_flows.iterrows():
            f.write(f"   o={int(r[origin_id_col])} -> d={int(r[dest_id_col])}: flow={r[flow_col]:.5f}, d={r[report_cost_col]:.1f}m\n")
        f.write("\n4. Top 10 Destination Hubs (D_j) [destination-side aggregation]:\n")
        for did, val in D_j.head(10).items():
            f.write(f"   dest_id={int(did)}: D_j={val:.4f}\n")

    return {
        "status": "success",
        "output_path": out_path,
        "total_pairs": total,
        "reachable_pairs": reach,
        "unreachable_pairs": total - reach,
        "mean_cost": float(reach_costs.mean()),
        "median_cost": float(reach_costs.median()),
        "n_for_50_pct": n50,
        "n_for_80_pct": n80,
        "top_hub_dest_id": int(D_j.head(1).index[0]) if len(D_j) else None,
    }

# @mcp.tool()
def assign_flows_to_roads(
    graph_path: str,
    predicted_flow_csv: str,
    origins_snapped_csv: str,
    destinations_snapped_csv: str,
    roads_path: str,
    output_name: str,
    weight_attr: str = "length_m",
    origin_id_col: str = "origin_id",
    dest_id_col: str = "dest_id",
    flow_col: str = "flow",
    flow_load_field: str = "flow_load",
    road_id_field: Optional[str] = None,
    encoding: str = "gbk",
    origin_point_id_col: str = "point_id",
    origin_node_col: str = "node_id",
    dest_point_id_col: str = "point_id",
    dest_node_col: str = "node_id",
) -> Dict[str, Any]:
    """
    Traffic Assignment: Map OD flows back onto the actual road network.

    Args:
        graph_path: Path to the GraphML file.
        predicted_flow_csv: CSV of predicted flows.
        origins_snapped_csv: Snapped origins CSV.
        destinations_snapped_csv: Snapped destinations CSV.
        roads_path: Road network Shapefile.
        output_name: Filename for output GeoJSON with flow loads.
        weight_attr: Attribute for shortest path calculation.
        origin_id_col: Flow-table column name for origin IDs.
        dest_id_col: Flow-table column name for destination IDs.
        flow_col: Flow-table column name for flow values.
        flow_load_field: New column name for accumulated flow load on roads.
        road_id_field: Optional unique identifier field in the road SHP for matching.
        encoding: Encoding for the road SHP file.
        origin_point_id_col: Origin snap-table column name for point IDs.
        origin_node_col: Origin snap-table column name for graph node IDs.
        dest_point_id_col: Destination snap-table column name for point IDs.
        dest_node_col: Destination snap-table column name for graph node IDs.

    Returns:
        status: str, "success"
        output_path: str, path to the output GeoJSON
        n_loaded_segments: int, number of segments with flow > 0
        max_flow_load: float, maximum flow assigned to any segment
        total_flow_load: float, total flow volume assigned
    """
    import geopandas as gpd
    import networkx as nx
    import pandas as pd

    gp = os.path.abspath(graph_path)
    pp = os.path.abspath(predicted_flow_csv)
    op = os.path.abspath(origins_snapped_csv)
    dp = os.path.abspath(destinations_snapped_csv)
    rp = os.path.abspath(roads_path)
    for p in [gp, pp, op, dp, rp]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))

    try:
        G = nx.read_graphml(gp, force_multigraph=True)
    except TypeError:
        G = nx.read_graphml(gp)
    G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})
    for u, v, data in G.edges(data=True):
        if weight_attr in data:
            try:
                data[weight_attr] = float(data[weight_attr])
            except Exception:
                data[weight_attr] = float("inf")
        # 注意：road_idx 经 GraphML 读回后可能是 str，统一用 str 做查找以避免类型不匹配
        if "road_idx" in data:
            try:
                data["road_idx"] = int(data["road_idx"])
            except (ValueError, TypeError):
                pass  # 保留原值（可能为非整数 ID）

    pred = pd.read_csv(pp)
    origins = pd.read_csv(op).set_index(origin_point_id_col)
    dests = pd.read_csv(dp).set_index(dest_point_id_col)
    try:
        roads = gpd.read_file(rp, encoding=encoding)
    except Exception:
        roads = gpd.read_file(rp)

    roads[flow_load_field] = 0.0

    # 建立 road_id → roads index 映射：键统一 cast 为 str 以避免 int/str 不匹配
    if road_id_field and road_id_field in roads.columns:
        id_to_idx = {str(val): idx for idx, val in roads[road_id_field].items()}
    else:
        id_to_idx = {str(idx): idx for idx in roads.index}

    for oid, grp in pred.groupby(origin_id_col):
        source_node = int(origins.loc[oid, origin_node_col])
        try:
            paths = nx.single_source_dijkstra_path(G, source_node, weight=weight_attr)
        except Exception:
            continue
        for _, row in grp.iterrows():
            target_node = int(dests.loc[row[dest_id_col], dest_node_col])
            if target_node in paths:
                path = paths[target_node]
                for u, v in zip(path[:-1], path[1:]):
                    edge_data = G.get_edge_data(u, v)
                    if not edge_data:
                        continue
                    first_val = next(iter(edge_data.values()), None)
                    if isinstance(first_val, dict):
                        candidates = list(edge_data.values())
                    else:
                        candidates = [edge_data]
                    best = min(candidates, key=lambda x: float(x.get(weight_attr, float("inf"))))
                    if "road_idx" in best:
                        rid_key = str(best["road_idx"])
                        if rid_key in id_to_idx:
                            roads.at[id_to_idx[rid_key], flow_load_field] += float(row[flow_col])

    roads.to_file(out_path, driver="GeoJSON")
    return {
        "status": "success",
        "output_path": out_path,
        "n_loaded_segments": int((roads[flow_load_field] > 0).sum()),
        "max_flow_load": float(roads[flow_load_field].max()),
        "total_flow_load": float(roads[flow_load_field].sum()),
    }

# @mcp.tool()
def compute_network_betweenness(
    graph_path: str,
    output_name: str,
    k: Optional[int] = 100,
    weight_attr: str = "length_m",
    node_id_col: str = "node_id",
    betweenness_col: str = "betweenness",
    x_col: str = "x",
    y_col: str = "y",
) -> Dict[str, Any]:
    """
    Calculate Betweenness Centrality for nodes in the network.

    Args:
        graph_path: Path to the GraphML file.
        output_name: Filename for results CSV.
        k: Sample size for approximation (None = exact, slower).
        weight_attr: Attribute to use as distance weight.
        node_id_col: Output column name for node ids.
        betweenness_col: Output column name for centrality values.
        x_col: Output column name for x coordinates.
        y_col: Output column name for y coordinates.

    Returns:
        status: str, "success"
        output_path: str, path to the centrality CSV
        n_nodes: int, total nodes processed
        max_betweenness: float, maximum centrality value
        mean_betweenness: float, average centrality value
        top_5_nodes: list, records of top 5 central nodes
    """
    import networkx as nx
    import pandas as pd

    gp = os.path.abspath(graph_path)
    if not os.path.isfile(gp):
        raise FileNotFoundError(f"Graph not found: {graph_path}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))

    G = nx.read_graphml(gp)
    G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})
    for u, v, data in G.edges(data=True):
        if weight_attr in data:
            try:
                data[weight_attr] = float(data[weight_attr])
            except Exception:
                data[weight_attr] = float("inf")

    bc = nx.betweenness_centrality(G, k=k, weight=weight_attr) if k else nx.betweenness_centrality(G, weight=weight_attr)
    rows = []
    for n, b in bc.items():
        nd = G.nodes[int(n)]
        rows.append({
            node_id_col: int(n),
            betweenness_col: float(b),
            x_col: float(nd.get("x", 0.0)),
            y_col: float(nd.get("y", 0.0)),
        })
    df = pd.DataFrame(rows).sort_values(betweenness_col, ascending=False).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return {
        "status": "success",
        "output_path": out_path,
        "n_nodes": len(df),
        "max_betweenness": float(df[betweenness_col].max()),
        "mean_betweenness": float(df[betweenness_col].mean()),
        "top_5_nodes": df.head(5).to_dict("records"),
    }

# @mcp.tool()
def add_service_time_field(
    facilities_path: str,
    type_field: str,
    service_minutes_by_type: Dict[str, float],
    default_service_minutes: float,
    output_name: str,
) -> Dict[str, Any]:
    """
    Add service-time fields to a point layer.

    Args:
        facilities_path: Path to the facility point vector file.
        type_field: Field containing the facility type or level.
        service_minutes_by_type: Mapping from type_field values to service-time thresholds in minutes.
        default_service_minutes: Service-time threshold used when a type value is not present in service_minutes_by_type.
        output_name: Output GeoJSON filename.

    Returns:
        status: str, "success" if the file is written.
        output_path: str, absolute path to the generated GeoJSON file.
        feature_count: int, number of facility features.
        service_field: str, generated service-time field name.
        type_label_field: str, generated type-label field name.
        type_counts: dict, feature counts by facility type value.
        service_minutes_by_type: dict, service-minute mapping used.
        default_service_minutes: float, fallback service threshold used.
    """
    if not os.path.isfile(facilities_path):
        raise FileNotFoundError(f"Vector file not found: {facilities_path}")
    try:
        facilities = gpd.read_file(facilities_path, engine="fiona")
    except Exception:
        facilities = gpd.read_file(facilities_path)
    if type_field not in facilities.columns:
        raise ValueError(f"Type field not found: {type_field}")
    facilities = facilities.copy()
    type_label_map = {"一级": "Level 1", "二级": "Level 2", "三级": "Level 3", "四级": "Level 4"}
    facilities["service_min"] = facilities[type_field].astype(str).map(service_minutes_by_type).fillna(default_service_minutes)
    facilities["type_en"] = facilities[type_field].astype(str).map(type_label_map).fillna(facilities[type_field].astype(str))
    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    facilities.to_file(out_path, driver="GeoJSON")
    saved_path = out_path
    counts = {str(k): int(v) for k, v in facilities[type_field].value_counts(dropna=False).to_dict().items()}
    return {
        "status": "success",
        "output_path": saved_path,
        "feature_count": int(len(facilities)),
        "service_field": "service_min",
        "type_label_field": "type_en",
        "type_counts": counts,
        "service_minutes_by_type": service_minutes_by_type,
        "default_service_minutes": float(default_service_minutes),
    }

# @mcp.tool()
def filter_facilities_by_names(
    facilities_path: str,
    name_field: str,
    selected_names: List[str],
    output_name: str,
) -> Dict[str, Any]:
    """
    Filter a point layer to a caller-specified name list and add map labels.

    Args:
        facilities_path: Path to the facility point vector file.
        name_field: Field containing facility names.
        selected_names: List of names to keep.
        output_name: Output GeoJSON filename.

    Returns:
        status: str, "success" if matching facilities are written.
        output_path: str, absolute path to the generated GeoJSON file.
        feature_count: int, number of selected facilities.
        selected_names: list, selected facility names in output order.
        label_field: str, generated field used for compact map labels.
    """
    if not os.path.isfile(facilities_path):
        raise FileNotFoundError(f"Vector file not found: {facilities_path}")
    try:
        facilities = gpd.read_file(facilities_path, engine="fiona")
    except Exception:
        facilities = gpd.read_file(facilities_path)
    if name_field not in facilities.columns:
        raise ValueError(f"Name field not found: {name_field}")

    selected = facilities[facilities[name_field].astype(str).isin([str(n) for n in selected_names])].copy()
    if selected.empty:
        raise ValueError("No facilities matched selected_names")

    order = {str(name): i for i, name in enumerate(selected_names)}
    selected["_order"] = selected[name_field].astype(str).map(order)
    selected = selected.sort_values("_order").drop(columns=["_order"])
    selected["facility_l"] = [f"F{i + 1}" for i in range(len(selected))]

    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    selected.to_file(out_path, driver="GeoJSON")
    saved_path = out_path
    return {
        "status": "success",
        "output_path": saved_path,
        "feature_count": int(len(selected)),
        "selected_names": [str(v) for v in selected[name_field].tolist()],
        "label_field": "facility_l",
    }

# @mcp.tool()
def compute_service_area_lines(
    network_path: str,
    facilities_path: str,
    facility_name_field: str,
    facility_type_field: str,
    service_time_field: str,
    cost_field: str,
    output_name: str,
    report_output_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute network service-area line segments for selected points.

    Args:
        network_path: Path to the road network vector file.
        facilities_path: Path to selected facility point layer.
        facility_name_field: Field containing facility names.
        facility_type_field: Field containing facility types.
        service_time_field: Field containing service thresholds in minutes.
        cost_field: Road-network cost field used for service-area travel cost.
        output_name: Output GeoJSON filename.
        report_output_name: Optional report JSON filename.

    Returns:
        status: str, "success" if service-area line segments are generated.
        output_path: str, absolute path to the service-area GeoJSON file.
        graph: dict, generated directed-graph summary.
        facility_count: int, number of facilities evaluated.
        service_segment_count: int, number of reachable line segments.
        facility_summaries: list, per-facility service-area statistics.
        report_path: str, present when report_output_name is provided.
    """
    if not os.path.isfile(network_path):
        raise FileNotFoundError(f"Vector file not found: {network_path}")
    if not os.path.isfile(facilities_path):
        raise FileNotFoundError(f"Vector file not found: {facilities_path}")
    try:
        network = gpd.read_file(network_path, engine="fiona")
    except Exception:
        network = gpd.read_file(network_path)
    try:
        facilities = gpd.read_file(facilities_path, engine="fiona")
    except Exception:
        facilities = gpd.read_file(facilities_path)
    if cost_field not in network.columns:
        raise ValueError(f"Cost field not found: {cost_field}")
    graph = nx.DiGraph()
    directed_edge_count = 0
    for road_index, row in network.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or not isinstance(geom, LineString):
            continue
        coords = [(round(float(c[0]), 3), round(float(c[1]), 3)) for c in geom.coords]
        if len(coords) < 2 or float(geom.length) <= 0:
            continue
        geom_length = float(geom.length)
        total_distance = float(row["METERS"]) if "METERS" in network.columns else geom_length
        oneway = str(row["ONEWAY"]).upper() if row.get("ONEWAY") is not None else ""
        road_name = row["NAME"] if "NAME" in network.columns else None
        if hasattr(road_name, "item"):
            road_name = road_name.item()
        if isinstance(road_name, float) and math.isnan(road_name):
            road_name = None
        if cost_field.upper() == "MINUTES" and "FT_MINUTES" in row.index and row["FT_MINUTES"] is not None and not (isinstance(row["FT_MINUTES"], float) and math.isnan(row["FT_MINUTES"])):
            forward_total_cost = float(row["FT_MINUTES"])
        else:
            forward_total_cost = float(row[cost_field])
        if cost_field.upper() == "MINUTES" and "TF_MINUTES" in row.index and row["TF_MINUTES"] is not None and not (isinstance(row["TF_MINUTES"], float) and math.isnan(row["TF_MINUTES"])):
            reverse_total_cost = float(row["TF_MINUTES"])
        else:
            reverse_total_cost = float(row[cost_field])
        for start, end in zip(coords[:-1], coords[1:]):
            segment_geom = LineString([start, end])
            ratio = segment_geom.length / geom_length
            segment_distance = total_distance * ratio
            forward_attrs = {"road_index": int(road_index), "road_name": road_name, "distance_m": float(segment_distance), "cost": float(forward_total_cost * ratio), "geometry": segment_geom, "direction": "FT"}
            reverse_attrs = {"road_index": int(road_index), "road_name": road_name, "distance_m": float(segment_distance), "cost": float(reverse_total_cost * ratio), "geometry": segment_geom, "direction": "TF"}
            if oneway == "FT":
                graph.add_edge(start, end, **forward_attrs)
                directed_edge_count += 1
            elif oneway == "TF":
                graph.add_edge(end, start, **reverse_attrs)
                directed_edge_count += 1
            else:
                graph.add_edge(start, end, **forward_attrs)
                graph.add_edge(end, start, **reverse_attrs)
                directed_edge_count += 2
    if graph.number_of_nodes() == 0:
        raise RuntimeError("Road network graph is empty")
    graph_info = {"node_count": int(graph.number_of_nodes()), "directed_edge_count": int(directed_edge_count), "road_feature_count": int(len(network)), "cost_field": cost_field, "uses_directional_minutes": bool(cost_field.upper() == "MINUTES" and "FT_MINUTES" in network.columns and "TF_MINUTES" in network.columns)}
    nodes = list(graph.nodes)

    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    seen_keys = set()
    for facility_index, facility in facilities.iterrows():
        threshold = float(facility[service_time_field])
        origin_node = nodes[0]
        snap_distance = math.inf
        for node in nodes:
            distance = facility.geometry.distance(Point(node))
            if distance < snap_distance:
                origin_node = node
                snap_distance = float(distance)
        distances = nx.single_source_dijkstra_path_length(graph, origin_node, cutoff=threshold, weight="cost")
        reachable_edges = 0
        service_length = 0.0
        for u, v, attrs in graph.edges(data=True):
            if u not in distances or v not in distances:
                continue
            if distances[u] > threshold or distances[v] > threshold:
                continue
            edge_key = (int(facility_index), int(attrs.get("road_index", -1)), tuple(u), tuple(v))
            if edge_key in seen_keys:
                continue
            seen_keys.add(edge_key)
            reachable_edges += 1
            service_length += float(attrs.get("distance_m", 0.0))
            rows.append(
                {
                    "facility_i": int(facility_index),
                    "facility_n": str(facility[facility_name_field]) if facility_name_field in facilities.columns else str(facility_index),
                    "facility_t": str(facility[facility_type_field]) if facility_type_field in facilities.columns else None,
                    "facility_e": str(facility["type_en"]) if "type_en" in facilities.columns else None,
                    "facility_l": str(facility["facility_l"]) if "facility_l" in facilities.columns else str(facility_index),
                    "service_mi": threshold,
                    "road_index": int(attrs.get("road_index", -1)),
                    "road_name": attrs.get("road_name"),
                    "edge_cost": float(attrs.get("cost", 0.0)),
                    "edge_dist": float(attrs.get("distance_m", 0.0)),
                    "geometry": attrs["geometry"],
                }
            )
        summary_rows.append(
            {
                "facility_index": int(facility_index),
                "facility_name": str(facility[facility_name_field]) if facility_name_field in facilities.columns else str(facility_index),
                "facility_type": str(facility[facility_type_field]) if facility_type_field in facilities.columns else None,
                "facility_type_en": str(facility["type_en"]) if "type_en" in facilities.columns else None,
                "service_minutes": threshold,
                "snap_distance": round(snap_distance, 3),
                "reachable_node_count": int(len(distances)),
                "reachable_edge_count": int(reachable_edges),
                "service_length_m": round(service_length, 3),
            }
        )

    if not rows:
        raise RuntimeError("No service-area line segments generated")
    service_gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    service_gdf.to_file(out_path, driver="GeoJSON")
    saved_path = out_path

    result = {
        "status": "success",
        "output_path": saved_path,
        "graph": graph_info,
        "facility_count": int(len(facilities)),
        "service_segment_count": int(len(service_gdf)),
        "facility_summaries": summary_rows,
    }
    if report_output_name:
        report_path = os.path.abspath(os.path.join(OUTPUT_DIR, report_output_name))
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result["report_path"] = report_path
    return result

# @mcp.tool()
def evaluate_school_coverage(
    network_path: str,
    facilities_path: str,
    schools_path: str,
    facility_name_field: str,
    service_time_field: str,
    school_name_field: str,
    cost_field: str,
    output_name: str,
    summary_output_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate whether reference points are covered by selected service ranges.

    Args:
        network_path: Path to the road network vector file.
        facilities_path: Path to selected facility point layer with service thresholds.
        schools_path: Path to the reference point vector file.
        facility_name_field: Field containing facility names.
        service_time_field: Field containing service thresholds in minutes.
        school_name_field: Field containing reference point names.
        cost_field: Road-network cost field used as travel-time weight.
        output_name: Output CSV filename.
        summary_output_name: Optional summary JSON filename.

    Returns:
        status: str, "success" if coverage is evaluated.
        output_path: str, absolute path to the generated CSV file.
        school_count: int, number of schools evaluated.
        covered_school_count: int, number of covered schools.
        uncovered_school_count: int, number of uncovered schools.
        coverage_rate: float, covered schools divided by total schools.
        uncovered_schools: list, names of schools outside the service range.
        summary_output_path: str, present when summary_output_name is provided.
    """
    if not os.path.isfile(network_path):
        raise FileNotFoundError(f"Vector file not found: {network_path}")
    if not os.path.isfile(facilities_path):
        raise FileNotFoundError(f"Vector file not found: {facilities_path}")
    if not os.path.isfile(schools_path):
        raise FileNotFoundError(f"Vector file not found: {schools_path}")
    try:
        network = gpd.read_file(network_path, engine="fiona")
    except Exception:
        network = gpd.read_file(network_path)
    try:
        facilities = gpd.read_file(facilities_path, engine="fiona")
    except Exception:
        facilities = gpd.read_file(facilities_path)
    try:
        schools = gpd.read_file(schools_path, engine="fiona")
    except Exception:
        schools = gpd.read_file(schools_path)
    if cost_field not in network.columns:
        raise ValueError(f"Cost field not found: {cost_field}")
    graph = nx.DiGraph()
    for road_index, row in network.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or not isinstance(geom, LineString):
            continue
        coords = [(round(float(c[0]), 3), round(float(c[1]), 3)) for c in geom.coords]
        if len(coords) < 2 or float(geom.length) <= 0:
            continue
        geom_length = float(geom.length)
        total_distance = float(row["METERS"]) if "METERS" in network.columns else geom_length
        oneway = str(row["ONEWAY"]).upper() if row.get("ONEWAY") is not None else ""
        if cost_field.upper() == "MINUTES" and "FT_MINUTES" in row.index and row["FT_MINUTES"] is not None and not (isinstance(row["FT_MINUTES"], float) and math.isnan(row["FT_MINUTES"])):
            forward_total_cost = float(row["FT_MINUTES"])
        else:
            forward_total_cost = float(row[cost_field])
        if cost_field.upper() == "MINUTES" and "TF_MINUTES" in row.index and row["TF_MINUTES"] is not None and not (isinstance(row["TF_MINUTES"], float) and math.isnan(row["TF_MINUTES"])):
            reverse_total_cost = float(row["TF_MINUTES"])
        else:
            reverse_total_cost = float(row[cost_field])
        for start, end in zip(coords[:-1], coords[1:]):
            segment_geom = LineString([start, end])
            ratio = segment_geom.length / geom_length
            if oneway == "FT":
                graph.add_edge(start, end, cost=float(forward_total_cost * ratio), geometry=segment_geom)
            elif oneway == "TF":
                graph.add_edge(end, start, cost=float(reverse_total_cost * ratio), geometry=segment_geom)
            else:
                graph.add_edge(start, end, cost=float(forward_total_cost * ratio), geometry=segment_geom)
                graph.add_edge(end, start, cost=float(reverse_total_cost * ratio), geometry=segment_geom)
    if graph.number_of_nodes() == 0:
        raise RuntimeError("Road network graph is empty")
    nodes = list(graph.nodes)

    facility_nodes = []
    for facility_index, facility in facilities.iterrows():
        node = nodes[0]
        snap_distance = math.inf
        for candidate in nodes:
            distance = facility.geometry.distance(Point(candidate))
            if distance < snap_distance:
                node = candidate
                snap_distance = float(distance)
        facility_nodes.append(
            {
                "index": int(facility_index),
                "name": str(facility[facility_name_field]) if facility_name_field in facilities.columns else str(facility_index),
                "service_minutes": float(facility[service_time_field]),
                "node": node,
                "snap_distance": float(snap_distance),
            }
        )

    rows: List[Dict[str, Any]] = []
    covered_count = 0
    for school_index, school in schools.iterrows():
        school_node = nodes[0]
        school_snap = math.inf
        for candidate in nodes:
            distance = school.geometry.distance(Point(candidate))
            if distance < school_snap:
                school_node = candidate
                school_snap = float(distance)
        best_time = math.inf
        best_facility = None
        best_threshold = None
        for facility in facility_nodes:
            try:
                travel_time = nx.shortest_path_length(graph, facility["node"], school_node, weight="cost")
            except nx.NetworkXNoPath:
                continue
            if travel_time < best_time:
                best_time = float(travel_time)
                best_facility = facility["name"]
                best_threshold = facility["service_minutes"]
        covered = bool(best_facility is not None and best_time <= float(best_threshold))
        if covered:
            covered_count += 1
        rows.append(
            {
                "school_index": int(school_index),
                "school_name": str(school[school_name_field]) if school_name_field in schools.columns else str(school_index),
                "covered": covered,
                "nearest_facility": best_facility,
                "travel_time_min": round(best_time, 4) if best_time < math.inf else None,
                "facility_service_min": best_threshold,
                "time_margin_min": round(float(best_threshold) - best_time, 4) if best_threshold is not None and best_time < math.inf else None,
                "school_snap_m": round(float(school_snap), 3),
            }
        )

    csv_path_obj = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))
    os.makedirs(os.path.dirname(csv_path_obj), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path_obj, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    csv_path = csv_path_obj
    result = {
        "status": "success",
        "output_path": csv_path,
        "school_count": int(len(schools)),
        "covered_school_count": int(covered_count),
        "uncovered_school_count": int(len(schools) - covered_count),
        "coverage_rate": round(float(covered_count) / float(len(schools)), 4) if len(schools) else 0.0,
        "uncovered_schools": [row["school_name"] for row in rows if not row["covered"]],
    }
    if summary_output_name:
        summary_path = os.path.abspath(os.path.join(OUTPUT_DIR, summary_output_name))
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result["summary_output_path"] = summary_path
    return result

# @mcp.tool()
def select_start_end_from_candidates(
    network_path: str,
    start_points_path: str,
    end_points_path: str,
    start_name_field: str,
    end_name_field: str,
    cost_field: str,
    selection_strategy: str,
    output_name: str,
) -> Dict[str, Any]:
    """
    Select the best pair from two candidate point layers.

    Args:
        network_path: Path to the road network vector file.
        start_points_path: Path to the candidate start point layer.
        end_points_path: Path to the candidate end point layer.
        start_name_field: Field name used to label selected start points.
        end_name_field: Field name used to label selected end points.
        cost_field: Road-network cost field used for network strategies.
        selection_strategy: Selection rule. Supported values are
            "longest_network_cost", "longest_network_distance", and
            "longest_euclidean_distance".
        output_name: Output GeoJSON filename.

    Returns:
        status: str, "success" if a pair is selected and written.
        output_path: str, absolute path to the selected start/end GeoJSON.
        score: float, selection score for the chosen pair.
        selection_strategy: str, strategy used for selection.
        start_index: int, source row index of the selected start point.
        end_index: int, source row index of the selected end point.
        start_name: str, label of the selected start point.
        end_name: str, label of the selected end point.
        start_x: float, selected start X coordinate.
        start_y: float, selected start Y coordinate.
        end_x: float, selected end X coordinate.
        end_y: float, selected end Y coordinate.
        start_snap_distance: float or None, start-to-network snap distance.
        end_snap_distance: float or None, end-to-network snap distance.
        route_cost_minutes: float or None, shortest-path cost in minutes.
        route_distance_m: float or None, shortest-path distance in meters.
        route_node_count: int or None, number of nodes in the candidate route.
    """
    try:
        starts = gpd.read_file(start_points_path, engine="fiona")
    except Exception:
        starts = gpd.read_file(start_points_path)
    try:
        ends = gpd.read_file(end_points_path, engine="fiona")
    except Exception:
        ends = gpd.read_file(end_points_path)
    if len(starts) == 0 or len(ends) == 0:
        raise ValueError("Candidate point layers must not be empty")

    graph = None
    nodes = None
    if selection_strategy in {"longest_network_cost", "longest_network_distance"}:
        try:
            network = gpd.read_file(network_path, engine="fiona")
        except Exception:
            network = gpd.read_file(network_path)
        if cost_field not in network.columns:
            raise ValueError(f"Cost field not found: {cost_field}")
        graph = nx.DiGraph()
        for road_index, row in network.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or not isinstance(geom, LineString):
                continue
            coords = [(round(float(c[0]), 3), round(float(c[1]), 3)) for c in geom.coords]
            if len(coords) < 2 or float(geom.length) <= 0:
                continue
            geom_length = float(geom.length)
            total_distance = float(row["METERS"]) if "METERS" in network.columns else geom_length
            oneway = str(row["ONEWAY"]).upper() if row.get("ONEWAY") is not None else ""
            if cost_field.upper() == "MINUTES" and "FT_MINUTES" in row.index and row["FT_MINUTES"] is not None and not (isinstance(row["FT_MINUTES"], float) and math.isnan(row["FT_MINUTES"])):
                forward_total_cost = float(row["FT_MINUTES"])
            else:
                forward_total_cost = float(row[cost_field])
            if cost_field.upper() == "MINUTES" and "TF_MINUTES" in row.index and row["TF_MINUTES"] is not None and not (isinstance(row["TF_MINUTES"], float) and math.isnan(row["TF_MINUTES"])):
                reverse_total_cost = float(row["TF_MINUTES"])
            else:
                reverse_total_cost = float(row[cost_field])
            for start, end in zip(coords[:-1], coords[1:]):
                segment_length = LineString([start, end]).length
                ratio = segment_length / geom_length
                attrs = {"distance_m": float(total_distance * ratio), "cost": float((forward_total_cost if oneway != "TF" else reverse_total_cost) * ratio)}
                if oneway == "FT":
                    graph.add_edge(start, end, **attrs)
                elif oneway == "TF":
                    graph.add_edge(end, start, **attrs)
                else:
                    graph.add_edge(start, end, distance_m=float(total_distance * ratio), cost=float(forward_total_cost * ratio))
                    graph.add_edge(end, start, distance_m=float(total_distance * ratio), cost=float(reverse_total_cost * ratio))
        if graph.number_of_nodes() == 0:
            raise RuntimeError("Road network graph is empty")
        nodes = list(graph.nodes)

    best: Optional[Dict[str, Any]] = None
    for start_idx, start_row in starts.iterrows():
        for end_idx, end_row in ends.iterrows():
            start_geom = start_row.geometry
            end_geom = end_row.geometry
            if start_geom is None or start_geom.is_empty or end_geom is None or end_geom.is_empty:
                continue

            if selection_strategy == "longest_euclidean_distance":
                score = float(start_geom.distance(end_geom))
                route_cost = None
                route_distance = None
                route_node_count = None
                start_node = None
                end_node = None
                start_snap_distance = None
                end_snap_distance = None
            else:
                assert graph is not None and nodes is not None
                start_node = nodes[0]
                start_snap_distance = math.inf
                for node in nodes:
                    distance = start_geom.distance(Point(node))
                    if distance < start_snap_distance:
                        start_node = node
                        start_snap_distance = float(distance)
                end_node = nodes[0]
                end_snap_distance = math.inf
                for node in nodes:
                    distance = end_geom.distance(Point(node))
                    if distance < end_snap_distance:
                        end_node = node
                        end_snap_distance = float(distance)
                try:
                    route_nodes = nx.shortest_path(graph, start_node, end_node, weight="cost")
                except nx.NetworkXNoPath:
                    continue

                route_cost = float(nx.shortest_path_length(graph, start_node, end_node, weight="cost"))
                route_distance = 0.0
                for u, v in zip(route_nodes[:-1], route_nodes[1:]):
                    route_distance += float(graph[u][v].get("distance_m", 0.0))
                route_node_count = int(len(route_nodes))
                score = route_cost if selection_strategy == "longest_network_cost" else route_distance

            candidate = {
                "score": float(score),
                "selection_strategy": selection_strategy,
                "start_index": int(start_idx),
                "end_index": int(end_idx),
                "start_name": str(start_row[start_name_field]) if start_name_field in starts.columns else str(start_idx),
                "end_name": str(end_row[end_name_field]) if end_name_field in ends.columns else str(end_idx),
                "start_x": float(start_geom.x),
                "start_y": float(start_geom.y),
                "end_x": float(end_geom.x),
                "end_y": float(end_geom.y),
                "start_snap_distance": round(float(start_snap_distance), 3) if start_snap_distance is not None else None,
                "end_snap_distance": round(float(end_snap_distance), 3) if end_snap_distance is not None else None,
                "route_cost_minutes": round(float(route_cost), 4) if route_cost is not None else None,
                "route_distance_m": round(float(route_distance), 3) if route_distance is not None else None,
                "route_node_count": route_node_count,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        raise RuntimeError("No reachable start/end candidate pair found")

    selected = gpd.GeoDataFrame(
        [
            {
                "role": "start",
                "source_index": best["start_index"],
                "name": best["start_name"],
                "x": best["start_x"],
                "y": best["start_y"],
                "snap_distance": best["start_snap_distance"],
                "geometry": Point(best["start_x"], best["start_y"]),
            },
            {
                "role": "end",
                "source_index": best["end_index"],
                "name": best["end_name"],
                "x": best["end_x"],
                "y": best["end_y"],
                "snap_distance": best["end_snap_distance"],
                "geometry": Point(best["end_x"], best["end_y"]),
            },
        ],
        geometry="geometry",
    )
    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    selected.to_file(out_path, driver="GeoJSON")
    best["output_path"] = out_path
    best["status"] = "success"
    return best

# @mcp.tool()
def shortest_path_start_end(
    network_path: str,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    cost_field: str,
    start_label: Optional[str] = None,
    end_label: Optional[str] = None,
    output_name: Optional[str] = None,
    report_output_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate the shortest route between two points.

    Args:
        network_path: Path to the road network vector file.
        start_x: X coordinate of the start point.
        start_y: Y coordinate of the start point.
        end_x: X coordinate of the end point.
        end_y: Y coordinate of the end point.
        cost_field: Road-network cost field used as the shortest-path weight.
        start_label: Optional label for the start point.
        end_label: Optional label for the end point.
        output_name: Optional route GeoJSON filename.
        report_output_name: Optional route report JSON filename.

    Returns:
        status: str, "success" if the route is computed.
        graph: dict, generated directed-graph summary.
        start: dict, original and snapped start-point information.
        end: dict, original and snapped end-point information.
        route: dict, route cost, distance, node count, and road-name summary.
        outputs: dict, generated route GeoJSON and report JSON paths.
    """
    try:
        network = gpd.read_file(network_path, engine="fiona")
    except Exception:
        network = gpd.read_file(network_path)
    if cost_field not in network.columns:
        raise ValueError(f"Cost field not found: {cost_field}")
    graph = nx.DiGraph()
    directed_edge_count = 0
    for road_index, row in network.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or not isinstance(geom, LineString):
            continue
        coords = [(round(float(c[0]), 3), round(float(c[1]), 3)) for c in geom.coords]
        if len(coords) < 2 or float(geom.length) <= 0:
            continue
        geom_length = float(geom.length)
        total_distance = float(row["METERS"]) if "METERS" in network.columns else geom_length
        oneway = str(row["ONEWAY"]).upper() if row.get("ONEWAY") is not None else ""
        road_name = row["NAME"] if "NAME" in network.columns else None
        if hasattr(road_name, "item"):
            road_name = road_name.item()
        if isinstance(road_name, float) and math.isnan(road_name):
            road_name = None
        if cost_field.upper() == "MINUTES" and "FT_MINUTES" in row.index and row["FT_MINUTES"] is not None and not (isinstance(row["FT_MINUTES"], float) and math.isnan(row["FT_MINUTES"])):
            forward_total_cost = float(row["FT_MINUTES"])
        else:
            forward_total_cost = float(row[cost_field])
        if cost_field.upper() == "MINUTES" and "TF_MINUTES" in row.index and row["TF_MINUTES"] is not None and not (isinstance(row["TF_MINUTES"], float) and math.isnan(row["TF_MINUTES"])):
            reverse_total_cost = float(row["TF_MINUTES"])
        else:
            reverse_total_cost = float(row[cost_field])
        for start, end in zip(coords[:-1], coords[1:]):
            segment_length = LineString([start, end]).length
            ratio = segment_length / geom_length
            distance_m = float(total_distance * ratio)
            if oneway == "FT":
                graph.add_edge(start, end, distance_m=distance_m, cost=float(forward_total_cost * ratio), road_name=road_name)
                directed_edge_count += 1
            elif oneway == "TF":
                graph.add_edge(end, start, distance_m=distance_m, cost=float(reverse_total_cost * ratio), road_name=road_name)
                directed_edge_count += 1
            else:
                graph.add_edge(start, end, distance_m=distance_m, cost=float(forward_total_cost * ratio), road_name=road_name)
                graph.add_edge(end, start, distance_m=distance_m, cost=float(reverse_total_cost * ratio), road_name=road_name)
                directed_edge_count += 2
    if graph.number_of_nodes() == 0:
        raise RuntimeError("Road network graph is empty")
    graph_info = {"node_count": int(graph.number_of_nodes()), "directed_edge_count": int(directed_edge_count), "road_feature_count": int(len(network)), "cost_field": cost_field, "uses_directional_minutes": bool(cost_field.upper() == "MINUTES" and "FT_MINUTES" in network.columns and "TF_MINUTES" in network.columns)}

    start_point = Point(float(start_x), float(start_y))
    end_point = Point(float(end_x), float(end_y))
    nodes = list(graph.nodes)
    start_node = nodes[0]
    start_snap_distance = math.inf
    for node in nodes:
        distance = start_point.distance(Point(node))
        if distance < start_snap_distance:
            start_node = node
            start_snap_distance = float(distance)
    end_node = nodes[0]
    end_snap_distance = math.inf
    for node in nodes:
        distance = end_point.distance(Point(node))
        if distance < end_snap_distance:
            end_node = node
            end_snap_distance = float(distance)

    route_nodes = nx.shortest_path(graph, start_node, end_node, weight="cost")
    total_cost = float(nx.shortest_path_length(graph, start_node, end_node, weight="cost"))

    total_distance = 0.0
    road_names: List[str] = []
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        attrs = graph[u][v]
        total_distance += float(attrs.get("distance_m", 0.0))
        road_name = attrs.get("road_name")
        if road_name and road_name not in road_names:
            road_names.append(str(road_name))

    route_line = LineString([(float(x), float(y)) for x, y in route_nodes])
    route_gdf = gpd.GeoDataFrame(
        [
            {
                "start_x": float(start_x),
                "start_y": float(start_y),
                "end_x": float(end_x),
                "end_y": float(end_y),
                "start_label": start_label,
                "end_label": end_label,
                "cost_field": cost_field,
                "cost_minutes": round(total_cost, 4),
                "distance_m": round(total_distance, 3),
                "start_snap_m": round(start_snap_distance, 3),
                "end_snap_m": round(end_snap_distance, 3),
                "node_count": int(len(route_nodes)),
                "road_names": ";".join(road_names[:20]),
                "geometry": route_line,
            }
        ],
        geometry="geometry",
    )

    route_path = None
    if output_name:
        route_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))
        os.makedirs(os.path.dirname(route_path), exist_ok=True)
        route_gdf.to_file(route_path, driver="GeoJSON")

    result = {
        "status": "success",
        "graph": graph_info,
        "start": {
            "x": float(start_x),
            "y": float(start_y),
            "label": start_label,
            "snapped_node": [float(start_node[0]), float(start_node[1])],
            "snap_distance": round(start_snap_distance, 3),
        },
        "end": {
            "x": float(end_x),
            "y": float(end_y),
            "label": end_label,
            "snapped_node": [float(end_node[0]), float(end_node[1])],
            "snap_distance": round(end_snap_distance, 3),
        },
        "route": {
            "cost_minutes": round(total_cost, 4),
            "distance_m": round(total_distance, 3),
            "node_count": int(len(route_nodes)),
            "road_names": road_names[:20],
        },
        "outputs": {
            "route_geojson": route_path,
        },
    }

    if report_output_name:
        report_path = os.path.abspath(os.path.join(OUTPUT_DIR, report_output_name))
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result["outputs"]["report_json"] = report_path
    return result

# @mcp.tool()
def summarize_route_result(
    route_geojson_path: str,
    report_json_path: Optional[str] = None,
    output_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Summarize a route GeoJSON and optional route report.

    Args:
        route_geojson_path: Path to the route GeoJSON file.
        report_json_path: Optional path to the detailed route report JSON.
        output_name: Optional summary JSON filename.

    Returns:
        status: str, "success" if the route layer is summarized.
        cost_minutes: float or None, total route cost in minutes.
        distance_m: float or None, total route distance in meters.
        node_count: int or None, number of route nodes.
        start_snap_m: float or None, start snap distance in meters.
        end_snap_m: float or None, end snap distance in meters.
        report_json_path: str or None, absolute route report path.
        graph_node_count: int, present when a report JSON is provided.
        graph_directed_edge_count: int, present when a report JSON is provided.
        road_names: list, present when a report JSON is provided.
        output_path: str, present when output_name is provided.
    """
    try:
        route = gpd.read_file(route_geojson_path, engine="fiona")
    except Exception:
        route = gpd.read_file(route_geojson_path)
    if len(route) == 0:
        raise RuntimeError("Route layer is empty")

    first = route.iloc[0]
    summary = {
        "status": "success",
        "cost_minutes": float(first["cost_minutes"]) if "cost_minutes" in route.columns else None,
        "distance_m": float(first["distance_m"]) if "distance_m" in route.columns else None,
        "node_count": int(first["node_count"]) if "node_count" in route.columns else None,
        "start_snap_m": float(first["start_snap_m"]) if "start_snap_m" in route.columns else None,
        "end_snap_m": float(first["end_snap_m"]) if "end_snap_m" in route.columns else None,
        "report_json_path": str(Path(report_json_path).resolve()) if report_json_path else None,
    }

    if report_json_path and Path(report_json_path).exists():
        with open(report_json_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        summary["graph_node_count"] = report.get("graph", {}).get("node_count")
        summary["graph_directed_edge_count"] = report.get("graph", {}).get("directed_edge_count")
        summary["road_names"] = report.get("route", {}).get("road_names", [])

    if output_name:
        output_path = os.path.abspath(os.path.join(OUTPUT_DIR, output_name))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        summary["output_path"] = output_path
    return summary