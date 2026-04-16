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

