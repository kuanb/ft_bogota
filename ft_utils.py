from typing import List, Tuple, Union

import abc
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Polygon, Point
from shapely.geometry.polygon import orient
from shapely.ops import triangulate

import ft_polyskel
import importlib
importlib.reload(ft_polyskel)


# TODO: Working in 4326 project is dumb. Should be using meter projection.
# Note: This is roughly 50km resolution
ACCURACY = 0.0005


def clean_base_df(df: pd.DataFrame, bbox: Tuple) -> pd.DataFrame:
    # Convert column names to easier strings
    df.columns = ['date',
                  'lon',
                  'lat',
                  'alt',
                  'speed',
                  'username',
                  'survey_id',
                  'trip_id',
                  'session_id']

    # Convert date col to a pandas datetime type
    df['date'] = pd.to_datetime(df['date'])

    # Sort by the date column
    df.sort_values(by='date', axis=0, inplace=True)

    # Let's check data that has bad coordinate
    lo_null = df['lon'].isnull()
    la_null = df['lat'].isnull()

    bad = len(df[(lo_null | la_null)])
    good = len(df[~(lo_null | la_null)])

    # Results aren't too bad
    print('{} bad data points out of {} total; or {}%.'.format(bad, good, round(bad/good * 100, 2)))

    # Create a cleaned dataset to work with moving forward
    dfc = df[~(lo_null | la_null)]

    too_high = dfc['lat'] > bbox[3]
    too_low = dfc['lat'] < bbox[1]
    too_west = dfc['lon'] < bbox[0]
    too_east = dfc['lon'] > bbox[2]

    dp_len = len(dfc[(too_high | too_low | too_west | too_east)])
    print('{} out of bounds data points.'.format(dp_len))

    # so we need to drop those from the cleaned data set as well
    dfc = dfc[~(too_high | too_low | too_west | too_east)]

    return dfc


def extract_single_trip(df_orig: pd.DataFrame, target_trip: str) -> MultiPolygon:
    """
    Extracts all trips that intersect with the target trip id
    """

    # Always control against mutation out of scope with Pandas dataframes
    df = df_orig.copy()

    single_trip_sub = df[df['trip_id'] == target_trip]

    # Get all lat and lon values from parent dataframe
    single_trip_lon = single_trip_sub['lon'].values
    single_trip_lat = single_trip_sub['lat'].values

    # Convert all coord values to Shapely Point objects
    single_tr_xys = [Point(x, y) for x, y in zip(single_trip_lon, single_trip_lat)]

    # Convert the new Points array to a GeoSeries and buffer
    gs = gpd.GeoSeries(single_tr_xys)
    gsb = gs.buffer(ACCURACY)

    # Now we can extract a single MultiPoly of the trip path
    return gsb.unary_union


def route_simplify(tdf):
    keep_pts = [tdf.geometry.values[0]]
    for v in tdf.geometry.values:
        if not keep_pts[-1] == v:
            keep_pts.append(v)
            
    if len(keep_pts) < 4:
        return None
    else:
        ls = LineString(keep_pts)
        return ls.simplify(0.005).buffer(0.005)

    
def ensure_sufficient_overlap(reference_trip, series):
    rtrip_bufff = reference_trip.buffer(ACCURACY)
    geoms = list(series)
    g_index = list(series.index)
    
    ok_list = []
    for ix, g in zip(g_index, geoms):
        i_area = g.intersection(rtrip_bufff).area
        proportion_satisfied = i_area/g.area > 0.125
        ok_list.append(proportion_satisfied)
            
    return series[np.array(ok_list)]


def get_similar_routes(reference_trip: MultiPolygon,
                       df_orig: pd.DataFrame) -> pd.Series:
    # Always control against mutation out of scope with Pandas dataframes
    df = df_orig.copy()
    
    # Make a GeoDataFrame of the entire point cloud
    df_xys = [Point(x, y) for x, y in zip(df['lon'].values, df['lat'].values)]
    df_gdf = gpd.GeoDataFrame(df, geometry=df_xys)
    
    new_geoms = df_gdf.groupby('trip_id').apply(route_simplify)

    clean_lines_ids = []
    clean_lines = []
    for ls_id, ls in zip(new_geoms.index, new_geoms.values):
        if ls is not None:
            clean_lines_ids.append(ls_id)
            clean_lines.append(ls)
            
    df_gdf = gpd.GeoDataFrame({'trip_id': clean_lines_ids}, geometry=clean_lines)

    df_int_gdf = df_gdf[df_gdf.intersects(reference_trip)]

    # Get the IDs of those that intersect by Trip ID
    unique_trip_ids = list(set(df_int_gdf['trip_id'].values))
    print('Subset that intersect target trip: ', len(unique_trip_ids))

    # Only keep trips that overlap at least 1/3 with reference trip
    grouped_trips = (df_int_gdf.groupby('trip_id')
                        .apply(lambda g: g.geometry.unary_union))

    valid_subset = ensure_sufficient_overlap(reference_trip, grouped_trips)
    print('Of that subset: {} are valid'.format(len(valid_subset)))

    return valid_subset


def unify_trips(grouped_trips: pd.Series) -> MultiPolygon:
    unioned = gpd.GeoSeries(grouped_trips).unary_union

    # If it is just a Poly, make it a MultiPoly
    if not isinstance(unioned, MultiPolygon):
        unioned = MultiPolygon([unioned])
    
    polys = [p for p in unioned]
    print('Unified polygons: {}'.format(len(polys)))

    # Drop the polygons in single MultoPolygon object
    return MultiPolygon(polys).simplify(0.0001)


def get_next_target_trip(df: pd.DataFrame, use_vals: List[str]) -> str:
    curr_best = None
    curr_best_len = 0
    for tid in use_vals:
        sub = df[df['trip_id'] == tid]
        if len(sub) > curr_best_len:
            curr_best = tid
            curr_best_len = len(sub)
    
    return curr_best


def triangulate_path_shape(path_shape: Union[Polygon, MultiPolygon]) -> List[Polygon]:
    # Convert a Polygon into a MultiPoly if it is submitted as the path_shape
    if not isinstance(path_shape, MultiPolygon):
        path_shape = MultiPolygon([path_shape])

    # Flatten triangulations of each polygon in shape into a
    # one-level array of triangles
    t_list = []
    for p in path_shape:
        triangles = MultiPolygon(triangulate(p, tolerance=0.0))
        t_list = t_list + [t for t in triangles]

    # Toss 
    tri_cleaned = []
    for poly in t_list:
        if poly.centroid.buffer(0.0001).intersects(path_shape):
            tri_cleaned.append(poly)

    return tri_cleaned

class NamedPoints(abc.ABC):
    def __init__(self):
        self.all_pts = [] # list of dicts

    def get_nearest_node_id(self, curr_pt):
        threshold = 0.00000001
        this_pt_id = None

        for pt in self.all_pts:
            # If this is a point that already exists, then
            # do not add a new point
            if pt['pt'].distance(curr_pt) < threshold:
                this_pt_id = pt['id']
        return this_pt_id

    def get_node_by_id(self, want_id):
        for p in self.all_pts:
            if p['id'] == want_id:
                return p['pt']

        return None
    
    def add(self, new_node):
        self.all_pts.append(new_node)
        return None
    
    def get_len(self):
        return len(self.all_pts)

def run_skeletonization(shape: Union[Polygon, MultiPolygon]):
    # Convert a Polygon into a MultiPoly if it is submitted as the shape
    if not isinstance(shape, MultiPolygon):
        shape = MultiPolygon([shape])

    # Prep the shape by pulling out the largest area from all
    # of the shapes that are in the MPoly
    curr_area = 0
    curr_poly = None
    for p in shape:
        if p.area > curr_area:
            curr_area = p.area
            curr_poly = p
    simple = curr_poly

    # Convert the simple shape to only its exterior, so that we don't
    # preserve any interior 'islands'
    simple = Polygon(simple.exterior).buffer(0.005).simplify(0.005)

    # Ensure that the shape's exterior ring is counter-clockwise
    shape_oriented = orient(simple, sign=1.0)

    sh_bounds = list(simple.exterior.coords)

    try:
        new_skeleton = ft_polyskel.skeletonize(sh_bounds, [])
    except TypeError as e:
        print('Caught error: {}'.format(e))
        print('Attempting a try on a more simplified shape.')
        new_skeleton = run_skeletonization(simple.buffer(0.005))

    return new_skeleton


def get_farthest_point_pair(skeleton) -> Tuple[Point]:
    sg = nx.Graph()
        
    # First we need to create all unique points and update edges
    init_point_id = 0

    named_points = NamedPoints()
    node_pos = {}
    for edge in skeleton:
        source = edge.source
        
        # Convert Point2 Pyeuclid types to standard Point
        source_converted = Point(source[0], source[1])
        
        pts_to_eval = [source_converted]
        for sink in edge.sinks:
            sink_converted = Point(sink[0], sink[1])
            pts_to_eval.append(sink_converted)
        
        for eval_pt in pts_to_eval:
            this_pt_id = named_points.get_nearest_node_id(eval_pt)

            if this_pt_id is None:
                init_point_id += 1
                this_pt_id = init_point_id

                # Update the list
                named_points.add({'id': this_pt_id, 'pt': eval_pt})

                # And update the graph
                sg.add_node(this_pt_id)
                node_pos[this_pt_id] = (eval_pt.x, eval_pt.y)


    # Now go ahead and add the edges by repeating the same loop, 
    # with the named points all populated already
    for edge in skeleton:
        # Convert Point2 Pyeuclid types to standard Point
        source = edge.source
        source_converted = Point(source[0], source[1])
        
        # Start value
        start_pt_id = named_points.get_nearest_node_id(source_converted)
        
        # Iterate through the end values
        for sink in edge.sinks:
            sink_converted = Point(sink[0], sink[1])
            end_pt_id = named_points.get_nearest_node_id(sink_converted)
            
            # Make sure to add pathways in both directions
            # to ensure possibility of bi-directional flow
            edge_dist = source_converted.distance(sink_converted)
            sg.add_edge(start_pt_id, end_pt_id, weight=edge_dist)
            sg.add_edge(end_pt_id, start_pt_id, weight=edge_dist)

    p = nx.shortest_path(sg)

    curr_max_wt = 0
    curr_path = None
    for p1 in p.keys():
        for p2 in p[p1].keys():
            curr_p = p[p1][p2]

            tot_wt = 0
            for e1, e2 in zip(curr_p[:-1], curr_p[1:]):
                wt = sg.edge[e1][e2]['weight']
                tot_wt += wt
        
        if curr_max_wt < tot_wt:
            curr_max_wt = tot_wt
            curr_path = curr_p

    first_point = Point(node_pos[curr_path[0]])
    last_point = Point(node_pos[curr_path[-1]])

    return (first_point, last_point)


def great_circle_vec(lat1, lng1, lat2, lng2, earth_radius=6371009):
    """
    Vectorized function to calculate the great-circle distance between two points or between vectors of points.
    Parameters
    ----------
    lat1 : float or array of float
    lng1 : float or array of float
    lat2 : float or array of float
    lng2 : float or array of float
    earth_radius : numeric
        radius of earth in units in which distance will be returned (default is meters)
    Returns
    -------
    distance : float or array of float
        distance or vector of distances from (lat1, lng1) to (lat2, lng2) in units of earth_radius
    """

    phi1 = np.deg2rad(90 - lat1)
    phi2 = np.deg2rad(90 - lat2)

    theta1 = np.deg2rad(lng1)
    theta2 = np.deg2rad(lng2)

    cos = (np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2) + np.cos(phi1) * np.cos(phi2))

    arc = np.arccos(cos)

    # return distance in units of earth_radius
    distance = arc * earth_radius
    return distance


def generate_multidigraph(tri_cleaned: List,
                          start_pt: Point,
                          end_pt: Point) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    node_count = 0

    id_of_end_pt = None
    curr_lowest_dist_end = 100000000
    id_of_start_pt = None
    curr_lowest_dist_start = 100000000

    for tri in tri_cleaned:
        coords = list(tri.exterior.coords)

        ids = list(map(lambda x: node_count + x, [1, 2, 3]))
        
        for one_id, coord in zip(ids, coords[0:3]):
            
            end_pt_dist = end_pt.distance(Point(*coord))
            if end_pt_dist < curr_lowest_dist_end:
                id_of_end_pt = one_id
                curr_lowest_dist_end = end_pt_dist

            start_pt_dist = start_pt.distance(Point(*coord))
            if start_pt_dist < curr_lowest_dist_start:
                id_of_start_pt = one_id
                curr_lowest_dist_start = start_pt_dist
            
            G.add_node(one_id, x=coord[0], y=coord[1])

        for a, b, i in zip(coords[:-1], coords[1:], ids):
            d = great_circle_vec(a[1], a[0], b[1], b[0])

            a_id = i
            b_id = i + 1

            if b_id > ids[-1]:
                b_id = ids[0]

            # Make sure to add pathways in both directions
            # to ensure possibility of bi-directional flow
            G.add_edge(a_id, b_id, length = d)
            G.add_edge(b_id, a_id, length = d)

        node_count += 4

    # Now we need to add connectors between each of the triangles
    # from within the triangle collection
    nodes_as_pts = []
    node_ids = []

    for node_id, xy in G.nodes(data=True):
        nodes_as_pts.append(Point(xy['x'], xy['y']))
        node_ids.append(node_id)

    node_gdf = gpd.GeoDataFrame(node_ids, geometry=nodes_as_pts, columns=['node_id'])
        
    for node_id, xy in G.nodes(data=True):
        node_pt = Point(xy['x'], xy['y'])
        ds = node_gdf.distance(node_pt)
        sub = node_gdf[ds < 0.0001]
        
        for to_id in sub['node_id'].values:
            G.add_edge(node_id, to_id, length = 0.00001)
            G.add_edge(to_id, node_id, length = 0.00001)
    
    # Update with some information for OSMnx if we need it
    G.graph['crs'] = {'init':'epsg:4326'}
    G.graph['name'] = 'gtfs_network'

    return (G, id_of_end_pt, id_of_start_pt)




