from typing import Tuple, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Point
from shapely.ops import triangulate


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


def get_similar_routes(reference_trip: MultiPolygon,
                       df_orig: pd.DataFrame) -> pd.Series:
    # Always control against mutation out of scope with Pandas dataframes
    df = df_orig.copy()
    
    # Make a GeoDataFrame of the entire point cloud
    df_xys = [Point(x, y) for x, y in zip(df['lon'].values, df['lat'].values)]
    df_gdf = gpd.GeoDataFrame(df, geometry=df_xys)

    print('Starting total GDF length: ', len(df_gdf))
    df_gdf.geometry = df_gdf.buffer(ACCURACY)
    df_int_gdf = df_gdf[df_gdf.intersects(reference_trip)]
    print('Subset that intersect target trip: ', len(df_int_gdf))

    # Get the IDs of those that intersect by Trip ID
    unique_trip_ids = list(set(df_int_gdf['trip_id'].values))
    print('These trips intersect: {}'.format(len(unique_trip_ids)))

    grouped_trips = (df_int_gdf.groupby('trip_id')
                        .apply(lambda g: g.geometry.unary_union))
    
    return grouped_trips


def unify_trips(grouped_trips: pd.Series):
    unioned = gpd.GeoSeries(grouped_trips).unary_union

    # If it is just a Poly, make it a MultiPoly
    if not isinstance(unioned, MultiPolygon):
        unioned = MultiPolygon([unioned])
    
    polys = [p for p in unioned]
    print('Unified polygons: '.format(len(polys)))

    # Drop the polygons in single MultoPolygon object
    return MultiPolygon(polys).simplify(0.0001)


def get_next_target_trip(df, use_vals):
    curr_best = None
    curr_best_len = 0
    for tid in use_vals:
        sub = df[df['trip_id'] == tid]
        if len(sub) > curr_best_len:
            curr_best = tid
            curr_best_len = len(sub)
    
    return curr_best


def triangulate_path_shape(path_shape: Union[Polygon, MultiPolygon]) -> gpd.GeoSeries:
    # Flatten triangulations of each polygon in shape into a
    # one-level array of triangles
    t_list = []
    for p in path_shape:
        triangles = MultiPolygon(triangulate(p, tolerance=0.0))
        t_list = t_list + [t for t in triangles]
        
    # Cast triangle array as a GeoSeries
    tl_gs = gpd.GeoSeries(t_list)

    # Toss 
    tri_cleaned = []
    for poly in triangles:
        if poly.centroid.intersects(MultiPolygon(polys)):
            tri_cleaned.append(poly)

    return gpd.GeoSeries(tri_cleaned)

