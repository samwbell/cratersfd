from .target_properties_module import *
import pycrs
import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Point, Polygon, MultiPolygon, LineString

moon_geod = Geod(a=1737400, b=1738100)

def densify_linestring(line, resolution=0.01):
    coords = np.array(line.coords)
    new_coords = [coords[0]]
    for i in range(1, len(coords)):
        start, end = coords[i - 1], coords[i]
        dist = np.linalg.norm(end - start)
        n_pts = max(2, int(dist / resolution))
        interp = np.linspace(start, end, n_pts)
        new_coords.extend(interp[1:].tolist())
    return LineString(new_coords)

def densify_polygon(polygon, resolution=0.01):
    exterior = densify_linestring(polygon.exterior, resolution)
    interiors = [
        densify_linestring(ring, resolution) 
        for ring in polygon.interiors
    ]
    return Polygon(exterior, interiors)

def compute_geodetic_area(polygon):
    global moon_geod
    total_area = 0
    lon, lat = polygon.exterior.xy
    area, _ = moon_geod.polygon_area_perimeter(lon, lat)
    total_area += area
    for interior in polygon.interiors:
        lon, lat = zip(*interior.coords)
        area, _ = moon_geod.polygon_area_perimeter(lon, lat)
        total_area -= area
    return abs(total_area)

def lunar_area(geom, resolution=0.01):
    if geom.is_empty:
        return 0.0
    if geom.geom_type == 'Polygon':
        poly = densify_polygon(geom, resolution)
        return compute_geodetic_area(poly) / 1e6
    elif geom.geom_type == 'MultiPolygon':
        total = 0
        for part in geom.geoms:
            poly = densify_polygon(part, resolution)
            total += compute_geodetic_area(poly)
        return total / 1e6
    else:
        raise TypeError('Input must be a Polygon or MultiPolygon.')

def width_at(polygon, y):
    xmin, xmax = polygon.bounds[0], polygon.bounds[2]
    buffer = xmax - xmin / 5
    line = LineString([(xmin - buffer, y), (xmax + buffer, y)])
    
    intersection = polygon.intersection(line)

    if intersection.is_empty:
        return 0

    if intersection.geom_type == 'LineString':
        return intersection.length
    elif intersection.geom_type == 'MultiLineString':
        return sum(line.length for line in intersection.geoms)
    else:
        raise ValueError('Must be a LineString or a MultiLineString')

def integrated_area(polygon, ny):
    r_moon = 1738.1
    ymin, ymax = polygon.bounds[1], polygon.bounds[3]
    ypad = (ymax - ymin) / ny / 2
    y_array = np.linspace(ymin + ypad, ymax - ypad, ny)
    y_widths = np.array([width_at(polygon, y) for y in y_array])
    latdeg = math.pi * r_moon / 180
    y_widths_km = np.array([
        np.cos(np.radians(y)) * latdeg * w
        for y, w in zip(y_array, y_widths)
    ])
    y_areas = y_widths_km * (ymax - ymin) / ny * latdeg
    return y_areas.sum()

orientale_center = Point(-92.8, -19.4)

def wedge_area(angle1, angle2):
    global orientale_center
    x1 = orientale_center.x + math.cos(angle1) * 800
    y1 = orientale_center.y + math.sin(angle1) * 800
    point1 = Point(x1, y1)
    x2 = orientale_center.x + math.cos(angle2) * 800
    y2 = orientale_center.y + math.sin(angle2) * 800
    point2 = Point(x2, y2)
    wedge = Polygon([orientale_center, point1, point2])
    wedge_gdf = gpd.GeoDataFrame(geometry=[wedge], crs=lunar_crs)
    regions['overlap'] = regions.geometry.intersection(wedge)
    areas = np.array([
        lunar_area(geom) 
        for geom in regions['overlap'].geometry
    ])
    return areas.sum()

