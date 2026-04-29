from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple


Coordinate = Tuple[float, float]
Ring = List[Coordinate]
Polygon = List[Ring]


def load_geojson_polygons(path: Path) -> list[Polygon]:
    data = json.loads(path.read_text(encoding="utf-8"))
    polygons: list[Polygon] = []

    for feature in data.get("features", []):
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates") or []

        if geometry_type == "Polygon":
            polygons.append(_parse_polygon(coordinates))
        elif geometry_type == "MultiPolygon":
            polygons.extend(_parse_polygon(polygon) for polygon in coordinates)

    if not polygons:
        raise ValueError(f"No Polygon or MultiPolygon features found in {path}")
    return polygons


def load_geojson(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_polygon(coordinates: list[Any]) -> Polygon:
    return [[(float(lon), float(lat)) for lon, lat in ring] for ring in coordinates]


def polygon_bbox(polygons: list[Polygon]) -> tuple[float, float, float, float]:
    points = [point for polygon in polygons for ring in polygon for point in ring]
    min_lon = min(lon for lon, _ in points)
    min_lat = min(lat for _, lat in points)
    max_lon = max(lon for lon, _ in points)
    max_lat = max(lat for _, lat in points)
    return min_lon, min_lat, max_lon, max_lat


def point_in_ring(lon: float, lat: float, ring: Ring) -> bool:
    inside = False
    previous_lon, previous_lat = ring[-1]

    for current_lon, current_lat in ring:
        crosses = (current_lat > lat) != (previous_lat > lat)
        if crosses:
            slope_lon = (previous_lon - current_lon) * (lat - current_lat)
            slope_lon = slope_lon / (previous_lat - current_lat) + current_lon
            if lon < slope_lon:
                inside = not inside
        previous_lon, previous_lat = current_lon, current_lat

    return inside


def point_in_polygon(lon: float, lat: float, polygon: Polygon) -> bool:
    if not polygon or not point_in_ring(lon, lat, polygon[0]):
        return False
    return not any(point_in_ring(lon, lat, hole) for hole in polygon[1:])


def point_in_polygons(lon: float | None, lat: float | None, polygons: list[Polygon]) -> bool:
    if lon is None or lat is None:
        return False
    return any(point_in_polygon(float(lon), float(lat), polygon) for polygon in polygons)
