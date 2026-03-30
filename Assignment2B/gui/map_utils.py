import folium
from folium import plugins
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

def create_traffic_map(
    edges: List[Any],
    node_coords: Dict[str, Tuple[float, float]],
    edge_flow: Dict[Tuple[str, str], float],
    edge_weights: Dict[Tuple[str, str], float],
    center: Tuple[float, float] = (-37.83, 145.07),
    zoom: int = 14,
) -> folium.Map:
    """
    Creates a Folium map showing the road network colored by predicted flow/congestion.
    """
    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")
    
    # Add a legend for traffic colors
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 110px; 
     border:2px solid grey; z-index:9999; font-size:12px;
     background-color:white; opacity: 0.9; padding: 10px; border-radius:8px;">
     <b>Traffic Flow (veh/hr)</b><br>
     <i style="background:green; width:10px; height:10px; display:inline-block"></i> < 400 (Free)<br>
     <i style="background:gold; width:10px; height:10px; display:inline-block"></i> 400-800 (Mod)<br>
     <i style="background:orange; width:10px; height:10px; display:inline-block"></i> 800-1200 (Hvy)<br>
     <i style="background:red; width:10px; height:10px; display:inline-block"></i> > 1200 (Cong)<br>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    network_layer = folium.FeatureGroup(name="Road Network (Colored by Flow)")
    
    for edge in edges:
        # Support both class objects and dictionaries
        u = getattr(edge, "source", edge.get("source") if isinstance(edge, dict) else None)
        v = getattr(edge, "target", edge.get("target") if isinstance(edge, dict) else None)
        
        if u in node_coords and v in node_coords:
            p1 = node_coords[u]
            p2 = node_coords[v]
            
            flow = edge_flow.get((u, v), 0.0)
            if flow < 400: color = "green"
            elif flow < 800: color = "gold"
            elif flow < 1200: color = "orange"
            else: color = "red"
            
            folium.PolyLine(
                locations=[p1, p2],
                color=color, weight=4, opacity=0.7,
                tooltip=f"{u} -> {v}: {flow:.0f} veh/hr"
            ).add_to(network_layer)

    network_layer.add_to(m)
    folium.LayerControl().add_to(m)
    return m

def create_network_overview_map(
    edges: List[Any],
    node_coords: Dict[str, Tuple[float, float]],
    edge_flow: Dict[Tuple[str, str], float],
    center: Tuple[float, float] = (-37.83, 145.07),
    zoom: int = 13,
) -> folium.Map:
    """
    Creates a large overview map of all SCATS sites and predicted flows.
    """
    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")
    
    # Add Markers for all sites
    marker_cluster = plugins.MarkerCluster(name="SCATS sites").add_to(m)
    for nid, coord in node_coords.items():
        folium.CircleMarker(
            location=coord,
            radius=5,
            color="black",
            fill=True,
            fill_color="white",
            fill_opacity=1,
            tooltip=f"Site {nid}"
        ).add_to(marker_cluster)

    # Add edges
    edge_layer = folium.FeatureGroup(name="Network Flow").add_to(m)
    for edge in edges:
        u = getattr(edge, "source", edge.get("source") if isinstance(edge, dict) else None)
        v = getattr(edge, "target", edge.get("target") if isinstance(edge, dict) else None)
        if u in node_coords and v in node_coords:
            flow = edge_flow.get((u, v), 0.0)
            if flow < 400: color = "green"
            elif flow < 800: color = "gold"
            elif flow < 1200: color = "orange"
            else: color = "red"
            
            folium.PolyLine(
                locations=[node_coords[u], node_coords[v]],
                color=color, weight=3, opacity=0.5
            ).add_to(edge_layer)

    folium.LayerControl().add_to(m)
    return m

def highlight_routes(
    m: folium.Map,
    routes: List[Dict[str, Any]],
    node_coords: Dict[str, Tuple[float, float]],
) -> folium.Map:
    """
    Highlights the selected routes on the map with distinctive colors and markers.
    """
    colors = ["#6c63ff", "#ff6363", "#48c6ef", "#9d63ff", "#ffbd63"]
    route_layer = folium.FeatureGroup(name="Highlighted Routes").add_to(m)
    
    for idx, route in enumerate(routes[:5]):
        path = route["path"]
        points = [node_coords[node] for node in path if node in node_coords]
        
        if len(points) < 2:
            continue
            
        color = colors[idx % len(colors)]
        folium.PolyLine(
            locations=points,
            color=color, weight=8, opacity=0.8,
            popup=f"Route {idx+1}: {route['total_time_min']:.1f} min"
        ).add_to(route_layer)
        
        # Markers for origin/destination (always visible)
        if idx == 0:
            folium.Marker(
                location=points[0],
                icon=folium.Icon(color='blue', icon='play', prefix='fa'),
                tooltip=f"ORIGIN: {path[0]}"
            ).add_to(m)
            folium.Marker(
                location=points[-1],
                icon=folium.Icon(color='darkred', icon='flag-checkered', prefix='fa'),
                tooltip=f"DESTINATION: {path[-1]}"
            ).add_to(m)

    return m
