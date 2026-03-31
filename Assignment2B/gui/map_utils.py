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
        u = str(getattr(edge, "source", edge.get("source") if isinstance(edge, dict) else ""))
        v = str(getattr(edge, "target", edge.get("target") if isinstance(edge, dict) else ""))
        
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
                color=color, weight=3, opacity=0.8,
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
        u = str(getattr(edge, "source", edge.get("source") if isinstance(edge, dict) else ""))
        v = str(getattr(edge, "target", edge.get("target") if isinstance(edge, dict) else ""))
        if u in node_coords and v in node_coords:
            flow = edge_flow.get((u, v), 0.0)
            if flow < 400: color = "green"
            elif flow < 800: color = "gold"
            elif flow < 1200: color = "orange"
            else: color = "red"
            
            folium.PolyLine(
                locations=[node_coords[u], node_coords[v]],
                color=color, weight=3, opacity=0.7,
                tooltip=f"{u} -> {v}: {flow:.0f} veh/hr"
            ).add_to(edge_layer)

    # Add a legend for traffic colors to the overview map too
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; right: 50px; width: 120px; height: 90px; 
     border:2px solid grey; z-index:9999; font-size:11px;
     background-color:white; opacity: 0.8; padding: 10px; border-radius:8px;">
     <b>Flow (veh/hr)</b><br>
     <i style="background:green; border-radius:50%; width:8px; height:8px; display:inline-block"></i> < 400<br>
     <i style="background:gold; border-radius:50%; width:8px; height:8px; display:inline-block"></i> < 800<br>
     <i style="background:orange; border-radius:50%; width:8px; height:8px; display:inline-block"></i> < 1200<br>
     <i style="background:red; border-radius:50%; width:8px; height:8px; display:inline-block"></i> > 1200
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    return m

def create_focused_route_map(
    routes: List[Dict[str, Any]],
    node_coords: Dict[str, Tuple[float, float]],
    edge_flow: Dict[Tuple[str, str], float],
    center: Tuple[float, float] = (-37.83, 145.07),
    zoom: int = 14,
) -> folium.Map:
    """
    Creates a clean map showing ONLY the Top 5 routes, with segments 
    colored by their actual predicted traffic flow.
    """
    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")
    
    # Add the Traffic Legend (Consistent with overview)
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 160px; height: 110px; 
     border:2px solid grey; z-index:9999; font-size:12px;
     background-color:white; opacity: 0.9; padding: 10px; border-radius:8px;">
     <b>Traffic Flow (veh/hr)</b><br>
     <i style="background:green; width:10px; height:10px; display:inline-block"></i> < 400 (Free)<br>
     <i style="background:gold; width:10px; height:10px; display:inline-block"></i> 400-800 (Mod)<br>
     <i style="background:orange; width:10px; height:10px; display:inline-block"></i> 800-1200 (Hvy)<br>
     <i style="background:red; width:10px; height:10px; display:inline-block"></i> > 1200 (Congest)<br>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    route_layer = folium.FeatureGroup(name="Top 5 Routes (Traffic Colored)").add_to(m)
    
    # Process all unique segments across top 5 routes to avoid double drawing
    seen_segments = set()
    
    for route in routes[:5]:
        for seg in route.get("segments", []):
            u, v = str(seg["from"]), str(seg["to"])
            if (u, v) in seen_segments: continue
            seen_segments.add((u, v))
            
            if u in node_coords and v in node_coords:
                flow = seg.get("predicted_flow_veh_per_hour", 0.0)
                if flow is None: flow = 0.0
                
                if flow < 400: color = "green"
                elif flow < 800: color = "gold"
                elif flow < 1200: color = "orange"
                else: color = "red"
                
                folium.PolyLine(
                    locations=[node_coords[u], node_coords[v]],
                    color=color, weight=7, opacity=0.9,
                    tooltip=f"{u} -> {v}: {flow:.0f} veh/hr"
                ).add_to(route_layer)

    # Add high-visibility Start/End markers
    if routes and routes[0]["path"]:
        start_node = str(routes[0]["path"][0])
        end_node = str(routes[0]["path"][-1])
        
        if start_node in node_coords:
            folium.Marker(
                location=node_coords[start_node],
                icon=folium.Icon(color='blue', icon='play', prefix='fa'),
                tooltip="ORIGIN"
            ).add_to(m)
            
        if end_node in node_coords:
            folium.Marker(
                location=node_coords[end_node],
                icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa'),
                tooltip="DESTINATION"
            ).add_to(m)

    return m
