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
    zoom: int = 13,
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
     background-color:white; opacity: 0.8; padding: 10px;">
     <b>Traffic Flow Index</b><br>
     <i style="background:green; width:10px; height:10px; display:inline-block"></i> Free Flow<br>
     <i style="background:gold; width:10px; height:10px; display:inline-block"></i> Moderate<br>
     <i style="background:orange; width:10px; height:10px; display:inline-block"></i> Heavy<br>
     <i style="background:red; width:10px; height:10px; display:inline-block"></i> Congested<br>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    for edge in edges:
        u, v = edge.source, edge.target
        if u in node_coords and v in node_coords:
            p1 = node_coords[u]
            p2 = node_coords[v]
            
            flow = edge_flow.get((u, v), 0.0)
            # Determine color based on flow (vehicles per hour)
            # Thresholds: < 400 (Green), 400-800 (Yellow), 800-1200 (Orange), > 1200 (Red)
            if flow < 400:
                color = "green"
            elif flow < 800:
                color = "gold"
            elif flow < 1200:
                color = "orange"
            else:
                color = "red"
            
            popup_text = f"Edge {u}->{v}<br>Flow: {flow:.1f} veh/hr<br>Time: {edge_weights.get((u,v), 0):.1f}s"
            
            folium.PolyLine(
                locations=[p1, p2],
                color=color,
                weight=4,
                opacity=0.7,
                popup=popup_text
            ).add_to(m)

    return m

def highlight_routes(
    m: folium.Map,
    routes: List[Dict[str, Any]],
    node_coords: Dict[str, Tuple[float, float]],
) -> folium.Map:
    """
    Highlights the selected routes on the map with distinctive colors.
    """
    colors = ["#6c63ff", "#ff6363", "#48c6ef", "#9d63ff", "#ffbd63"]
    
    for idx, route in enumerate(routes[:5]):
        path = route["path"]
        points = [node_coords[node] for node in path if node in node_coords]
        
        if len(points) < 2:
            continue
            
        folium.PolyLine(
            locations=points,
            color=colors[idx % len(colors)],
            weight=6,
            opacity=0.9,
            dash_array='10',
            popup=f"Route {idx+1} ({route['total_time_min']:.1f} min)"
        ).add_to(m)
        
        # Add start/end markers for the first route
        if idx == 0:
            folium.Marker(
                location=points[0],
                icon=folium.Icon(color='green', icon='play'),
                tooltip="Start"
            ).add_to(m)
            folium.Marker(
                location=points[-1],
                icon=folium.Icon(color='red', icon='stop'),
                tooltip="Destination"
            ).add_to(m)

    return m
