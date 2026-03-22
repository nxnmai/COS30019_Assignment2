from __future__ import annotations

import argparse
from typing import Union

import numpy as np

FLOW_SPEED_A = 1.4648375
FLOW_SPEED_B = 93.75
FREE_FLOW_THRESHOLD = 351.0
DEFAULT_SPEED_LIMIT_KMH = 60.0
DEFAULT_MIN_SPEED_KMH = 5.0
DEFAULT_INTERSECTION_DELAY_SEC = 30.0

ArrayLike = Union[float, np.ndarray]


def flow_to_speed_kmh(
    flow_veh_per_hour: ArrayLike,
    speed_limit_kmh: float = DEFAULT_SPEED_LIMIT_KMH,
    free_flow_threshold: float = FREE_FLOW_THRESHOLD,
    min_speed_kmh: float = DEFAULT_MIN_SPEED_KMH,
) -> ArrayLike:
    """
    Convert flow (veh/hr) to speed (km/h) using the assignment formula:
      flow = -1.4648375 * speed^2 + 93.75 * speed

    Branching:
      flow <= 351  -> speed = 60 (free flow, capped by speed_limit)
      flow >  351  -> lower quadratic root (congested branch)
    """

    is_scalar = np.isscalar(flow_veh_per_hour)
    flow = np.asarray(flow_veh_per_hour, dtype=np.float64)
    flow = np.clip(flow, 0.0, None)

    speed = np.full(flow.shape, speed_limit_kmh, dtype=np.float64)
    congested_mask = flow > free_flow_threshold

    if np.any(congested_mask):
        congested_flow = flow[congested_mask]
        discriminant = (FLOW_SPEED_B**2) - (4.0 * FLOW_SPEED_A * congested_flow)
        invalid_mask = discriminant < 0.0
        discriminant = np.clip(discriminant, 0.0, None)

        # Lower root = congested branch.
        congested_speed = (FLOW_SPEED_B - np.sqrt(discriminant)) / (2.0 * FLOW_SPEED_A)
        congested_speed = np.clip(congested_speed, min_speed_kmh, speed_limit_kmh)
        congested_speed[invalid_mask] = min_speed_kmh

        speed[congested_mask] = congested_speed

    speed = np.clip(speed, min_speed_kmh, speed_limit_kmh)
    if is_scalar:
        return float(speed.item())
    return speed


def travel_time_seconds(
    flow_veh_per_hour: ArrayLike,
    distance_km: ArrayLike,
    intersection_delay_sec: float = DEFAULT_INTERSECTION_DELAY_SEC,
    intersections_count: int = 1,
    speed_limit_kmh: float = DEFAULT_SPEED_LIMIT_KMH,
    free_flow_threshold: float = FREE_FLOW_THRESHOLD,
    min_speed_kmh: float = DEFAULT_MIN_SPEED_KMH,
) -> ArrayLike:
    """
    Compute travel time in seconds:
      travel_time = (distance / speed) * 3600 + 30 seconds per intersection
    """

    is_scalar = np.isscalar(flow_veh_per_hour) and np.isscalar(distance_km)
    distance = np.asarray(distance_km, dtype=np.float64)
    distance = np.clip(distance, 0.0, None)

    speed = flow_to_speed_kmh(
        flow_veh_per_hour=flow_veh_per_hour,
        speed_limit_kmh=speed_limit_kmh,
        free_flow_threshold=free_flow_threshold,
        min_speed_kmh=min_speed_kmh,
    )
    speed_arr = np.asarray(speed, dtype=np.float64)

    base_time_sec = np.divide(
        distance,
        speed_arr,
        out=np.zeros_like(distance, dtype=np.float64),
        where=speed_arr > 0.0,
    ) * 3600.0
    total_time_sec = base_time_sec + (intersection_delay_sec * max(intersections_count, 0))

    if is_scalar:
        return float(np.asarray(total_time_sec).item())
    return total_time_sec


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flow to travel-time converter.")
    parser.add_argument("--flow", type=float, required=True, help="Predicted flow in vehicles/hour")
    parser.add_argument("--distance-km", type=float, required=True, help="Road edge distance in km")
    parser.add_argument("--intersections", type=int, default=1, help="Number of intersections on route segment")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    speed = flow_to_speed_kmh(args.flow)
    time_sec = travel_time_seconds(args.flow, args.distance_km, intersections_count=args.intersections)

    print(f"Flow: {args.flow:.2f} veh/hr")
    print(f"Speed: {speed:.2f} km/h")
    print(f"Travel time: {time_sec:.2f} sec")


if __name__ == "__main__":
    main()
