import os
import sys
import math

# --- AUTO-PATH SETTING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from prediction.travel_time import (
    flow_to_speed_kmh,
    travel_time_seconds,
    FREE_FLOW_THRESHOLD,
    DEFAULT_SPEED_LIMIT_KMH,
    DEFAULT_MIN_SPEED_KMH,
)

DISTANCE_KM       = 1.0
NUM_INTERSECTIONS = 1
TOLERANCE         = 0.001

def run_tc03() -> bool:
    """
    TC03: Flow-to-Travel-Time Conversion Logic Check.
    """
    print("\n>>> Running TC03: Flow-to-Travel-Time Logic Check...")

    all_passed = True
    test_flows = [0, 100, 351, 352, 600, 1000]
    labels = [
        "zero flow",
        "light traffic (free-flow)",
        "boundary: 351 veh/hr (still free-flow)",
        "boundary: 352 veh/hr (enters congested branch)",
        "heavy congestion",
        "extreme congestion",
    ]

    speeds       = []
    travel_times = []

    print("\n  {:>15}  {:>14}  {:>17}  {}".format(
        "Flow (veh/hr)", "Speed (km/h)", "Travel time (s)", "Label"))
    print("  " + "-" * 80)

    for flow, label in zip(test_flows, labels):
        try:
            spd = float(flow_to_speed_kmh(flow))
            tt  = float(travel_time_seconds(
                flow_veh_per_hour   = flow,
                distance_km         = DISTANCE_KM,
                intersections_count = NUM_INTERSECTIONS,
            ))
        except Exception as exc:
            print(f"[TC03] FAILED (crash): flow={flow} veh/hr raised: {exc}")
            all_passed = False
            speeds.append(None)
            continue

        if not math.isfinite(spd) or spd <= 0:
            print(f"[TC03] FAILED (speed invalid): flow={flow} -> speed={spd}")
            all_passed = False
        
        speeds.append(spd)
        travel_times.append(tt)
        print(f"  {flow:>15}  {spd:>14.4f}  {tt:>17.4f}  {label}")

    if None in speeds:
        print("[TC03] Result: FAILED -- one or more flows caused a crash.")
        return False

    # Check 1: Free-flow branch
    for i, flow in enumerate(test_flows):
        if flow <= FREE_FLOW_THRESHOLD:
            if abs(speeds[i] - DEFAULT_SPEED_LIMIT_KMH) > TOLERANCE:
                print(f"[TC03] FAILED: flow={flow} did not reach free-flow speed {DEFAULT_SPEED_LIMIT_KMH}")
                all_passed = False

    status = "PASSED" if all_passed else "FAILED"
    print(f"[TC03] Result: {status}")
    return all_passed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm")
    args = parser.parse_args()
    
    # Model name is ignored in this specific test as it validates fixed physics formulas
    success = run_tc03()
    sys.exit(0 if success else 1)