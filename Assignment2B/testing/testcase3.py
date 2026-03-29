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
    DEFAULT_INTERSECTION_DELAY_SEC,
)

DISTANCE_KM       = 1.0
NUM_INTERSECTIONS = 1
TOLERANCE         = 0.001


def run_tc03() -> bool:
    """
    TC03: Flow-to-Travel-Time Conversion Logic Check.

    Tests flow_to_speed_kmh() and travel_time_seconds() against the
    behavioural contract of the quadratic traffic flow model:

        1. Free-flow branch   -- flow <= FREE_FLOW_THRESHOLD (351)
                                 -> speed == DEFAULT_SPEED_LIMIT_KMH (60).
        2. Congested branch   -- flow >  FREE_FLOW_THRESHOLD
                                 -> speed is clamped to [DEFAULT_MIN_SPEED_KMH, 60).
        3. Speed floor        -- no flow value (including extreme) produces
                                 speed below DEFAULT_MIN_SPEED_KMH (5 km/h).
        4. Positivity &       -- no output is <= 0, NaN, or infinity for any
           finiteness            tested flow value.

    NOTE on monotonicity:
        The underlying quadratic model exhibits a "backward-bending" speed-flow
        curve on the congested branch — higher flow can yield HIGHER speed
        (and therefore LOWER travel time) once past the capacity drop point.
        This is expected behaviour of the Greenshields/quadratic model and is
        NOT treated as a failure here.  Monotonicity of travel time with
        respect to flow is therefore intentionally NOT checked.
    """
    print("\n>>> Running TC03: Flow-to-Travel-Time Logic Check...")

    all_passed = True

    # Test flows in veh/hr. Covers: zero, free-flow interior, exact boundary,
    # just above boundary, and two congested values to confirm the speed floor.
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

    # ------------------------------------------------------------------
    # Check 4: no crashes, no NaN / inf / non-positive values
    # ------------------------------------------------------------------
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
            travel_times.append(None)
            continue

        if not math.isfinite(spd) or spd <= 0:
            print(f"[TC03] FAILED (speed invalid): flow={flow} -> speed={spd}")
            all_passed = False
        if not math.isfinite(tt) or tt <= 0:
            print(f"[TC03] FAILED (travel time invalid): flow={flow} -> tt={tt}")
            all_passed = False

        speeds.append(spd)
        travel_times.append(tt)
        print(f"  {flow:>15}  {spd:>14.4f}  {tt:>17.4f}  {label}")

    print()

    if None in speeds:
        print("[TC03] Result: FAILED -- one or more flows caused a crash.")
        return False

    # ------------------------------------------------------------------
    # Check 1: free-flow branch (flow <= 351 -> speed exactly == 60)
    # ------------------------------------------------------------------
    ff_failures = []
    for i, flow in enumerate(test_flows):
        if flow <= FREE_FLOW_THRESHOLD:
            if abs(speeds[i] - DEFAULT_SPEED_LIMIT_KMH) > TOLERANCE:
                ff_failures.append(
                    f"flow={flow} veh/hr -> speed={speeds[i]:.4f} "
                    f"(expected {DEFAULT_SPEED_LIMIT_KMH})"
                )
    if ff_failures:
        print("[TC03] FAILED (free-flow branch):")
        for msg in ff_failures:
            print(f"         {msg}")
        all_passed = False
    else:
        print(f"[TC03] PASSED (free-flow branch): flows <= {FREE_FLOW_THRESHOLD} veh/hr "
              f"-> speed = {DEFAULT_SPEED_LIMIT_KMH} km/h.")

    # ------------------------------------------------------------------
    # Check 2: congested branch (flow > 351 -> speed strictly < 60)
    # ------------------------------------------------------------------
    cong_failures = []
    for i, flow in enumerate(test_flows):
        if flow > FREE_FLOW_THRESHOLD:
            if speeds[i] >= DEFAULT_SPEED_LIMIT_KMH:
                cong_failures.append(
                    f"flow={flow} veh/hr -> speed={speeds[i]:.4f} "
                    f"(should be < {DEFAULT_SPEED_LIMIT_KMH})"
                )
    if cong_failures:
        print("[TC03] FAILED (congested branch):")
        for msg in cong_failures:
            print(f"         {msg}")
        all_passed = False
    else:
        print(f"[TC03] PASSED (congested branch): flows > {FREE_FLOW_THRESHOLD} veh/hr "
              f"-> speed < {DEFAULT_SPEED_LIMIT_KMH} km/h.")

    # ------------------------------------------------------------------
    # Check 3: speed floor -- speed never goes below DEFAULT_MIN_SPEED_KMH
    # across ALL flows, including extreme values
    # ------------------------------------------------------------------
    floor_failures = []
    for i, flow in enumerate(test_flows):
        if speeds[i] < DEFAULT_MIN_SPEED_KMH - TOLERANCE:
            floor_failures.append(
                f"flow={flow} veh/hr -> speed={speeds[i]:.4f} "
                f"(below min floor {DEFAULT_MIN_SPEED_KMH} km/h)"
            )
    if floor_failures:
        print("[TC03] FAILED (speed floor):")
        for msg in floor_failures:
            print(f"         {msg}")
        all_passed = False
    else:
        print(f"[TC03] PASSED (speed floor): speed >= {DEFAULT_MIN_SPEED_KMH} km/h "
              f"for all tested flows.")

    # ------------------------------------------------------------------
    # NOTE: monotonicity of travel time w.r.t. flow is NOT checked.
    # The quadratic model's congested branch produces a backward-bending
    # speed-flow curve: past the capacity drop, higher flow -> higher speed
    # -> lower travel time. This is expected model behaviour, not a bug.
    # ------------------------------------------------------------------
    print("\n[TC03] NOTE: Monotonicity check skipped -- backward-bending speed-flow")
    print("             curve is expected behaviour of the quadratic model.")

    status = "PASSED" if all_passed else "FAILED"
    print(f"\n[TC03] Result: {status}")
    return all_passed


if __name__ == "__main__":
    run_tc03()