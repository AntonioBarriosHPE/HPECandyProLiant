# camera_fps_test.py
"""A small utility to
1. Detect available USB cameras
2. Safely negotiate a working resolution (preferring 1080p → 720p → 480p)
3. Try several FPS targets (default / 30 / 60 / 120) at that resolution
4. Optionally down-shift to 720p and retry at 120FPS if the first
   resolution is higher than 720p
5. Display the live stream with an FPS overlay and print a summary table

Press **q** in the preview window to abort an individual FPS run or kill
with ^C to exit completely.
"""

from __future__ import annotations

import time
from typing import List, Tuple, Dict

import cv2

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def set_camera_resolution_safely(cap: cv2.VideoCapture, width: int, height: int) -> Tuple[int, int]:
    """Try to set a resolution and roll back if the camera refuses it.

    Returns the *actual* width/height the camera settled on.
    """
    print(f"Attempting resolution {width}x{height} …", end=" ")

    # Keep original to restore if necessary
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    time.sleep(0.4)  # give the driver a moment

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Test a read to ensure frames arrive
    ok, _ = cap.read()
    if not ok:
        print("failed - reverting")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, orig_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, orig_h)
        time.sleep(0.3)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        print("ok")

    return actual_w, actual_h


def test_camera_fps(cap: cv2.VideoCapture, label: str, duration: float = 10.0) -> float:
    """Run a live loop for *duration* seconds and return the realised FPS."""

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n--- {label} ---")
    print(f"Resolution: {width}x{height} | Requested FPS: {target_fps}")

    # Flush a few frames
    for _ in range(5):
        cap.read()

    frames = 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost frame - aborting test")
            break

        frames += 1
        elapsed = time.time() - start
        current_fps = frames / elapsed if elapsed > 0 else 0.0

        # overlay
        cv2.putText(frame, f"{current_fps:5.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, label, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Camera FPS Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or elapsed >= duration:
            break

    final_fps = frames / (time.time() - start) if frames else 0.0
    print(f"→ {final_fps:.2f} FPS over {frames} frames\n")
    return final_fps


def find_cameras(max_index: int = 10) -> List[int]:
    """Return indices of working cameras (0‑based)."""
    print("Scanning for cameras …")
    result: List[int] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  [{idx}] {w}x{h} @ {fps} FPS")
                result.append(idx)
            cap.release()
    return result

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    print("USB-camera FPS test tool  -  press 'q' to skip a run\n")

    cam_indices = find_cameras()
    if not cam_indices:
        print("No usable camera found.")
        return

    cam_id = cam_indices[0]
    if len(cam_indices) > 1:
        choice = input(f"Multiple cams {cam_indices}. Use which? [{cam_id}] → ").strip()
        if choice.isdigit() and int(choice) in cam_indices:
            cam_id = int(choice)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    # -------------------------------------------------------------------
    # Negotiate resolution
    # -------------------------------------------------------------------
    for desired in [(1920, 1080), (1280, 720), (640, 480)]:
        actual = set_camera_resolution_safely(cap, *desired)
        if actual == desired:
            break  # perfect
    working_w, working_h = actual
    print(f"→ Using {working_w}x{working_h}\n")

    fps_results: Dict[str, float] = {}

    try:
        # Default driver FPS
        fps_results["Default"] = test_camera_fps(cap, "Driver default")

        # Explicit 30
        cap.set(cv2.CAP_PROP_FPS, 30)
        fps_results["30 FPS"] = test_camera_fps(cap, "Set 30 FPS")

        # Explicit 60
        cap.set(cv2.CAP_PROP_FPS, 60)
        fps_results["60 FPS"] = test_camera_fps(cap, "Set 60 FPS")

        # Explicit 120
        cap.set(cv2.CAP_PROP_FPS, 120)
        fps_results["120 FPS"] = test_camera_fps(cap, "Set 120 FPS")

        # If we’re running above 720p, drop to 720p and retry 120 FPS
        if working_h > 720:
            print("\nTrying 720p @ 120 FPS …")
            set_camera_resolution_safely(cap, 1280, 720)
            cap.set(cv2.CAP_PROP_FPS, 120)
            fps_results["120 FPS (720p)"] = test_camera_fps(cap, "120 FPS @ 720p")

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # ----------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------
        if fps_results:
            print("\n=======  Summary  =======")
            for label, fps in fps_results.items():
                print(f"{label:20}: {fps:6.2f} FPS")
            best = max(fps_results, key=fps_results.get)
            print(f"\nBest: {best} → {fps_results[best]:.2f} FPS")


if __name__ == "__main__":
    main()
