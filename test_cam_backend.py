import cv2
import time

def test_camera_backends():
    """Test different OpenCV backends"""
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "Auto")
    ]

    for backend_id, backend_name in backends:
        print(f"\n--- Testing {backend_name} backend ---")
        try:
            cap = cv2.VideoCapture(0, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✓ {backend_name} works! Frame shape: {frame.shape}")

                    # Test a few more frames
                    success_count = 0
                    for i in range(10):
                        ret, frame = cap.read()
                        if ret:
                            success_count += 1

                    print(f"  Read {success_count}/10 test frames successfully")

                    # Try to display one frame
                    if success_count > 0:
                        cv2.imshow(f'Test - {backend_name}', frame)
                        cv2.waitKey(1000)  # Show for 1 second
                        cv2.destroyAllWindows()

                    cap.release()
                    return backend_id, backend_name
                else:
                    print(f"✗ {backend_name} opens but can't read frames")
            else:
                print(f"✗ {backend_name} won't open camera")
            cap.release()
        except Exception as e:
            print(f"✗ {backend_name} error: {e}")

    return None, None

def simple_fps_test(backend_id=None):
    """Simple FPS test with working backend"""
    print(f"\n--- Simple FPS Test ---")

    if backend_id is not None:
        cap = cv2.VideoCapture(0, backend_id)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open camera")
        return

    print("Camera opened. Testing frame rate...")
    frame_count = 0
    start_time = time.time()
    duration = 5  # 5 second test

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            # Add FPS overlay
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('FPS Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to read frame")
            break

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    print(f"Results: {fps:.2f} FPS ({frame_count} frames in {elapsed:.2f}s)")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Testing camera backends to find working one...")

    # Test backends
    working_backend, backend_name = test_camera_backends()

    if working_backend is not None:
        print(f"\n✓ Found working backend: {backend_name}")
        simple_fps_test(working_backend)
    else:
        print("\n✗ No working backend found!")
        print("\nTroubleshooting suggestions:")
        print("1. Unplug and reconnect your Razer Kiyo Pro")
        print("2. Close all apps that might use the camera")
        print("3. Restart your computer")
        print("4. Check Windows Camera app works first")
        print("5. Update camera drivers")