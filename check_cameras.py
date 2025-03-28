import cv2

def check_cameras(max_to_check=10):
    """Check which camera indices are available on the system."""
    print("Checking available cameras...")
    available_cameras = []
    
    for i in range(max_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera index {i} is available")
                h, w = frame.shape[:2]
                print(f"  Resolution: {w}x{h}")
                available_cameras.append(i)
            else:
                print(f"Camera index {i} opened but failed to read frame")
            cap.release()
        else:
            print(f"Camera index {i} not available")
    
    if not available_cameras:
        print("No cameras were found")
    else:
        print(f"Available camera indices: {available_cameras}")
    
    return available_cameras

if __name__ == "__main__":
    check_cameras()