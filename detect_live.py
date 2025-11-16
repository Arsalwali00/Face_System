import cv2
import face_recognition
import pickle
import json
from datetime import datetime
import time

# Import ZKTeco library
try:
    from zk import ZK
    ZKTECO_AVAILABLE = True
except ImportError:
    print("⚠ Warning: pyzk not installed. Install with: pip install pyzk")
    ZKTECO_AVAILABLE = False

# ZKTeco K50 Configuration
ZKTECO_IP = "192.168.18.230"  # Change to your K50 IP
ZKTECO_PORT = 4370
UNLOCK_DURATION = 3  # Seconds to keep door unlocked

# Load encodings
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Load person info (must have "access": true for authorized users)
with open("persons.json", "r") as f:
    persons = json.load(f)

# Connect to ZKTeco K50
def connect_k50():
    if not ZKTECO_AVAILABLE:
        return None
    try:
        zk = ZK(ZKTECO_IP, port=ZKTECO_PORT, timeout=5)
        conn = zk.connect()
        conn.disable_device()
        print(f"✓ Connected to ZKTeco K50 at {ZKTECO_IP}")
        firmware = conn.get_firmware_version()
        print(f"  Firmware: {firmware}")
        conn.enable_device()
        return conn
    except Exception as e:
        print(f"✗ K50 Connection failed: {e}")
        print("  Running in offline mode...")
        return None

# Unlock door
def unlock_door(conn):
    try:
        if conn:
            conn.unlock(UNLOCK_DURATION)
            print(f"🔓 Door unlocked for {UNLOCK_DURATION} seconds")
            return True
        else:
            print("🔓 [DEMO MODE] Door would be unlocked")
            return True
    except Exception as e:
        print(f"✗ Unlock failed: {e}")
        return False

# Mark attendance
def mark_attendance(conn, emp_code, name):
    try:
        if conn:
            # Extract UID from employee code (e.g., "EMP-001" -> 1)
            uid = int(emp_code.split('-')[1]) if '-' in emp_code else hash(emp_code) % 10000
            
            # Add/update user in K50
            conn.set_user(uid=uid, name=name, privilege=0, password='', 
                         group_id='', user_id=str(uid))
            
            print(f"✓ Attendance marked for {name} (UID: {uid})")
            return True
    except Exception as e:
        print(f"✗ Attendance marking failed: {e}")
    return False

# Log access attempts
def log_access(name, emp_code, granted, reason=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "name": name,
        "emp_code": emp_code,
        "access_granted": granted,
        "reason": reason
    }
    
    with open("access_log.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    status = "✓ GRANTED" if granted else "✗ DENIED"
    print(f"\n{'='*60}")
    print(f"[{timestamp}] {name} ({emp_code})")
    print(f"Access: {status}")
    if reason:
        print(f"Reason: {reason}")
    print(f"{'='*60}\n")

# Main function
def main():
    print("\n" + "="*60)
    print("  ZKTeco K50 + Face Recognition Door Access System")
    print("="*60 + "\n")
    
    # Connect to K50
    k50_conn = connect_k50()
    
    # Initialize camera
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video.set(cv2.CAP_PROP_FPS, 30)

    if not video.isOpened():
        print("✗ Error: Cannot open camera")
        return

    print("✓ Camera initialized")
    print("Press 'q' to quit | Press 'l' to manually unlock")
    print("="*60 + "\n")

    # Performance optimizations
    frame_count = 0
    process_every_n_frames = 5
    scale_factor = 0.25

    # Cache last detection results
    cached_faces = []
    last_printed_name = None
    recently_processed = {}  # Prevent multiple unlocks for same person

    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Clean up old entries (cooldown: 10 seconds)
        recently_processed = {k: v for k, v in recently_processed.items() 
                            if current_time - v < 10}
        
        # Only process every Nth frame
        if frame_count % process_every_n_frames == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)

            cached_faces = []
            
            for (top, right, bottom, left), encoding in zip(boxes, encodings):
                # Scale coordinates back to original size
                top = int(top / scale_factor)
                right = int(right / scale_factor)
                bottom = int(bottom / scale_factor)
                left = int(left / scale_factor)
                
                # Use face_distance for faster matching
                face_distances = face_recognition.face_distance(data["encodings"], encoding)
                best_match_index = face_distances.argmin() if len(face_distances) > 0 else -1

                name = "Unknown"
                color = (0, 0, 255)
                status_text = "ACCESS DENIED"
                info = {"name": "Unknown", "code": "N/A"}

                if best_match_index != -1 and face_distances[best_match_index] < 0.5:
                    name = data["names"][best_match_index]
                    info = persons.get(name, {"name": name, "code": "N/A"})
                    emp_code = info.get("code", "N/A")
                    
                    # Check if user has access permission
                    has_access = info.get("access", False)
                    
                    if has_access:
                        color = (0, 255, 0)
                        status_text = "AUTHORIZED"
                        
                        # Only process if not recently authenticated
                        if emp_code not in recently_processed:
                            # Print user info
                            if name != last_printed_name:
                                print(json.dumps(info, indent=4))
                                last_printed_name = name
                            
                            # Unlock door via K50
                            if unlock_door(k50_conn):
                                # Mark attendance
                                mark_attendance(k50_conn, emp_code, name)
                                
                                # Log access
                                log_access(name, emp_code, True, "Face recognized + Access granted")
                                
                                recently_processed[emp_code] = current_time
                    else:
                        color = (255, 165, 0)  # Orange for unauthorized
                        status_text = "NO PERMISSION"
                        
                        if emp_code not in recently_processed:
                            log_access(name, emp_code, False, "User recognized but no access permission")
                            recently_processed[emp_code] = current_time
                else:
                    # Unknown person
                    if "UNKNOWN" not in recently_processed:
                        log_access("Unknown", "N/A", False, "Face not recognized")
                        recently_processed["UNKNOWN"] = current_time
                
                cached_faces.append({
                    "box": (top, right, bottom, left),
                    "info": info,
                    "color": color,
                    "status": status_text
                })
        
        # Display system status
        k50_status = f"K50: {ZKTECO_IP}" if k50_conn else "K50: OFFLINE"
        cv2.putText(frame, k50_status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw cached faces on every frame (for smooth video)
        for face in cached_faces:
            top, right, bottom, left = face["box"]
            color = face["color"]
            info = face["info"]
            status = face["status"]
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw name
            cv2.putText(frame, info["name"], (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw status
            cv2.putText(frame, status, (left, bottom + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("K50 Door Access System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("l"):
            # Manual unlock
            print("\n[MANUAL UNLOCK]")
            unlock_door(k50_conn)
            log_access("Manual", "ADMIN", True, "Manual unlock via keyboard")

    # Cleanup
    video.release()
    cv2.destroyAllWindows()
    if k50_conn:
        k50_conn.disconnect()
    print("\n✓ System shutdown complete")

if __name__ == "__main__":
    main()