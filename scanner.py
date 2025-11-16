import socket
import threading
import time

print("🔍 Scanning your WiFi (192.168.18.x) for ZKTeco K50...")

def check_k50(ip):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex((ip, 4370))  # ZKTeco uses port 4370
        if result == 0:
            print(f"\n🎯 K50 FOUND! IP Address: {ip}")
            print(f"   → Use this in your script: ZKTECO_IP = \"{ip}\"")
        sock.close()
    except:
        pass

# Scan your network (based on your PC IP: 192.168.18.225)
network = "192.168.18"
threads = []

for i in range(1, 255):
    ip = f"{network}.{i}"
    t = threading.Thread(target=check_k50, args=(ip,))
    t.start()
    threads.append(t)

# Wait for all to finish
for t in threads:
    t.join()

print("\n✅ Scan complete!")