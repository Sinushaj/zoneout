import numpy as np
import matplotlib.pyplot as plt

def moving_average(points, window_size=5):
    points = np.array(points)
    smoothed = []
    
    for i in range(len(points)):
        start = max(0, i - window_size // 2)
        end = min(len(points), i + window_size // 2 + 1)
        smoothed.append(points[start:end].mean(axis=0))
    
    return np.array(smoothed)


def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)


def find_serve_and_receive(points):
    points = np.array(points)
    
    # --- 1. Find split index (y sign change) ---
    split_idx = None
    for i in range(1, len(points)):
        if points[i-1][1] * points[i][1] < 0:
            split_idx = i
            print('split_idx =', split_idx)
            break
    
    if split_idx is None:
        raise ValueError("No y sign change found.")
    
    first_half = points[:split_idx]
    second_half = points[split_idx:]
    
    # --- 2. Serve location (closest to |y| = 9) ---
    serve_idx = np.argmin(np.abs(np.abs(first_half[:, 1]) - 9))
    print('serve idx =', serve_idx)
    serve_point = first_half[serve_idx]
    print('serve_point =', serve_point)
    
    # --- 3. Smooth second half ---
    smoothed = moving_average(second_half, window_size=5)
    
    # --- 4. Detect direction change ---
    window = 3
    threshold = np.deg2rad(45)  # adjust if needed
    
    receive_point = None

    angles_temp_check = []
    for i in range(window, len(smoothed) - window):
        v1 = smoothed[i] - smoothed[i - window]
        v2 = smoothed[i + window] - smoothed[i]
        
        angle = angle_between(v1, v2)
        angles_temp_check.append(angle)
        
        if angle > threshold:
            receive_point = second_half[i]
            print('receive idx =', i)
            print('receive point =', receive_point)
            plt.plot(range(len(angles_temp_check)), angles_temp_check)
            plt.show()
            break
    
    # fallback if nothing detected
    if receive_point is None:
        print('no reception detected')
        receive_point = second_half[0] #fixa sen
    
    return serve_point, receive_point