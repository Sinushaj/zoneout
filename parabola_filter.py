import numpy as np

def fit_parabola(points, times):
    points = np.array(points)
    t = np.array(times)
    
    x = points[:, 0]
    y = points[:, 1]
    
    px = np.polyfit(t, x, 1)
    py = np.polyfit(t, y, 2)
    
    return px, py


def predict_position(px, py, t):
    x = np.polyval(px, t)
    y = np.polyval(py, t)
    return np.array([x, y])


def parabola_filter(points, window=5, max_error=50, reset_threshold=7):
    filtered = []
    
    error_streak = 0
    trajectory_points = []  # current segment
    
    for i in range(len(points)):
        p = points[i]
        
        # Add valid points to current trajectory buffer
        if p is not None:
            trajectory_points.append(p)
        else:
            trajectory_points.append(None)
        
        # Keep only recent points
        recent_valid = [pt for pt in trajectory_points if pt is not None][-window:]
        
        # Not enough data → just pass through
        if len(recent_valid) < window:
            filtered.append(p)
            continue
        
        # Fit parabola
        times = list(range(len(recent_valid)))
        px, py = fit_parabola(recent_valid, times)
        
        pred = predict_position(px, py, len(recent_valid))
        
        if p is not None:
            error = np.linalg.norm(np.array(p) - pred)
            
            if error < max_error:
                # Good point → reset error streak
                error_streak = 0
                filtered.append(p)
            else:
                # Bad point
                error_streak += 1
                
                if error_streak >= reset_threshold:
                    # 🔥 RESET TRAJECTORY
                    trajectory_points = []
                    error_streak = 0
                    filtered.append(p)  # accept as new start
                else:
                    # Replace with prediction
                    filtered.append(tuple(pred))
        else:
            # Missing detection → use prediction
            error_streak += 1
            
            if error_streak >= reset_threshold:
                trajectory_points = []
                error_streak = 0
                filtered.append(None)
            else:
                filtered.append(tuple(pred))
    
    return filtered