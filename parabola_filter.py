import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def ransac_ball_trajectory(points, threshold=1):
    """
    points: list of (x, y, z) ordered by time
    returns: list of inlier points belonging to best parabola
    """
    pts = np.asarray(points)
    t = np.arange(len(pts)).reshape(-1, 1)  # time index
    z = pts[:, 2]

    # Quadratic model z(t)
    model = make_pipeline(
        PolynomialFeatures(2),
        RANSACRegressor(residual_threshold=threshold)
    )
    
    model.fit(t, z)

    inliers = model.named_steps['ransacregressor'].inlier_mask_
    
    return pts[inliers].tolist()