import numpy as np

# Gopro 8 black with 1080p Linear and 16:9
gopro_cameramatrix = np.array([
    [960.0,   0.0, 960.0],
    [  0.0, 960.0, 540.0],
    [  0.0,   0.0,   1.0]
])