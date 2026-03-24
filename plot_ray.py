import numpy as np
import matplotlib.pyplot as plt


def plot_ray_and_squares(origin, direction, ray_length=20):
    """
    Plots a 3D ray and two squares in the XY plane (z=0).

    Parameters:
        origin: (3,) array-like
        direction: (3,) array-like (does not need to be normalized)
        ray_length: how far to extend the ray for visualization
    """

    origin = np.array(origin)
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)

    # Generate ray points
    t = np.linspace(0, ray_length, 100)
    ray_points = origin.reshape(3,1) + direction.reshape(3,1) * t

    # Define squares
    square1 = np.array([
        [0, 0, 0],
        [0, -9, 0],
        [9, -9, 0],
        [9, 0, 0],
        [0, 0, 0]
    ])

    square2 = np.array([
        [0, 0, 0],
        [9, 0, 0],
        [9, 9, 0],
        [0, 9, 0],
        [0, 0, 0]
    ])

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot ray
    ax.plot(ray_points[0], ray_points[1], ray_points[2])

    # Plot origin
    ax.scatter(origin[0], origin[1], origin[2])

    # Plot squares
    ax.plot(square1[:,0], square1[:,1], square1[:,2])
    ax.plot(square2[:,0], square2[:,1], square2[:,2])

    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Equal aspect ratio
    max_range = 20
    ax.set_xlim([-5, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-5, 15])

    plt.title("3D Ray and Squares")
    plt.show()




def plot_rays_and_squares(rays, ray_length=20):
    """
    Plots one or multiple 3D rays and two squares in the XY plane (z=0).

    Parameters:
        rays: list of (origin, direction) tuples
              origin: (3,)
              direction: (3,)
        ray_length: length of rays for visualization
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # --- Plot each ray ---
    for origin, direction in rays:
        origin = np.array(origin)
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)

        t = np.linspace(0, ray_length, 100)
        ray_points = origin.reshape(3,1) + direction.reshape(3,1) * t

        # Plot ray
        ax.plot(ray_points[0], ray_points[1], ray_points[2])

        # Plot origin
        ax.scatter(origin[0], origin[1], origin[2])

    # --- Squares ---
    square1 = np.array([
        [0, 0, 0],
        [0, -9, 0],
        [9, -9, 0],
        [9, 0, 0],
        [0, 0, 0]
    ])

    square2 = np.array([
        [0, 0, 0],
        [9, 0, 0],
        [9, 9, 0],
        [0, 9, 0],
        [0, 0, 0]
    ])

    ax.plot(square1[:,0], square1[:,1], square1[:,2])
    ax.plot(square2[:,0], square2[:,1], square2[:,2])

    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Fixed view range
    ax.set_xlim([-5, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-5, 15])

    plt.title("3D Rays and Squares")
    plt.show()