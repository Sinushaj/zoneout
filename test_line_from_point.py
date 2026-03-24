import numpy as np
import matplotlib.pyplot as plt
from court_points import *
from get_line_from_point import screen_point_to_world_ray, intersect_rays
from plot_ray import plot_rays_and_squares
from point_finder import get_screen_coordinates
from save_to_csv import read_csv_to_tuples_np


# ----------------
# constant info
# ----------------


gopro_video_name = 'GH011611.MP4'
zve10_video_name = 'C0600.MP4'


'''
court_points = [
    (0,-3, 0),
    (9,-3, 0),
    (9,3, 0),
    (0,3, 0),
    (0,-9, 0),
    (0,9, 0),
]



gopro_cameramatrix = np.array([
    [960.0,   0.0, 960.0],
    [  0.0, 960.0, 540.0],
    [  0.0,   0.0,   1.0]
])


# Not yet tested to see if it is reasonable
zve10_cameramatrix = np.array([
    [2371,    0, 960],
    [   0, 2008, 540],
    [   0,    0,   1]
], dtype=float)

# -------------
# new code
# -------------


# GOPRO:
court_screen_points = read_csv_to_tuples_np('gopro_points.csv')
# court_screen_points = get_screen_coordinates(video_name, 6)
new_screen_point = get_screen_coordinates(video_name, 1)[0]
point_origin, point_ray = screen_point_to_world_ray(court_screen_points, court_points, gopro_cameramatrix, new_screen_point)
print(point_origin, point_ray)

zv_court_screen_points = read_csv_to_tuples_np('zve10_points.csv')
# zv_court_screen_points = get_screen_coordinates(zve10_video_name, 6)
zv_new_screen_point = get_screen_coordinates(zve10_video_name, 1)[0]
zv_point_origin, zv_point_ray = screen_point_to_world_ray(zv_court_screen_points, court_points, zve10_cameramatrix, zv_new_screen_point)
print(zv_point_origin, zv_point_ray)

ray_list = [
    (point_origin, point_ray),
    (zv_point_origin, zv_point_ray),
]

intersection = intersect_rays(ray_list)

plot_rays_and_squares(ray_list, ray_length = 30)
print(intersection)
'''

def point_from_camera_coordinates(gopro_coordinate, zve10_coordinate, plot_rays = False):
    court_points = [
        (0,-3, 0),
        (9,-3, 0),
        (9,3, 0),
        (0,3, 0),
        (0,-9, 0),
        (0,9, 0),
    ]
    gopro_cameramatrix = np.array([
        [960.0,   0.0, 960.0],
        [  0.0, 960.0, 540.0],
        [  0.0,   0.0,   1.0]
    ])
    zve10_cameramatrix = np.array([
        [2371,    0, 960],
        [   0, 2008, 540],
        [   0,    0,   1]
    ], dtype=float)

    gopro_court_points = read_csv_to_tuples_np('gopro_points.csv')
    gopro_origin, gopro_ray = screen_point_to_world_ray(gopro_court_points, court_points, gopro_cameramatrix, gopro_coordinate)

    zve10_court_points = read_csv_to_tuples_np('zve10_points.csv')
    zve10_origin, zve10_ray = screen_point_to_world_ray(zve10_court_points, court_points, zve10_cameramatrix, zve10_coordinate)

    ray_list = [
        (gopro_origin, gopro_ray),
        (zve10_origin, zve10_ray),
    ]

    if plot_rays:
        plot_rays_and_squares(ray_list, ray_length = 30)

    intersection_estimate = intersect_rays(ray_list)
    return intersection_estimate

'''
new_gopro_screen_point = get_screen_coordinates(gopro_video_name, 1)[0]
new_zve10_screen_point = get_screen_coordinates(zve10_video_name, 1)[0]

real_world_point = point_from_camera_coordinates(new_gopro_screen_point, new_zve10_screen_point, plot_rays = True)
print(real_world_point)
'''