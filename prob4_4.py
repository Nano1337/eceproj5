import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

from prob4_3 import *
from Project5_4driver import *

# Assuming the following functions and variables are already defined:
# camera_calibration, load_and_undistort_image, create_ideal_intrinsic_matrix, perform_pnp_registration
# ...

def read_tumor_data(json_file_path):
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    return data['tumor']['verts'], data['tumor']['faces']

def compute_camera_pose(rvecs, tvecs):
    # Convert rotation vector to rotation matrix
    rmat = R.from_rotvec(rvecs.ravel()).as_matrix()

    # Form the extrinsic matrix
    extrinsic_matrix = np.hstack((rmat, tvecs))
    return extrinsic_matrix

def extract_camera_parameters(extrinsic_matrix):
    # Camera position (last column of extrinsic matrix)
    cam_position = -extrinsic_matrix[:, :3].T @ extrinsic_matrix[:, 3]

    # Camera projection direction (negative Z axis)
    cam_projection_direction = -extrinsic_matrix[:3, 2]

    # Camera view up direction (Y axis)
    cam_view_up = extrinsic_matrix[:3, 1]

    return cam_position, cam_projection_direction, cam_view_up

def compute_view_angle(image_height, focal_length_y):
    return 2 * np.arctan(image_height / (2 * focal_length_y))

if __name__ == "__main__":
    json_file_path = 'Project5_4.json'
    image_path = 'AR0.png'

    intrinsic_matrix, distortion_coefficients = camera_calibration('.', json_file_path)
    undistorted_image = load_and_undistort_image(image_path, intrinsic_matrix, distortion_coefficients)
    ideal_intrinsic_matrix = create_ideal_intrinsic_matrix(intrinsic_matrix, undistorted_image.shape)
    rvecs, tvecs = perform_pnp_registration(json_file_path, undistorted_image, ideal_intrinsic_matrix)

    verts, faces = read_tumor_data(json_file_path)
    extrinsic_matrix = compute_camera_pose(rvecs, tvecs)
    cam_position, cam_projection_direction, cam_view_up = extract_camera_parameters(extrinsic_matrix)
    view_angle = compute_view_angle(undistorted_image.shape[0], ideal_intrinsic_matrix[1, 1])

    # Report camera pose and view angle estimates
    print("Camera Position:", cam_position)
    print("Camera Projection Direction:", cam_projection_direction)
    print("Camera View Up Direction:", cam_view_up)
    print("View Angle:", np.degrees(view_angle), "degrees")

    # Call to Project5_4driver (assuming it's properly imported and available)
    Project5_4driver(undistorted_image, verts, faces,
                     cam_position, cam_projection_direction, cam_view_up, view_angle)
