import cv2
import numpy as np
import json

from prob4_2 import *

def perform_pnp_registration(json_file_path, undistorted_image, ideal_intrinsic_matrix):
    # Load the AR grid points
    with open(json_file_path) as json_file:
        calibration_data = json.load(json_file)
    objp_ar = np.array(calibration_data["objp_AR"], dtype=np.float32)

    # Convert undistorted image to grayscale and find corners
    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)

    if not ret:
        raise ValueError("Chessboard corners not found in the image")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    # Perform PnP registration
    ret, rvecs, tvecs = cv2.solvePnP(objp_ar, corners2, ideal_intrinsic_matrix, np.zeros(5))

    return rvecs, tvecs

if __name__ == "__main__":
    image_folder_path = '.'
    json_file_path = 'Project5_4.json'
    image_path = 'AR0.png'

    # Perform the existing steps
    intrinsic_matrix, distortion_coefficients = camera_calibration(image_folder_path, json_file_path)
    undistorted_image = load_and_undistort_image(image_path, intrinsic_matrix, distortion_coefficients)
    ideal_intrinsic_matrix = create_ideal_intrinsic_matrix(intrinsic_matrix, undistorted_image.shape)

    # Perform PnP registration
    rvecs, tvecs = perform_pnp_registration(json_file_path, undistorted_image, ideal_intrinsic_matrix)

    print("Rotation Vectors:")
    print(rvecs)
    print("Translation Vectors:")
    print(tvecs)
