from prob4_1 import *
from Project5_4driver import *

import cv2
import numpy as np

# Assume intrinsic_matrix and distortion_coefficients are obtained from the camera_calibration function

# Function to load and undistort an image
def load_and_undistort_image(image_path, intrinsic_matrix, distortion_coefficients):
    # Load the image
    img = cv2.imread(image_path)

    # Undistort the image
    undistorted_img = cv2.undistort(img, intrinsic_matrix, distortion_coefficients)

    return undistorted_img

# Function to create a new 'ideal' intrinsic matrix for AR
def create_ideal_intrinsic_matrix(intrinsic_matrix, image_shape):
    # Use the y-direction focal length for both x and y directions
    focal_length = intrinsic_matrix[1, 1]

    # Set the new optical center to the center of the image plane
    cx = image_shape[1] / 2
    cy = image_shape[0] / 2

    # Create the new ideal intrinsic matrix
    ideal_intrinsic_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])

    return ideal_intrinsic_matrix

if __name__ == "__main__":

    image_path = 'AR0.png'
    image_folder_path = '.'
    json_file_path = 'Project5_4.json'
    intrinsic_matrix, distortion_coefficients = camera_calibration(image_folder_path, json_file_path)

    undistorted_image = load_and_undistort_image(image_path, intrinsic_matrix, distortion_coefficients)
    ideal_intrinsic_matrix = create_ideal_intrinsic_matrix(intrinsic_matrix, undistorted_image.shape)

    print("Ideal Intrinsic matrix:")
    print(ideal_intrinsic_matrix)