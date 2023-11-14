import cv2
import numpy as np
import json

# Function to perform camera calibration
def camera_calibration(image_folder_path, json_file_path):
    # Load the JSON data for the true grid points
    with open(json_file_path) as json_file:
        calibration_data = json.load(json_file)

    # Retrieve the object points for calibration from the JSON dictionary
    objp_calib = np.array(calibration_data["objp_calib"], dtype=np.float32)

    # Prepare object points and image points for camera calibration
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    # Prepare the criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Process each image
    for i in range(0, 20): 
        img = cv2.imread(f'{image_folder_path}/Frame{i}.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp_calib)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2)

    # Camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

if __name__ == "__main__":

    image_folder_path = '.'
    json_file_path = 'Project5_4.json'
    intrinsic_matrix, distortion_coefficients = camera_calibration(image_folder_path, json_file_path)

    print("Intrinsic matrix:")
    print(intrinsic_matrix)
    print("Distortion coefficients:")
    print(distortion_coefficients)