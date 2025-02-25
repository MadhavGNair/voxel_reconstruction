import glob
import json
import os

import cv2
import numpy as np
from lxml import etree
import matplotlib.pyplot as plt

from camera_calibrator import CameraCalibrator


def get_mouse_click(event, x, y, flags, param):
    corners = param["global_corners"]
    win_title = param["win_title"]
    image = param["image"]
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(win_title, image)
        if len(corners) >= 4:
            cv2.destroyWindow(win_title)
    return corners


def select_global_corners(image):
    win_title = "Select Corners in order top-left, top-right, bottom-right, bottom-left"
    global_corners = []
    img = image.copy()

    params = {
        "image": img,
        "global_corners": global_corners,
        "win_title": win_title,
    }

    cv2.namedWindow(win_title)
    cv2.setMouseCallback(win_title, get_mouse_click, params)
    cv2.imshow(win_title, image)
    cv2.waitKey(0)
    return global_corners


def compute_local_corners(image, width, height, square_size):
    global_corners = select_global_corners(image)
    tl, tr, br, bl = global_corners

    corners = np.float32([tl, tr, br, bl])

    # compute the orthogonal rectangle size (when viewed straight ahead)
    ort_width = (width - 1) * square_size
    ort_height = (height - 1) * square_size

    # define the orthogonal rectangle corners
    ort_corners = np.float32(
        [[0, 0], [ort_width, 0], [ort_width, ort_height], [0, ort_height]]
    )

    # CHOICE TASK 3: calculate perspective transform matrix (P) that converts between the physical
    # rectangle (straight) to the manually provided rectangle (tilted)
    perspective_matrix = cv2.getPerspectiveTransform(ort_corners, corners)

    # compute the local corners
    local_corners = np.zeros((width * height, 1, 2), np.float32)
    for i in range(width):
        for j in range(height):
            x = i * square_size
            y = j * square_size

            # [x' y' w']^T = P^T * [x y 1]^T
            point = np.array([x, y, 1.0])
            trans_point = perspective_matrix.dot(point)
            # normalize the homogeneous coordinates
            trans_point = trans_point / trans_point[2]

            # calculate the index based on the cv2 ordering (bottom-up, left-right)
            index = (height - 1 - j) + i * height
            local_corners[index, 0] = trans_point[:2]
    return local_corners


def load_intrinsic_data(filepath):
    with open(filepath, "r") as f:
        calibration_data = json.load(f)
    return (
        np.array(calibration_data["camera_matrix"]),
        np.array(calibration_data["dist_coeffs"]),
        np.array(calibration_data["rvecs"]),
        np.array(calibration_data["tvecs"]),
    )


def load_extrinsic_data(filepath):
    with open(filepath, "r") as f:
        extrinsic_data = json.load(f)
    return (
        np.array(extrinsic_data["rvec"]),
        np.array(extrinsic_data["tvec"]),
    )


def draw_axes_and_cube(
    image_source,
    intrinsic_path,
    extrinsic_path,
    obj_points,
    square_size,
    save=False,
    save_root="./data",
):
    camera_matrix, dist_coeffs, _, _ = load_intrinsic_data(intrinsic_path)
    rvecs, tvecs = load_extrinsic_data(extrinsic_path)

    img = cv2.imread(image_source)

    # define axis points (origin and points along x, y, z axes)
    axis_length = 3 * square_size
    axis_points = np.float32(
        [
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, -axis_length],
        ]
    )

    # project axis points to image plane
    axis_imgpts, _ = cv2.projectPoints(
        axis_points, rvecs, tvecs, camera_matrix, dist_coeffs
    )

    # define axis colors
    color_x = (90, 90, 219)
    color_y = (124, 219, 90)
    color_z = (219, 0, 0)

    # draw axis lines
    origin = tuple(map(int, axis_imgpts[0].ravel()))

    cv2.arrowedLine(
        img,
        origin,
        tuple(map(int, axis_imgpts[1].ravel())),
        color_x,
        2,
        tipLength=0.03,
    )

    cv2.arrowedLine(
        img,
        origin,
        tuple(map(int, axis_imgpts[2].ravel())),
        color_y,
        2,
        tipLength=0.03,
    )

    cv2.arrowedLine(
        img,
        origin,
        tuple(map(int, axis_imgpts[3].ravel())),
        color_z,
        2,
        tipLength=0.03,
    )

    # add axis labels
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    offset = 15

    img = cv2.putText(
        img,
        "x",
        tuple(map(int, axis_imgpts[1].ravel() + offset)),
        font,
        1,
        color_x,
        2,
    )

    img = cv2.putText(
        img,
        "y",
        tuple(map(int, axis_imgpts[2].ravel() + offset)),
        font,
        1,
        color_y,
        2,
    )

    img = cv2.putText(
        img,
        "z",
        tuple(map(int, axis_imgpts[3].ravel() + offset)),
        font,
        1,
        color_z,
        2,
    )

    cv2.imshow("Axes and Cube on Chessboard", img)
    cv2.waitKey(0)

    # save image with axes drawn if needed
    if save:
        fname = os.path.basename(intrinsic_path).split(".")[0]
        save_path = os.path.join(save_root, f"{fname}/{fname}.png")
        cv2.imwrite(save_path, img)
        print(f"Image with axes drawn saved to {save_path}")

def extract_frames(video_path, output_folder, frame_interval):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps

    count = 0
    success = True
    frame_num = 0
    while success:
        success, image = video.read()
        if success:
            if frame_num % frame_interval == 0:
                output_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
                cv2.imwrite(output_path, image)
                print(f"Extracted frame: {output_path}")
                count += 1
            frame_num += 1
    video.release()
    print(f"Extracted {count} frames from {video_path}")


def extract_first_frame(video_path, output_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file: {video_path}")

    success, image = video.read()
    if success:
        cv2.imwrite(output_path, image)
        print(f"Extracted first frame: {output_path}")
        video.release()
    else:
        print(f"Error: Could not read frame from {video_path}")
        video.release()


def get_intrinsics_values():
    extract_frame = False
    calibrate = False
    test = False

    tree = etree.parse("./data/checkerboard.xml")
    root = tree.getroot()

    width = int(root.find("CheckerBoardWidth").text)
    height = int(root.find("CheckerBoardHeight").text)
    square_size = int(root.find("CheckerBoardSquareSize").text)

    # extract frames from all intrinsics.avi files in the data folder
    if extract_frame:
        intrinsics_video_paths = glob.glob("./data/**/intrinsics.avi", recursive=True)
        for video_path in intrinsics_video_paths:
            camera_name = os.path.basename(os.path.dirname(video_path))
            extract_frames(video_path, f"./data/{camera_name}/intrinsic_frames", 100)

    camera_names = [
        d for d in os.listdir("./data") if os.path.isdir(os.path.join("./data", d))
    ]

    # calibrate the camera
    if calibrate:
        for camera_name in camera_names:
            print(f"Calibrating camera: {camera_name}")
            image_path = f"./data/{camera_name}/intrinsic_frames"
            calibrator = CameraCalibrator(image_path, (width, height), square_size)
            calibrator.calibrate(display=False, save=True)

    # test the camera by drawing the axes and cube
    if test:
        intrinsics_paths = glob.glob("./data/intrinsic_values/*.json")
        for intrinsics_path in intrinsics_paths:
            camera_name = os.path.basename(intrinsics_path).split(".")[0]
            image_path = f"./data/intrinsic_values/test.jpg"
            print(f"Testing camera: {camera_name}")
            calibrator = CameraCalibrator(image_path, (width, height), square_size)
            calibrator.draw_axes_and_cube(image_path, intrinsics_path, save=False)

def get_extrinsics_values():
    extract_frame = False
    calibrate = False
    display = False
    draw_axes = False

    tree = etree.parse("./data/checkerboard.xml")
    root = tree.getroot()

    width = int(root.find("CheckerBoardWidth").text)
    height = int(root.find("CheckerBoardHeight").text)
    square_size = int(root.find("CheckerBoardSquareSize").text)

    if extract_frame:
        extrinsics_video_paths = glob.glob("./data/**/checkerboard.avi", recursive=True)
        for video_path in extrinsics_video_paths:
            camera_name = os.path.basename(os.path.dirname(video_path))
            extract_first_frame(
                video_path, f"./data/{camera_name}/extrinsic_frames/frame_0000.jpg"
            )

    camera_names = [
        d
        for d in os.listdir("./data")
        if os.path.isdir(os.path.join("./data", d)) and d.startswith("cam")
    ]

    # define the 3D corner points of the chessboard
    obj_points = np.zeros((width * height, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:height, 0:width].T.reshape(-1, 2) * square_size

    # load the intrinsic values
    if calibrate:
        for camera_name in camera_names:
            cam_matrix, dist_coeffs, _, _ = load_intrinsic_data(
                f"./data/intrinsic_values/{camera_name}.json"
            )
            image_path = f"./data/{camera_name}/extrinsic_frames/frame_0000.jpg"
            img = cv2.imread(image_path)
            local_corners = compute_local_corners(img, width, height, square_size)

            if display:
                cv2.drawChessboardCorners(img, (height, width), local_corners, True)
                cv2.imshow(os.path.basename(image_path), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            ret, rvec, tvec = cv2.solvePnP(
                obj_points, local_corners, cam_matrix, dist_coeffs
            )

            # save the extrinsic values
            extrinsic_values = {"rvec": rvec.tolist(), "tvec": tvec.tolist()}
            with open(f"./data/extrinsic_values/{camera_name}.json", "w") as f:
                json.dump(extrinsic_values, f)
            print(f"Saved extrinsic values for {camera_name}")

    # test the camera by drawing the axes and cube
    if draw_axes:
        for camera_name in camera_names:
            image_path = f"./data/{camera_name}/extrinsic_frames/frame_0000.jpg"
            print(f"Testing camera: {camera_name}")
            draw_axes_and_cube(
                image_path,
                f"./data/intrinsic_values/{camera_name}.json",
                f"./data/extrinsic_values/{camera_name}.json",
                obj_points,
                square_size,
                save=True,
            )


if __name__ == "__main__":
    get_intrinsics_values()
    # get_extrinsics_values()
