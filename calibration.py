from camera_calibrator import CameraCalibrator
from lxml import etree
import glob
import os
import cv2


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

    print(f"Video: {video_path}")
    print(f"  Frame count: {frame_count}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {duration:.2f} seconds")

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


if __name__ == "__main__":
    extract_frames = False
    calibrate = False
    test = False

    tree = etree.parse("./data/checkerboard.xml")
    root = tree.getroot()

    width = int(root.find("CheckerBoardWidth").text)
    height = int(root.find("CheckerBoardHeight").text)
    square_size = int(root.find("CheckerBoardSquareSize").text)

    # extract frames from all intrinsics.avi files in the data folder
    if extract_frames:
        intrinsics_video_paths = glob.glob("./data/**/intrinsics.avi", recursive=True)
        for video_path in intrinsics_video_paths:
            camera_name = os.path.basename(os.path.dirname(video_path))
            extract_frames(video_path, f"./data/{camera_name}/calibration_frames", 100)

    camera_names = [
        d for d in os.listdir("./data") if os.path.isdir(os.path.join("./data", d))
    ]

    # calibrate the camera
    if calibrate:   
        for camera_name in camera_names:
            print(f"Calibrating camera: {camera_name}")
            image_path = f"./data/{camera_name}/calibration_frames"
            calibrator = CameraCalibrator(image_path, (width, height), square_size)
            calibrator.calibrate(display=False, save=True)

    # test the camera by drawing the axes and cube
    if test:
        intrinsics_paths = glob.glob("./data/intrinsic_values/*.json")
        for intrinsics_path in intrinsics_paths:
            camera_name = os.path.basename(intrinsics_path).split('.')[0]
            image_path = f"./data/intrinsic_values/test.jpg"
            print(f"Testing camera: {camera_name}")
            calibrator = CameraCalibrator(image_path, (width, height), square_size)
            calibrator.draw_axes_and_cube(image_path, intrinsics_path, save=False)
