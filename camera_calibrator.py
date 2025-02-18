import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

TILE_SIZE = 16  # size of the chessboard square in mm


class CameraCalibrator:
    def __init__(self, images_root, chessboard_size, square_size):
        self.width, self.height = chessboard_size
        self.square_size = square_size
        self.image_root = images_root
        self.image_paths = glob.glob(images_root + "/*.jpg")
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # define the 3D corner points of the chessboard
        self.obj_points = np.zeros((self.width * self.height, 3), np.float32)
        self.obj_points[:, :2] = (
            np.mgrid[0 : self.height, 0 : self.width].T.reshape(-1, 2)
            * self.square_size
        )

        # initialize list to store global object and image points for every image
        self.global_obj_points = []
        self.global_img_points = []

    def __get_mouse_click(self, event, x, y, flags, param):
        """
        Function to get the (x, y) coordinates of the left mouse button click on an image.
        :param event: Event type
        :param x: x-coordinate of the mouse click
        :param y: y-coordinate of the mouse click
        :param flags: Flags
        :param param: Parameters
        :return: List of (x, y) coordinates of the mouse clicks
        """
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

    def __select_global_corners(self, image):
        """
        Function to manually select the global corners of the chessboard in the image.
        :param image: Image of the chessboard
        :return: List of selected global corners in the order top-left, top-right, bottom-right, bottom-left
        """
        win_title = (
            "Select Corners in order top-left, top-right, bottom-right, bottom-left"
        )
        global_corners = []
        img = image.copy()

        params = {
            "image": img,
            "global_corners": global_corners,
            "win_title": win_title,
        }

        cv2.namedWindow(win_title)
        cv2.setMouseCallback(win_title, self.__get_mouse_click, params)
        cv2.imshow(win_title, image)
        cv2.waitKey(0)
        return global_corners

    def __compute_local_corners(self, image):
        """
        Function to compute the local corners of the chessboard in the image based on the global corners.
        :param image: Image of the chessboard
        :return: Local corners of the chessboard in the image
        """
        global_corners = self.__select_global_corners(image)
        tl, tr, br, bl = global_corners

        corners = np.float32([tl, tr, br, bl])

        # compute the orthogonal rectangle size (when viewed straight ahead)
        ort_width = (self.width - 1) * self.square_size
        ort_height = (self.height - 1) * self.square_size

        # define the orthogonal rectangle corners
        ort_corners = np.float32(
            [[0, 0], [ort_width, 0], [ort_width, ort_height], [0, ort_height]]
        )

        # CHOICE TASK 3: calculate perspective transform matrix (P) that converts between the physical
        # rectangle (straight) to the manually provided rectangle (tilted)
        perspective_matrix = cv2.getPerspectiveTransform(ort_corners, corners)

        # compute the local corners
        local_corners = np.zeros((self.width * self.height, 1, 2), np.float32)
        for i in range(self.width):
            for j in range(self.height):
                x = i * self.square_size
                y = j * self.square_size

                # [x' y' w']^T = P^T * [x y 1]^T
                point = np.array([x, y, 1.0])
                trans_point = perspective_matrix.dot(point)
                # normalize the homogeneous coordinates
                trans_point = trans_point / trans_point[2]

                # calculate the index based on the cv2 ordering (bottom-up, left-right)
                index = (self.height - 1 - j) + i * self.height
                local_corners[index, 0] = trans_point[:2]
        return local_corners

    def __save_calibration_data(self, cam_matrix, coeffs, rvecs, tvecs, filepath):
        """
        Function to save the camera calibration data to a JSON file.
        :param cam_matrix: Camera matrix
        :param coeffs: Distortion coefficients
        :param rvecs: Rotation vectors
        :param tvecs: Translation vectors
        :param filepath: Path to save the calibration data
        :return: None
        """
        calibration_data_json = {
            "camera_matrix": cam_matrix.tolist(),
            "dist_coeffs": coeffs.tolist(),
            "rvecs": [rvec.tolist() for rvec in rvecs],
            "tvecs": [tvec.tolist() for tvec in tvecs],
        }
        with open(filepath + ".json", "w") as f:
            json.dump(calibration_data_json, f, indent=4)
        print(f"Calibration data saved to {filepath}.json")

    def __load_calibration_data(self, filepath):
        """
        Function to load the camera calibration data from a JSON file.
        :param filepath: Path to the calibration data JSON file
        :return: Camera matrix, distortion coefficients, rotation vectors, translation vectors
        """
        with open(filepath, "r") as f:
            calibration_data = json.load(f)
        return (
            np.array(calibration_data["camera_matrix"]),
            np.array(calibration_data["dist_coeffs"]),
            np.array(calibration_data["rvecs"]),
            np.array(calibration_data["tvecs"]),
        )

    def draw_axes_and_cube(
        self, image_source, params_path, save=False, save_root="./output"
    ):
        """
        Function to draw X, Y, Z axes on the chessboard in the image based on estimated camera parameters.
        :param image_source: Path to the image, camera index (int) for webcam, or URL for IP camera
        :param params_path: Path to the calibration parameters file
        :param save: Boolean flag to save the image with axes drawn
        :param save_root: Root path to save the image with axes drawn
        :return: Image with axes drawn on the chessboard
        """
        camera_matrix, dist_coeffs, _, _ = self.__load_calibration_data(params_path)

        # check if image_source is a path, camera index, or URL
        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                cap = cv2.VideoCapture(image_source)
                if not cap.isOpened():
                    raise Exception("Could not open IP camera stream!")
            elif image_source == 0:
                cap = cv2.VideoCapture(image_source)
                if not cap.isOpened():
                    raise Exception("Could not open webcam!")
            else:
                cap = None
                img = cv2.imread(image_source)
        else:
            cap = cv2.VideoCapture(image_source)
            if not cap.isOpened():
                raise Exception("Could not open webcam!")

        try:
            while True:
                if cap is not None:
                    ret, img = cap.read()
                    if not ret:
                        break

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(
                    gray, (self.height, self.width), None
                )

                if ret:
                    # refine corner detection
                    corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), self.criteria
                    )

                    # estimate rotation and translation vectors
                    ret, rvecs, tvecs = cv2.solvePnP(
                        self.obj_points, corners, camera_matrix, dist_coeffs
                    )

                    # define axis points (origin and points along x, y, z axes)
                    axis_length = 6 * self.square_size

                    # NOTE: the z-axis is negated for aesthetic reasons so that it aligns with the cube. It was mentioned by Prof Poppe
                    # that this is fine as long as it is consistent across all the images. If not simply remove the negation for original view.
                    axis_points = np.float32(
                        [
                            [0, 0, 0],
                            [axis_length, 0, 0],
                            [0, axis_length, 0],
                            [0, 0, -axis_length],
                        ]
                    )

                    # define cube points
                    cube_length = 3 * self.square_size
                    cube_points = np.float32(
                        [
                            [0, 0, 0],
                            [0, cube_length, 0],
                            [cube_length, cube_length, 0],
                            [cube_length, 0, 0],
                            [0, 0, -cube_length],
                            [0, cube_length, -cube_length],
                            [cube_length, cube_length, -cube_length],
                            [cube_length, 0, -cube_length],
                        ]
                    )

                    # project axis points to image plane
                    axis_imgpts, _ = cv2.projectPoints(
                        axis_points, rvecs, tvecs, camera_matrix, dist_coeffs
                    )

                    # project cube points to image plane
                    cube_imgpts, _ = cv2.projectPoints(
                        cube_points, rvecs, tvecs, camera_matrix, dist_coeffs
                    )

                    # define axis colors
                    color_x = (90, 90, 219)
                    color_y = (124, 219, 90)
                    color_z = (219, 194, 90)

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

                    # define cube colors
                    pillar_shade = (0, 153, 255)
                    cube_imgpts = np.int32(cube_imgpts).reshape(-1, 2)

                    # draw pillars
                    for i, j in zip(range(4), range(4, 8)):
                        img = cv2.line(
                            img,
                            tuple(cube_imgpts[i]),
                            tuple(cube_imgpts[j]),
                            pillar_shade,
                            1,
                        )

                    # draw bottom borders
                    img = cv2.drawContours(img, [cube_imgpts[:4]], -1, pillar_shade, 1)

                    # compute the center of the top face of the cube
                    top_face_center = np.mean(cube_points[4:], axis=0)

                    # project center point to get its position relative to camera
                    center_imgpt, _ = cv2.projectPoints(
                        top_face_center, rvecs, tvecs, camera_matrix, dist_coeffs
                    )

                    # get rotation matrix from rotation vector
                    R, _ = cv2.Rodrigues(rvecs)

                    # calculate distance to camera (using translation and rotation)
                    camera_position = -np.dot(R.T, tvecs)
                    center_in_camera = np.dot(R, top_face_center.reshape(3, 1)) + tvecs
                    distance = np.linalg.norm(center_in_camera)

                    # calculate value (intensity) based on distance
                    max_distance = 4000
                    value = int(max(0, min(255, 255 * (1 - distance / max_distance))))

                    # compute the normal vector of the top face in camera coordinates
                    normal_vector = np.dot(R, np.array([0, 0, 1]))

                    # compute the angle between the normal vector (z-axis of the cube) and the z-axis (of the camera)
                    camera_axis = np.array([0, 0, 1])
                    cos_angle = np.dot(normal_vector, camera_axis)
                    angle = np.arccos(cos_angle) * 180 / np.pi

                    # compute the saturation based on angle
                    max_angle = 45
                    saturation = int(max(0, min(255, 255 * (1 - angle / max_angle))))

                    # compute the hue based on relative position using the azimuth angle in the x-y plane of the camera coordinates
                    azimuth_angle = np.arctan2(center_in_camera[1], center_in_camera[0])
                    hue = int(((azimuth_angle.item() + np.pi) * 180 / np.pi) % 180)

                    # convert HSV to BGR
                    hsv_color = np.uint8([[[hue, saturation, value]]])
                    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

                    # shade the top face
                    img = cv2.fillConvexPoly(img, cube_imgpts[4:], bgr_color.tolist())

                    # draw top borders
                    img = cv2.drawContours(img, [cube_imgpts[4:]], -1, pillar_shade, 1)

                    if cap is None:
                        cv2.imshow("Axes and Cube on Chessboard", img)
                        cv2.waitKey(0)
                        break
                    else:
                        cv2.imshow("Axes and Cube on Chessboard", img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                else:
                    if cap is None:
                        raise Exception("Chessboard corners not found!")
                    else:
                        cv2.imshow("Axes and Cube on Chessboard", img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                # save image with axes drawn if needed
                if save and ret:
                    fname = os.path.basename(params_path).split(".")[0]
                    save_path = os.path.join(save_root, f"{fname}.png")
                    cv2.imwrite(save_path, img)
                    print(f"Image with axes drawn saved to {save_path}")
        finally:
            cv2.destroyAllWindows()
            if cap is not None:
                cap.release()

        return img if ret else None

    def plot_camera_locations(self, params_path):
        """
        Function to plot the 3D locations of the camera relative to the chessboard and the camera's viewing direction.
        :param params_path: Path to the calibration parameters file
        :return: None
        """
        camera_matrix, dist_coeffs, rvecs, tvecs = self.__load_calibration_data(
            params_path
        )
        rvecs = [np.array(rvec) for rvec in rvecs]
        tvecs = [np.array(tvec) for tvec in tvecs]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # plot the axes at the origin
        axis_length = 30

        # x-axis
        ax.plot([0, axis_length], [0, 0], [0, 0], "r-", linewidth=2)
        ax.text(axis_length, 0, 0, "X", color="red")

        # y-axis
        ax.plot([0, 0], [0, axis_length], [0, 0], "g-", linewidth=2)
        ax.text(0, axis_length, 0, "Y", color="green")

        # z-axis (negated to match with axes plotting from ONLINE part of assignment)
        ax.plot([0, 0], [0, 0], [0, -axis_length], "b-", linewidth=2)
        ax.text(0, 0, -axis_length, "Z", color="blue")

        # plot each camera position
        for i in range(len(rvecs)):
            # convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvecs[i])

            # camera position is the inverse transformation of the chessboard to camera
            camera_position = -np.dot(R.T, tvecs[i])

            ax.plot(
                [camera_position[0, 0]],
                [camera_position[1, 0]],
                [camera_position[2, 0]],
                "ro",
                markersize=5,
            )

        # plot the chessboard grid
        x = np.arange(0, self.height) * self.square_size
        y = np.arange(0, self.width) * self.square_size
        xv, yv = np.meshgrid(x, y)

        # set z-coordinates to 0
        zv = np.zeros_like(xv)

        # flatten the coordinate arrays for plotting
        x_flat = xv.flatten()
        y_flat = yv.flatten()
        z_flat = zv.flatten()

        # plot the grid points
        ax.scatter(x_flat, y_flat, z_flat, c="k", marker=".")

        # set labels and title
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title("Camera Positions and Chessboard Grid")
        ax.set_aspect("equal")

        # show the plot
        plt.show()

    def calibrate(self, display=False, save=True):
        """
        Function to obtain intrinsic and extrinsic parameters of camera using images of the chessboard.
        :param display: Boolean flag to display the images with detected corners
        :param save: Boolean flag to save the calibration data
        :return: None
        """
        for image_path in self.image_paths:
            # read the image and convert it to grayscale
            img = cv2.imread(image_path)
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(
                gray_img, (self.height, self.width), None
            )

            # if the corners are found, refine them using cornerSubPix
            if ret == True:
                local_corners = cv2.cornerSubPix(
                    gray_img, corners, (11, 11), (-1, -1), self.criteria
                )
            # if the corners are not found, compute them manually
            else:
                continue
                # local_corners = self.__compute_local_corners(img)

            # append the object points and image points
            self.global_obj_points.append(self.obj_points)
            self.global_img_points.append(local_corners)

            # display the result if needed
            if display:
                cv2.drawChessboardCorners(
                    img, (self.height, self.width), local_corners, True
                )
                cv2.imshow(os.path.basename(image_path), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # calibrate the camera with no flags for default behavior of not fixing cx, cy, fx, or fy
        # this is not needed but ensures CALIB_FIX_PRINCIPAL_POINT and CALIB_FIX_FOCAL_LENGTH are not set
        c_ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.global_obj_points,
            self.global_img_points,
            gray_img.shape[::-1],
            None,
            None,
            flags=0,
        )

        print("Calibration successful." if c_ret else "Calibration failed.")

        # save the calibration data if needed
        if c_ret and save:
            print("Calibration successful. Saving data...")
            save_root = "./data"
            save_path = os.path.join(
                save_root, os.path.basename(os.path.dirname(self.image_root))
            )
            if os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            self.__save_calibration_data(
                camera_matrix, dist_coeffs, rvecs, tvecs, save_path
            )


if __name__ == "__main__":
    # OFFLINE STEP:
    root_path = "./images/"
    for i in range(1, 4):
        IMAGE_PATH = os.path.join(root_path, f"run_{i}")
        calibrator = CameraCalibrator(IMAGE_PATH, (6, 9), TILE_SIZE)
        calibrator.calibrate(display=True, save=False)
        print(f"Calibration for Run {i} complete.")

    # ONLINE STEP:
    # # load the calibration data
    # TEST_IMG_PATH = os.path.join(root_path, "test")
    # calibrator = CameraCalibrator(TEST_IMG_PATH, (6, 9), TILE_SIZE)

    # # select camera source (choose from "webcam", "ip camera" or "static image")
    # camera_input_choice = "static image"

    # camera_source = None
    # if camera_input_choice == "webcam":
    #     camera_source = 0
    # elif camera_input_choice == "ip camera":
    #     camera_source = "http://192.168.X.Y:PORT/video"
    # else:
    #     camera_source = os.path.join(TEST_IMG_PATH, "test.png")

    # # run visualization
    # for i in range(1, 4):
    #     params_path = f"./output/run_{i}.json"
    #     calibrator.draw_axes_and_cube(camera_source, params_path, save=False)
    #     # calibrator.plot_camera_locations(params_path)
