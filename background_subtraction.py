import cv2
import numpy as np

class BackgroundSubtractor:
    def __init__(self, video_path, background_path):
        self.video_path = video_path
        self.background_path = background_path
        self.height = 0
        self.width = 0
        self.background_model = None

    def __create_background_model(self, var_threshold=64):
        cap = cv2.VideoCapture(self.background_path)

        gmm = cv2.createBackgroundSubtractorMOG2(varThreshold=var_threshold, detectShadows=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gmm.apply(frame)

        cap.release()

        self.background_model = gmm
    # 50, 10, 40
    def __background_subtraction(self, frame, h_threshold=50, s_threshold=10, v_threshold=40):
        gmm_mask = self.background_model.apply(frame, learningRate=0.0001)
        ret, gmm_mask = cv2.threshold(gmm_mask, 250, 255, cv2.THRESH_BINARY)

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        background = self.background_model.getBackgroundImage()
        background_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)

        h_diff = cv2.absdiff(frame_hsv[:,:,0], background_hsv[:,:,0])
        s_diff = cv2.absdiff(frame_hsv[:,:,1], background_hsv[:,:,1])
        v_diff = cv2.absdiff(frame_hsv[:,:,2], background_hsv[:,:,2])

        h_mask = (h_diff > h_threshold).astype(np.uint8) * 255
        s_mask = (s_diff > s_threshold).astype(np.uint8) * 255
        v_mask = (v_diff > v_threshold).astype(np.uint8) * 255
        
        ret, h_mask = cv2.threshold(h_mask, h_threshold, 255, cv2.THRESH_BINARY)
        ret, s_mask = cv2.threshold(s_mask, s_threshold, 255, cv2.THRESH_BINARY)
        ret, v_mask = cv2.threshold(v_mask, v_threshold, 255, cv2.THRESH_BINARY)
        
        # or and and
        hsv_mask = cv2.bitwise_or(cv2.bitwise_and(v_mask, s_mask), h_mask)

        combined_mask = cv2.bitwise_and(gmm_mask, hsv_mask)
        
        # apply morphological operations to remove noise
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)

        internal_contours, _ = cv2.findContours(combined_mask, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        # set the threshold area to the average of the two largest internal contours
        internal_areas = [cv2.contourArea(contour) for contour in internal_contours]
        internal_areas.sort()
        if len(internal_areas) >= 2:
            internal_threshold_area = internal_areas[-2] + (internal_areas[-1] - internal_areas[-2]) / 2
        else:
            internal_threshold_area = 7000

        # fill holes inside foreground
        for contour in internal_contours:
            area = cv2.contourArea(contour)
            if area < internal_threshold_area:
                cv2.drawContours(combined_mask, [contour], 0, 255, -1)

        external_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # set the threshold area to the average of the two largest external contours
        external_areas = [cv2.contourArea(contour) for contour in external_contours]
        external_areas.sort()
        if len(external_areas) >= 2:
            external_threshold_area = external_areas[-2] + (external_areas[-1] - external_areas[-2]) / 2
        else:
            external_threshold_area = 7000

        # remove pixels outside foreground
        for contour in external_contours:
            area = cv2.contourArea(contour)
            if area < external_threshold_area:
                cv2.drawContours(combined_mask, [contour], -1, 0, cv2.FILLED)
        return combined_mask, frame
        
    def isolate_foreground(self, output_folder="./output"):
        self.__create_background_model()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        # while True:
        ret, frame = cap.read()
        if not ret:
            return

        combined_mask, frame = self.__background_subtraction(frame)
        
        cv2.imshow('Combined Mask', combined_mask)
        cv2.imshow('Frame', frame)
        cv2.waitKey(0)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    for i in range(1, 5):
        background_path = f"./data/cam{i}/background.avi"
        video_path = f"./data/cam{i}/video.avi"

        background_subtractor = BackgroundSubtractor(video_path, background_path)
        background_subtractor.isolate_foreground()
        