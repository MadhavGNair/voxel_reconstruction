import cv2
import numpy as np

class BackgroundSubtractor:
    def __init__(self, video_path, background_path):
        self.video_path = video_path
        self.background_path = background_path
        self.height = 0
        self.width = 0
        self.background_model = None

    def __create_background_model(self, var_threshold=16, random_state=42):
        cap = cv2.VideoCapture(self.background_path)

        gmm = cv2.createBackgroundSubtractorMOG2(varThreshold=var_threshold, detectShadows=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gmm.apply(frame)

        cap.release()

        self.background_model = gmm

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
        
        # kernel = np.ones((3,3), np.uint8)
        # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        threshold_area = 500

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < threshold_area:
                cv2.drawContours(combined_mask, [contour], 0, 255, -1)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < threshold_area:
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
        # cv2.imshow('Frame', frame)
        cv2.waitKey(0)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    for i in range(1, 5):
        background_path = f"./data/cam{i}/background.avi"
        video_path = f"./data/cam{i}/video.avi"

        background_subtractor = BackgroundSubtractor(video_path, background_path)
        background_subtractor.isolate_foreground()
        
