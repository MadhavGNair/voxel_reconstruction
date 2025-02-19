import cv2
import numpy as np

class BackgroundSubtractor:
    def __init__(self, video_path, background_path):
        self.video_path = video_path
        self.background_path = background_path
        self.height = 0
        self.width = 0
        self.background_model = None

    def __create_background_model(self, var_threshold=50, random_state=42):
        cap = cv2.VideoCapture(self.background_path)

        gmm = cv2.createBackgroundSubtractorMOG2(varThreshold=var_threshold)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gmm.apply(frame)

        cap.release()

        self.background_model = gmm

    def __background_subtraction(self, frame, threshold=10):
        gmm_mask = self.background_model.apply(frame)

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        background = self.background_model.getBackgroundImage()
        background_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        
        h_diff = cv2.absdiff(frame_hsv[:,:,0], background_hsv[:,:,0])
        s_diff = cv2.absdiff(frame_hsv[:,:,1], background_hsv[:,:,1])
        v_diff = cv2.absdiff(frame_hsv[:,:,2], background_hsv[:,:,2])
        
        h_mask = (h_diff > threshold).astype(np.uint8) * 255
        s_mask = (s_diff > threshold).astype(np.uint8) * 255
        v_mask = (v_diff > threshold).astype(np.uint8) * 255

        # only foreground if both s and v are above threshold or h is above threshold
        # this works best because hue changes are more important than s or v changes
        hsv_mask = cv2.bitwise_or(cv2.bitwise_and(s_mask, v_mask), h_mask)
        
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                
        return combined_mask
        
    def isolate_foreground(self, output_folder="./output"):
        self.__create_background_model()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            combined_mask = self.__background_subtraction(frame, threshold=10)
            
            cv2.imshow('Combined Mask', combined_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    background_path = "./data/cam1/background.avi"
    video_path = "./data/cam1/video.avi"

    background_subtractor = BackgroundSubtractor(video_path, background_path)
    background_subtractor.isolate_foreground()
        
