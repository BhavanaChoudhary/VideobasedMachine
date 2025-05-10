import cv2
import numpy as np

def detect_motion(prev_frame, curr_frame, threshold_ratio=0.02):
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(gray_prev, gray_curr)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    motion_score = np.sum(thresh > 0)
    total_pixels = gray_prev.shape[0] * gray_prev.shape[1]
    threshold = total_pixels * threshold_ratio

    return motion_score > threshold, int(motion_score)

def analyze_image(frame):
    """
    Placeholder function for image analysis.
    Currently returns a dummy result.
    """
    # Future image analysis logic goes here
    return "No issues detected"
