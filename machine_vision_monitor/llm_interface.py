import cv2
import numpy as np
from PIL import Image
from models.manual_analysis import identify_machine

class ImageAnalysisLLM:
    """
    Simplified LLM interface for image analysis.
    Provides image type and rust-marked image.
    """

    def __init__(self):
        pass

    def analyze_image(self, pil_image):
        """
        Analyze the image to get type and rust-marked image.
        Args:
            pil_image: PIL Image object
        Returns:
            tuple: (image_type_str, rust_marked_pil_image)
        """
        # Get image type using identify_machine
        image_type = identify_machine(pil_image)

        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Rust detection and marking (improved)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # Define range for rust-like color (brownish-red)
        lower_rust = np.array([5, 50, 50])
        upper_rust = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_rust, upper_rust)

        # Morphological operations to clean mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours of rust areas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw red contours on the image
        cv2.drawContours(cv_image, contours, -1, (0, 0, 255), 3)

        # Convert back to RGB for PIL
        marked_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_marked_image = Image.fromarray(marked_image)

        # Calculate rust percentage
        rust_area = cv2.countNonZero(mask)
        total_area = mask.shape[0] * mask.shape[1]
        rust_percentage = (rust_area / total_area) * 100 if total_area > 0 else 0

        # Determine rust status based on percentage thresholds
        if rust_percentage < 5:
            rust_status = "Low"
        elif rust_percentage < 20:
            rust_status = "Medium"
        else:
            rust_status = "High"

        return image_type, pil_marked_image, rust_percentage, rust_status
