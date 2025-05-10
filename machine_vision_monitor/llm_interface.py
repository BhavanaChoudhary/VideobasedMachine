import sys
import os
import requests

# Add parent directory to sys.path to import config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import api_key

class ImageAnalysisLLM:
    """
    A class representing a Large Language Model (LLM) interface for image analysis using AIMLAPI.
    """

    def __init__(self, api_key=api_key):
        """
        Initialize with the API key.
        """
        self.api_key = api_key
        self.api_url = "https://api.aimlapi.com/v1/images/analyze"  # Replace with new API endpoint

    def analyze_image(self, image_path):
        """
        Call the AIMLAPI image analysis endpoint with the given image.
        Args:
            image_path: Path to the image file to analyze.
        Returns:
            Detailed analysis string or error message.
        """
        if not self.api_key:
            return "API key not provided."

        if not image_path or not isinstance(image_path, str):
            return "Invalid image path."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

        try:
            with open(image_path, "rb") as image_file:
                files = {"image": image_file}
                response = requests.post(self.api_url, headers=headers, files=files)
            print("Response Status Code:", response.status_code)
            print("Response Text:", response.text)
            response.raise_for_status()
            result = response.json()
            # Assuming the API returns a 'description' field with the detailed analysis
            if "description" in result:
                return result["description"]
            else:
                return "No analysis data found in response."
        except requests.exceptions.RequestException as e:
            return f"API request failed: {str(e)}"

    def load_model(self, model_path):
        """
        Placeholder method to load a pre-trained LLM model.
        Args:
            model_path: Path to the model file.
        """
        # Implement model loading logic here
        pass

    def close(self):
        """
        Placeholder method to release resources if needed.
        """
        pass
