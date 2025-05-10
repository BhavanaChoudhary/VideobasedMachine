import cv2
import tkinter as tk
from tkinter import Label, filedialog, Button
from PIL import Image, ImageTk
from utils.video_utils import detect_motion  # Import your motion detection

# Initialize root window
root = tk.Tk()
root.title("Real-time Machine Monitor")
root.geometry("720x580")
root.config(bg="#f0f4f8")

# Title label
title_label = tk.Label(root, text="Real-time Machine Monitor", font=("Helvetica", 24, "bold"), bg="#f0f4f8", fg="#333")
title_label.pack(pady=(20, 10))

# Status label with padding and rounded border effect
status_label = tk.Label(root, text="Machine: Normal", font=("Helvetica", 16), bg="#4caf50", fg="white", padx=15, pady=8)
status_label.pack(pady=10)
status_label.config(relief="groove", bd=2)

# Frame with border around canvas
canvas_frame = tk.Frame(root, bg="#ccc", bd=2, relief="sunken")
canvas_frame.pack(pady=10)

# Canvas for video feed inside the frame
canvas = tk.Canvas(canvas_frame, width=640, height=360, bg="black")
canvas.pack()

# Globals
cap = None
img_ref = None
prev_frame = None
using_webcam = True

# Function to update the frame
def update_frame():
    global cap, img_ref, prev_frame, using_webcam

    if not cap:
        return

    ret, frame = cap.read()
    if not ret:
        if not using_webcam:
            status_label.config(text="Video playback ended", bg="gray", fg="white")
        return

    if prev_frame is not None:
        motion_detected, score = detect_motion(prev_frame, frame)
        if motion_detected:
            status_label.config(text=f"Machine: ⚠ Warning (Score: {score})", bg="red", fg="white")
        else:
            status_label.config(text=f"Machine: ✅ Normal (Score: {score})", bg="green", fg="white")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    img_ref = photo
    prev_frame = frame

    delay = 10 if using_webcam else 30
    root.after(delay, update_frame)

# Button to use live camera
def start_live():
    global cap, using_webcam, prev_frame
    if cap:
        cap.release()
    cap = cv2.VideoCapture(0)
    using_webcam = True
    prev_frame = None
    update_frame()

# Button to open uploaded video
def load_video():
    global cap, using_webcam, prev_frame
    file_path = filedialog.askopenfilename(filetypes=[("Video files", ".mp4 *.avi"), ("All files", ".*")])
    if file_path:
        if cap:
            cap.release()
        cap = cv2.VideoCapture(file_path)
        using_webcam = False
        prev_frame = None
        update_frame()

# Buttons for control
button_frame = tk.Frame(root, bg="#f0f4f8")
button_frame.pack(pady=15, fill=tk.X, expand=True)

from llm_interface import ImageAnalysisLLM

import tkinter.simpledialog as simpledialog
import tkinter.messagebox as messagebox

from PIL import Image
from models.manual_analysis import identify_machine, assess_condition, detect_rust, suggest_repurpose

from llm_interface import ImageAnalysisLLM

def analyze_image():
    # Prompt user to upload an image file
    file_path = filedialog.askopenfilename(title="Select Image for Analysis", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
    if not file_path:
        messagebox.showerror("Error", "No image file selected.")
        return

    # Load image
    try:
        image = Image.open(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open image: {str(e)}")
        return

    # Use simplified LLM interface for analysis
    llm = ImageAnalysisLLM()
    image_type, rust_marked_image, rust_percentage, rust_status = llm.analyze_image(image)

    # Open a new window for displaying analysis result
    llm_window = tk.Toplevel(root)
    llm_window.title("Image Analysis Result")
    llm_window.geometry("800x600")
    llm_window.config(bg="#f0f4f8")

    # Display the rust-marked image
    rust_marked_image = rust_marked_image.resize((400, 300), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(rust_marked_image)
    image_label = tk.Label(llm_window, image=img_tk, bg="#f0f4f8")
    image_label.image = img_tk  # Keep a reference
    image_label.pack(side=tk.LEFT, padx=20, pady=20)

    # Create a frame for text info on the right
    info_frame = tk.Frame(llm_window, bg="#f0f4f8")
    info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Display the image type text
    type_label = tk.Label(info_frame, text=f"Image Type: {image_type}", 
                          font=("Helvetica", 18, "bold"), bg="#f0f4f8", fg="#333", anchor="w")
    type_label.pack(fill=tk.X, pady=(0, 15))

    # Display rust percentage
    rust_percentage_label = tk.Label(info_frame, text=f"Rust Percentage: {rust_percentage:.2f}%", 
                                    font=("Helvetica", 16), bg="#f0f4f8", fg="#333", anchor="w")
    rust_percentage_label.pack(fill=tk.X, pady=5)

    # Display rust status
    rust_status_label = tk.Label(info_frame, text=f"Rust Status: {rust_status}", 
                                 font=("Helvetica", 16, "bold"), bg="#f0f4f8", fg="#b22222", anchor="w")
    rust_status_label.pack(fill=tk.X, pady=5)

    # Close button at bottom right
    close_button = tk.Button(info_frame, text="Close", command=llm_window.destroy, 
                             font=("Helvetica", 12), bg="#1976d2", fg="white", relief="raised", bd=3, cursor="hand2")
    close_button.pack(side=tk.BOTTOM, pady=20)

button_style = {
    "width": 20,
    "padx": 10,
    "pady": 6,
    "bg": "#1976d2",
    "fg": "white",
    "font": ("Helvetica", 12, "bold"),
    "activebackground": "#1565c0",
    "activeforeground": "white",
    "relief": "raised",
    "bd": 3,
    "cursor": "hand2"
}

Button(button_frame, text="Use Live Camera", command=start_live, **button_style).pack(side=tk.LEFT, padx=10)
Button(button_frame, text="Load Video File", command=load_video, **button_style).pack(side=tk.LEFT, padx=10)
Button(button_frame, text="Analyze Image", command=analyze_image, **button_style).pack(side=tk.LEFT, padx=10)

# Close cleanly
def on_closing():
    if cap:
        cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start with webcam
start_live()
root.mainloop()