import cv2
import tkinter as tk
from tkinter import Label, filedialog, Button
from PIL import Image, ImageTk
from utils.video_utils import detect_motion  # Import your motion detection

# Initialize root window
root = tk.Tk()
root.title("Real-time Machine Monitor")
root.geometry("700x540")
root.config(bg="white")

# Status label
status_label = Label(root, text="Machine: Normal", font=("Helvetica", 16), bg="green", fg="white")
status_label.pack(pady=10)

# Canvas for video feed
canvas = tk.Canvas(root, width=640, height=360)
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
button_frame = tk.Frame(root, bg="white")
button_frame.pack(pady=10)

Button(button_frame, text="Use Live Camera", command=start_live, width=20).pack(side=tk.LEFT, padx=10)
Button(button_frame, text="Load Video File", command=load_video, width=20).pack(side=tk.LEFT, padx=10)

# Close cleanly
def on_closing():
    if cap:
        cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start with webcam
start_live()
root.mainloop()