import tkinter as tk
from tkinter import Entry, Button, filedialog, Label, messagebox
import subprocess
import cv2
from PIL import Image, ImageTk
import threading
import time
import os

class YOLOv7DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv7 Detector")
        self.root.geometry("1280x1000")

        # List to store variable types
        self.variable_types = [str, str, str, int, float, float]

        # List to store variable values with default values
        default_values = [
            'D:\\college\\ai video recognition project\\yolov7\\yolov7-e6e.pt',
            '0',
            'D:\\college\\ai video recognition project\\yolov7\\cfg\\deploy\\yolov7-e6e.yaml',
            640,
            0.25,
            0.45
        ]
        self.variable_values = [tk.StringVar(value=default) for default in default_values]

        # Flag to indicate if the video is playing or paused
        self.video_playing = False

        # Flag to indicate if the inferred video is playing or paused
        self.video_playing_inferred = False

        self.create_widgets()

    def create_widgets(self):
        # Create labels, entry widgets, and browse buttons
        headers = ['Weight Path', 'Source Path', 'CFG Path', 'Size', 'Confidence', 'IOU Threshold']
        for i in range(6):
            label = Label(self.root, text=f"{headers[i]}:")
            label.grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)

            entry = Entry(self.root, textvariable=self.variable_values[i], width=30)
            entry.grid(row=i, column=1, padx=10, pady=5)

            # Only create browse button for 'Weight Path', 'Source Path', and 'CFG Path'
            if i in [0, 1, 2]:
                browse_button = Button(self.root, text=f"Browse {headers[i]}", command=lambda i=i: self.browse_file(i))
                browse_button.grid(row=i, column=2, padx=10, pady=5)

        # Run Detection Button
        run_button = Button(self.root, text="Run Detection", command=self.run_detection)
        run_button.grid(row=6, column=0, columnspan=3, pady=10)

        # Canvas for source view
        self.canvas_source = tk.Canvas(self.root, width=640, height=360, bg="gray")
        self.canvas_source.grid(row=0, column=3, padx=10, pady=10, rowspan=8, columnspan=2)

        # Play/Pause Button
        play_pause_button = Button(self.root, text="Play/Pause", command=self.play_pause)
        play_pause_button.grid(row=8, column=3, pady=5)

        # Rewind Button
        rewind_button = Button(self.root, text="Rewind", command=self.rewind)
        rewind_button.grid(row=8, column=4, pady=5)

        # Canvas for inferred view
        self.canvas_inferred = tk.Canvas(self.root, width=640, height=360, bg="gray")
        self.canvas_inferred.grid(row=9, column=3, padx=10, pady=10, rowspan=8, columnspan=2)

        # Play/Resume Button for Inference View
        play_pause_button_inferred = Button(self.root, text="Play/Pause", command=self.play_pause_inferred)
        play_pause_button_inferred.grid(row=18, column=3, pady=5)

        # Rewind Button for Inference View
        rewind_button_inferred = Button(self.root, text="Rewind", command=self.rewind_inferred)
        rewind_button_inferred.grid(row=18, column=4, pady=5)

    def browse_file(self, index):
        # Open file dialog to select a file
        file_path = filedialog.askopenfilename()
        if file_path:
            # Update the file path in the corresponding Entry widget
            self.variable_values[index].set(file_path)

            # Display the source image or video
            self.display_source(file_path)

    def run_detection(self):
        # Get the current values of variables
        values = [var.get() for var in self.variable_values]

        # Modify this part to call your YOLOv7 detection script with the selected paths
        command = [
            "python", "detect.py",
            "--weights", values[0],
            "--source", values[1],
            "--cfg", values[2],
            "--img-size", str(values[3]),
            "--conf-thres", str(values[4]),
            "--iou-thres", str(values[5])
        ]

        # Run the command and capture output
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            output = result.stdout
            error = result.stderr
            self.display_inferred(output)
        except subprocess.CalledProcessError as e:
            output = e.stdout
            error = e.stderr

        # Display the output on the GUI
        # Note: Removed text output section

    def display_source(self, source_path):
        # Capture the source video or image
        self.cap = cv2.VideoCapture(source_path)

        # Get the dimensions of the source video or image
        width = int(self.cap.get(3))
        height = int(self.cap.get(4))

        # Read the first frame
        ret, frame = self.cap.read()

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to fit the source view canvas
        frame_resized = cv2.resize(frame_rgb, (self.canvas_source.winfo_width(), self.canvas_source.winfo_height()))

        # Convert the frame to ImageTk format
        img = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the source view canvas
        source_img = self.canvas_source.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas_source.img_tk = img_tk  # Keep a reference to prevent garbage collection

        # Function to continuously update the source view canvas with video frames
        def update_source_view():
            nonlocal frame
            while self.cap.isOpened():
                if self.video_playing:
                    ret, frame = self.cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, (self.canvas_source.winfo_width(), self.canvas_source.winfo_height()))
                        img = Image.fromarray(frame_resized)
                        img_tk = ImageTk.PhotoImage(image=img)
                        self.canvas_source.itemconfig(source_img, image=img_tk)
                        self.canvas_source.img_tk = img_tk
                        self.root.update_idletasks()
                        time.sleep(0.033)  # Introduce a delay of 33 milliseconds (approx. 30 fps)
                    else:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.cap.release()
            self.video_playing = False

        # Start the thread to continuously update the source view
        self.video_playing = True
        source_thread = threading.Thread(target=update_source_view)
        source_thread.start()

        # Release the video capture when the GUI is closed
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.release_capture())

    def display_inferred(self, output):
        # Convert the output string to a list of lines
        output_lines = output.splitlines()

        inferred_path = None

        # Find the line containing information about saving the result
        for line in output_lines:
            if "The image with the result is saved in:" in line:
                inferred_path = line.split("The image with the result is saved in:")[1].strip()
                break
            elif "Results saved to" in line:
                inferred_path = line.split("Results saved to")[1].strip()
                break

        if inferred_path:
            # Check if the inferred file exists
            if os.path.exists(inferred_path):
                # Check if the inferred file is an image or video
                if inferred_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    # Load and display inferred image
                    self.display_image_inferred(inferred_path)
                elif inferred_path.lower().endswith((".mp4", ".avi", ".mkv")):
                    # Load and display inferred video
                    self.display_video_inferred(inferred_path)
                else:
                    messagebox.showwarning("Unsupported File Type", "Unsupported file type for display.")
            else:
                messagebox.showerror("File Not Found", "The inferred file does not exist.")
        else:
            messagebox.showerror("Path Not Found", "Path to the inferred file could not be extracted from the output.")

    def display_image_inferred(self, inferred_path):
        # Load the inferred image
        inferred_img = cv2.imread(inferred_path)

        # Convert the image to RGB format
        inferred_img_rgb = cv2.cvtColor(inferred_img, cv2.COLOR_BGR2RGB)

        # Resize the image to fit the inference view canvas
        img_resized = cv2.resize(inferred_img_rgb, (self.canvas_inferred.winfo_width(), self.canvas_inferred.winfo_height()))

        # Convert the image to ImageTk format
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_resized))

        # Update the inference view canvas
        self.canvas_inferred.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas_inferred.img_tk = img_tk  # Keep a reference to prevent garbage collection

    def display_video_inferred(self, inferred_path):
        self.video_playing_inferred = True

        # Load and display inferred video
        self.inferred_cap = cv2.VideoCapture(inferred_path)

        # Check if the video file is opened successfully
        if not self.inferred_cap.isOpened():
            messagebox.showerror("Video Load Error", "Failed to load the inferred video file.")
            self.inferred_cap.release()
            return

        # Get the dimensions of the inferred video
        width = int(self.inferred_cap.get(3))
        height = int(self.inferred_cap.get(4))

        # Read the first frame
        ret, frame = self.inferred_cap.read()

        if not ret:
            messagebox.showerror("Video Load Error", "Failed to read the inferred video frames.")
            self.inferred_cap.release()
            return

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to fit the inference view canvas
        frame_resized = cv2.resize(frame_rgb, (self.canvas_inferred.winfo_width(), self.canvas_inferred.winfo_height()))

        # Convert the frame to ImageTk format
        img = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the inference view canvas
        inferred_img = self.canvas_inferred.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas_inferred.img_tk = img_tk  # Keep a reference to prevent garbage collection

        # Function to continuously update the inference view canvas with video frames
        def update_inferred_view():
            while self.inferred_cap.isOpened():
                if self.video_playing_inferred:
                    ret, frame = self.inferred_cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, (self.canvas_inferred.winfo_width(), self.canvas_inferred.winfo_height()))
                        img = Image.fromarray(frame_resized)
                        img_tk = ImageTk.PhotoImage(image=img)

                        # Update the inference view canvas with a new PhotoImage object
                        self.canvas_inferred.itemconfig(inferred_img, image=img_tk)
                        self.canvas_inferred.img_tk = img_tk
                        self.root.update_idletasks()
                        time.sleep(0.033)  # Introduce a delay of 33 milliseconds (approx. 30 fps)
                    else:
                        self.inferred_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Start the thread to continuously update the inference view
        self.video_playing_inferred = True
        inferred_thread = threading.Thread(target=update_inferred_view)
        inferred_thread.start()

        # Release the video capture when the GUI is closed
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.release_inferred_capture())

    
    def play_pause(self):
        # Toggle the play/pause state
        self.video_playing = not self.video_playing

    def rewind(self):
        if hasattr(self, 'cap'):
            # Set the video capture to the beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def play_pause_inferred(self):
        # Toggle the play/pause state for the inference view
        self.video_playing_inferred = not self.video_playing_inferred

    def rewind_inferred(self):
        if hasattr(self, 'inferred_cap'):
            # Set the inferred video capture to the beginning
            self.inferred_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release_capture(self):
        if hasattr(self, 'cap'):
            # Release the video capture when the GUI is closed
            self.cap.release()

    def release_inferred_capture(self):
        if hasattr(self,'inferred_cap'):
            self.inferred_cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv7DetectorApp(root)
    root.mainloop()
