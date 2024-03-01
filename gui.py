import tkinter as tk
from tkinter import Entry, Button, filedialog, Label, scrolledtext, messagebox
import subprocess
import os
import platform


class YOLOv7DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv7 Detector")

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

        self.create_widgets()

    def create_widgets(self):
        # Create labels, entry widgets, and browse buttons
        headers = ['Weight Path', 'Source Path', 'CFG Path', 'Size', 'Confidence', 'IOU Threshold']
        for i in range(6):
            label = Label(self.root, text=f"{headers[i]}:")
            label.grid(row=i, column=0, pady=5)

            entry = Entry(self.root, textvariable=self.variable_values[i], width=30)
            entry.grid(row=i, column=1, pady=5)

            # Only create browse button for 'Weight Path', 'Source Path', and 'CFG Path'
            if i in [0, 1, 2]:
                browse_button = Button(self.root, text=f"Browse {headers[i]}", command=lambda i=i: self.browse_file(i))
                browse_button.grid(row=i, column=2, pady=10)

        # Run Detection Button
        run_button = Button(self.root, text="Run Detection", command=self.run_detection)
        run_button.grid(row=6, column=1, pady=10)

        # Text widget to display output
        self.output_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=50, height=10)
        self.output_text.grid(row=7, column=1, pady=10)

        # View Original Button
        view_original_button = Button(self.root, text="View Original", command=self.view_original)
        view_original_button.grid(row=10, column=0, pady=10)

        # View Inferred Button
        view_inferred_button = Button(self.root, text="View Inferred", command=self.view_inferred)
        view_inferred_button.grid(row=10, column=1, pady=10)

        # Exit Button
        exit_button = Button(self.root, text="Exit", command=self.root.destroy)
        exit_button.grid(row=10, column=2, pady=10)

    def browse_file(self, index):
        # Open file dialog to select a file
        file_path = filedialog.askopenfilename()
        if file_path:
            # Update the file path in the corresponding Entry widget
            self.variable_values[index].set(file_path)

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
            # Display a message box after successful detection
            messagebox.showinfo("Detection Done", "Object detection is complete!")
        except subprocess.CalledProcessError as e:
            output = e.stdout
            error = e.stderr

        # Display the output on the GUI
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Output:\n{output}\nError:\n{error}")

    def view_original(self):
        source_path = self.variable_values[1].get()
        if os.path.exists(source_path):
            if platform.system() == "Windows":
                try:
                    # Open the file with the default associated program on Windows
                    subprocess.run(['start', '', source_path], shell=True)
                except Exception as e:
                    print(f"Error opening file: {e}")
            else:
                print("Opening files is only supported on Windows in this example.")
        if ((source_path == '0') or (source_path == None) or (source_path == '')):
            messagebox.showwarning("No Source Path", "No path to source file found")

    def view_inferred(self):
        try:
            output_text = self.output_text.get(1.0, tk.END)
            paths = self.extract_paths(output_text, "The result is saved in:")
            if os.path.exists(paths):
                if platform.system() == "Windows":
                    try:
                        # Open the file with the default associated program on Windows
                        subprocess.run(['start', '', paths], shell=True)
                    except Exception as e:
                        print(f"Error opening file: {e}")
                else:
                    print("Opening files is only supported on Windows in this example.")
        except:
            messagebox.showwarning("No Inferred Path", "No path to inferred file found")

    def extract_paths(self, text, file_type):
        lines = text.split("\n")
        paths = [line.split(":")[1].strip() for line in lines if f"{file_type}" in line]
        return paths[0]

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv7DetectorApp(root)
    root.mainloop()
