import sys
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

class FileSelected:
    def __init__(self):
        self.file = "G:/My Drive/PHD RESEARCH/Manuscript - Dehazing/TEX Proposal/figures/cycle_consistency.png"

    def get_file(self):
        return self.file

    def set_file(self, filename):
        self.file = filename

file_selected = FileSelected()

class AppWindow():
    def __init__(self):
        self.title = "Dehazing UI"
        # setup UI
        self.app_window = tk.Tk()
        self.app_window.title(self.title)

        # create UI elements
        self.canvas = tk.Canvas(self.app_window, width=1440, height=900)
        self.canvas.pack()

        button = tk.Button(text="Press", width=10, height=3, command=self.on_open_file)
        button.pack()

    def on_open_file(self):
        file_name = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png")))
        file_selected.set_file(file_name)

        self.img_canvas = ImageTk.PhotoImage(Image.open(file_selected.get_file()))
        self.canvas.create_image(300, 300, anchor = "center", image=self.img_canvas)
        #self.canvas.itemconfig(self.img_canvas, image = self.img_canvas)

    def initialize(self):
        self.app_window.mainloop()


def main(args):
    print("Hello")
    app_window = AppWindow()
    app_window.initialize()

if __name__ == "__main__":
    main(sys.argv)