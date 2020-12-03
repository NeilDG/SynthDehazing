import sys
import tkinter as tk
from tkinter import filedialog
import constants
from torchvision import transforms
from model import ffa_net as ffa
from model import latent_network
import torch
from PIL import ImageTk, Image
import numpy as np
from utils import tensor_utils

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
        self.input_img_resource = None

        # setup UI
        self.app_window = tk.Tk()
        self.app_window.title(self.title)

        #setup useful attributes for layouting
        self.IMG_WIDTH = 40
        self.IMG_HEIGHT = 20

        # create UI elements
        self.display_frame = tk.Frame(self.app_window, borderwidth = 1)
        self.display_frame.pack(fill = tk.BOTH, expand = True)
        self.display_frame.configure(bg = "#333")

        self.loaded_img_display = tk.Label(self.display_frame, text = "Input image")
        self.loaded_img_display.pack(side = tk.LEFT, expand = True)

        self.output_img_display = tk.Label(self.display_frame, text = "Output Image")
        self.output_img_display.pack(side = tk.RIGHT, expand = True)

        self.z_slider = tk.Scale(self.app_window, from_=-1.0, to = 1.0, length = 600, resolution=0.00001, orient=tk.HORIZONTAL, command = self.on_slider_changed)
        self.z_slider.set(0.0)
        self.z_slider.pack()

        self.button_frame = tk.Frame(self.app_window, relief = tk.RAISED, borderwidth = 1)
        self.button_frame.pack(fill = tk.BOTH, expand = True)

        process_btn = tk.Button(self.button_frame, text="Dehaze", width=10, height=3, command=self.on_process)
        process_btn.pack(side = tk.RIGHT, padx = 5, pady =5)

        load_img_btn = tk.Button(self.button_frame, text="Load Image", width=10, height=3, command=self.on_open_file)
        load_img_btn.pack(side=tk.RIGHT, padx=5, pady=5)

    def on_open_file(self):
        file_name = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png")))
        file_selected.set_file(file_name)

        rgb_transform_op = transforms.Compose([transforms.ToPILImage(),
                                               transforms.Resize(constants.TEST_IMAGE_SIZE),
                                               transforms.CenterCrop(constants.TEST_IMAGE_SIZE)])
        self.input_img_resource = Image.open(file_selected.get_file())
        self.input_img_resource = rgb_transform_op(np.asarray(self.input_img_resource))
        self.input_img_canvas = ImageTk.PhotoImage(self.input_img_resource)
        self.loaded_img_display.configure(image = self.input_img_canvas)

    def on_slider_changed(self, event):
        print(self.z_slider.get())

    def on_process(self):
        if(self.input_img_resource != None):
            print("Initiated dehazing network")

            DEHAZER_CHECKPATH = "./checkpoint/dehazer_v1.11_2.pt"
            device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
            dehazer = ffa.FFA(gps=3, blocks=19).to(device)
            dehazer_checkpt = torch.load(DEHAZER_CHECKPATH)
            dehazer.load_state_dict(dehazer_checkpt[constants.GENERATOR_KEY])

            LN = latent_network.LatentNetwork().to(device)
            latent_checkpoint = torch.load(constants.LATENT_CHECKPATH)
            LN.load_state_dict(latent_checkpoint[constants.GENERATOR_KEY])

            rgb_transform_op = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Resize(constants.TEST_IMAGE_SIZE),
                                                   transforms.CenterCrop(constants.TEST_IMAGE_SIZE),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            with torch.no_grad():
                rgb_tensor = rgb_transform_op(np.asarray(self.input_img_resource))
                rgb_tensor = torch.unsqueeze(rgb_tensor, 0).to(device)
                z_signal = tensor_utils.compute_z_signal(self.z_slider.get(), 100, constants.TEST_IMAGE_SIZE).to(device)
                #rgb_tensor_clean = dehazer(rgb_tensor, LN(z_signal))
                rgb_tensor_clean = LN(z_signal)
                rgb_tensor_clean = tensor_utils.normalize_to_matplotimg(rgb_tensor_clean.cpu(), 0, 0.5, 0.5)
                to_pil_op = transforms.ToPILImage()
                output_img_resource = to_pil_op(rgb_tensor_clean)

            self.output_img_canvas = ImageTk.PhotoImage(output_img_resource)
            self.output_img_display.configure(image = self.output_img_canvas)
        else:
            print("Provide input image first.")
        
    def initialize(self):
        self.app_window.mainloop()


def main(args):
    print("Hello")
    app_window = AppWindow()
    app_window.initialize()

if __name__ == "__main__":
    main(sys.argv)