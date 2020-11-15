# -*- coding: utf-8 -*-
# Template trainer. Do not use this for actual training.

import os
from model import vanilla_cycle_gan as cg
import constants
import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.utils as vutils
from utils import logger
from utils import plot_utils

class TemplateTrainer:
    
    def __init__(self, gan_version, gan_iteration, gpu_device, lr = 0.0002):
        self.gpu_device = gpu_device
        self.lr = lr
        self.gan_version = gan_version
        self.gan_iteration = gan_iteration
        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()
        
    
    def initialize_dict(self):
        #what to store in visdom?
        self.losses_dict = {}
        
    
    def update_penalties(self):
        #what penalties to use for losses?
        self.cycle_weight = 10.0
    
    def train(self, tensor_a, tensor_b):
        print(tensor_a, tensor_b)
        
        #what to put to losses dict for visdom reporting?
    
    def visdom_report(self, train_a, train_b, test_a, test_b):
        with torch.no_grad():
            #infer
            print()
        
        #report to visdom
    
    def load_saved_state(self, iteration, checkpoint, model_key, optimizer_key):
        self.iteration = iteration
        #load model
    
    def save_states(self, epoch, iteration, path, model_key, optimizer_key):
        print()
        #save model