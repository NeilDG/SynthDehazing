# -*- coding: utf-8 -*-
# Class for early stopping mechanism
from enum import Enum
import torch.nn as nn
import kornia
import torch.cuda.amp as amp
import torch

class EarlyStopperMethod(Enum):
    L1_TYPE = 0,
    SSIM_TYPE = 1,
    PSNR_TYPE = 2,

class EarlyStopper():
    def __init__(self, min_epochs, early_stopper_method):
        self.min_epochs = min_epochs
        self.early_stop_tolerance = 2000
        self.stop_counter = 0
        self.last_metric = 10000.0
        self.stop_condition_met = False
        self.network = None

        if(early_stopper_method is EarlyStopperMethod.L1_TYPE):
            self.loss_op = nn.L1Loss()
        elif(early_stopper_method is EarlyStopperMethod.SSIM_TYPE):
            self.loss_op = kornia.losses.SSIMLoss(5)

    def test(self, epoch, input_tensor, gt_tensor):
        if(epoch < self.min_epochs):
            return

        with torch.no_grad(), amp.autocast():
            D_loss = self.loss_op(input_tensor, gt_tensor)

        if(self.last_metric < D_loss):
            self.stop_counter += 1

        elif(self.last_metric >= D_loss):
            self.last_metric = D_loss
            self.stop_counter = 0
            print("Early stopping mechanism reset. Best metric is now ", self.last_metric)

        if (self.stop_counter == self.early_stop_tolerance):
            self.stop_condition_met = True
            print("Met stopping condition with best metric of: ", self.last_metric, ". Latest metric: ", D_loss)

        return self.stop_condition_met

    def did_stop_condition_met(self):
        return self.stop_condition_met









