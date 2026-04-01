import cv2
import os

def downsampling(gt, SCALE):
    h, w = gt.shape[:2]
    lr = cv2.resize(gt, (w // SCALE, h // SCALE), interpolation=cv2.INTER_CUBIC)
    return lr

def upsampling(lr):
    h, w = lr.shape[:2]
    sr = cv2.resize(lr, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    return sr
