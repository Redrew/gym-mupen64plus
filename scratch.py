#!/bin/python
import gym, gym_mupen64plus
import matplotlib.pyplot as plt
import cv2

def saveimg(img, name='image.png'):
    plt.imshow(img)
    plt.savefig(name)

env = gym.make('Smash-mario-v0')
state = env.reset()
state = cv2.resize(state, (64, 64))
saveimg(state)