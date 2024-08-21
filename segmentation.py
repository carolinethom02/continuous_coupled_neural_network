import numpy as np
import matplotlib.pyplot as plt

from classic_model import ClassicalPCNN
from ccnn_classic_model import ClassicalCCNN


import cv2


def choose_pcnn_model(pcnn_type, input_shape, kernel_type):
    if pcnn_type == 'PCNN':
        model = ClassicalPCNN(input_shape, kernel_type)
    elif pcnn_type == 'CCNN':
        model = ClassicalCCNN(input_shape, kernel_type)

    return model


def run_image_segm(gamma, beta, v_theta, kernel='gaussian'):

    colour_image = cv2.imread('drone_a.jpg')
    colour_image = np.uint8(colour_image)
    image = cv2.cvtColor(colour_image, cv2.COLOR_BGR2GRAY)

    model = ClassicalCCNN(image.shape, kernel)
    # segm_image = classifier.segment_image(image, gamma, beta, v_theta, kernel_type='gaussian')
    segm_image = model.segment_image(image, gamma=1, beta=2, v_theta=40, kernel_type='gaussian')
    print(image)
    print(segm_image)

    plt.imshow(np.hstack((image, segm_image)), cmap = 'gray')
    plt.colorbar()
    plt.show()

    # plt.imshow(image)
    # plt.colorbar()
    # plt.show()

    # plt.imshow(segm_image)
    # plt.colorbar()
    # plt.show()

def run_pcnn_model(pcnn_type):
    colour_image = cv2.imread('drone_a.jpg')
    # colour_image = np.uint8(colour_image)
    image = cv2.cvtColor(colour_image, cv2.COLOR_BGR2GRAY)
    print("og shape", image.shape)
    # 'ICM'
    # 'SCM'
    # 'SLM'
    # 'FLM'
    # 'PCNN'

    model = choose_pcnn_model(pcnn_type, input_shape=image.shape, kernel_type='gaussian')
    model.simulate(image)


    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()  # Flatten the array to make it 1D

    # Plot the histogram
    plt.figure(figsize=(10, 5))
    plt.bar(range(256), hist, color='gray')
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

def bounding_boxes(image_path):
    return None

def superimpose_images(og_image, segm_image):
    return None

import time
start = time.time()
# run_pcnn_model('SCM')
# run_pcnn_model('ICM')
# run_pcnn_model('SLM')
# run_pcnn_model('FLM')
# run_pcnn_model('PCNN')
# run_pcnn_model('CCNN')

# run_image_segm(10, 1.1, 40)
print(time.time() - start)

