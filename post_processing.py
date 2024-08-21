import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def morph_processing(image_path):
    # Load the image
    image = cv2.imread(image_path)  
    # image = cv2.bitwise_not(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)  # Adjust threshold value as needed
    
    # Define kernels for morphological operations
    kernel = np.ones((10, 10), np.uint8)  # Adjust size for your specific image

    # Apply dilation to merge parts of the drone
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Apply erosion to remove smaller noise
    eroded = cv2.erode(dilated, kernel, iterations=10)

    # Find contours in the binary image to identify the drone
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the drone
    mask = np.zeros_like(gray)

    # Draw contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Optionally apply morphological closing to fill small holes in the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Create a 3-channel mask for the original image
    mask_3channel = cv2.merge([mask, mask, mask])

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, mask_3channel)

    # Display results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 3, 2)
    plt.title('Binary Image')
    plt.imshow(binary, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Dilated Image')
    plt.imshow(dilated, cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title('Eroded Image')
    plt.imshow(eroded, cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title('Mask')
    plt.imshow(mask, cmap='gray')

    plt.subplot(2, 3, 6)
    plt.title('Result')
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imwrite("j_morph.png", result)
    return result

# morph_processing('a_5_cropped.png')
# morph_processing('b_5_cropped.png')
# morph_processing('c_5_cropped.png')
# morph_processing('d_5_cropped.png')
# morph_processing('e_5_cropped.png')
# morph_processing('f_5_cropped.png')
# morph_processing('g_5_cropped.png')
# morph_processing('h_5_cropped.png')
# morph_processing('i_5_cropped.png')
# morph_processing('j_5_cropped.png')


def find_bounding_boxes(image_path):
    """
    Find bounding boxes of objects in the image.
    
    Parameters:
    - image_path: Path to the input image
    
    Returns:
    - Image
    - Biggest bounding box
    """
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # print(image)

    # Crop the image to remove white borders
    height, width = image.shape
    image = image[1 : height - 1, 1 : width - 1]

    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print('contours', contours)

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
    
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("Bounding Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print('FOUND BOUNDING BOXES:', bounding_boxes)
    print('last bounding box', bounding_boxes[-1])
    return image, bounding_boxes[-1] # Drone is biggest object in image


# Global variables
drawing = False
box_start = (-1, -1)
boxes = []

def manually_select_bounding_boxes(image_path):
    global drawing, box_start, boxes  # Declare the global variables
    image = cv2.imread(image_path)
    clone = image.copy()  # Initialize clone with a copy of the image

    def mouse_callback(event, x, y, flags, param):
        global drawing, box_start, boxes, clone

        if event == cv2.EVENT_LBUTTONDOWN:
            box_start = (x, y)
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                clone = image.copy()  # Update the clone image during drawing
                cv2.rectangle(clone, box_start, (x, y), (0, 255, 0), 2)
                cv2.imshow("drawn bb", clone)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(clone, box_start, (x, y), (0, 255, 0), 2)
            boxes.append((box_start, (x, y)))
            cv2.imshow("drawn bb", clone)

    # Show the image and set the mouse callback function
    cv2.imshow("drawn bb", image)
    cv2.setMouseCallback("drawn bb", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert bounding boxes to (x, y, w, h)
    bbox_list = []
    for start, end in boxes:
        x1, y1 = start
        x2, y2 = end
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x1 - x2)
        h = abs(y1 - y2)
        bbox_list.append((x, y, w, h))

    # Draw box on image
    x, y, w, h = bbox_list[-1]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print("CHOSEN BOUNDING BOX", bbox_list)
    return image, bbox_list[-1]


def superimpose_images(image1, image2, alpha=0.5, beta=0.5):
    """
    Superimpose two images with a specified alpha and beta for blending.

    :param image1_path: Path to the first image.
    :param image2_path: Path to the second image.
    :param alpha: Weight of the first image.
    :param beta: Weight of the second image.
    :return: The superimposed image.
    """

    # Resize images to the same size 
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Superimpose images
    superimposed_image = cv2.addWeighted(image1, alpha, image2, beta, 0)

    # Display the result
    cv2.imshow('Superimposed Image', superimposed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

   


def calculate_iou(image_path1, image_path2):
    image1, box1 = find_bounding_boxes(image_path1)
    image2, box2 = manually_select_bounding_boxes(image_path2)

    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
   
    box1_area = w1 * h1
    box2_area = w2 * h2

    x_inter1 = max(x1, x2)
    y_inter1 = max(y1, y2)
    x_inter2 = min(x1 + w1, x2 + w2)
    y_inter2 = min(y1 + h1, y2 + h2)

    w_inter = max(0, x_inter2 - x_inter1)
    h_inter = max(0, y_inter2 - y_inter1)
    
    intersection = w_inter * h_inter
    union = box1_area + box2_area - intersection

    iou = intersection / union
    print('IoU is', iou)

    superimpose_images(image1, image2)

    return iou

calculate_iou('b_morph.png', 'b_og_cropped.png')

cv2.waitKey(0)
cv2.destroyAllWindows()