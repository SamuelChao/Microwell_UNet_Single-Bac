import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def detect_circles(image, min_radius=40, max_radius=60, dp=1, min_dist=50, exclusion_margin=100):
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # if image is None:
    #     print(f"Error: Unable to load image at {image_path}")
    #     return None, None, None

    img_height, img_width = image.shape

    # Apply GaussianBlur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(image, (9, 9), 2)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, 
                               param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)

    if circles is None:
        print("No circles detected.")
        return None, None, None

    circles = np.uint16(np.around(circles))

    circle_data = []
    bounding_boxes = []

    center_x = []
    center_y = []
    radii = []
    top_left = []
    bottom_right = []

    for circle in circles[0, :]:
        x, y, radius = circle
        
        if (x - radius > exclusion_margin and x + radius < img_width - exclusion_margin and 
            y - radius > exclusion_margin and y + radius < img_height - exclusion_margin):
            x_min = x - radius
            y_min = y - radius
            x_max = x + radius
            y_max = y + radius
            
            bounding_box = (x_min, y_min, x_max, y_max)
            circle_data.append((x, y, radius))
            bounding_boxes.append(bounding_box)

            center_x.append(x)
            center_y.append(y)
            radii.append(radius)
            top_left.append([x-90, y-90])
            bottom_right.append([x+90, y+90])



        
    return center_x, center_y, radii, top_left, bottom_right




def detect_circles_frompath(image_path, min_radius=40, max_radius=60, dp=1, min_dist=50, exclusion_margin=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None, None

    img_height, img_width = image.shape

    # Apply GaussianBlur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(image, (9, 9), 2)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, 
                               param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)

    if circles is None:
        print("No circles detected.")
        return None, None, None

    circles = np.uint16(np.around(circles))

    circle_data = []
    bounding_boxes = []

    for circle in circles[0, :]:
        x, y, radius = circle
        
        if (x - radius > exclusion_margin and x + radius < img_width - exclusion_margin and 
            y - radius > exclusion_margin and y + radius < img_height - exclusion_margin):
            x_min = x - radius
            y_min = y - radius
            x_max = x + radius
            y_max = y + radius
            
            bounding_box = (x_min, y_min, x_max, y_max)
            circle_data.append((x, y, radius))
            bounding_boxes.append(bounding_box)

    return circle_data, bounding_boxes, image

def draw_circles(image, circles, bounding_boxes, exclusion_margin=100):
    output_image = image.copy()
    img_height, img_width = output_image.shape

    for (x, y, radius), (x_min, y_min, x_max, y_max) in zip(circles, bounding_boxes):
        if (x  > exclusion_margin and x  < img_width - exclusion_margin and 
            y  > exclusion_margin and y  < img_height - exclusion_margin):
            cv2.circle(output_image, (x, y), radius, (0, 255, 0), 2)
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    return output_image

def save_image(image, output_path):
    # Save the processed image with detected circles
    cv2.imwrite(output_path, image)
    print(f"Overlap image saved to {output_path}")


def save_slice(image, center, slice_size, output_folder, slice_index):
    # Calculate the slice boundaries (ensure they stay within image bounds)
    x, y = center
    half_size = slice_size // 2
    y_min = max(0, y - half_size)
    y_max = min(image.shape[0], y + half_size)
    x_min = max(0, x - half_size)
    x_max = min(image.shape[1], x + half_size)

    slice = image[y_min:y_max, x_min:x_max]

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the slice as a PNG file
    slice_filename = os.path.join(output_folder, f"slice_{slice_index}.png")
    cv2.imwrite(slice_filename, slice)
    print(f"Saved slice {slice_index} to {slice_filename}")

def main():
    image_path = 'image/Ec-DH5alpha_1E7_BF_5ms_20X-partial-rotate.png'
    output_folder = 'output/microwell_slice'

    circles, bounding_boxes, image = detect_circles_frompath(image_path)

    if circles is not None:
        # Draw circles on the image
        overlap_image = draw_circles(image, circles, bounding_boxes)
        save_image(overlap_image, 'output/marked_image.png')  # Save the overlap image

        # Extract 180x180 slices for each detected circle and save
        slice_size = 180
        for i, (x, y, radius) in enumerate(circles):
            save_slice(image, (x, y), slice_size, output_folder, i)
    else:
        print("No circles detected.")

if __name__ == "__main__":
    main()
