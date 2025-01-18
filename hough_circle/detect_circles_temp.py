import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_circles(image_path, min_radius=40, max_radius=60, dp=1, min_dist=50, exclusion_margin=100):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None, None

    # Get image dimensions
    img_height, img_width = image.shape

    # Apply GaussianBlur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(image, (9, 9), 2)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, 
                               param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)

    # If no circles are detected, return empty results
    if circles is None:
        print("No circles detected.")
        return None, None, None

    # Convert circle coordinates to integer
    circles = np.uint16(np.around(circles))

    # Prepare data for circles and bounding boxes
    circle_data = []
    bounding_boxes = []

    # For each detected circle, extract the center, radius, and check exclusion conditions
    for circle in circles[0, :]:
        x, y, radius = circle
        
        # Exclude circles if their center is too close to the edges (within exclusion_margin)
        if (x - radius > exclusion_margin and x + radius < img_width - exclusion_margin and 
            y - radius > exclusion_margin and y + radius < img_height - exclusion_margin):
            # Calculate the bounding box for the circle
            x_min = x - radius
            y_min = y - radius
            x_max = x + radius
            y_max = y + radius
            
            bounding_box = (x_min, y_min, x_max, y_max)
            circle_data.append((x, y, radius))
            bounding_boxes.append(bounding_box)
        else:
            # Debug: print when a circle is excluded because its center is too close to an image edge
            print(f"Excluding circle at ({x}, {y}) with radius {radius} - Center too close to the image edge.")
    
    # Debugging: Print all circle locations and their bounding boxes
    print("Detected Circles and Bounding Boxes (after boundary check):")
    for i, (circle, bbox) in enumerate(zip(circle_data, bounding_boxes)):
        print(f"Circle {i+1}: Center = ({circle[0]}, {circle[1]}), Radius = {circle[2]}")
        print(f"Bounding Box {i+1}: {bbox}")
    
    return circle_data, bounding_boxes, image

def draw_circles(image, circles, bounding_boxes, exclusion_margin=100):
    # Create a copy of the original image to overlay the circles
    output_image = image.copy()
        # Get image dimensions
    img_height, img_width = output_image.shape

    # Draw the circles and bounding boxes
    for (x, y, radius), (x_min, y_min, x_max, y_max) in zip(circles, bounding_boxes):
        if (x  > exclusion_margin and x  < img_width - exclusion_margin and 
            y  > exclusion_margin and y  < img_height - exclusion_margin):
            # Draw the circle perimeter
            cv2.circle(output_image, (x, y), radius, (0, 255, 0), 2)
            # Draw the bounding box
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    return output_image

def show_images(original_image, overlap_image):
    # Show the original image and the one with overlapped circles
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(overlap_image, cmap='gray')
    plt.title('Detected Circles')

    plt.show()

def save_image(image, output_path):
    # Save the processed image with detected circles
    cv2.imwrite(output_path, image)
    print(f"Overlap image saved to {output_path}")

def main():
    image_path = 'image/Ec-DH5alpha_1E7_BF_5ms_20X-partial-rotate.png'  # Change this to the path of your image
    output_path = 'output/marked_Ec-DH5alpha_1E7_BF_5ms_20X-partial-rotate.png'  # Output path to save the overlap image

    # Detect circles
    circles, bounding_boxes, image = detect_circles(image_path)

    if circles is not None:
        # Draw circles on the image
        overlap_image = draw_circles(image, circles, bounding_boxes)
        save_image(overlap_image, output_path)  # Save the overlap image

        # # Show images (optional)
        # show_images(image, overlap_image)

        # # Print circle locations and bounding boxes
        # print(f"Detected {len(circles)} circles:")
        # for i, (x, y, radius) in enumerate(circles):
        #     print(f"Circle {i+1}: Center = ({x}, {y}), Radius = {radius}")
        #     print(f"Bounding Box {i+1}: {bounding_boxes[i]}")
    else:
        print("No circles detected.")

if __name__ == "__main__":
    main()
