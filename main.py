import os
import cv2
import argparse
import numpy as np

# Process the images
def process_images(input_dir):
    # Create the Results directory if it does not exist
    output_dir = "Results"
    os.makedirs(output_dir, exist_ok=True)

    # Load all images from the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for i, filename in enumerate(image_files):
        # Read the image
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        # De-noising using Gaussian blur
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 20, 10, 7, 21)

        # Define source and destination points for perspective transformation
        src_points = np.float32([[8, 14], [234, 4], [29, 235], [251, 228]])
        dst_points = np.float32([[0, 0], [denoised_image.shape[1], 0], [0, denoised_image.shape[0]], [denoised_image.shape[1], denoised_image.shape[0]]])

        # Calculate perspective transformation matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply perspective transformation
        warped_image = cv2.warpPerspective(denoised_image, perspective_matrix, (256, 256))

        # Define mask for the missing region
        gray_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            mask = np.zeros_like(gray_image)
            cv2.drawContours(mask, [contour], 0, 255, 60)
            inpainting_image = cv2.inpaint(warped_image, mask, 30, cv2.INPAINT_TELEA)

        # Adjusting color contrast
        contrasted_image = cv2.convertScaleAbs(inpainting_image, alpha=1.1, beta=-10)

        # Enhancing color saturation
        contrasted_hsv = cv2.cvtColor(contrasted_image, cv2.COLOR_BGR2HSV)
        contrasted_hsv[:,:,1] = np.clip(contrasted_hsv[:,:,1]*1.5, 0, 255)
        enhanced_image = cv2.cvtColor(contrasted_hsv, cv2.COLOR_HSV2BGR)

        # Save the processed image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, enhanced_image)

        # Print the filename
        print(f"Processed: {filename}")


# Define the main function
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Image processing for chest X-ray images")
    parser.add_argument("input_dir", help="Directory containing input images")
    args = parser.parse_args()

    # Process the images
    process_images(args.input_dir)


if __name__ == "__main__":
    main()
