import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        sys.exit(1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_and_show(plt, title):
    filename = title.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "") + ".png"
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    
    if plt.get_backend().lower() != 'agg':
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
    else:
        print(f"Non-interactive backend ({plt.get_backend()}); skipping display.")
    plt.close()

def plot_group(original, images, titles, group_title):
    n = len(images) + 1
    plt.figure(figsize=(4 * n, 5))
    
    plt.subplot(1, n, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis('off')
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, n, i + 2)
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')
    
    plt.suptitle(group_title, fontsize=16)
    save_and_show(plt, group_title)

def get_flipping_transformations(img):
    h_flipped = cv2.flip(img, 1)
    v_flipped = cv2.flip(img, 0)
    return [h_flipped, v_flipped], ["Horizontal Flip", "Vertical Flip"]

def get_padding_transformation(img):
    rows, cols = img.shape[:2]
    padded = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    padded_rescaled = cv2.resize(padded, (cols, rows), interpolation=cv2.INTER_LINEAR)
    return [padded_rescaled], ["3px Border & Rescaled"]

def get_rotation_transformation(img):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    return [rotated], ["Rotated 45°"]

def get_scaling_transformations(img):
    rows, cols = img.shape[:2]
    upscaled = cv2.resize(img, (cols * 2, rows * 2), interpolation=cv2.INTER_CUBIC)
    downscaled = cv2.resize(img, (cols // 2, rows // 2), interpolation=cv2.INTER_AREA)
    return [upscaled, downscaled], ["Upscaled (2x)", "Downscaled (0.5x)"]

def get_normalization_transformations(img):
    img_float = img.astype(np.float32) / 255.0
    normalized = img_float
    mean = np.mean(img_float, axis=(0, 1))
    std = np.std(img_float, axis=(0, 1))
    standardized = (img_float - mean) / (std + 1e-7)
    standardized_viz = np.clip((standardized + 3) / 6, 0, 1)
    return [normalized, standardized_viz], ["Normalized (0-1)", "Standardized (Z-score)"]

def get_blur_transformation(img):
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    return [blurred], ["Gaussian Blur"]

def compute_fft(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = np.uint8(cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX))
    return magnitude_spectrum

def main():
    if len(sys.argv) < 2:
        print("Usage: python image_processor.py <image_path1> [image_path2]")
        return

    img1_path = sys.argv[1]
    original1 = load_image(img1_path)

    # 1-6. Standard Transformations for the first image
    plot_group(original1, *get_flipping_transformations(original1), "Flipping Transformations")
    plot_group(original1, *get_padding_transformation(original1), "Padding Transformation")
    plot_group(original1, *get_rotation_transformation(original1), "Rotation Transformation")
    plot_group(original1, *get_scaling_transformations(original1), "Scaling Transformations")
    plot_group(original1, *get_normalization_transformations(original1), "Normalization and Standardization")
    plot_group(original1, *get_blur_transformation(original1), "Blurring")

    # 7. Frequency Domain
    if len(sys.argv) > 2:
        # Comparison mode
        img2_path = sys.argv[2]
        original2 = load_image(img2_path)
        
        fft1 = compute_fft(original1)
        fft2 = compute_fft(original2)
        
        plt.figure(figsize=(16, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(original1)
        plt.title("Original 1")
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(fft1, cmap='gray')
        plt.title("FFT 1")
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(original2)
        plt.title("Original 2")
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(fft2, cmap='gray')
        plt.title("FFT 2")
        plt.axis('off')
        
        plt.suptitle("Frequency Domain Comparison", fontsize=16)
        save_and_show(plt, "Frequency Domain")
    else:
        # Single image mode
        fft1 = compute_fft(original1)
        plot_group(original1, [fft1], ["Frequency Spectrum"], "Frequency Domain")

if __name__ == "__main__":
    main()
