import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def compute_fft(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = np.uint8(cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX))
    return magnitude_spectrum

def get_images_from_dir(dir_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = []
    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a directory.")
        return []
    
    # Sort files to ensure consistent order
    for filename in sorted(os.listdir(dir_path)):
        if filename.lower().endswith(valid_extensions):
            path = os.path.join(dir_path, filename)
            img = load_image(path)
            if img is not None:
                images.append((img, filename))
    return images

def main():
    if len(sys.argv) < 3:
        print("Usage: python freq_processor.py <dir1> <dir2>")
        return

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]

    images1 = get_images_from_dir(dir1)
    images2 = get_images_from_dir(dir2)

    if not images1 and not images2:
        print("No valid images found in either directory.")
        return

    num_rows = 2
    max_imgs = max(len(images1), len(images2))
    num_cols = 2 * max_imgs # 2 columns per image (Original and FFT)

    # Dynamic figure size based on number of images
    plt.figure(figsize=(4 * num_cols, 5 * num_rows))

    # Row 1: First Directory
    for i, (img, name) in enumerate(images1):
        fft = compute_fft(img)
        # Plot Original
        plt.subplot(num_rows, num_cols, 2 * i + 1)
        plt.imshow(img)
        plt.title(f"D1 - {name}")
        plt.axis('off')
        # Plot FFT
        plt.subplot(num_rows, num_cols, 2 * i + 2)
        plt.imshow(fft, cmap='gray')
        plt.title(f"FFT: {name}")
        plt.axis('off')

    # Row 2: Second Directory
    for i, (img, name) in enumerate(images2):
        fft = compute_fft(img)
        # Plot Original (Starts at row 2)
        plt.subplot(num_rows, num_cols, num_cols + 2 * i + 1)
        plt.imshow(img)
        plt.title(f"D2 - {name}")
        plt.axis('off')
        # Plot FFT
        plt.subplot(num_rows, num_cols, num_cols + 2 * i + 2)
        plt.imshow(fft, cmap='gray')
        plt.title(f"FFT: {name}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("FREQS.png")
    print(f"Processed {len(images1)} images from {dir1} and {len(images2)} images from {dir2}.")
    print("Saved frequency comparison grid to FREQS.png")
    
    if plt.get_backend().lower() != 'agg':
        try:
            plt.show()
        except:
            pass

if __name__ == "__main__":
    main()
