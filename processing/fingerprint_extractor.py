import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def extract_noise_residual(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to float32 for precision
    img_float = img.astype(np.float32)
    
    # Estimate the scene content through a denoising filter
    # Using Gaussian Blur as a simpler proxy for the denoising filter mentioned in the paper
    denoised = cv2.GaussianBlur(img_float, (3, 3), 0)
    
    # Noise Residual: Ri = Xi - f(Xi)
    residual = img_float - denoised
    return residual

def compute_fft_amplitude(fingerprint):
    # Convert to grayscale if it's 3-channel (average across channels)
    if len(fingerprint.shape) == 3:
        fingerprint = np.mean(fingerprint, axis=2)
    
    # Compute 2D FFT
    f = np.fft.fft2(fingerprint)
    fshift = np.fft.fftshift(f)
    
    # Amplitude (Magnitude)
    amplitude = np.abs(fshift)
    
    # Log scale for visualization
    magnitude_spectrum = 20 * np.log(amplitude + 1)
    
    # Normalize for display
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude_spectrum.astype(np.uint8)

def process_directory(dir_path, limit):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    if not os.path.exists(dir_path):
        print(f"Warning: Directory {dir_path} does not exist.")
        return None, 0
        
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(valid_extensions)]
    files = sorted(files)[:limit]
    
    if not files:
        print(f"Warning: No images found in {dir_path}.")
        return None, 0

    print(f"Extracting fingerprints from {len(files)} images in {dir_path}...")
    
    sum_residual = None
    count = 0
    target_shape = None
    
    for i, file_path in enumerate(files):
        residual = extract_noise_residual(file_path)
        if residual is not None:
            if target_shape is None:
                target_shape = residual.shape
                sum_residual = residual
            else:
                # Resize if images have different dimensions
                if residual.shape != target_shape:
                    residual = cv2.resize(residual, (target_shape[1], target_shape[0]))
                sum_residual += residual
            count += 1
            if count % 100 == 0:
                print(f"  Processed {count}/{len(files)} images...")

    if count == 0:
        return None, 0

    # Average the residuals to estimate the fingerprint
    fingerprint = sum_residual / count
    
    # Compute FFT of the fingerprint
    spectrum = compute_fft_amplitude(fingerprint)
    return spectrum, count

def main():
    if len(sys.argv) < 2:
        print("Usage: python fingerprint_extractor.py <num_images>")
        return

    try:
        limit = int(sys.argv[1])
    except ValueError:
        print("Error: The argument <num_images> must be an integer.")
        return

    fake_dir = "../dataset/train/FAKE/"
    real_dir = "../dataset/train/REAL/"
    
    fake_spectrum, fake_count = process_directory(fake_dir, limit)
    real_spectrum, real_count = process_directory(real_dir, limit)
    
    if fake_spectrum is None and real_spectrum is None:
        print("Error: Failed to process images from both directories.")
        return

    # Set up the comparison plot
    plt.figure(figsize=(12, 6))
    
    if fake_spectrum is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(fake_spectrum, cmap='magma')
        plt.title(f"FAKE Fingerprint FFT\n(N={fake_count} images)")
        plt.axis('off')
        
    if real_spectrum is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(real_spectrum, cmap='magma')
        plt.title(f"REAL Fingerprint FFT\n(N={real_count} images)")
        plt.axis('off')
        
    plt.suptitle("Artificial Fingerprint Frequency Analysis Comparison", fontsize=16)
    plt.tight_layout()
    
    output_filename = "fingerprint_comparison.png"
    plt.savefig(output_filename)
    print(f"\nSaved fingerprint comparison to {output_filename}")
    
    if plt.get_backend().lower() != 'agg':
        try:
            plt.show()
        except:
            pass

if __name__ == "__main__":
    main()
