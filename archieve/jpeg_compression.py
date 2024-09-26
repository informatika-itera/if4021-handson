import numpy as np
from skimage import color
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

def create_rgb_image():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:128, :128] = [255, 0, 0]  # Red block
    image[:128, 128:] = [0, 255, 0]  # Green block
    image[128:, :128] = [0, 0, 255]  # Blue block
    image[128:, 128:] = [255, 255, 0]  # Yellow block
    return image

def rgb_to_ycbcr(image):
    """Convert RGB to YCbCr."""
    return color.rgb2ycbcr(image)

def ycbcr_to_rgb(image):
    """Convert YCbCr to RGB."""
    return color.ycbcr2rgb(image)

def dct2(block):
    """2D DCT transform."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """2D inverse DCT transform."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def quantize(block, quantization_matrix):
    """Quantize the DCT coefficients."""
    return np.round(block / quantization_matrix) * quantization_matrix

def jpeg_compress(image, quality_factor=50):
    """Perform JPEG-like compression on the image."""
    # Convert to YCbCr
    ycbcr = rgb_to_ycbcr(image)
    
    # Define quantization matrix
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    Q = Q * (quality_factor / 50) if quality_factor > 50 else Q * (50 / quality_factor)
    
    height, width = image.shape[:2]
    compressed = np.zeros_like(ycbcr)
    
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            for k in range(3):  # For each channel
                block = ycbcr[i:i+8, j:j+8, k]
                dct_block = dct2(block)
                quantized = quantize(dct_block, Q)
                compressed[i:i+8, j:j+8, k] = idct2(quantized)
    
    # Convert back to RGB
    return ycbcr_to_rgb(compressed)

def main():
    # Create the sample image
    original = create_rgb_image()
    
    # Compress the image with different quality factors
    quality_factors = [10, 50, 90]
    compressed_images = [jpeg_compress(original, qf) for qf in quality_factors]
    
    # Display the results
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    for i, (compressed, qf) in enumerate(zip(compressed_images, quality_factors), 1):
        axes[i].imshow(compressed)
        axes[i].set_title(f'Compressed (Quality: {qf})')
        axes[i].axis('off')
        
        # Calculate and display PSNR
        mse = np.mean((original - compressed) ** 2)
        psnr = 20 * np.log10(255 / np.sqrt(mse))
        print(f"PSNR (Quality {qf}): {psnr:.2f} dB")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()