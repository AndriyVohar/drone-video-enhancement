"""
Test image enhancement - підбір параметрів на одному зображенні

Використання:
1. Помістіть тестове зображення в input/test_frame.jpg (або вкажіть інший шлях)
2. Налаштуйте параметри нижче
3. Запустіть: python test_image.py
4. Подивіться результат: output/test_result.jpg
5. Коли результат ОК - скопіюйте параметри в config.py
"""
import cv2
import numpy as np
import os
from utils.fft_tools import psf2otf
from numpy.fft import fft2, ifft2

# ========================================
# НАЛАШТУВАННЯ ДЛЯ ТЕСТУВАННЯ
# ========================================

# Шлях до тестового зображення
INPUT_IMAGE = "input/test_frame.jpg"  # ← ЗМІНІТЬ на своє зображення
OUTPUT_IMAGE = "output/test_result.jpg"

# --- DENOISING ---
DENOISE_METHOD = "nlm"  # "none", "gaussian", "bilateral", "nlm"
GAUSSIAN_SIGMA = 0.01
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 20
BILATERAL_SIGMA_SPACE = 20

# --- PSF (Point Spread Function) ---
PSF_TYPE = "motion"  # "motion" або "gaussian"
MOTION_LENGTH = 1.2    # ← ГОЛОВНИЙ ПАРАМЕТР! Почніть з 5, пробуйте 3, 7, 10
MOTION_ANGLE = 0     # ← ВАЖЛИВО! 0=горизонтально, 90=вертикально, 45=діагональ

# --- DEBLURRING ---
DEBLUR_METHOD = "wiener"  # "none", "wiener", "richardson_lucy"
WIENER_K = 0.03         # ← Якщо багато шуму - збільште до 0.05
RL_ITERATIONS = 10

# --- POST-PROCESSING ---
APPLY_CLAHE = False      # Контраст (безпечно)
CLAHE_CLIP_LIMIT = 2.0
APPLY_SHARPENING = False  # Не використовуйте з deblur одночасно!
SHARPENING_AMOUNT = 0.3

# ========================================
# КОД (не змінюйте, якщо не знаєте що робите)
# ========================================

def motion_psf(length, angle, size):
    psf = np.zeros((size, size), np.float32)
    center = size // 2
    rad = np.deg2rad(angle)
    x = int(center + length * np.cos(rad))
    y = int(center + length * np.sin(rad))
    cv2.line(psf, (center, center), (x, y), 1, 1)
    psf = psf / (psf.sum() + 1e-8)
    return psf


def gaussian_psf(size, sigma):
    """Generate Gaussian PSF."""
    kernel = cv2.getGaussianKernel(size[0], sigma)
    psf = kernel @ kernel.T
    return psf / np.sum(psf)


def denoise_image(img, method):
    """Apply denoising."""
    if method == "gaussian":
        return cv2.GaussianBlur(img, (0, 0), GAUSSIAN_SIGMA)
    elif method == "bilateral":
        return cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    elif method == "nlm":
        if len(img.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, GAUSSIAN_SIGMA * 10,
                                                   GAUSSIAN_SIGMA * 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(img, None, GAUSSIAN_SIGMA * 10, 7, 21)
    return img




def wiener_deconvolution(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy


def richardson_lucy(image, psf, iterations=10):
    """Richardson-Lucy deconvolution."""
    psf_flipped = np.flip(psf)
    estimate = image.copy()

    for _ in range(iterations):
        # Convolve estimate with PSF
        conv = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REPLICATE)

        # Avoid division by zero
        conv = np.maximum(conv, 1e-10)

        # Ratio
        ratio = image / conv

        # Convolve ratio with flipped PSF
        ratio_conv = cv2.filter2D(ratio, -1, psf_flipped, borderType=cv2.BORDER_REPLICATE)

        # Update estimate
        estimate = estimate * ratio_conv

    return estimate


def apply_clahe(img):
    """Apply CLAHE (correctly for color images)."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))

    if len(img.shape) == 2:
        return clahe.apply(img)
    else:
        # Convert to LAB, apply to L channel only
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def apply_sharpening(img, amount):
    """Apply unsharp mask."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    img_float = img.astype(np.float32)
    blurred_float = blurred.astype(np.float32)
    sharpened = img_float + amount * (img_float - blurred_float)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def process_channel(channel, psf):
    """Process single channel for deblurring."""
    channel_float = channel.astype(np.float32) / 255.0

    if DEBLUR_METHOD == "wiener":
        result = wiener_deconvolution(channel_float, psf, WIENER_K)
    elif DEBLUR_METHOD == "richardson_lucy":
        result = richardson_lucy(channel_float, psf, RL_ITERATIONS)
    else:
        result = channel_float

    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result


def main():
    """Main processing."""
    print("=" * 60)
    print("ТЕСТУВАННЯ ПАРАМЕТРІВ НА ЗОБРАЖЕННІ")
    print("=" * 60)

    # Check input file
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ Помилка: Файл не знайдено: {INPUT_IMAGE}")
        print("\n💡 Інструкція:")
        print("1. Витягніть кадр з відео:")
        print("   ffmpeg -i input/drone_auto.mp4 -vf 'select=eq(n\\,100)' -vframes 1 input/test_frame.jpg")
        print("2. Або помістіть будь-яке зображення в input/test_frame.jpg")
        print("3. Запустіть знову: python test_image.py")
        return

    # Load image
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"❌ Не вдалося завантажити зображення: {INPUT_IMAGE}")
        return

    print(f"✓ Завантажено: {INPUT_IMAGE}")
    print(f"  Розмір: {img.shape[1]}x{img.shape[0]}")

    print("\nПараметри обробки:")
    print(f"  Denoising: {DENOISE_METHOD}")
    print(f"  Deblur: {DEBLUR_METHOD}")
    if DEBLUR_METHOD != "none":
        print(f"  PSF: {PSF_TYPE}, Length={MOTION_LENGTH}, Angle={MOTION_ANGLE}°")
        if DEBLUR_METHOD == "wiener":
            print(f"  Wiener K: {WIENER_K}")
        elif DEBLUR_METHOD == "richardson_lucy":
            print(f"  RL iterations: {RL_ITERATIONS}")
    print(f"  CLAHE: {APPLY_CLAHE}")
    print(f"  Sharpening: {APPLY_SHARPENING}")
    print()

    # Process
    result = img.copy()

    # 1. Denoise
    if DENOISE_METHOD != "none":
        print("⏳ Denoising...")
        result = denoise_image(result, DENOISE_METHOD)

    # 2. Deblur
    if DEBLUR_METHOD != "none":
        print("⏳ Deblurring...")

        # Generate PSF
        if PSF_TYPE == "motion":
            psf = motion_psf(MOTION_LENGTH, MOTION_ANGLE, 65)  # або 41, або 51, але не tuple
        else:
            psf = gaussian_psf((15, 15), 3.0)

        # Process each color channel separately
        b, g, r = cv2.split(result)
        b_deblur = process_channel(b, psf)
        g_deblur = process_channel(g, psf)
        r_deblur = process_channel(r, psf)
        result = cv2.merge([b_deblur, g_deblur, r_deblur])

    # 3. CLAHE
    if APPLY_CLAHE:
        print("⏳ Applying CLAHE...")
        result = apply_clahe(result)

    # 4. Sharpening
    if APPLY_SHARPENING:
        print("⏳ Sharpening...")
        result = apply_sharpening(result, SHARPENING_AMOUNT)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_IMAGE) if os.path.dirname(OUTPUT_IMAGE) else ".", exist_ok=True)
    cv2.imwrite(OUTPUT_IMAGE, result)

    # Also save side-by-side comparison
    comparison = np.hstack([img, result])
    comparison_path = OUTPUT_IMAGE.replace('.jpg', '_comparison.jpg')
    cv2.imwrite(comparison_path, comparison)

    print("✅ Готово!")
    print(f"   Результат: {OUTPUT_IMAGE}")
    print(f"   Порівняння: {comparison_path}")
    print()
    print("💡 Що робити далі:")
    print("1. Подивіться результат")
    print("2. Якщо ОК - скопіюйте параметри в config.py")
    print("3. Якщо НЕ ОК - змініть параметри вгорі файлу і запустіть знову")
    print("=" * 60)


if __name__ == "__main__":
    main()

