import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

real_images = [f for f in glob.glob('/kaggle/input/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/train/REAL/*.jpg') if ' ' not in f]
ai_images = [f for f in glob.glob('/kaggle/input/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/train/FAKE/*.jpg') if ' ' not in f]

real_img_path = real_images[0]
ai_img_path = ai_images[0]

print(f"Тестируем реальное фото: {real_img_path}")
print(f"Тестируем ИИ-картинку: {ai_img_path}")

def extract_image_features(image_path, title_prefix=""):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Ошибка загрузки: {image_path}")
        return None

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    noise = cv2.absdiff(img, blurred) 
    noise_level = np.mean(noise)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    h, w = img.shape
    center_h, center_w = h//2, w//2
    high_freq_energy = np.mean(magnitude_spectrum[center_h-50:center_h+50, center_w-50:center_w+50])

    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    lbp_variance = np.var(lbp)

    print(f"\n📊 {title_prefix}:")
    print(f"1. Уровень шума: {noise_level:.2f}")
    print(f"2. Энергия высоких частот (FFT): {high_freq_energy:.2f}")
    print(f"3. Разброс текстур (LBP Variance): {lbp_variance:.2f}")
    
    return noise, magnitude_spectrum, lbp

res_real = extract_image_features(real_img_path, "РЕАЛЬНОЕ ФОТО")
res_ai = extract_image_features(ai_img_path, "ИИ-КАРТИНКА")
# --- ВИЗУАЛИЗАЦИЯ ДЛЯ СРАВНЕНИЯ ---
if res_real and res_ai:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    axes[0, 0].imshow(cv2.imread(real_img_path), cmap='gray')
    axes[0, 0].set_title("Реальное: Оригинал")
    axes[0, 1].imshow(res_real[0], cmap='gray')
    axes[0, 1].set_title("Реальное: Шум")
    axes[0, 2].imshow(res_real[1], cmap='gray')
    axes[0, 2].set_title("Реальное: FFT")
    axes[0, 3].imshow(res_real[2], cmap='gray')
    axes[0, 3].set_title("Реальное: LBP")

    axes[1, 0].imshow(cv2.imread(ai_img_path), cmap='gray')
    axes[1, 0].set_title("ИИ: Оригинал")
    axes[1, 1].imshow(res_ai[0], cmap='gray')
    axes[1, 1].set_title("ИИ: Шум")
    axes[1, 2].imshow(res_ai[1], cmap='gray')
    axes[1, 2].set_title("ИИ: FFT")
    axes[1, 3].imshow(res_ai[2], cmap='gray')
    axes[1, 3].set_title("ИИ: LBP")

    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
