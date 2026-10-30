import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_ai_signs(image_path):
    #в оттенках серого
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Картинка не найдена")
        return

    #Анализ высокочастотного шума 
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    noise = cv2.absdiff(img, blurred) 
    
    # Считаем среднimport cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_ai_signs(image_path):
    #в оттенках серого
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Картинка не найдена")
        return

    #Анализ высокочастотного шума 
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    noise = cv2.absdiff(img, blurred) 

    
    # Считаем средний уровень шума
    noise_level = np.mean(noise)
    
    
    edges = cv2.Canny(img, 100, 200)
    
    
    print(f"📊 Уровень шума (PRNU): {noise_level:.2f}")
    print(f"📊 Количество пикселей на границах: {np.sum(edges > 0)}")ий уровень шума
    noise_level = np.mean(noise)
    
    
    edges = cv2.Canny(img, 100, 200)
    
    
    print(f"📊 Уровень шума (PRNU): {noise_level:.2f}")
    print(f"📊 Количество пикселей на границах: {np.sum(edges > 0)}")
    # Отрисовка для наглядности
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Оригинал")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Карта шума (ИИ = черный фон)")
    plt.imshow(noise, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Границы (Canny)")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


test_img = '/kaggle/input/datasets/yassaabram/qt-detect-yolo/QR-1/valid/images/003_jpg.rf.62920359c53e8497c278fefabe4d6bac.jpg'
analyze_ai_signs(test_img)