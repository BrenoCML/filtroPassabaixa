
'''
import cv2
import numpy as np

def butterworth_lowpass_filter(image, cutoff_frequency):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Cria um filtro passa-baixa Butterworth
    mask = np.zeros((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow) ** 2 + (y - ccol) ** 2 <= cutoff_frequency ** 2
    mask[mask_area] = 1

    # Aplica a transformada de Fourier na imagem
    f_image = np.fft.fft2(image)
    fshift = np.fft.fftshift(f_image)

    # Aplica o filtro na frequência
    fshift_filtered = fshift * mask

    # Faz a inversa da transformada de Fourier
    f_image_filtered = np.fft.ifftshift(fshift_filtered)
    image_filtered = np.fft.ifft2(f_image_filtered)
    image_filtered = np.abs(image_filtered)

    return image_filtered


# Carrega a imagem
image = cv2.imread('OIP.jpeg', cv2.IMREAD_GRAYSCALE)

# Aplica o filtro passa-baixa
cutoff_frequency = 30
filtered_image = butterworth_lowpass_filter(image, cutoff_frequency)

# Mostra a imagem original e a imagem filtrada
cv2.imshow('Original', image)
cv2.imshow('Filtrada', filtered_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def butterworth_lowpass_filter(image, cutoff_frequency):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Cria um filtro passa-baixa Butterworth
    mask = np.zeros((rows, cols), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow) ** 2 + (y - ccol) ** 2 <= cutoff_frequency ** 2
    mask[mask_area] = 1

    # Aplica a transformada de Fourier na imagem
    f_image = np.fft.fft2(image)
    fshift = np.fft.fftshift(f_image)

    # Aplica o filtro na frequência
    fshift_filtered = fshift * mask

    # Faz a inversa da transformada de Fourier
    f_image_filtered = np.fft.ifftshift(fshift_filtered)
    image_filtered = np.fft.ifft2(f_image_filtered)
    image_filtered = np.abs(image_filtered)

    return image_filtered


# Abre uma caixa de diálogo para selecionar a imagem
Tk().withdraw()  # Esconde a janela principal do Tkinter
image_path = askopenfilename(title="Selecione uma imagem")

# Verifica se o usuário selecionou uma imagem
if image_path:
    # Carrega a imagem
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Aplica o filtro passa-baixa
    cutoff_frequency = 30
    filtered_image = butterworth_lowpass_filter(image, cutoff_frequency)

    # Mostra a imagem original e a imagem filtrada
    cv2.imshow('Original', image)
    cv2.imshow('Filtrada', filtered_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nenhuma imagem selecionada.")