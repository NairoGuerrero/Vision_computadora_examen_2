"""
Módulo de procesamiento y anotación de video usando clasificadores entrenados por bits.

Este script procesa un video frame por frame, predice la clase de acción
basándose en clasificadores binarios SVM y genera un nuevo video anotado.
"""

import cv2
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern

class VideoAnnotator:
    """Clase para anotar video usando clasificadores binarios."""

    def __init__(self, video_path, modelos_path_prefix='modelo_bit_', resize_shape=(64, 64)):
        self.video_path = video_path
        self.resize_shape = resize_shape
        self.modelos = [joblib.load(f'{modelos_path_prefix}{i}.pkl') for i in range(2)]
        self.bit_code = {
            0: [1, 0],  # golf
            1: [0, 1],  # nada
            2: [0, 0]   # tennis
        }
        self.orden_clases = ['golf', 'nada', 'tennis']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.orden_clases)

    def extraer_lbp(self, img_gray):
        """Extrae histograma LBP desde imagen en escala de grises."""
        img = cv2.resize(img_gray, self.resize_shape)
        lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist