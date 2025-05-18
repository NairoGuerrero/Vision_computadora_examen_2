"""
Módulo de entrenamiento de clasificadores binarios por codificación de bits
para la clasificación de acciones deportivas en video.

Este script procesa un conjunto de imágenes etiquetadas, extrae características (LBP y color),
aplica codificación binaria para múltiples clases y entrena clasificadores SVM por bit.


"""

import os
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern
import joblib

class FeatureExtractor:
    """Clase para extracción de características desde imágenes."""

    def _init_(self, resize_shape=(64, 64)):
        self.resize_shape = resize_shape

    def extraer_lbp(self, path):
        """Extrae histograma LBP de una imagen en escala de grises."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.resize_shape)
        lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist

    def extraer_histograma_color(self, path):
        """Extrae histograma de color de una imagen RGB."""
        img = cv2.imread(path)
        img = cv2.resize(img, self.resize_shape)
        chans = cv2.split(img)
        features = []
        for chan in chans:
            hist = cv2.calcHist([chan], [0], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        return np.array(features)