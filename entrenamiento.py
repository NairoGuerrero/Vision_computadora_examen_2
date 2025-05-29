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

    def __init__(self, resize_shape=(64, 64)):
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

class BinaryBitClassifier:
    """Clase encargada del entrenamiento de clasificadores por bit binario."""

    def __init__(self, frames_dir='frames_etiquetados', k=10):
        self.frames_dir = frames_dir
        self.k = k
        self.feature_extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.bit_code = {
            0: [1, 0],  # golf
            1: [0, 1],  # nada
            2: [0, 0]   # tennis
        }
        self.modelos = []

    def cargar_datos(self):
        """Carga imágenes, extrae características y codifica etiquetas."""
        X, y = [], []
        for clase in os.listdir(self.frames_dir):
            clase_path = os.path.join(self.frames_dir, clase)
            if not os.path.isdir(clase_path):
                continue
            for file in os.listdir(clase_path):
                if file.endswith('.jpg'):
                    img_path = os.path.join(clase_path, file)
                    lbp = self.feature_extractor.extraer_lbp(img_path)
                    color = self.feature_extractor.extraer_histograma_color(img_path)
                    features = np.concatenate([lbp, color])
                    X.append(features)
                    y.append(clase)

        X = np.array(X)
        y_encoded = self.label_encoder.fit_transform(y)
        Y_bits = np.array([self.bit_code[i] for i in y_encoded])
        return X, y_encoded, Y_bits

    def entrenar_modelos(self, X, y_encoded, Y_bits):
        """Entrena modelos SVM por cada bit de la codificación."""
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)

        for bit_idx in range(2):
            print(f"\n=== Clasificador para bit {bit_idx} ===")
            y_bit = Y_bits[:, bit_idx]
            accs = []

            # Entrenamiento con validación cruzada
            for train_idx, test_idx in skf.split(X, y_encoded):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_bit[train_idx], y_bit[test_idx]

                clf = SVC(kernel='linear')
                clf.fit(X_train, y_train)
                acc = clf.score(X_test, y_test)
                accs.append(acc)

            print(f"📊 Bit {bit_idx} - Accuracy promedio: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

            # Entrenamiento final con todo el conjunto
            final_model = SVC(kernel='linear')
            final_model.fit(X, y_bit)
            self.modelos.append(final_model)
            joblib.dump(final_model, f"modelo_bit_{bit_idx}.pkl")

    def evaluar_modelos(self, X, y_encoded):
        """Evalúa el desempeño global del sistema usando los dos clasificadores."""
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)
        global_y_true, global_y_pred = [], []

        for train_idx, test_idx in skf.split(X, y_encoded):
            X_test = X[test_idx]
            true_classes = y_encoded[test_idx]

            pred_bit0 = self.modelos[0].predict(X_test)
            pred_bit1 = self.modelos[1].predict(X_test)
            pred_bits = np.vstack((pred_bit0, pred_bit1)).T

            pred_classes = []
            for bits in pred_bits:
                for class_idx, code in self.bit_code.items():
                    if list(bits) == code:
                        pred_classes.append(class_idx)
                        break
                else:
                    pred_classes.append(-1)

            global_y_true.extend(true_classes)
            global_y_pred.extend(pred_classes)

        # Filtrado de errores de decodificación
        y_true_filtrado, y_pred_filtrado = [], []
        for yt, yp in zip(global_y_true, global_y_pred):
            if yp != -1:
                y_true_filtrado.append(yt)
                y_pred_filtrado.append(yp)

        print("\n=== Evaluación Global ===")
        print(classification_report(y_true_filtrado, y_pred_filtrado, target_names=self.label_encoder.classes_))

if __name__ == "__main__":
    clasificador = BinaryBitClassifier()
    X, y_encoded, Y_bits = clasificador.cargar_datos()
    clasificador.entrenar_modelos(X, y_encoded, Y_bits)
    clasificador.evaluar_modelos(X, y_encoded)
