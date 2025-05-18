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

    def extraer_histograma_color(self, img):
        """Extrae histograma de color desde imagen RGB."""
        img = cv2.resize(img, self.resize_shape)
        chans = cv2.split(img)
        features = []
        for chan in chans:
            hist = cv2.calcHist([chan], [0], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        return np.array(features)

    def decodificar_bits(self, bits):
        """Decodifica una lista de bits en una clase."""
        for clase_idx, code in self.bit_code.items():
            if bits == code:
                return self.label_encoder.inverse_transform([clase_idx])[0]
        return "desconocido"

    def procesar_video(self, salida_path='video_anotado.mp4'):
        """Procesa el video y genera una salida anotada."""
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("❌ No se pudo abrir el video.")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo leer el primer frame.")
            cap.release()
            return

        # Rotar y definir dimensiones del frame de salida
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        rotated_height, rotated_width = frame.shape[:2]

        print(f"🎥 FPS: {fps}, Ancho: {rotated_width}, Alto: {rotated_height}")

        # Configurar el video de salida
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(salida_path, fourcc, fps, (rotated_width, rotated_height))

        if not out.isOpened():
            print("❌ No se pudo crear el archivo de salida.")
            cap.release()
            return

        while True:
            # Extracción de características
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lbp = self.extraer_lbp(gray)
            color = self.extraer_histograma_color(frame)
            features = np.concatenate([lbp, color]).reshape(1, -1)

            # Clasificación binaria por bit
            pred_bits = [modelo.predict(features)[0] for modelo in self.modelos]
            clase_predicha = self.decodificar_bits(pred_bits)

            # Anotar frame
            cv2.putText(frame, f"Prediccion: {clase_predicha}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow("Prediccion", frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Finalizar
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Video anotado guardado como '{salida_path}'")