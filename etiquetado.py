"""
Módulo de etiquetado de frames a partir de un video segmentado.

Este script extrae frames de un video dado y los guarda en carpetas
según su clase, basándose en intervalos de tiempo definidos.

"""

import cv2
import os

class VideoLabeler:
    """Clase que gestiona la extracción y etiquetado de frames desde un video."""

    def _init_(self, video_path, output_folder, fps, intervalos):
        self.video_path = video_path
        self.output_folder = output_folder
        self.fps = fps
        self.intervalos = intervalos
        self.intervalos_frame = [
            (int(start * fps), int(end * fps), label)
            for start, end, label in intervalos
        ]

    def crear_carpetas_clases(self):
        """Crea directorios de salida para cada clase."""
        clases = list(set([label for _, _, label in self.intervalos]))
        for clase in clases:
            os.makedirs(os.path.join(self.output_folder, clase), exist_ok=True)

    def obtener_etiqueta_para_frame(self, frame_idx):
        """Devuelve la etiqueta correspondiente a un frame según los intervalos definidos."""
        for inicio, fin, label in self.intervalos_frame:
            if inicio <= frame_idx < fin:
                return label
        return 'desconocido'
        
    def etiquetar_video(self):
        """Procesa el video y guarda los frames en carpetas según su etiqueta."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("❌ No se pudo abrir el video.")
            return

        self.crear_carpetas_clases()

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            etiqueta = self.obtener_etiqueta_para_frame(frame_idx)

            filename = f'frame_{frame_idx:04d}.jpg'
            path_guardado = os.path.join(self.output_folder, etiqueta, filename)
            cv2.imwrite(path_guardado, frame)

            frame_idx += 1

        cap.release()
        print("✅ Proceso de etiquetado completado con éxito.")

if __name__ == "__main__":
    # Definición de parámetros y ejecución
    intervalos = [
        (0, 10, 'golf'),
        (10, 21, 'nada'),
        (21, 32, 'tennis'),
        (32, 41, 'nada'),
        (41, 52, 'golf'),
        (52, 60, 'nada')
    ]

    labeler = VideoLabeler(
        video_path='Video1.mp4',
        output_folder='frames_etiquetados',
        fps=30,
        intervalos=intervalos
    )
    labeler.etiquetar_video()