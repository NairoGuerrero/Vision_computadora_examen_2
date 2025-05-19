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