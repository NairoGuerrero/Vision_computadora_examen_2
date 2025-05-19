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
