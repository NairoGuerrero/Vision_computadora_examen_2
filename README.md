# 🏷️ Etiquetado y Clasificación de Objetos en Video (Examen 2 - Visión por Computadora)

Este repositorio corresponde al Examen 2 de la asignatura de Visión por Computadora. En este proyecto,
utilizamos máquinas de vectores de soporte (SVM) con OpenCV y scikit-learn para desarrollar un modelo
multiclase que identifica cuál de dos objetos aparece en cada instante de un video demostrativo de al menos
1 minuto de duración a 30 fps.

Cada grupo debe mostrar alternadamente uno de los dos objetos frente a la cámara, asegurando que el objeto
sea claramente visible y luego retirado antes de mostrar el siguiente. El sistema procesa el video, extrae
frames, entrena clasificadores binarios por codificación de bits y genera un video anotado con la etiqueta
correspondiente en la parte superior.

## 📁 Estructura del Proyecto

```
├── entrenamiento.py           # Entrenamiento de clasificadores SVM binarios por codificación de bits
├── etiquetado.py              # Extracción y almacenamiento de frames etiquetados según intervalos de video
├── video_procesado.py         # Anotación de video con predicciones de los clasificadores
├── frames_etiquetados/        # Carpeta de salida con subcarpetas por clase y frames etiquetados
    ├── tennis/
    ├── golf/
    ├── nada/
├── modelo_bit_0.pkl           # Modelo SVM entrenado para el bit 0
├── modelo_bit_1.pkl           # Modelo SVM entrenado para el bit 1
├── video_anotado.mp4          # Video resultante con anotaciones de predicción
└── requirements.txt           # Dependencias necesarias para ejecutar el proyecto
```

## ⚙️ Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/Proyecto_VisionDeportiva.git
   cd Proyecto_VisionDeportiva
   ```

2. Crea y activa un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Uso

### 1. Etiquetado de Frames
Extrae y organiza los frames del video en carpetas según los intervalos definidos:
```bash
python etiquetado.py
```

### 2. Entrenamiento de Clasificadores
Procesa las imágenes etiquetadas, extrae características y entrena los modelos SVM:
```bash
python entrenamiento.py
```

### 3. Anotación de Video
Aplica los clasificadores al video original y genera una versión anotada:
```bash
python main.py
```

## 👥 Autores

- **Nairo Guerrero Márquez** - [nairo.guerrero@utp.edu.co](mailto:nairo.guerrero@utp.edu.co)
- **Juan David Perdomo Quintero** - [juandavid.perdomo@utp.edu.co](mailto:juandavid.perdomo@utp.edu.co)
- **Andres Felipe Zambrano Torres** - [a.zambrano1@utp.edu.co](mailto:a.zambrano1@utp.edu.co)
- **Fabian Esteban Quintero Arias** - [esteban.quintero@utp.edu.co](mailto:esteban.quintero@utp.edu.co)
- **Santiago Rojas Diez** - [santiago.rojas@utp.edu.co](mailto:santiago.rojas@utp.edu.co)
- **Juan Esteban Osorio Montoya** - [juanesteban.osorio@utp.edu.co](mailto:juanesteban.osorio@utp.edu.co)

## 📄 Licencia

Este proyecto está bajo la [MIT License](LICENSE).
