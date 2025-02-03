import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import io
from PIL import Image


# Inicializar la app de Flask
app = Flask(__name__)

# Habilitar CORS en todas las rutas
CORS(app)

# Inicializar MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # Cambiar a True si procesas imágenes estáticas
    max_num_faces=1,  # Número máximo de rostros a detectar
    refine_landmarks=True,  # Refinar la detección de puntos faciales
    min_detection_confidence=0.5,  # Confianza mínima para la detección
    min_tracking_confidence=0.5  # Confianza mínima para el seguimiento
)

# image = cv2.imread('ruta/a/tu/imagen.jpg')
# if image is None:
#     print("Error al cargar la imagen.")
# else:
#     print("Imagen cargada correctamente.")

# import sys


@app.route('/', methods=['GET'])
def index():
    # print(sys.executable)
    return render_template('index.html')



@app.route('/api/detect_face', methods=['POST'])
def detect_faces():
    # Obtener la imagen enviada en el request
    file = request.files['image']

    # Abrir la imagen usando PIL y convertirla a formato de OpenCV
    pil_image = Image.open(file)
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Procesar la imagen para obtener los puntos faciales
    landmarks = process_image(image)

    # Devolver los puntos faciales como JSON
    if landmarks:
        return jsonify({'landmarks': landmarks}), 200
    else:
        return jsonify({'error': 'No face landmarks detected'}), 400



# Función para procesar la imagen y obtener los puntos faciales
def process_image(image):
    # Convertir la imagen de BGR a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pasar la imagen al modelo de MediaPipe
    results = face_mesh.process(image_rgb)

    # Si se detectan puntos faciales, devolver sus coordenadas
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i, landmark in enumerate(face_landmarks.landmark):
                # Etiquetamos algunos puntos clave
                if i == 1:  # Puntos de la nariz
                    landmark_name = 'Nose'
                elif 33 <= i <= 133:  # Ojo izquierdo
                    landmark_name = 'Left Eye'
                elif 362 <= i <= 463:  # Ojo derecho
                    landmark_name = 'Right Eye'
                elif 61 <= i <= 91:  # Boca
                    landmark_name = 'Mouth'
                elif i <= 16:  # Mandíbula
                    landmark_name = 'Jaw'
                else:
                    landmark_name = f'Point {i}'

                landmarks.append({
                    'index': i,
                    'name': landmark_name,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })

    return landmarks





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3030, debug=True)
