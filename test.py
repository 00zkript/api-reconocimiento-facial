import cv2
import mediapipe as mp

def detect_face_landmarks_image_mediapipe(image_path):
    # Inicializar módulos de MediaPipe
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    # Leer la imagen y convertirla a RGB
    image = cv2.imread(image_path)
    if image is None:
        print("No se pudo cargar la imagen:", image_path)
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Usar MediaPipe para detección y landmarks en modo imagen estática
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection, \
         mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

        # Procesar la imagen para detectar rostros
        results_detection = face_detection.process(image_rgb)
        if results_detection.detections:
            for detection in results_detection.detections:
                # Dibujar el rectángulo y la confianza sobre la imagen
                mp_drawing.draw_detection(image, detection)
        else:
            print("No se detectaron rostros.")

        # Procesar la imagen para obtener los landmarks faciales
        results_mesh = face_mesh.process(image_rgb)
        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
                )
        else:
            print("No se detectaron landmarks faciales.")

    # Mostrar la imagen resultante
    cv2.imshow("Rostro y Landmarks (MediaPipe)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_face_landmarks_video_mediapipe():
    # Inicializar módulos de MediaPipe
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    # Iniciar la captura de video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la cámara.")
        return

    # Usar MediaPipe en modo video (procesamiento en tiempo real)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el frame.")
                break

            # Convertir el frame a RGB para MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detección de rostros
            results_detection = face_detection.process(frame_rgb)
            if results_detection.detections:
                for detection in results_detection.detections:
                    mp_drawing.draw_detection(frame, detection)

            # Detección de landmarks faciales
            results_mesh = face_mesh.process(frame_rgb)
            if results_mesh.multi_face_landmarks:
                for face_landmarks in results_mesh.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
                    )

            cv2.imshow("Rostro y Landmarks en Video (MediaPipe)", frame)

            # Salir con la tecla Esc (ASCII 27)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Para procesar una imagen:
    image_path = "718ab0e7537f97ef56e5fdba8afc6327.jpg"  # Cambia por la ruta de tu imagen
    # detect_face_landmarks_image_mediapipe(image_path)

    # Para procesar video, descomenta la siguiente línea:
    detect_face_landmarks_video_mediapipe()
