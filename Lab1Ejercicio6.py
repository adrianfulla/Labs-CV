"""
Ejercicio 6:

Implementar un detector de color YELLOW que funciones en tiempo real en un video. Para ello, implementar el algoritmo en
OpenCV, de modo que capture las imágenes directamente de la cámara de su PC, y que muestre el resultado de la detección
en tiempo real en pantalla.

"""
import cv2
import numpy as np

# Definir los rangos de color para el amarillo en HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])


# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame de BGR a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crear una máscara para el color amarillo
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar un rectángulo verde alrededor de los contornos detectados
    for contour in contours:
        # Obtener el rectángulo delimitador del contorno
        x, y, w, h = cv2.boundingRect(contour)
        
        # Dibujar el rectángulo en el frame original
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde: (0, 255, 0), grosor: 2

    # Mostrar el frame original con los rectángulos verdes
    cv2.imshow('Original', frame)

    # Mostrar la máscara de detección de amarillo
    cv2.imshow('Máscara del amarillo', mask)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()