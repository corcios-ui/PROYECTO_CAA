import cv2
import math
import os
import numpy as np
from datetime import datetime

# --------- Función de error ---------
def E(x, y):
    """E(x,y) = cos(sqrt(xy))."""
    if x * y <= 0:
        return None
    return math.cos(math.sqrt(x * y))

def grad_E(x, y):
    """Gradiente de E(x,y) = cos(sqrt(xy)). Devuelve (Ex, Ey)."""
    if x * y <= 0:
        return None, None
    raiz = math.sqrt(x * y)
    if raiz == 0:
        return None, None
    sen_term = math.sin(raiz)
    Ex = - (y * sen_term) / (2.0 * raiz)
    Ey = - (x * sen_term) / (2.0 * raiz)
    return Ex, Ey


# --------- Parámetros teóricos del ejercicio ---------
x0 = 2.0
y0 = 0.5
E_teorico = E(x0, y0)   # ≈ cos(1) ≈ 0.5403

# Mínimo teórico de la función de error (cos entre -1 y 1)
E_MINIMO = -1.0

# Error máximo posible entre dos valores de coseno (de -1 a 1)
ERROR_MAX = 2.0  # lo usamos para normalizar porcentajes

# Gradiente máximo de referencia: usamos el del punto inicial
Ex0, Ey0 = grad_E(x0, y0)
if Ex0 is not None and Ey0 is not None:
    GRAD_BASE = math.sqrt(Ex0**2 + Ey0**2)
else:
    GRAD_BASE = 1.0  # valor de respaldo para evitar división entre 0

# FACTOR INTERACTIVO de sensibilidad del gradiente (lo ajustamos con teclas)
grad_sensibilidad = 1.0  # 1.0x al inicio


# --------- Configuración de carpeta de salida ---------
OUTPUT_DIR = "capturas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------- Clasificador de caras de OpenCV ---------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------- Cámara ---------
cap = cv2.VideoCapture(0)

# Escala: cuántos píxeles equivalen a 1 unidad en x,y
PIXELS_POR_UNIDAD = 300.0  # ajusta según qué tan sensible lo quieras

# Umbrales de distancia para captura (en píxeles)
UMBRAL_CAPTURA = 25        # si la nariz está a menos de 25 px del punto rojo -> captura
UMBRAL_RESET = 60          # si se aleja más de esto, se habilita nueva captura

captura_realizada = False  # para no guardar muchas capturas seguidas

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- mirror / efecto espejo ---
    frame = cv2.flip(frame, 1)


    # Tamaño de la imagen original
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2   # centro (punto rojo)

    # Dibujamos el punto rojo objetivo (más grande)
    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)

    # Convertimos a gris para detección de cara
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectamos caras
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)  # para evitar caras muy pequeñas
    )

    nose_x, nose_y = None, None
    x = y = valor_E = None

    # errores / métricas
    error_teorico = None          # |E - E_teorico|
    error_minimo = None           # |E - (-1)|
    dist_pix = None
    cercania_pct = None           # cercanía a E_teorico
    cercania_min_pct = None       # cercanía a E_minimo = -1
    grad_norm = None              # ||grad E||
    grad_cercania_pct = None      # cercanía a gradiente cero

    if len(faces) > 0:
        # Tomamos la cara más grande (la que está más cerca)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (fx, fy, fw, fh) = faces[0]

        # Dibujamos recuadro de la cara
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

        # Aproximamos la nariz como el centro del rostro
        nose_x = fx + fw // 2
        nose_y = fy + fh // 2

        # Marcamos la "nariz" con un punto azul
        cv2.circle(frame, (nose_x, nose_y), 8, (255, 0, 0), -1)

        # Distancia desde la "nariz" al centro (en píxeles)
        dx_pix = nose_x - cx
        dy_pix = nose_y - cy
        dist_pix = math.sqrt(dx_pix**2 + dy_pix**2)

        # Píxeles -> unidades del plano (x,y)
        dx = dx_pix / PIXELS_POR_UNIDAD
        dy = dy_pix / PIXELS_POR_UNIDAD

        # Definimos x,y alrededor de (2, 0.5)
        x = x0 + dx
        y = y0 + dy

        valor_E = E(x, y)
        if valor_E is not None:
            # --------- Error vs punto teórico del ejercicio ---------
            error_teorico = abs(valor_E - E_teorico)

            # Cercanía normalizada respecto a E_teorico
            diff_norm = min(error_teorico, ERROR_MAX) / ERROR_MAX
            cercania_pct = max(0.0, 100.0 * (1.0 - diff_norm))

            # --------- Error y cercanía respecto al mínimo -1 ---------
            error_minimo = abs(valor_E - E_MINIMO)  # |E - (-1)|

            diff_norm_min = min(error_minimo, ERROR_MAX) / ERROR_MAX
            cercania_min_pct = max(0.0, 100.0 * (1.0 - diff_norm_min))
            # cercania_min_pct = 100% cuando E ≈ -1

            # --------- Gradiente y cercanía a gradiente cero ---------
            Ex, Ey = grad_E(x, y)
            if Ex is not None and Ey is not None:
                grad_norm = math.sqrt(Ex**2 + Ey**2)

                # NORMALIZACIÓN INTERACTIVA:
                # usamos un gradiente máximo efectivo = GRAD_BASE * grad_sensibilidad
                grad_max_efectivo = GRAD_BASE * grad_sensibilidad
                if grad_max_efectivo <= 0:
                    grad_max_efectivo = 1.0

                grad_norm_clipped = min(grad_norm, grad_max_efectivo)
                grad_cercania_pct = max(
                    0.0,
                    100.0 * (1.0 - grad_norm_clipped / grad_max_efectivo)
                )
                # grad_cercania_pct = 100% cuando ||grad E|| ~ 0

            # ----- Lógica de captura cuando está muy cerca del punto rojo -----
            if dist_pix <= UMBRAL_CAPTURA and not captura_realizada:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captura_{ts}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)

                # Hacemos una copia del frame para escribir los datos encima
                frame_to_save = frame.copy()

                # Escribimos datos del "ejercicio" dentro de la imagen
                y_text_save = 40
                dy_text_save = 30

                cv2.putText(frame_to_save, "Resultados ejercicio E(x,y)=cos(raiz(xy))",
                            (10, y_text_save), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
                y_text_save += dy_text_save

                cv2.putText(frame_to_save,
                            "Punto de partida (2,1/2), E(2,0.5)=cos(1)",
                            (10, y_text_save), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)
                y_text_save += dy_text_save * 2

                cv2.putText(frame_to_save, f"x calculado = {x:.6f}",
                            (10, y_text_save), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                y_text_save += dy_text_save

                cv2.putText(frame_to_save, f"y calculado = {y:.6f}",
                            (10, y_text_save), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                y_text_save += dy_text_save

                cv2.putText(frame_to_save, f"E(x,y) = {valor_E:.6f}",
                            (10, y_text_save), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                y_text_save += dy_text_save

                cv2.putText(frame_to_save,
                            f"E teorica = E(2,0.5) = {E_teorico:.6f}",
                            (10, y_text_save), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)
                y_text_save += dy_text_save

                cv2.putText(frame_to_save,
                            f"Error |E - E_teor| = {error_teorico:.6f}",
                            (10, y_text_save), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 200, 255), 2)
                y_text_save += dy_text_save

                cv2.putText(frame_to_save,
                            f"Cercania a E_teor = {cercania_pct:.2f}%",
                            (10, y_text_save), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 200, 255), 2)
                y_text_save += dy_text_save

                # Info respecto al error mínimo -1
                cv2.putText(frame_to_save, f"E minimo teorico = {E_MINIMO:.1f}",
                            (10, y_text_save), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)
                y_text_save += dy_text_save

                cv2.putText(frame_to_save,
                            f"|E - (-1)| = {error_minimo:.6f}",
                            (10, y_text_save), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 150, 255), 2)
                y_text_save += dy_text_save

                if cercania_min_pct is not None:
                    cv2.putText(frame_to_save,
                                f"Cercania al minimo = {cercania_min_pct:.2f}%",
                                (10, y_text_save), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 150, 255), 2)
                    y_text_save += dy_text_save

                # Info de gradiente
                if grad_norm is not None:
                    cv2.putText(frame_to_save,
                                f"||grad E|| = {grad_norm:.6f}",
                                (10, y_text_save),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (200, 200, 255), 2)
                    y_text_save += dy_text_save

                if grad_cercania_pct is not None:
                    cv2.putText(frame_to_save,
                                f"Cercania a grad 0 = {grad_cercania_pct:.2f}%",
                                (10, y_text_save),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (200, 200, 255), 2)
                    y_text_save += dy_text_save

                # Sensibilidad usada
                cv2.putText(frame_to_save,
                            f"Sens grad = {grad_sensibilidad:.2f}x",
                            (10, y_text_save),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 2)
                y_text_save += dy_text_save

                # Guardamos la imagen anotada
                cv2.imwrite(filepath, frame_to_save)

                captura_realizada = True

                # Mensaje visual de captura en la ventana en vivo
                cv2.putText(frame, "CAPTURA GUARDADA",
                            (int(w*0.25), int(h*0.5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 255), 3)

        # Reset de captura cuando te alejas otra vez
        if dist_pix is not None and dist_pix > UMBRAL_RESET:
            captura_realizada = False

    # --------- Panel lateral derecho (interfaz en vivo) ---------
    panel_width = 380
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

    y_text = 40
    dy_text = 28

    cv2.putText(panel, "Calculo + IA", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y_text += dy_text

    cv2.putText(panel, "E(x,y)=cos(raiz(xy)) punto (2,1/2)", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    y_text += dy_text * 2

    if x is not None and y is not None:
        cv2.putText(panel, f"x (estimado) = {x:.4f}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_text += dy_text
        cv2.putText(panel, f"y (estimado) = {y:.4f}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_text += dy_text

    if valor_E is not None:
        cv2.putText(panel, f"E(x,y) = {valor_E:.4f}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_text += dy_text

    cv2.putText(panel, f"E(2,0.5) = {E_teorico:.4f}", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_text += dy_text

    cv2.putText(panel, f"E minimo teorico = {E_MINIMO:.1f}", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_text += dy_text

    if error_teorico is not None:
        cv2.putText(panel, f"|E - E_teor| = {error_teorico:.4f}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y_text += dy_text

    if cercania_pct is not None:
        cv2.putText(panel, f"Cercania a E_teor: {cercania_pct:.1f}%",
                    (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 200, 255), 2)
        y_text += dy_text


    if grad_norm is not None:
        cv2.putText(panel, f"||grad E|| = -{grad_norm:.4f}",
                    (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 255), 2)
        y_text += dy_text

    if grad_cercania_pct is not None:
        cv2.putText(panel, f"Cercania a grad 0: {grad_norm*100:.1f}%",
                    (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 255), 2)
        y_text += dy_text

    if dist_pix is not None:
        cv2.putText(panel, f"Dist. nariz-punto = {dist_pix:.1f}px", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y_text += dy_text

    y_text += dy_text
    cv2.putText(panel, f"Umbral captura: {UMBRAL_CAPTURA}px", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_text += dy_text
    cv2.putText(panel, "Acerca la nariz al punto rojo", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_text += dy_text


    # Mensaje abajo en el frame principal
    cv2.putText(frame, "ESC: salir",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    # Unimos frame + panel lateral
    frame_panel = np.hstack((frame, panel))

    cv2.imshow("Demo Calculo + IA (cara + E(x,y))", frame_panel)

    # ------------ Manejo de teclas (interactividad) ------------
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('k') or key == ord('K'):
        grad_sensibilidad *= 1.2   # aumenta sensibilidad
    elif key == ord('j') or key == ord('J'):
        grad_sensibilidad /= 1.2   # disminuye sensibilidad
        if grad_sensibilidad < 0.1:
            grad_sensibilidad = 0.1

cap.release()
cv2.destroyAllWindows()
