# Demo Cálculo Avanzado + IA (OpenCV)

## Descripción
Este script usa la cámara web para:
- Detectar tu rostro (OpenCV + Haar Cascade).
- Tomar la posición de la nariz y convertirla en un punto \((x,y)\) cercano a \((2, 1/2)\).
- Evaluar la función:
  \[
  E(x,y) = \cos(\sqrt{xy})
  \]
- Mostrar en tiempo real:
  - \(E(x,y)\)
  - Error respecto a \(E(2, 0.5)\)
  - Cercanía al mínimo teórico \(-1\)
  - Norma del gradiente \(\|\nabla E\|\)

## Parte matemática
- Función de error:
  \[
  E(x,y) = \cos(\sqrt{xy})
  \]
- Punto teórico:
  \[
  (x_0, y_0) = (2, 0.5), \quad E(2,0.5) = \cos(1)
  \]
- Gradiente (implementado en `grad_E(x, y)`):
  - \(\dfrac{\partial E}{\partial x}\) y \(\dfrac{\partial E}{\partial y}\) miden cómo cambia el error al mover \(x\) o \(y\).
  - Cuando \(\|\nabla E\|\) es pequeña, estamos cerca de un punto crítico (posible mínimo).

## Lógica básica del programa
1. Abre la cámara (`cv2.VideoCapture(0)`) y aplica efecto espejo.
2. Detecta la cara y aproxima la nariz como el centro del rectángulo.
3. Convierte el desplazamiento de la nariz a coordenadas \((x,y)\) alrededor de \((2, 0.5)\).
4. Calcula:
   - \(E(x,y)\)
   - Error \(|E - E_{\text{teorico}}|\)
   - Error \(|E - (-1)|\)
   - Gradiente \(\nabla E\) y su norma.
5. Muestra todo en un panel lateral.
6. Si la nariz se acerca lo suficiente al punto rojo, guarda una captura en la carpeta `capturas/`.

## Requisitos
- Python 3.x  
- Librerías:
  ```bash
  pip install opencv-python numpy
