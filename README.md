# Demo Cálculo Avanzado + IA (OpenCV)

## Descripción
Este script usa la cámara web para:
- Detectar tu rostro (OpenCV + Haar Cascade).
- Tomar la posición de la nariz y convertirla en un punto (x, y) cercano a (2, 0.5).
- Evaluar la función de error:
  𝑬(𝒙,𝒚)=𝒄𝒐𝒔(√𝒙𝒚) 𝒕𝒐𝒎𝒂𝒏𝒅𝒐 𝒄𝒐𝒎𝒐 𝒑𝒖𝒏𝒕𝒐 𝒅𝒆 𝒑𝒂𝒓𝒕𝒊𝒅𝒂 (𝟐, 𝟏 𝟐) 
- Mostrar en tiempo real:
  - E(x, y)
  - Error respecto a E(2, 0.5)
  - Cercanía al mínimo teórico -1
  - Norma del gradiente ||∇E||

## Parte matemática
- Función de error:
  E(x, y) = cos( sqrt(x * y) )

- Punto teórico:
  (x0, y0) = (2, 0.5)
  E(2, 0.5) = cos(1)

- Gradiente (implementado en la función `grad_E(x, y)`):
  - dE/dx y dE/dy miden cómo cambia el error al mover x o y.
  - Cuando ||∇E|| es pequeña, estamos cerca de un punto crítico (posible mínimo).

## Lógica básica del programa
1. Abre la cámara (cv2.VideoCapture(0)) y aplica efecto espejo.
2. Detecta la cara y aproxima la nariz como el centro del rectángulo.
3. Convierte el desplazamiento de la nariz a coordenadas (x, y) alrededor de (2, 0.5).
4. Calcula:
   - E(x, y)
   - Error |E(x, y) - E_teorico|
   - Error |E(x, y) - (-1)|
   - Gradiente ∇E y su norma ||∇E||.
5. Muestra todo en un panel lateral (valores, errores, porcentajes de cercanía).
6. Si la nariz se acerca lo suficiente al punto rojo, guarda una captura en la carpeta `capturas/`.

## Requisitos
- Python 3.x  
- Librerías:
  ```bash
  pip install opencv-python numpy
