# Demo Cálculo Avanzado + IA con OpenCV

Este proyecto muestra una integración sencilla entre **Cálculo Avanzado** y **Visión por Computadora** usando Python y OpenCV.

La idea principal es:
- Detectar tu rostro con la cámara.
- Tomar la posición de la nariz y mapearla a un punto \((x,y)\) cercano a \((2,\tfrac{1}{2})\).
- Evaluar una función de error \(E(x,y)\) típica de Cálculo Avanzado.
- Mostrar en tiempo real qué tan cerca estás del valor esperado y del mínimo teórico, además de información del gradiente.

---

## 1. Función Matemática que se Usa

La función de error que se trabaja en el código es:

\[
E(x,y) = \cos\big(\sqrt{xy}\big)
\]

- El punto de partida del ejercicio es:
  \[
  (x_0, y_0) = \left(2, \frac{1}{2}\right)
  \]
- En el código se calcula:
  ```python
  E_teorico = E(2.0, 0.5)   # cos(1)
