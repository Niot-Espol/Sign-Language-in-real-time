# Protocolo común de captura

## Antes de grabar

- Confirmar en `config/catalogo.json` la etiqueta, tipo y cantidad de manos.
- Comparar la postura con `docs/referencias/<slug>.jpg`.
- Usar fondo sin distracciones, luz frontal y manos completas dentro del cuadro.
- Registrar al menos tres sesiones separadas por palabra, en vez de una única ráfaga idéntica.

## Durante la grabación

- Esperar el indicador verde `ESTABLE`.
- Mantener Espacio para capturar frames estáticos.
- Para señas dinámicas, usar el recolector de secuencias; cada repetición tiene cuenta regresiva y 30 frames.
- Pulsar `E` para guardar una evidencia del visor.
- Variar moderadamente distancia, orientación y posición entre sesiones, sin cambiar la seña.

## Control de calidad

- Cada CSV válido tiene 127 columnas: 126 valores y `label`.
- Los CSV de 63 valores son legado y no pueden mezclarse con el esquema actual.
- Los respaldos anteriores se conservan en `data/legacy_63/`; nunca se consolidan automáticamente.
- El consolidador rechaza nombres, etiquetas, columnas y valores inválidos.
- La evaluación separa integrantes para evitar que frames casi idénticos aparezcan en entrenamiento y prueba.
