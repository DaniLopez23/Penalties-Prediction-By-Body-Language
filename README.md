# Penalties Prediction By Body Language

Proyecto de vision por computador para analizar videos de penaltis a partir de deteccion de objetos, tracking y estimacion de pose. El sistema procesa cada frame del video, identifica los elementos principales de la escena, asigna los roles de lanzador y portero, sigue el balon y calcula metricas simples de lenguaje corporal y resultado del disparo.

El repositorio incluye un pipeline ejecutable por script y una interfaz en Streamlit para subir videos, generar un video anotado y visualizar KPIs como zona de porteria, direccion del portero, segundo de disparo y evolucion de angulos corporales.

## Pipeline y Flujo

El flujo principal esta implementado en `src/pipeline.py` y se ejecuta desde `run_pipeline.py` o desde la app de Streamlit:

1. Se carga el video de entrada con OpenCV.
2. Se detecta la porteria en el frame.
3. Se construye una ROI del area util de juego, desde la porteria hacia la zona del lanzador.
4. Se aplica YOLO sobre la ROI para detectar personas y balon.
5. Se filtran detecciones fuera de la ROI.
6. Se actualiza el tracking del balon combinando YOLO, movimiento entre frames y prediccion.
7. Se asignan roles a las personas detectadas: lanzador y portero.
8. Se estima la pose con un modelo YOLO pose y se asocia cada pose con su rol.
9. Se calculan metricas corporales y estado del penalti.
10. Se dibujan anotaciones sobre el frame y se guarda el video procesado.

La salida por defecto se guarda en `data/output_videos/` con el sufijo `_annotated`.

## Detectores y Tracking

### Jugadores: lanzador y portero

La deteccion de jugadores usa modelos YOLO de Ultralytics sobre la clase `person` de COCO. El detector trabaja con `model.track(...)`, por lo que tambien recibe IDs de tracking cuando el tracker configurado puede asignarlos. Por defecto se usa `botsort.yaml`.

La asignacion de roles se realiza en `src/tracking/roles.py` mediante scoring heuristico:

- El portero se prioriza por cercania a la porteria, alineacion con el centro de la porteria y posicion vertical dentro de la escena.
- El lanzador se prioriza por estar mas abajo en la imagen, cerca del centro, con mayor tamano relativo y cerca del balon.
- Cuando existen IDs de tracking, el sistema bloquea el ID asignado a cada rol y lo reutiliza mientras siga siendo plausible.
- Si un rol se pierde durante varios frames, se permite reasignarlo usando las puntuaciones anteriores.

### Porteria

La porteria se detecta en `src/detectors/goal.py` con tecnicas clasicas de vision:

- Conversion a HSV.
- Mascara de blancos usando baja saturacion y alto valor.
- Operaciones morfologicas de erosion, dilatacion y median blur.
- Extraccion de contornos.
- Seleccion del mejor contorno por area, posicion central y proporcion esperada de la porteria.
- Suavizado temporal para evitar saltos entre frames.
- Uso de la ultima porteria valida durante algunos frames si se pierde la deteccion.
- Fallback proporcional al tamano del frame cuando no se detecta una porteria fiable.

La porteria tambien se divide en tres zonas horizontales: izquierda, centro y derecha. Esta division se usa para estimar la zona del disparo cuando el balon entra en el area de la porteria.

### Balon

El tracking del balon esta en `src/tracking/ball.py` y combina varias fuentes:

- Detecciones YOLO de la clase `sports ball`.
- Candidatos por movimiento calculados con diferencia entre frames, umbralizacion, apertura morfologica y contornos.
- Filtro de Kalman para predecir la siguiente posicion cuando el balon no se observa claramente.
- Penalizaciones para candidatos solapados con jugadores o zonas de pies.
- Restricciones de salto maximo, tamano, circularidad y pertenencia a la ROI.
- Trail temporal para visualizar la trayectoria reciente.

El estado del disparo se calcula en `src/analysis/penalty.py` observando la velocidad del balon entre frames. Cuando la velocidad supera un umbral durante varios frames, el estado pasa de `pre-shot` a `shot`.

### Pose

La pose se estima en `src/detectors/yolo.py` con un modelo YOLO pose de Ultralytics. Las poses detectadas se asocian a lanzador y portero en `src/analysis/pose.py` comparando la caja de la pose con la caja del jugador mediante IoU y distancia entre centros.

A partir de los keypoints COCO se calculan metricas como:

- Angulo de hombros.
- Angulo de caderas.
- Inclinacion corporal.
- Angulo entre brazo y tronco.

Estas metricas se usan para anotar el video y estimar, de forma simple, la direccion del portero segun su inclinacion corporal.

## Ejecucion

Antes de ejecutar, coloca los videos en `data/` o subelos desde la app Streamlit. El archivo `run_pipeline.py` incluye parametros editables como video de entrada, modelos, dispositivo, tamano de imagen, confianzas y numero maximo de frames.

### Opcion 1: Conda

```bash
conda create -n penalties-body-language python=3.10 -y
conda activate penalties-body-language
pip install -r requirements.txt
```

Ejecutar el pipeline por script:

```bash
python run_pipeline.py
```

Ejecutar la interfaz Streamlit:

```bash
streamlit run streamlit_app.py
```

### Opcion 2: venv

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ejecutar el pipeline por script:

```bash
python run_pipeline.py
```

Ejecutar la interfaz Streamlit:

```bash
streamlit run streamlit_app.py
```

## Modelos y Dependencias

Las dependencias estan definidas en `requirements.txt`:

- `opencv-python`: lectura, escritura y procesamiento de video.
- `ultralytics`: deteccion YOLO y estimacion de pose.
- `numpy` y `pandas`: calculo numerico y manejo de historiales.
- `streamlit`: interfaz web local.
- `imageio-ffmpeg`: soporte para conversion de video reproducible en navegador.
- `mediapipe`, `supervision` y `lap`: dependencias de apoyo para vision y tracking.

El repositorio incluye varios pesos YOLO (`.pt`) en la raiz. Los modelos activos se configuran en `run_pipeline.py` mediante `DETECTOR_MODEL` y `POSE_MODEL`.
