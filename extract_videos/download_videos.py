import json
import subprocess
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Calidad/FPS de salida de clips (puedes ajustar estos valores según tu hardware)
TARGET_FPS = 60
VIDEO_CODEC = "libx264"
ENCODE_PRESET = "slow"
CRF = 18

# Directorios
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
JSON_FILE = SCRIPT_DIR / "videos.json"
TEMP_VIDEOS_DIR = SCRIPT_DIR / ".temp_videos"
CLIPS_DIR = PROJECT_ROOT / "data"

# Crear directorios si no existen
TEMP_VIDEOS_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True)


def descargar_video(url, video_id):
    """
    Descarga un video de YouTube usando yt-dlp a un archivo temporal.
    
    Args:
        url (str): URL del video de YouTube
        video_id (str): ID único para el video
        
    Returns:
        str: Ruta del archivo descargado o None si falló
    """
    output_path = TEMP_VIDEOS_DIR / f"{video_id}.%(ext)s"
    
    try:
        logger.info(f"Descargando video: {url}")
        comando = [
            "yt-dlp",
            "--no-playlist",
            "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            url
        ]
        
        subprocess.run(comando, check=True, capture_output=True, text=True)

        # Determinar el archivo final descargado por yt-dlp
        candidates = sorted(TEMP_VIDEOS_DIR.glob(f"{video_id}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            logger.error(f"No se encontró archivo descargado para {video_id}")
            return None

        downloaded_path = candidates[0]
        logger.info(f"Video descargado exitosamente en temporal: {downloaded_path.name}")
        return str(downloaded_path)
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error descargando video {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error inesperado descargando {url}: {e}")
        return None


def obtener_duracion_video(video_path):
    """
    Obtiene la duración de un video usando ffmpeg.
    
    Args:
        video_path (str): Ruta del archivo de video
        
    Returns:
        float: Duración en segundos o None si falló
    """
    try:
        # Usar ffprobe para obtener la duración
        ffprobe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1:nokey=1",
            video_path
        ]
        
        resultado = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
        duracion = float(resultado.stdout.strip())
        return duracion
    except Exception as e:
        logger.warning(f"No se pudo obtener duración con ffprobe: {e}")
        return None


def extraer_clip(video_path, video_id, clip_index, inicio_min, inicio_seg, fin_min, fin_seg):
    """
    Extrae un clip de un video usando ffmpeg.
    
    Args:
        video_path (str): Ruta del archivo de video
        video_id (str): ID del video
        clip_index (int): Índice del clip
        inicio_min (int): Minutos de inicio
        inicio_seg (int): Segundos de inicio
        fin_min (int): Minutos de fin
        fin_seg (int): Segundos de fin
        
    Returns:
        str: Ruta del clip extraído o None si falló
    """
    try:
        # Convertir tiempos a segundos
        tiempo_inicio = inicio_min * 60 + inicio_seg
        tiempo_fin = fin_min * 60 + fin_seg
        
        # Validar tiempos
        if tiempo_inicio >= tiempo_fin:
            logger.error(f"Tiempo de inicio ({tiempo_inicio}s) no puede ser mayor o igual al tiempo de fin ({tiempo_fin}s)")
            return None
        
        # Calcular duración del clip
        duracion_clip = tiempo_fin - tiempo_inicio
        
        logger.info(f"Extrayendo clip {clip_index} del video {video_id} ({tiempo_inicio}s - {tiempo_fin}s)")
        
        # Guardar clip con formato: {video_id}_{clip_index}.mp4
        clip_filename = f"{video_id}_{clip_index}.mp4"
        clip_path = CLIPS_DIR / clip_filename
        
        # Re-encode de alta calidad para controlar FPS y calidad final.
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", str(tiempo_inicio),
            "-i", str(video_path),
            "-t", str(duracion_clip),
            "-map", "0:v:0",
            "-an",
            "-vf", f"fps={TARGET_FPS},unsharp=lx=3:ly=3:la=0.8:cx=3:cy=3:ca=0.4",
            "-c:v", VIDEO_CODEC,
            "-preset", ENCODE_PRESET,
            "-crf", str(CRF),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-y",
            str(clip_path)
        ]
        
        resultado = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if resultado.returncode != 0:
            logger.error(f"Error ffmpeg al extraer clip: {resultado.stderr}")
            return None
        
        logger.info(f"Clip extraído exitosamente: {clip_path}")
        return str(clip_path)
    
    except Exception as e:
        logger.error(f"Error extrayendo clip {clip_index} del video {video_id}: {e}")
        return None


def procesar_videos():
    """
    Procesa todos los videos y clips definidos en videos.json.
    """
    try:
        # Cargar configuración
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        
        if not isinstance(datos, list):
            logger.error("videos.json debe contener un array de objetos")
            return
        
        logger.info(f"Procesando {len(datos)} videos...")
        
        for idx, video_config in enumerate(datos):
            # Validar estructura
            if "url" not in video_config:
                logger.warning(f"Video {idx} no tiene campo 'url', saltando...")
                continue
            
            if "clips" not in video_config or not isinstance(video_config["clips"], list):
                logger.warning(f"Video {idx} no tiene campo 'clips' válido, saltando...")
                continue
            
            url = video_config["url"]
            video_id = video_config.get("id", f"video_{idx:03d}")
            clips = video_config["clips"]
            
            # Descargar video
            video_path = descargar_video(url, video_id)
            
            if not video_path:
                logger.error(f"No se pudo descargar el video {video_id}, saltando clips...")
                continue
            
            # Procesar clips
            for clip_idx, clip_config in enumerate(clips):
                # Validar estructura del clip
                if not all(key in clip_config for key in ["inicio_min", "inicio_seg", "fin_min", "fin_seg"]):
                    logger.warning(f"Clip {clip_idx} del video {video_id} no tiene estructura válida, saltando...")
                    continue
                
                extraer_clip(
                    video_path,
                    video_id,
                    clip_idx,
                    clip_config["inicio_min"],
                    clip_config["inicio_seg"],
                    clip_config["fin_min"],
                    clip_config["fin_seg"]
                )
            
            # Eliminar video temporal después de procesar todos los clips
            try:
                Path(video_path).unlink()
                logger.info(f"Video temporal {video_id} eliminado")
            except Exception as e:
                logger.warning(f"No se pudo eliminar el video temporal {video_id}: {e}")
        
        logger.info("Procesamiento completado")
    
    except FileNotFoundError:
        logger.error(f"Archivo {JSON_FILE} no encontrado")
    except json.JSONDecodeError as e:
        logger.error(f"Error al parsear {JSON_FILE}: {e}")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")


if __name__ == "__main__":
    procesar_videos()
