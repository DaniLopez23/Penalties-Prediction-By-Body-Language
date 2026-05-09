from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import altair as alt
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_pipeline import build_config
from src.config import PipelineConfig
from src.pipeline import PenaltyAnalysisPipeline


APP_VIDEO_DIR = APP_DIR / "videos"
UPLOAD_DIR = APP_VIDEO_DIR / "uploads"
OUTPUT_DIR = APP_VIDEO_DIR / "outputs"
SUPPORTED_VIDEO_TYPES = ("mp4", "mov", "avi", "mkv")


def main() -> None:
    st.set_page_config(page_title="Análisis de penalti", layout="wide")
    _inject_styles()

    st.title("Análisis de penalti")
    st.caption("Sube un video y procésalo con el mismo pipeline de detección, pose y métricas.")

    video_col, info_col = st.columns([2.2, 1.0])

    with video_col:
        uploaded_file = st.file_uploader(
            "Video del penalti",
            type=SUPPORTED_VIDEO_TYPES,
            accept_multiple_files=False,
            help="Formatos soportados: MP4, MOV, AVI y MKV.",
        )
        _clear_stale_result(uploaded_file)

        if uploaded_file is not None:
            st.info(f"Video seleccionado para analizar: {uploaded_file.name}")
            st.video(uploaded_file.getvalue())

        progress_bar = st.progress(0)
        status_box = st.empty()
        process_clicked = st.button(
            "Procesar video",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )

        if process_clicked and uploaded_file is not None:
            _process_uploaded_video(uploaded_file, progress_bar, status_box)

        result = st.session_state.get("analysis_result")
        if result is not None:
            _render_processed_media(result)

    with info_col:
        _render_side_panel(st.session_state.get("analysis_result"), uploaded_file)

    _render_charts(st.session_state.get("analysis_result"))


def _process_uploaded_video(uploaded_file, progress_bar, status_box) -> None:
    _ensure_app_dirs()

    input_path = _save_upload(uploaded_file)
    output_path = OUTPUT_DIR / f"{input_path.stem}_annotated.mp4"
    config = _build_app_config(input_path)
    pipeline = PenaltyAnalysisPipeline(config)

    def on_progress(frame_index, total_frames, analysis_state) -> None:
        if frame_index % 5 != 0 and frame_index != total_frames:
            return
        if total_frames:
            progress = min(frame_index / total_frames, 1.0)
            status = (
                f"Analizando {uploaded_file.name} · "
                f"frame {frame_index}/{total_frames} · {analysis_state.shot_state}"
            )
        else:
            progress = 0.0
            status = f"Analizando {uploaded_file.name} · {frame_index} frames · {analysis_state.shot_state}"
        progress_bar.progress(progress)
        status_box.info(status)

    try:
        status_box.info("Cargando modelos y preparando el análisis...")
        processed_path = pipeline.process_video(
            input_path=input_path,
            output_path=output_path,
            show_window=False,
            progress_callback=on_progress,
        )
    except Exception as exc:
        progress_bar.progress(0)
        status_box.error(f"No se pudo procesar el video: {exc}")
        return

    playback = _prepare_playback_video(processed_path)
    goalkeeper_direction = _goalkeeper_direction_near_shot(
        pipeline.analysis_history,
        pipeline.shot_time_sec,
    )
    st.session_state.analysis_result = {
        "input_name": uploaded_file.name,
        "input_path": str(input_path),
        "output_path": str(processed_path),
        "playback": playback,
        "video_info": pipeline.video_info,
        "final_state": asdict(pipeline.last_analysis_state) if pipeline.last_analysis_state else {},
        "history": [asdict(record) for record in pipeline.analysis_history],
        "goalkeeper_direction": goalkeeper_direction,
        "shot_frame_index": pipeline.shot_frame_index,
        "shot_time_sec": pipeline.shot_time_sec,
    }
    progress_bar.progress(1.0)
    status_box.success("Procesamiento completado.")


def _build_app_config(input_path: Path) -> PipelineConfig:
    base_config = build_config()
    video_config = replace(
        base_config.video,
        input_path=input_path,
        output_dir=OUTPUT_DIR,
        show_window=False,
        max_frames=None,
    )
    return replace(base_config, video=video_config)


def _save_upload(uploaded_file) -> Path:
    original = Path(uploaded_file.name)
    safe_stem = _safe_stem(original.stem)
    suffix = original.suffix.lower() or ".mp4"
    path = UPLOAD_DIR / f"{safe_stem}_{uuid4().hex[:8]}{suffix}"
    path.write_bytes(uploaded_file.getbuffer())
    return path


def _prepare_playback_video(source_path: Path) -> dict[str, str | None]:
    ffmpeg_path = _ffmpeg_executable()
    if ffmpeg_path is None:
        return {
            "kind": "error",
            "path": str(source_path),
            "note": (
                "El output se ha generado, pero OpenCV lo guarda como MP4 mp4v y el navegador "
                "necesita convertirlo a H.264. Instala imageio-ffmpeg o ffmpeg para verlo aquí."
            ),
        }

    playback_path = source_path.with_name(f"{source_path.stem}_streamlit.mp4")
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(playback_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        return {
            "kind": "error",
            "path": str(source_path),
            "note": f"No se pudo convertir el output a H.264 para el navegador: {detail}",
        }

    return {
        "kind": "video",
        "path": str(playback_path),
        "note": None,
    }


def _ffmpeg_executable() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def _render_processed_media(result: dict) -> None:
    st.subheader("Video procesado")
    playback = result.get("playback", {})
    media_path = Path(playback.get("path") or result["output_path"])
    if not media_path.exists():
        st.warning("El video procesado ya no está disponible en disco.")
        return

    if playback.get("kind") == "error":
        st.error(playback.get("note") or "No se pudo preparar el video procesado para el navegador.")
        st.caption(f"Archivo generado: {media_path}")
        return
    if playback.get("kind") not in (None, "video"):
        st.error("El resultado anterior no está preparado como video. Vuelve a procesar el archivo.")
        return

    st.video(media_path.read_bytes(), format="video/mp4")


def _ensure_app_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _clear_stale_result(uploaded_file) -> None:
    upload_signature = None
    if uploaded_file is not None:
        upload_signature = (uploaded_file.name, uploaded_file.size)

    if st.session_state.get("upload_signature") != upload_signature:
        st.session_state.upload_signature = upload_signature
        st.session_state.pop("analysis_result", None)


def _safe_stem(stem: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in stem)
    return cleaned.strip("_") or "penalty"


def _goalkeeper_direction_near_shot(history, shot_time_sec) -> str:
    if shot_time_sec is None:
        return "unknown"

    valid_records = [
        record
        for record in history
        if record.goalkeeper_direction and record.goalkeeper_direction != "unknown"
    ]
    if not valid_records:
        return "unknown"

    closest = min(
        valid_records,
        key=lambda record: abs(record.time_sec - float(shot_time_sec)),
    )
    return closest.goalkeeper_direction
    return "unknown"


def _render_side_panel(result: dict | None, uploaded_file) -> None:
    st.subheader("Información")
    if result is None:
        if uploaded_file is None:
            st.info("Sube un video para ver aquí sus métricas.")
        else:
            st.info("Pulsa procesar para calcular los KPIs del penalti.")
        return

    info = result.get("video_info", {})
    final_state = result.get("final_state", {})
    st.markdown("**Video**")
    st.metric("Archivo", result.get("input_name", "No disponible"))
    st.metric("Resolución", _format_resolution(info))
    st.metric("FPS", _format_number(info.get("fps"), suffix=" fps"))
    st.metric("Duración", _format_seconds(info.get("duration_sec")))
    st.metric("Frames procesados", _format_int(info.get("processed_frame_count")))

    st.markdown("**KPIs del análisis**")
    st.metric("Zona de portería", _translate_zone(final_state.get("ball_zone")))
    st.metric("Dirección del portero", _translate_direction(result.get("goalkeeper_direction")))
    st.metric("Segundo de disparo", _format_seconds(result.get("shot_time_sec")))


def _render_charts(result: dict | None) -> None:
    st.subheader("Evolución de ángulos")
    if result is None:
        st.info("Las gráficas aparecerán cuando termine el procesamiento.")
        return

    history = result.get("history", [])
    if not history:
        st.warning("No hay historial de métricas para graficar.")
        return

    df = pd.DataFrame(history).set_index("time_sec")
    shot_time_sec = result.get("shot_time_sec")

    chart_col_a, chart_col_b = st.columns(2)
    with chart_col_a:
        st.markdown("**Ángulo de hombros del lanzador**")
        _line_chart_or_message(
            df,
            "striker_shoulder_angle_deg",
            "Ángulo de hombros del lanzador",
            "No se detectó el ángulo de hombros del lanzador.",
            shot_time_sec,
        )
    with chart_col_b:
        st.markdown("**Inclinación del portero**")
        _line_chart_or_message(
            df,
            "goalkeeper_lean_deg",
            "Inclinación del portero",
            "No se detectó la inclinación del portero.",
            shot_time_sec,
        )


def _line_chart_or_message(
    dataframe: pd.DataFrame,
    value_column: str,
    label: str,
    empty_message: str,
    shot_time_sec,
) -> None:
    values = dataframe.reset_index()[["time_sec", value_column]].dropna()
    if values.empty:
        st.info(empty_message)
        return
    values = values.rename(columns={value_column: "value"})
    chart = (
        alt.Chart(values)
        .mark_line(color="#1769aa", strokeWidth=2)
        .encode(
            x=alt.X("time_sec:Q", title="Segundo"),
            y=alt.Y("value:Q", title="Grados"),
            tooltip=[
                alt.Tooltip("time_sec:Q", title="Segundo", format=".2f"),
                alt.Tooltip("value:Q", title=label, format=".2f"),
            ],
        )
    )
    if shot_time_sec is not None:
        rule = (
            alt.Chart(pd.DataFrame({"time_sec": [float(shot_time_sec)]}))
            .mark_rule(color="#c7362f", strokeDash=[6, 4], strokeWidth=2)
            .encode(x="time_sec:Q")
        )
        chart = chart + rule
        st.caption(f"Línea vertical: disparo en {_format_seconds(shot_time_sec)}")
    st.altair_chart(chart.properties(height=280), use_container_width=True)


def _format_resolution(info: dict) -> str:
    width = info.get("width")
    height = info.get("height")
    if width is None or height is None:
        return "No disponible"
    return f"{int(width)} x {int(height)}"


def _format_seconds(value) -> str:
    if value is None:
        return "No detectado"
    return f"{float(value):.2f} s"


def _format_number(value, suffix: str = "") -> str:
    if value is None:
        return "No disponible"
    return f"{float(value):.2f}{suffix}"


def _format_int(value) -> str:
    if value is None:
        return "No disponible"
    return f"{int(value)}"


def _translate_zone(value: str | None) -> str:
    return {
        "left": "Izquierda",
        "center": "Centro",
        "right": "Derecha",
    }.get(value or "", "No detectada")


def _translate_direction(value: str | None) -> str:
    return {
        "left": "Izquierda",
        "center": "Centro",
        "right": "Derecha",
        "unknown": "No detectada",
    }.get(value or "", "No detectada")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stFileUploaderDropzone"] {
            min-height: 220px;
            border-width: 2px;
        }
        div[data-testid="stFileUploaderFile"] {
            display: none;
        }
        div[data-testid="stMetric"] {
            background: #f7f8fa;
            border: 1px solid #e6e8ec;
            border-radius: 8px;
            padding: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
