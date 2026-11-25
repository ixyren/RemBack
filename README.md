# RemBack

**RemBack** es una interfaz gráfica moderna, fluida y minimalista escrita en **Rust** para **[PeterL1n/BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2)**.

Permite eliminar el fondo de videos o reemplazarlo por pantalla verde, color sólido, transparencia RGBA o un nuevo video/imagen de fondo, con vista previa en vivo (incluyendo comparativa dividida *Antes / Después*) y todos los parámetros oficiales de `BackgroundMattingV2`.

---

## Características principales

- **Interfaz moderna en Rust (`eframe` / `egui`)**:
  - Animación de bienvenida al abrir la aplicación.
  - Controles suaves (*toggles*, controles segmentados animados, barras de progreso con brillo y transiciones fluidas).
  - **Tooltips explicativos suaves** en cada parámetro y botón.
- **Todos los parámetros de [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2)**:
  - `--model-type`: `mattingrefine` y `mattingbase`.
  - `--model-backbone`: `mobilenetv2`, `resnet50`, `resnet101`.
  - `--model-backbone-scale`: escala configurable (`0.125` a `1.0`).
  - `--model-checkpoint`: selector rápido de checkpoints `.pth` locales o descargados.
  - `--model-refine-mode`: `sampling`, `thresholding`, `full`.
  - `--model-refine-sample-pixels`, `--model-refine-threshold`, `--model-refine-kernel-size`.
  - `--device`: `cuda` (GPU) o `cpu`.
  - `--preprocess-alignment`: alineación homográfica para cámara en mano.
  - `--video-resize`: redimensionado rápido (`1080p`, `720p` o personalizado).
  - `--output-format`: `video` (`.mp4`) o `image_sequences` (`PNG`/`JPG`).
  - `--output-types`: `com`, `pha`, `fgr`, `err`, `ref`.
- **Modo sin frame de referencia (Reconstrucción automática de fondo)**:
  - `BackgroundMattingV2` requiere originalmente una fotografía limpia del fondo sin el sujeto; RemBack v1.1.0 incorpora el modo **Sin imagen de referencia (Auto)** que muestrea el video, segmenta el sujeto, calcula la mediana temporal de los píxeles de fondo y rellena zonas ocluidas mediante *inpainting* automáticamente (con caché local por video).
- **Vista previa interactiva y barras de progreso en vivo**:
  - Barra de progreso detallada en tiempo real tanto al generar la **Vista Previa** (carga de modelo, estimación automática de fondo e inferencia) como durante la **Exportación** completa.
  - Modos **Resultado**, **Dividido** (barra deslizable *Antes / Después*) y **Original**.
  - Selector de fotograma y capa de inspección (`com`, `pha`, `fgr`, `err`, `ref`).
  - Extractor automático del fotograma 0 cuando prefieras usar un frame de referencia manual.
- **Actualizador integrado de BackgroundMattingV2**:
  - Verifica nuevas actualizaciones directamente en el repositorio oficial `PeterL1n/BackgroundMattingV2` en GitHub.
  - Descarga/actualiza el motor, configura un entorno Python + PyTorch (con CUDA si detecta GPU NVIDIA) aislado en `%LOCALAPPDATA%\RemBack\venv` y descarga los checkpoints `.pth` oficiales con un clic.
- **Instalador `.exe` optimizado**:
  - Instalador nativo compilado en Rust con el mismo diseño visual, compresión zlib del binario principal, creación de accesos directos y desinstalador integrado.

---

## Instalación (Windows)

1. Descarga **`RemBack-v1.1.0-Setup.exe`** desde la sección [Releases](https://github.com/ixyren/RemBack/releases/latest).
2. Ejecuta el instalador y elige la carpeta de destino.
3. Al abrir **RemBack**, haz clic en **Actualizaciones** para descargar `BackgroundMattingV2`, configurar el entorno Python/PyTorch y bajar un modelo `.pth` (por ejemplo `pytorch_resnet50.pth` o `pytorch_mobilenetv2.pth`).

---

## Compilación desde código fuente

Requisitos:
- Rust (toolchain `x86_64-pc-windows-msvc`)
- Windows 10/11 SDK (`rc.exe`)

```powershell
powershell -ExecutionPolicy Bypass -File .\build_release.ps1
```

Genera:
- `target\release\remback.exe`
- `dist\RemBack-v1.1.0-Setup.exe`

---

## Créditos

- Motor de segmentación y matting: **[PeterL1n/BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2)** (Lin et al., *Real-Time High-Resolution Background Matting*).
- GUI e instalador en Rust: **ixyren/RemBack**.
