# RemBack

**RemBack** es una interfaz gráfica moderna, fluida y optimizada escrita en **Rust** para **[PeterL1n/BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2)** (con soporte integrado de **[PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)** para videos sin frame de referencia).

Permite eliminar el fondo de videos o reemplazarlo por pantalla verde, transparencia RGBA, color sólido o un nuevo video/imagen de fondo, con **Modo Simple** (ajustes recomendados listos para cualquier video) y **Modo Avanzado** (control total y descripciones detalladas).

---

## Características principales

- **Modo Simple y Modo Avanzado**:
  - **Modo Simple**: elige tu video, el fondo de salida y la calidad (`Alta Calidad` o `Ultra Rápido`), optimizado para funcionar sin configurar parámetros técnicos.
  - **Modo Avanzado**: acceso completo a cada parámetro de `BackgroundMattingV2` y `RobustVideoMatting`, precisión FP16 CUDA, escalas, refinamiento, formatos y capas con tooltips explicativos detallados.
- **Rápido y preciso con o sin frame de referencia**:
  - **Sin foto de referencia (Automático / RVM FP16)**: usa redes recurrentes TorchScript FP16 (`rvm_resnet50_fp16` / `rvm_mobilenetv3_fp16`) con memoria temporal para recortes limpios de cabello y bordes a alta velocidad en GPU.
  - **Con foto de referencia (BackgroundMattingV2)**: autodetección de arquitectura (`mobilenetv2`, `resnet50`, `resnet101`) desde el `.pth`, aceleración FP16 CUDA y reconstrucción opcional del fondo si no tienes captura limpia.
- **Ajuste de máscara, tolerancia y limpieza de artefactos**:
  - Controles de **Tolerancia al fondo**, **Suavizado de borde**, **Expansión/Contracción** y **Eliminación automática de artefactos sueltos** aplicables tanto en la vista previa como en la exportación final.
- **Vista previa en vivo, cambio de capas instantáneo y barras de progreso detalladas**:
  - Barra de progreso en tiempo real tanto en **Vista Previa** como en **Exportación** (con lector/escritor multihilo para aprovechar al máximo la GPU).
  - Cambio instantáneo entre **Color (`com`)**, **Máscara (`pha`)**, **Sujeto (`fgr`)**, **Error (`err`)** y **Refinamiento (`ref`)** sin re-procesar el fotograma.
  - Comparativa **Resultado**, **Dividido** (*Antes / Después* deslizable) y **Original**.
- **Auto-actualización completa**:
  - Detecta nuevas versiones de **RemBack** en GitHub y se actualiza automáticamente con 1 clic.
  - Detecta y actualiza **BackgroundMattingV2**, configura Python/PyTorch CUDA y descarga modelos `.pth` / `.torchscript` desde la ventana **Actualizaciones**.
- **Instalador `.exe` inteligente y optimizado**:
  - Detecta si ya tienes **RemBack** instalado, muestra la versión actual -> nueva, cierra automáticamente `remback.exe` si está abierto y actualiza sin fricción.

---

## Instalación (Windows)

1. Descarga **`RemBack-v1.3.0-Setup.exe`** desde la sección [Releases](https://github.com/ixyren/RemBack/releases/latest).
2. Ejecuta el instalador (si ya tienes una versión anterior, la detectará y la actualizará automáticamente).
3. Al abrir **RemBack**, haz clic en **Actualizaciones** si aún no tienes configurado el entorno Python/PyTorch.

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
- `dist\RemBack-v1.3.0-Setup.exe`

---

## Créditos

- Motores de matting neuronal: **[PeterL1n/BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2)** y **[PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)**.
- GUI, motor puente en Python e instalador en Rust: **ixyren/RemBack**.
