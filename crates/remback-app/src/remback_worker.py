"""
RemBack Worker Bridge.
Supports:
  - Engine 'rvm': PeterL1n/RobustVideoMatting TorchScript FP16/FP32 (no reference frame needed)
  - Engine 'bgmv2': PeterL1n/BackgroundMattingV2 with auto-backbone detection, FP16 CUDA autocast,
    and optional automatic clean background reconstruction.
"""

import argparse
import hashlib
import json
import os
import queue
import shutil
import sys
import time
import urllib.request
from threading import Thread


RVM_BASE_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0"


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def emit_stage(progress: float, message: str) -> None:
    emit({"type": "stage", "progress": round(float(progress), 4), "message": message})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RemBack Matting Worker")
    p.add_argument("--mode", required=True, choices=["info", "extract", "preview", "export"])
    p.add_argument("--engine", default="rvm", choices=["rvm", "bgmv2"])
    p.add_argument("--rvm-backbone", default="resnet50", choices=["resnet50", "mobilenetv3"])
    p.add_argument("--models-dir", default="")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bgmv2-dir", required=False, default="")
    p.add_argument("--video-src", required=True)
    p.add_argument("--auto-bgr", action="store_true")
    p.add_argument("--video-bgr", default="")
    p.add_argument("--video-target-bgr", default="")
    p.add_argument("--bg-mode", default="green", choices=["green", "transparent", "custom_color", "target_media"])
    p.add_argument("--bg-color", default="120,255,155")
    p.add_argument("--model-type", default="mattingrefine", choices=["mattingbase", "mattingrefine"])
    p.add_argument("--model-backbone", default="resnet50", choices=["resnet101", "resnet50", "mobilenetv2"])
    p.add_argument("--model-backbone-scale", type=float, default=0.25)
    p.add_argument("--model-checkpoint", default="")
    p.add_argument("--model-refine-mode", default="sampling", choices=["full", "sampling", "thresholding"])
    p.add_argument("--model-refine-sample-pixels", type=int, default=80000)
    p.add_argument("--model-refine-threshold", type=float, default=0.7)
    p.add_argument("--model-refine-kernel-size", type=int, default=3)
    p.add_argument("--video-resize", type=int, nargs=2, default=None)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--preprocess-alignment", action="store_true")
    p.add_argument("--output-dir", default="")
    p.add_argument("--output-types", nargs="+", default=["com"], choices=["com", "pha", "fgr", "err", "ref"])
    p.add_argument("--output-format", default="video", choices=["video", "image_sequences"])
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--preview-layer", default="com", choices=["com", "pha", "fgr", "err", "ref"])
    p.add_argument("--extract-out", default="")
    p.add_argument("--preview-dir", default="")
    return p.parse_args()


def probe_video(video_path: str) -> dict:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps if fps > 0 else 30.0,
        "frame_count": max(frame_count, 1),
    }


def extract_single_frame(video_path: str, frame_index: int, out_path: str) -> None:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = max(0, min(frame_index, max(total - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"No se pudo leer el fotograma {idx} de {video_path}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, frame)


def resolve_device(torch, requested: str):
    use_device = requested
    if use_device == "cuda" and not torch.cuda.is_available():
        emit({"type": "warn", "message": "CUDA no está disponible en PyTorch; usando CPU."})
        use_device = "cpu"
    return torch.device(use_device)


def resolve_models_dir(args: argparse.Namespace) -> str:
    if args.models_dir:
        os.makedirs(args.models_dir, exist_ok=True)
        return args.models_dir
    if args.bgmv2_dir:
        d = os.path.join(os.path.dirname(args.bgmv2_dir), "models")
        os.makedirs(d, exist_ok=True)
        return d
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    os.makedirs(d, exist_ok=True)
    return d


def ensure_rvm_model(args: argparse.Namespace, device, stage_start: float, stage_end: float) -> tuple[str, bool]:
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    prec = "fp16" if use_fp16 else "fp32"
    filename = f"rvm_{args.rvm_backbone}_{prec}.torchscript"
    models_dir = resolve_models_dir(args)
    target_path = os.path.join(models_dir, filename)

    if os.path.isfile(target_path) and os.path.getsize(target_path) > 100_000:
        return target_path, use_fp16

    url = f"{RVM_BASE_URL}/{filename}"
    part_path = target_path + ".part"
    emit_stage(stage_start, f"Descargando modelo automático {filename}...")

    req = urllib.request.Request(url, headers={"User-Agent": "RemBack-App"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        total_len = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        last_pct = -1.0
        with open(part_path, "wb") as out_f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if total_len > 0:
                    frac = min(1.0, downloaded / float(total_len))
                    if frac - last_pct >= 0.02:
                        last_pct = frac
                        p = stage_start + (stage_end - stage_start) * frac
                        mb = downloaded / 1_048_576.0
                        tot_mb = total_len / 1_048_576.0
                        emit_stage(p, f"Descargando {filename} ({mb:.1f}/{tot_mb:.1f} MB)...")

    os.replace(part_path, target_path)
    emit_stage(stage_end, f"Modelo {filename} listo.")
    return target_path, use_fp16


def detect_backbone_from_state(state: dict, ckpt_path: str, fallback: str) -> str:
    keys = list(state.keys())
    if any(k.startswith("backbone.features.") for k in keys):
        return "mobilenetv2"
    if any(k.startswith("backbone.layer3.22.") for k in keys):
        return "resnet101"
    if any(k.startswith("backbone.layer3.0.") for k in keys):
        return "resnet50"

    name = os.path.basename(ckpt_path).lower()
    if "resnet101" in name:
        return "resnet101"
    if "resnet50" in name:
        return "resnet50"
    if "mobilenetv2" in name or "mobilenet" in name:
        return "mobilenetv2"
    return fallback


def load_bgmv2_and_model(args: argparse.Namespace, stage_base: float = 0.10, stage_weights: float = 0.28):
    if not args.bgmv2_dir or not os.path.isdir(args.bgmv2_dir):
        raise RuntimeError("No se encontró BackgroundMattingV2. Instálalo en 'Actualizaciones'.")
    if args.bgmv2_dir not in sys.path:
        sys.path.insert(0, args.bgmv2_dir)

    emit_stage(stage_base, "Cargando PyTorch y BackgroundMattingV2...")
    import torch
    from model import MattingBase, MattingRefine
    from inference_utils import HomographicAlignment

    device = resolve_device(torch, args.device)
    emit_stage(stage_weights, "Cargando pesos (.pth) y verificando arquitectura...")

    ckpt = args.model_checkpoint
    is_torchscript = os.path.basename(ckpt).lower().startswith("torchscript_")
    if is_torchscript:
        model = torch.jit.load(ckpt, map_location=device).eval()
        ts_fp16 = "fp16" in os.path.basename(ckpt).lower()
        return torch, device, model, HomographicAlignment, True, ts_fp16

    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    except Exception:
        state = torch.load(ckpt, map_location=device)

    backbone = detect_backbone_from_state(state, ckpt, args.model_backbone)
    if backbone != args.model_backbone:
        emit({
            "type": "warn",
            "message": f"Backbone ajustado automáticamente a '{backbone}' para coincidir con el checkpoint.",
        })

    if args.model_type == "mattingbase":
        model = MattingBase(backbone)
    else:
        model = MattingRefine(
            backbone,
            args.model_backbone_scale,
            args.model_refine_mode,
            args.model_refine_sample_pixels,
            args.model_refine_threshold,
            args.model_refine_kernel_size,
        )

    model = model.to(device).eval()
    model.load_state_dict(state, strict=False)
    use_half = bool(args.fp16 and device.type == "cuda")
    if use_half:
        model = model.half()
    return torch, device, model, HomographicAlignment, False, use_half


def generate_auto_background(
    video_path: str,
    width: int,
    height: int,
    torch,
    device,
    preview_dir: str,
    stage_start: float,
    stage_end: float,
):
    import cv2
    import numpy as np
    from PIL import Image

    os.makedirs(preview_dir, exist_ok=True)
    st = os.stat(video_path)
    key_raw = f"{os.path.abspath(video_path)}|{st.st_mtime_ns}|{st.st_size}|{width}x{height}"
    cache_key = hashlib.sha1(key_raw.encode("utf-8")).hexdigest()[:16]
    cache_path = os.path.join(preview_dir, f"auto_bgr_{cache_key}.png")

    if os.path.isfile(cache_path):
        emit_stage(stage_end, "Usando fondo reconstruido en caché...")
        return Image.open(cache_path).convert("RGB").resize((width, height), Image.BILINEAR)

    emit_stage(stage_start, "Muestreando fotogramas para reconstruir el fondo...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video para estimar fondo: {video_path}")
    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    sample_count = min(16, total_frames)
    indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=int).tolist()

    sampled_rgb = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if ok and frame_bgr is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[1] != width or rgb.shape[0] != height:
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            sampled_rgb.append(rgb)
    cap.release()

    if not sampled_rgb:
        raise RuntimeError("No se pudieron leer fotogramas del video para reconstruir el fondo.")

    frames_arr = np.stack(sampled_rgb, axis=0)
    n_samples = frames_arr.shape[0]
    bg_valid = np.ones((n_samples, height, width), dtype=bool)

    emit_stage(stage_start + (stage_end - stage_start) * 0.35, "Segmentando sujeto para aislar el fondo...")
    seg_model = None
    try:
        from torchvision.models.segmentation import (
            LRASPP_MobileNet_V3_Large_Weights,
            lraspp_mobilenet_v3_large,
        )

        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        seg_model = lraspp_mobilenet_v3_large(weights=weights).to(device).eval()
    except Exception as exc:
        emit({
            "type": "warn",
            "message": f"Segmentador automático no disponible ({exc}); usando mediana temporal simple.",
        })

    if seg_model is not None:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        max_side = 512
        scale = min(1.0, max_side / float(max(width, height)))
        sw = max(64, int(round(width * scale)))
        sh = max(64, int(round(height * scale)))
        k_sz = max(9, int(round(min(width, height) * 0.025)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))

        with torch.no_grad():
            for i in range(n_samples):
                small = cv2.resize(frames_arr[i], (sw, sh), interpolation=cv2.INTER_LINEAR)
                t_in = torch.from_numpy(small).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                t_in = (t_in - mean) / std
                out = seg_model(t_in)["out"]
                fg_small = (out.argmax(1)[0] != 0).cpu().numpy().astype(np.uint8)
                fg_full = cv2.resize(fg_small, (width, height), interpolation=cv2.INTER_NEAREST)
                fg_dilated = cv2.dilate(fg_full, kernel, iterations=1) > 0
                bg_valid[i] = ~fg_dilated

    emit_stage(stage_start + (stage_end - stage_start) * 0.72, "Combinando fondo y rellenando oclusiones...")
    bg_out = np.zeros((height, width, 3), dtype=np.uint8)
    hole_mask = np.zeros((height, width), dtype=np.uint8)
    chunk_rows = 128
    for y0 in range(0, height, chunk_rows):
        y1 = min(height, y0 + chunk_rows)
        sub_frames = frames_arr[:, y0:y1, :, :].astype(np.float32)
        sub_valid = bg_valid[:, y0:y1, :]
        sub_frames[~sub_valid] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(sub_frames, axis=0)
        nan_pix = np.isnan(med).any(axis=2)
        hole_mask[y0:y1, :] = nan_pix.astype(np.uint8) * 255
        med = np.nan_to_num(med, nan=0.0)
        bg_out[y0:y1, :, :] = np.clip(med, 0, 255).astype(np.uint8)

    if np.any(hole_mask):
        work_scale = min(1.0, 960.0 / float(max(width, height)))
        if work_scale < 0.99:
            iw = max(64, int(round(width * work_scale)))
            ih = max(64, int(round(height * work_scale)))
            small_bg = cv2.resize(bg_out, (iw, ih), interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(hole_mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
            inpainted_small = cv2.inpaint(small_bg, small_mask, 7, cv2.INPAINT_TELEA)
            inpainted = cv2.resize(inpainted_small, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            inpainted = cv2.inpaint(bg_out, hole_mask, 7, cv2.INPAINT_TELEA)
        hole_bool = hole_mask > 0
        bg_out[hole_bool] = inpainted[hole_bool]

    bg_pil = Image.fromarray(bg_out)
    bg_pil.save(cache_path)
    emit_stage(stage_end, "Fondo limpio reconstruido.")
    return bg_pil


def safe_align(aligner, src_pil, bgr_pil):
    try:
        return aligner(src_pil, bgr_pil)
    except Exception:
        return src_pil, bgr_pil


def parse_rgb_tensor(torch, color_str: str, device, dtype):
    parts = [int(x.strip()) for x in color_str.split(",")]
    while len(parts) < 3:
        parts.append(0)
    r, g, b = [max(0, min(255, c)) / 255.0 for c in parts[:3]]
    return torch.tensor([r, g, b], device=device, dtype=dtype).view(1, 3, 1, 1)


class TargetMediaReader:
    """Reads either a static image or looping target video resized to (w, h)."""

    def __init__(self, path: str, width: int, height: int):
        import cv2
        import numpy as np
        from PIL import Image

        self.cv2 = cv2
        self.np = np
        self.width = width
        self.height = height
        self.cap = None
        self.static_rgb = None
        ext = os.path.splitext(path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}:
            img = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
            self.static_rgb = np.asarray(img)
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"No se pudo abrir el fondo destino: {path}")

    def read_tensor(self, torch, device, dtype, frame_idx: int | None = None):
        if self.static_rgb is not None:
            t = torch.from_numpy(self.static_rgb).to(device, non_blocking=True)
            return t.permute(2, 0, 1).unsqueeze(0).to(dtype).div_(255.0)

        assert self.cap is not None
        if frame_idx is not None:
            total = int(self.cap.get(self.cv2.CAP_PROP_FRAME_COUNT)) or 1
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, frame_idx % total)
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            return torch.zeros((1, 3, self.height, self.width), device=device, dtype=dtype)
        frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = self.cv2.resize(frame, (self.width, self.height), interpolation=self.cv2.INTER_LINEAR)
        t = torch.from_numpy(frame).to(device, non_blocking=True)
        return t.permute(2, 0, 1).unsqueeze(0).to(dtype).div_(255.0)

    def close(self):
        if self.cap is not None:
            self.cap.release()


def save_preview_rgb_or_rgba(tensor_bchw, path: str, max_w: int = 1280) -> None:
    import cv2

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = tensor_bchw[0].clamp(0.0, 1.0).mul(255).byte().cpu().permute(1, 2, 0).numpy()
    h, w, c = arr.shape
    if w > max_w:
        nh = max(2, int(round(h * (max_w / float(w)))))
        arr = cv2.resize(arr, (max_w, nh), interpolation=cv2.INTER_AREA)
    if c == 4:
        arr_cv = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    elif c == 1:
        arr_cv = arr[:, :, 0]
    else:
        arr_cv = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, arr_cv)


def save_preview_bgr_array(rgb_arr, path: str, max_w: int = 1280) -> None:
    import cv2

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    h, w, _ = rgb_arr.shape
    arr = rgb_arr
    if w > max_w:
        nh = max(2, int(round(h * (max_w / float(w)))))
        arr = cv2.resize(arr, (max_w, nh), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


class AsyncImageWriter:
    def __init__(self, path: str, extension: str):
        import cv2

        self.cv2 = cv2
        self.path = path
        self.extension = extension
        self.index = 0
        os.makedirs(path, exist_ok=True)
        self._q: queue.Queue = queue.Queue(maxsize=24)
        self._worker = Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _loop(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            idx, arr = item
            c = arr.shape[2]
            if c == 4:
                out_arr = self.cv2.cvtColor(arr, self.cv2.COLOR_RGBA2BGRA)
            elif c == 1:
                out_arr = arr[:, :, 0]
            else:
                out_arr = self.cv2.cvtColor(arr, self.cv2.COLOR_RGB2BGR)
            self.cv2.imwrite(os.path.join(self.path, f"{idx:05d}.{self.extension}"), out_arr)

    def add(self, tensor_bchw):
        arr = tensor_bchw[0].clamp(0.0, 1.0).mul(255).byte().cpu().permute(1, 2, 0).numpy()
        self._q.put((self.index, arr))
        self.index += 1

    def close(self):
        self._q.put(None)
        self._worker.join()


class AsyncMp4Writer:
    def __init__(self, path: str, fps: float, w: int, h: int):
        import cv2

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.cv2 = cv2
        self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self._q: queue.Queue = queue.Queue(maxsize=24)
        self._worker = Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _loop(self):
        while True:
            arr = self._q.get()
            if arr is None:
                break
            if arr.shape[2] == 1:
                bgr = self.cv2.cvtColor(arr, self.cv2.COLOR_GRAY2BGR)
            else:
                bgr = self.cv2.cvtColor(arr, self.cv2.COLOR_RGB2BGR)
            self.out.write(bgr)

    def add(self, tensor_bchw):
        arr = tensor_bchw[0].clamp(0.0, 1.0).mul(255).byte().cpu().permute(1, 2, 0).numpy()
        self._q.put(arr)

    def close(self):
        self._q.put(None)
        self._worker.join()
        self.out.release()


class ThreadedVideoReader:
    def __init__(self, video_path: str, width: int, height: int):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(video_path)
        self.width = width
        self.height = height
        self._q: queue.Queue = queue.Queue(maxsize=8)
        self._stop = False
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            ok, frame_bgr = self.cap.read()
            if not ok or frame_bgr is None:
                self._q.put(None)
                break
            rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
            if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
                rgb = self.cv2.resize(rgb, (self.width, self.height), interpolation=self.cv2.INTER_LINEAR)
            self._q.put(rgb)
        self.cap.release()

    def read_next(self):
        return self._q.get()

    def close(self):
        self._stop = True


def compose_output(torch, args, fgr, pha, solid_bg, target_reader, frame_idx=None):
    if args.bg_mode == "transparent":
        # Clean foreground premultiplied alpha + genuine alpha channel
        return torch.cat([fgr * (pha > 0.01).to(fgr.dtype), pha], dim=1)
    if args.bg_mode == "target_media" and target_reader is not None:
        tgt_t = target_reader.read_tensor(torch, fgr.device, fgr.dtype, frame_idx=frame_idx)
        return fgr * pha + tgt_t * (1.0 - pha)
    return fgr * pha + solid_bg * (1.0 - pha)


def setup_writers(args, fps: float, w: int, h: int, allow_ref: bool):
    writers = {}
    out_types = list(args.output_types)
    if "ref" in out_types and not allow_ref:
        out_types.remove("ref")
    if "err" in out_types and args.engine == "rvm":
        out_types.remove("err")
    if not out_types:
        out_types = ["com"]

    for tname in out_types:
        if args.output_format == "video":
            if tname == "com" and args.bg_mode == "transparent":
                seq_dir = os.path.join(args.output_dir, "com_rgba")
                if os.path.exists(seq_dir):
                    shutil.rmtree(seq_dir)
                writers[tname] = AsyncImageWriter(seq_dir, "png")
            else:
                writers[tname] = AsyncMp4Writer(os.path.join(args.output_dir, f"{tname}.mp4"), fps, w, h)
        else:
            ext = "png" if tname == "com" else "jpg"
            seq_dir = os.path.join(args.output_dir, tname)
            if os.path.exists(seq_dir):
                shutil.rmtree(seq_dir)
            writers[tname] = AsyncImageWriter(seq_dir, ext)
    return writers


# -------------------- RVM Engine --------------------


def handle_preview_rvm(args: argparse.Namespace) -> None:
    import cv2

    emit_stage(0.08, "Cargando PyTorch y motor rápido (RVM)...")
    import torch

    device = resolve_device(torch, args.device)
    model_path, use_fp16 = ensure_rvm_model(args, device, stage_start=0.14, stage_end=0.52)
    dtype = torch.float16 if use_fp16 else torch.float32

    emit_stage(0.56, "Inicializando red neuronal en GPU/CPU...")
    model = torch.jit.load(model_path, map_location=device).eval()

    info = probe_video(args.video_src)
    w = args.video_resize[0] if args.video_resize else info["width"]
    h = args.video_resize[1] if args.video_resize else info["height"]
    idx = max(0, min(args.frame_index, info["frame_count"] - 1))
    downsample_ratio = max(0.125, min(1.0, 512.0 / float(max(w, h))))

    emit_stage(0.66, "Analizando contexto temporal del video...")
    warmup_start = max(0, idx - 5)
    cap = cv2.VideoCapture(args.video_src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)

    target_reader = None
    if args.bg_mode == "target_media" and args.video_target_bgr:
        target_reader = TargetMediaReader(args.video_target_bgr, w, h)
    solid_bg = parse_rgb_tensor(torch, args.bg_color, device, dtype)

    rec = [None, None, None, None]
    last_rgb = None
    fgr = pha = None

    with torch.no_grad():
        for cur_i in range(warmup_start, idx + 1):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[1] != w or rgb.shape[0] != h:
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            last_rgb = rgb
            src_t = torch.from_numpy(rgb).to(device).permute(2, 0, 1).unsqueeze(0).to(dtype).div_(255.0)
            fgr, pha, *rec = model(src_t, *rec, downsample_ratio)
            if cur_i == warmup_start:
                emit_stage(0.78, "Refinando memoria temporal sobre el sujeto...")

    cap.release()
    if last_rgb is None or fgr is None or pha is None:
        raise RuntimeError(f"No se pudo leer el fotograma {idx}")

    emit_stage(0.90, "Componiendo vista previa...")
    layer = args.preview_layer
    if layer in ("err", "ref"):
        layer = "pha"

    if layer == "pha":
        out_t = pha.repeat(1, 3, 1, 1)
    elif layer == "fgr":
        out_t = fgr
    else:
        out_t = compose_output(torch, args, fgr, pha, solid_bg, target_reader, frame_idx=idx)

    if target_reader is not None:
        target_reader.close()

    preview_dir = args.preview_dir or os.path.join(resolve_models_dir(args), "..", "preview_cache")
    src_path = os.path.join(preview_dir, "preview_src.png")
    out_path = os.path.join(preview_dir, "preview_out.png")
    save_preview_bgr_array(last_rgb, src_path)
    save_preview_rgb_or_rgba(out_t, out_path)

    emit({
        "type": "preview_done",
        "frame": idx,
        "width": w,
        "height": h,
        "src_path": src_path,
        "out_path": out_path,
        "layer": layer,
    })


def handle_export_rvm(args: argparse.Namespace) -> None:
    emit_stage(0.05, "Cargando PyTorch y motor rápido (RVM)...")
    import torch

    device = resolve_device(torch, args.device)
    model_path, use_fp16 = ensure_rvm_model(args, device, stage_start=0.10, stage_end=0.32)
    dtype = torch.float16 if use_fp16 else torch.float32

    emit_stage(0.36, "Inicializando red neuronal en GPU/CPU...")
    model = torch.jit.load(model_path, map_location=device).eval()

    info = probe_video(args.video_src)
    w = args.video_resize[0] if args.video_resize else info["width"]
    h = args.video_resize[1] if args.video_resize else info["height"]
    fps = info["fps"]
    total = info["frame_count"]
    downsample_ratio = max(0.125, min(1.0, 512.0 / float(max(w, h))))

    os.makedirs(args.output_dir, exist_ok=True)
    preview_dir = args.preview_dir or os.path.join(resolve_models_dir(args), "..", "preview_cache")
    os.makedirs(preview_dir, exist_ok=True)
    live_out_path = os.path.join(preview_dir, "export_live.png")

    writers = setup_writers(args, fps, w, h, allow_ref=False)
    target_reader = None
    if args.bg_mode == "target_media" and args.video_target_bgr:
        target_reader = TargetMediaReader(args.video_target_bgr, w, h)
    solid_bg = parse_rgb_tensor(torch, args.bg_color, device, dtype)

    emit_stage(0.45, "Iniciando exportación en tiempo real...")
    reader = ThreadedVideoReader(args.video_src, w, h)
    rec = [None, None, None, None]
    t_start = time.time()
    frame_idx = 0

    with torch.no_grad():
        while True:
            rgb = reader.read_next()
            if rgb is None:
                break

            src_t = torch.from_numpy(rgb).to(device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).to(dtype).div_(255.0)
            fgr, pha, *rec = model(src_t, *rec, downsample_ratio)

            com_vis = None
            if "com" in writers:
                com_vis = compose_output(torch, args, fgr, pha, solid_bg, target_reader)
                writers["com"].add(com_vis)
            if "pha" in writers:
                writers["pha"].add(pha)
            if "fgr" in writers:
                writers["fgr"].add(fgr)

            frame_idx += 1
            elapsed = max(time.time() - t_start, 1e-4)
            cur_fps = frame_idx / elapsed
            rem = max(total - frame_idx, 0)
            eta = rem / max(cur_fps, 1e-4)

            if (frame_idx == 1) or (frame_idx % 5 == 0) or (frame_idx >= total):
                preview_path = ""
                if frame_idx % 15 == 1 or frame_idx >= total:
                    vis = com_vis if com_vis is not None else pha.repeat(1, 3, 1, 1)
                    save_preview_rgb_or_rgba(vis, live_out_path)
                    preview_path = live_out_path
                emit({
                    "type": "progress",
                    "frame": frame_idx,
                    "total": max(total, frame_idx),
                    "fps": round(cur_fps, 2),
                    "eta": round(eta, 1),
                    "preview_path": preview_path,
                })

    reader.close()
    if target_reader is not None:
        target_reader.close()
    for w_obj in writers.values():
        w_obj.close()

    emit({
        "type": "export_done",
        "output_dir": args.output_dir,
        "frames": frame_idx,
        "elapsed": round(time.time() - t_start, 2),
    })


# -------------------- BackgroundMattingV2 Engine --------------------


def run_bgmv2_frame(args, torch, F, model, is_torchscript, src_t, bgr_t, use_half: bool):
    if use_half:
        src_in = src_t.half()
        bgr_in = bgr_t.half()
    else:
        src_in = src_t
        bgr_in = bgr_t

    if is_torchscript:
        pha, fgr = model(src_in, bgr_in)[:2]
        err = torch.zeros_like(pha)
        ref = None
    elif args.model_type == "mattingbase":
        pha, fgr, err, _ = model(src_in, bgr_in)
        ref = None
    else:
        pha, fgr, _, _, err, ref = model(src_in, bgr_in)
    err_up = F.interpolate(err.float(), src_t.shape[2:], mode="bilinear", align_corners=False)
    ref_up = F.interpolate(ref.float(), src_t.shape[2:], mode="nearest") if ref is not None else None
    return pha.float(), fgr.float(), err_up, ref_up


def handle_preview_bgmv2(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    torch, device, model, HomographicAlignment, is_ts, use_half = load_bgmv2_and_model(
        args, stage_base=0.10, stage_weights=0.28
    )
    F = torch.nn.functional

    emit_stage(0.42, "Leyendo fotograma de video...")
    info = probe_video(args.video_src)
    cap = cv2.VideoCapture(args.video_src)
    idx = max(0, min(args.frame_index, info["frame_count"] - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"No se pudo leer el fotograma {idx}")

    w = args.video_resize[0] if args.video_resize else info["width"]
    h = args.video_resize[1] if args.video_resize else info["height"]

    preview_dir = args.preview_dir or os.path.join(os.path.dirname(args.bgmv2_dir), "preview_cache")
    os.makedirs(preview_dir, exist_ok=True)

    src_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    src_pil = Image.fromarray(src_rgb).resize((w, h), Image.BILINEAR)

    if args.auto_bgr or not args.video_bgr:
        bgr_pil = generate_auto_background(
            args.video_src, w, h, torch, device, preview_dir, stage_start=0.50, stage_end=0.74
        )
    else:
        emit_stage(0.65, "Cargando imagen de fondo...")
        bgr_pil = Image.open(args.video_bgr).convert("RGB").resize((w, h), Image.BILINEAR)

    if args.preprocess_alignment:
        emit_stage(0.77, "Aplicando alineación homográfica...")
        aligner = HomographicAlignment()
        src_pil, bgr_pil = safe_align(aligner, src_pil, bgr_pil)

    src_t = to_tensor(src_pil).unsqueeze(0).to(device)
    bgr_t = to_tensor(bgr_pil).unsqueeze(0).to(device)

    target_reader = None
    if args.bg_mode == "target_media" and args.video_target_bgr:
        target_reader = TargetMediaReader(args.video_target_bgr, w, h)
    solid_bg = parse_rgb_tensor(torch, args.bg_color, device, torch.float32)

    emit_stage(0.84, "Ejecutando inferencia BackgroundMattingV2...")
    with torch.no_grad():
        pha, fgr, err_up, ref_up = run_bgmv2_frame(args, torch, F, model, is_ts, src_t, bgr_t, use_half)

        layer = args.preview_layer
        if layer == "ref" and ref_up is None:
            layer = "err"

        if layer == "pha":
            out_t = pha.repeat(1, 3, 1, 1)
        elif layer == "fgr":
            out_t = fgr
        elif layer == "err":
            out_t = err_up.repeat(1, 3, 1, 1)
        elif layer == "ref":
            out_t = ref_up.repeat(1, 3, 1, 1)
        else:
            out_t = compose_output(torch, args, fgr, pha, solid_bg, target_reader, frame_idx=idx)

    if target_reader is not None:
        target_reader.close()

    emit_stage(0.95, "Guardando vista previa...")
    src_path = os.path.join(preview_dir, "preview_src.png")
    out_path = os.path.join(preview_dir, "preview_out.png")
    save_preview_bgr_array(np.asarray(src_pil), src_path)
    save_preview_rgb_or_rgba(out_t, out_path)

    emit({
        "type": "preview_done",
        "frame": idx,
        "width": w,
        "height": h,
        "src_path": src_path,
        "out_path": out_path,
        "layer": layer,
    })


def handle_export_bgmv2(args: argparse.Namespace) -> None:
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    torch, device, model, HomographicAlignment, is_ts, use_half = load_bgmv2_and_model(
        args, stage_base=0.05, stage_weights=0.14
    )
    F = torch.nn.functional

    info = probe_video(args.video_src)
    w = args.video_resize[0] if args.video_resize else info["width"]
    h = args.video_resize[1] if args.video_resize else info["height"]
    fps = info["fps"]
    total = info["frame_count"]

    os.makedirs(args.output_dir, exist_ok=True)
    preview_dir = args.preview_dir or os.path.join(os.path.dirname(args.bgmv2_dir), "preview_cache")
    os.makedirs(preview_dir, exist_ok=True)

    writers = setup_writers(args, fps, w, h, allow_ref=(args.model_type == "mattingrefine" and not is_ts))

    if args.auto_bgr or not args.video_bgr:
        bgr_base = generate_auto_background(
            args.video_src, w, h, torch, device, preview_dir, stage_start=0.18, stage_end=0.35
        )
    else:
        emit_stage(0.25, "Cargando imagen de fondo...")
        bgr_base = Image.open(args.video_bgr).convert("RGB").resize((w, h), Image.BILINEAR)

    emit_stage(0.40, "Iniciando exportación de fotogramas...")
    bgr_tensor_static = to_tensor(bgr_base).unsqueeze(0).to(device, non_blocking=True)
    aligner = HomographicAlignment() if args.preprocess_alignment else None

    target_reader = None
    if args.bg_mode == "target_media" and args.video_target_bgr:
        target_reader = TargetMediaReader(args.video_target_bgr, w, h)
    solid_bg = parse_rgb_tensor(torch, args.bg_color, device, torch.float32)

    reader = ThreadedVideoReader(args.video_src, w, h)
    live_out_path = os.path.join(preview_dir, "export_live.png")

    t_start = time.time()
    frame_idx = 0

    with torch.no_grad():
        while True:
            src_rgb = reader.read_next()
            if src_rgb is None:
                break

            if aligner is not None:
                src_aligned, bgr_aligned = safe_align(aligner, Image.fromarray(src_rgb), bgr_base)
                src_t = to_tensor(src_aligned).unsqueeze(0).to(device, non_blocking=True)
                bgr_t = to_tensor(bgr_aligned).unsqueeze(0).to(device, non_blocking=True)
            else:
                src_t = torch.from_numpy(src_rgb).to(device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
                bgr_t = bgr_tensor_static

            pha, fgr, err_up, ref_up = run_bgmv2_frame(args, torch, F, model, is_ts, src_t, bgr_t, use_half)

            com_vis = None
            if "com" in writers:
                com_vis = compose_output(torch, args, fgr, pha, solid_bg, target_reader)
                writers["com"].add(com_vis)
            if "pha" in writers:
                writers["pha"].add(pha)
            if "fgr" in writers:
                writers["fgr"].add(fgr)
            if "err" in writers:
                writers["err"].add(err_up)
            if "ref" in writers and ref_up is not None:
                writers["ref"].add(ref_up)

            frame_idx += 1
            elapsed = max(time.time() - t_start, 1e-4)
            cur_fps = frame_idx / elapsed
            rem = max(total - frame_idx, 0)
            eta = rem / max(cur_fps, 1e-4)

            if (frame_idx == 1) or (frame_idx % 5 == 0) or (frame_idx >= total):
                preview_path = ""
                if frame_idx % 15 == 1 or frame_idx >= total:
                    vis = com_vis if com_vis is not None else pha.repeat(1, 3, 1, 1)
                    save_preview_rgb_or_rgba(vis, live_out_path)
                    preview_path = live_out_path
                emit({
                    "type": "progress",
                    "frame": frame_idx,
                    "total": max(total, frame_idx),
                    "fps": round(cur_fps, 2),
                    "eta": round(eta, 1),
                    "preview_path": preview_path,
                })

    reader.close()
    if target_reader is not None:
        target_reader.close()
    for w_obj in writers.values():
        w_obj.close()

    emit({
        "type": "export_done",
        "output_dir": args.output_dir,
        "frames": frame_idx,
        "elapsed": round(time.time() - t_start, 2),
    })


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "info":
            info = probe_video(args.video_src)
            emit({"type": "info", **info})
        elif args.mode == "extract":
            out = args.extract_out
            if not out:
                raise RuntimeError("Falta --extract-out")
            extract_single_frame(args.video_src, args.frame_index, out)
            emit({"type": "extract_done", "path": out})
        elif args.mode == "preview":
            if args.engine == "rvm":
                handle_preview_rvm(args)
            else:
                handle_preview_bgmv2(args)
        elif args.mode == "export":
            if args.engine == "rvm":
                handle_export_rvm(args)
            else:
                handle_export_bgmv2(args)
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
