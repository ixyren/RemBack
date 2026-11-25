"""
RemBack Worker Bridge for PeterL1n/BackgroundMattingV2.
Handles video metadata inspection, frame extraction, single-frame preview, and full export
with streaming newline-delimited JSON progress.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from threading import Thread


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def emit_stage(progress: float, message: str) -> None:
    emit({"type": "stage", "progress": round(float(progress), 4), "message": message})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RemBack BackgroundMattingV2 Worker")
    p.add_argument("--mode", required=True, choices=["info", "extract", "preview", "export"])
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


def load_bgmv2_and_model(args: argparse.Namespace, stage_base: float = 0.10, stage_weights: float = 0.30):
    if not args.bgmv2_dir or not os.path.isdir(args.bgmv2_dir):
        raise RuntimeError("No se encontró la carpeta de BackgroundMattingV2.")
    if args.bgmv2_dir not in sys.path:
        sys.path.insert(0, args.bgmv2_dir)

    emit_stage(stage_base, "Cargando PyTorch y red neuronal...")
    import torch
    from model import MattingBase, MattingRefine
    from inference_utils import HomographicAlignment

    use_device = args.device
    if use_device == "cuda" and not torch.cuda.is_available():
        emit({"type": "warn", "message": "CUDA no está disponible en PyTorch; usando CPU."})
        use_device = "cpu"
    device = torch.device(use_device)

    if args.model_type == "mattingbase":
        model = MattingBase(args.model_backbone)
    else:
        model = MattingRefine(
            args.model_backbone,
            args.model_backbone_scale,
            args.model_refine_mode,
            args.model_refine_sample_pixels,
            args.model_refine_threshold,
            args.model_refine_kernel_size,
        )

    emit_stage(stage_weights, "Cargando pesos del modelo (.pth)...")
    model = model.to(device).eval()
    try:
        state = torch.load(args.model_checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.model_checkpoint, map_location=device)
    except Exception:
        state = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    return torch, device, model, HomographicAlignment


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
        emit_stage(stage_end, "Usando fondo limpio reconstruido (caché)...")
        return Image.open(cache_path).convert("RGB").resize((width, height), Image.BILINEAR)

    emit_stage(stage_start, "Muestreando fotogramas para reconstruir el fondo...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video para estimar fondo: {video_path}")
    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    sample_count = min(20, total_frames)
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

    mid_stage = stage_start + (stage_end - stage_start) * 0.35
    emit_stage(mid_stage, "Segmentando sujeto para aislar el fondo limpio...")

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

    emit_stage(
        stage_start + (stage_end - stage_start) * 0.72,
        "Calculando mediana temporal y rellenando oclusiones...",
    )
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
        # Fallback for any permanently occluded region: fill using Telea inpainting on working scale
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
    emit_stage(stage_end, "Fondo limpio reconstruido automáticamente.")
    return bg_pil


def safe_align(aligner, src_pil, bgr_pil):
    try:
        return aligner(src_pil, bgr_pil)
    except Exception:
        return src_pil, bgr_pil


def parse_rgb_tensor(torch, color_str: str, device):
    parts = [int(x.strip()) for x in color_str.split(",")]
    while len(parts) < 3:
        parts.append(0)
    r, g, b = [max(0, min(255, c)) / 255.0 for c in parts[:3]]
    return torch.tensor([r, g, b], device=device).view(1, 3, 1, 1)


class TargetMediaReader:
    """Reads either a single static image or a looping target video matching (w, h)."""

    def __init__(self, path: str, width: int, height: int):
        import cv2
        from PIL import Image

        self.width = width
        self.height = height
        self.cap = None
        self.static_rgb = None
        ext = os.path.splitext(path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}:
            img = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
            self.static_rgb = img
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"No se pudo abrir el fondo destino: {path}")

    def read_tensor(self, torch, to_tensor, device, frame_idx: int | None = None):
        import cv2
        from PIL import Image

        if self.static_rgb is not None:
            return to_tensor(self.static_rgb).unsqueeze(0).to(device, non_blocking=True)

        assert self.cap is not None
        if frame_idx is not None:
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx % total)
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            return torch.zeros((1, 3, self.height, self.width), device=device)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return to_tensor(Image.fromarray(frame)).unsqueeze(0).to(device, non_blocking=True)

    def close(self):
        if self.cap is not None:
            self.cap.release()


def save_tensor_image(tensor_bchw, path: str) -> None:
    from torchvision.transforms.functional import to_pil_image

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img = to_pil_image(tensor_bchw[0].clamp(0.0, 1.0).cpu())
    img.save(path)


def run_model_frame(args, torch, F, model, src_t, bgr_t):
    if args.model_type == "mattingbase":
        pha, fgr, err, _ = model(src_t, bgr_t)
        ref = None
    else:
        pha, fgr, _, _, err, ref = model(src_t, bgr_t)
    err_up = F.interpolate(err, src_t.shape[2:], mode="bilinear", align_corners=False)
    ref_up = F.interpolate(ref, src_t.shape[2:], mode="nearest") if ref is not None else None
    return pha, fgr, err_up, ref_up


def handle_preview(args: argparse.Namespace) -> None:
    import cv2
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    torch, device, model, HomographicAlignment = load_bgmv2_and_model(
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

    emit_stage(0.82, "Ejecutando inferencia BackgroundMattingV2...")
    with torch.no_grad():
        pha, fgr, err_up, ref_up = run_model_frame(args, torch, F, model, src_t, bgr_t)

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
            if args.bg_mode == "transparent":
                out_t = torch.cat([fgr * pha.ne(0), pha], dim=1)
            elif args.bg_mode == "target_media" and target_reader is not None:
                tgt_t = target_reader.read_tensor(torch, to_tensor, device, frame_idx=idx)
                out_t = fgr * pha + tgt_t * (1.0 - pha)
            else:
                tgt_t = parse_rgb_tensor(torch, args.bg_color, device)
                out_t = fgr * pha + tgt_t * (1.0 - pha)

    if target_reader is not None:
        target_reader.close()

    emit_stage(0.95, "Guardando vista previa...")
    src_path = os.path.join(preview_dir, "preview_src.png")
    out_path = os.path.join(preview_dir, "preview_out.png")
    src_pil.save(src_path)
    save_tensor_image(out_t, out_path)

    emit({
        "type": "preview_done",
        "frame": idx,
        "width": w,
        "height": h,
        "src_path": src_path,
        "out_path": out_path,
        "layer": layer,
    })


class AsyncImageWriter:
    def __init__(self, path: str, extension: str):
        self.path = path
        self.extension = extension
        self.index = 0
        os.makedirs(path, exist_ok=True)
        self._threads: list[Thread] = []

    def add(self, tensor_bchw):
        from torchvision.transforms.functional import to_pil_image

        idx = self.index
        self.index += 1
        frame = tensor_bchw[0].clamp(0.0, 1.0).cpu()

        def _save():
            img = to_pil_image(frame)
            img.save(os.path.join(self.path, f"{idx:05d}.{self.extension}"))

        t = Thread(target=_save)
        t.start()
        self._threads.append(t)
        if len(self._threads) > 16:
            for old in self._threads[:-8]:
                old.join()
            self._threads = self._threads[-8:]

    def close(self):
        for t in self._threads:
            t.join()
        self._threads.clear()


class Mp4Writer:
    def __init__(self, path: str, fps: float, w: int, h: int):
        import cv2

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.cv2 = cv2
        self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def add(self, tensor_bchw):
        frame = tensor_bchw[0].clamp(0.0, 1.0).mul(255).byte().cpu().permute(1, 2, 0).numpy()
        if frame.shape[2] == 1:
            frame = self.cv2.cvtColor(frame, self.cv2.COLOR_GRAY2BGR)
        else:
            frame = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)
        self.out.write(frame)

    def close(self):
        self.out.release()


def handle_export(args: argparse.Namespace) -> None:
    import cv2
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    torch, device, model, HomographicAlignment = load_bgmv2_and_model(
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

    writers = {}
    out_types = list(args.output_types)
    if "ref" in out_types and args.model_type == "mattingbase":
        out_types.remove("ref")

    for tname in out_types:
        if args.output_format == "video":
            if tname == "com" and args.bg_mode == "transparent":
                # MP4 cannot store alpha channel; export lossless RGBA PNG sequence for transparent mode
                seq_dir = os.path.join(args.output_dir, "com_rgba")
                if os.path.exists(seq_dir):
                    shutil.rmtree(seq_dir)
                writers[tname] = AsyncImageWriter(seq_dir, "png")
            else:
                writers[tname] = Mp4Writer(os.path.join(args.output_dir, f"{tname}.mp4"), fps, w, h)
        else:
            ext = "png" if tname == "com" else "jpg"
            seq_dir = os.path.join(args.output_dir, tname)
            if os.path.exists(seq_dir):
                shutil.rmtree(seq_dir)
            writers[tname] = AsyncImageWriter(seq_dir, ext)

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
    solid_bg = parse_rgb_tensor(torch, args.bg_color, device)

    cap = cv2.VideoCapture(args.video_src)
    live_out_path = os.path.join(preview_dir, "export_live.png")

    t_start = time.time()
    frame_idx = 0

    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            src_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            src_pil = Image.fromarray(src_rgb).resize((w, h), Image.BILINEAR)

            if aligner is not None:
                src_aligned, bgr_aligned = safe_align(aligner, src_pil, bgr_base)
                src_t = to_tensor(src_aligned).unsqueeze(0).to(device, non_blocking=True)
                bgr_t = to_tensor(bgr_aligned).unsqueeze(0).to(device, non_blocking=True)
            else:
                src_t = to_tensor(src_pil).unsqueeze(0).to(device, non_blocking=True)
                bgr_t = bgr_tensor_static

            pha, fgr, err_up, ref_up = run_model_frame(args, torch, F, model, src_t, bgr_t)

            if "com" in writers:
                if args.bg_mode == "transparent":
                    com_rgba = torch.cat([fgr * pha.ne(0), pha], dim=1)
                    writers["com"].add(com_rgba)
                    com_vis = com_rgba
                elif args.bg_mode == "target_media" and target_reader is not None:
                    tgt_t = target_reader.read_tensor(torch, to_tensor, device)
                    com_vis = fgr * pha + tgt_t * (1.0 - pha)
                    writers["com"].add(com_vis)
                else:
                    com_vis = fgr * pha + solid_bg * (1.0 - pha)
                    writers["com"].add(com_vis)
            else:
                com_vis = None

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

            should_emit = (frame_idx == 1) or (frame_idx % 5 == 0) or (frame_idx >= total)
            if should_emit:
                preview_path = ""
                if frame_idx % 15 == 1 or frame_idx >= total:
                    vis = com_vis if com_vis is not None else pha.repeat(1, 3, 1, 1)
                    save_tensor_image(vis, live_out_path)
                    preview_path = live_out_path
                emit({
                    "type": "progress",
                    "frame": frame_idx,
                    "total": max(total, frame_idx),
                    "fps": round(cur_fps, 2),
                    "eta": round(eta, 1),
                    "preview_path": preview_path,
                })

    cap.release()
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
            handle_preview(args)
        elif args.mode == "export":
            handle_export(args)
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
