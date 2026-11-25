use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver},
    Arc, Mutex,
};
use std::thread;

use serde::Deserialize;

use crate::config::{ComputeDevice, EngineMode, MattingConfig};
use crate::updater::UpdaterPaths;

const WORKER_PY: &str = include_str!("remback_worker.py");

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub frame_count: u32,
}

#[derive(Debug, Clone)]
pub enum WorkerEvent {
    Info(VideoInfo),
    ExtractDone { path: String },
    PreviewDone {
        frame: u32,
        width: u32,
        height: u32,
        src_path: String,
        out_path: String,
    },
    Progress {
        frame: u32,
        total: u32,
        fps: f32,
        eta: f32,
        preview_path: String,
    },
    Stage {
        progress: f32,
        message: String,
    },
    ExportDone {
        output_dir: String,
        frames: u32,
        elapsed: f32,
    },
    Warn(String),
    Error(String),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum RawWorkerMsg {
    #[serde(rename = "info")]
    Info {
        width: u32,
        height: u32,
        fps: f32,
        frame_count: u32,
    },
    #[serde(rename = "extract_done")]
    ExtractDone { path: String },
    #[serde(rename = "preview_done")]
    PreviewDone {
        frame: u32,
        width: u32,
        height: u32,
        src_path: String,
        out_path: String,
    },
    #[serde(rename = "progress")]
    Progress {
        frame: u32,
        total: u32,
        fps: f32,
        eta: f32,
        #[serde(default)]
        preview_path: String,
    },
    #[serde(rename = "stage")]
    Stage {
        progress: f32,
        message: String,
    },
    #[serde(rename = "export_done")]
    ExportDone {
        output_dir: String,
        frames: u32,
        elapsed: f32,
    },
    #[serde(rename = "warn")]
    Warn { message: String },
    #[serde(rename = "error")]
    Error { message: String },
}

pub struct WorkerHandle {
    pub rx: Receiver<WorkerEvent>,
    cancelled: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
}

impl WorkerHandle {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        if let Ok(mut guard) = self.child.lock() {
            if let Some(ref mut ch) = *guard {
                let _ = ch.kill();
            }
        }
    }
}

/// Ensure the embedded worker Python script is written to disk and up to date.
pub fn ensure_worker_script(paths: &UpdaterPaths) -> Result<PathBuf, String> {
    std::fs::create_dir_all(&paths.backend_dir)
        .map_err(|e| format!("No se pudo crear carpeta backend: {e}"))?;
    let script_path = paths.backend_dir.join("remback_worker.py");
    let needs_write = match std::fs::read_to_string(&script_path) {
        Ok(existing) => existing != WORKER_PY,
        Err(_) => true,
    };
    if needs_write {
        std::fs::write(&script_path, WORKER_PY)
            .map_err(|e| format!("No se pudo escribir remback_worker.py: {e}"))?;
    }
    Ok(script_path)
}

fn spawn_python_worker(
    python_exe: &Path,
    script_path: &Path,
    args: Vec<String>,
) -> WorkerHandle {
    let (tx, rx) = mpsc::channel();
    let cancelled = Arc::new(AtomicBool::new(false));
    let child_arc: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));

    let py = python_exe.to_path_buf();
    let script = script_path.to_path_buf();
    let cancelled_bg = Arc::clone(&cancelled);
    let child_bg = Arc::clone(&child_arc);

    thread::spawn(move || {
        let mut cmd = Command::new(&py);
        cmd.arg("-u")
            .arg(&script)
            .args(&args)
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut spawned = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(WorkerEvent::Error(format!(
                    "No se pudo iniciar Python ({}): {e}",
                    py.display()
                )));
                return;
            }
        };

        let stdout = spawned.stdout.take();
        let stderr = spawned.stderr.take();
        if let Ok(mut g) = child_bg.lock() {
            *g = Some(spawned);
        }

        let stderr_buf = Arc::new(Mutex::new(String::new()));
        let stderr_clone = Arc::clone(&stderr_buf);
        let stderr_thread = stderr.map(|err_pipe| {
            thread::spawn(move || {
                let reader = BufReader::new(err_pipe);
                for line in reader.lines().map_while(Result::ok) {
                    if let Ok(mut b) = stderr_clone.lock() {
                        if b.len() < 4000 {
                            b.push_str(&line);
                            b.push('\n');
                        }
                    }
                }
            })
        });

        let mut saw_error_or_done = false;
        if let Some(out_pipe) = stdout {
            let reader = BufReader::new(out_pipe);
            for line in reader.lines().map_while(Result::ok) {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(msg) = serde_json::from_str::<RawWorkerMsg>(trimmed) {
                    let evt = match msg {
                        RawWorkerMsg::Info {
                            width,
                            height,
                            fps,
                            frame_count,
                        } => {
                            saw_error_or_done = true;
                            WorkerEvent::Info(VideoInfo {
                                width,
                                height,
                                fps,
                                frame_count,
                            })
                        }
                        RawWorkerMsg::ExtractDone { path } => {
                            saw_error_or_done = true;
                            WorkerEvent::ExtractDone { path }
                        }
                        RawWorkerMsg::PreviewDone {
                            frame,
                            width,
                            height,
                            src_path,
                            out_path,
                        } => {
                            saw_error_or_done = true;
                            WorkerEvent::PreviewDone {
                                frame,
                                width,
                                height,
                                src_path,
                                out_path,
                            }
                        }
                        RawWorkerMsg::Progress {
                            frame,
                            total,
                            fps,
                            eta,
                            preview_path,
                        } => WorkerEvent::Progress {
                            frame,
                            total,
                            fps,
                            eta,
                            preview_path,
                        },
                        RawWorkerMsg::Stage { progress, message } => {
                            WorkerEvent::Stage { progress, message }
                        }
                        RawWorkerMsg::ExportDone {
                            output_dir,
                            frames,
                            elapsed,
                        } => {
                            saw_error_or_done = true;
                            WorkerEvent::ExportDone {
                                output_dir,
                                frames,
                                elapsed,
                            }
                        }
                        RawWorkerMsg::Warn { message } => WorkerEvent::Warn(message),
                        RawWorkerMsg::Error { message } => {
                            saw_error_or_done = true;
                            WorkerEvent::Error(message)
                        }
                    };
                    let _ = tx.send(evt);
                }
            }
        }

        if let Some(t) = stderr_thread {
            let _ = t.join();
        }

        let status = child_bg
            .lock()
            .ok()
            .and_then(|mut g| g.take())
            .and_then(|mut c| c.wait().ok());

        if cancelled_bg.load(Ordering::SeqCst) {
            let _ = tx.send(WorkerEvent::Error("Proceso cancelado por el usuario.".into()));
        } else if !saw_error_or_done {
            let err_text = stderr_buf.lock().map(|s| s.clone()).unwrap_or_default();
            let code = status.and_then(|s| s.code()).unwrap_or(-1);
            let msg = if err_text.trim().is_empty() {
                format!("El motor terminó inesperadamente (código {code}).")
            } else {
                format!("Error del motor (código {code}):\n{}", err_text.trim())
            };
            let _ = tx.send(WorkerEvent::Error(msg));
        }
    });

    WorkerHandle {
        rx,
        cancelled,
        child: child_arc,
    }
}

fn build_common_args(cfg: &MattingConfig, paths: &UpdaterPaths) -> Vec<String> {
    let rgb = cfg.effective_bg_rgb();
    let mut args = vec![
        "--engine".into(),
        cfg.engine_mode.as_arg().into(),
        "--rvm-backbone".into(),
        cfg.simple_quality.rvm_backbone_arg().into(),
        "--models-dir".into(),
        paths.models_dir.to_string_lossy().into_owned(),
        "--bgmv2-dir".into(),
        paths.bgmv2_dir.to_string_lossy().into_owned(),
        "--video-src".into(),
        cfg.video_src.clone(),
        "--bg-mode".into(),
        cfg.bg_mode.as_arg().into(),
        "--bg-color".into(),
        format!("{},{},{}", rgb[0], rgb[1], rgb[2]),
        "--model-type".into(),
        cfg.model_type.as_arg().into(),
        "--model-backbone".into(),
        cfg.model_backbone.as_arg().into(),
        "--model-backbone-scale".into(),
        format!("{:.3}", cfg.model_backbone_scale),
        "--model-checkpoint".into(),
        cfg.model_checkpoint.clone(),
        "--model-refine-mode".into(),
        cfg.refine_mode.as_arg().into(),
        "--model-refine-sample-pixels".into(),
        cfg.refine_sample_pixels.to_string(),
        "--model-refine-threshold".into(),
        format!("{:.3}", cfg.refine_threshold),
        "--model-refine-kernel-size".into(),
        cfg.refine_kernel_size.to_string(),
        "--device".into(),
        cfg.device.as_arg().into(),
        "--preview-dir".into(),
        paths.preview_dir.to_string_lossy().into_owned(),
    ];

    if cfg.use_fp16 && cfg.device == ComputeDevice::Cuda {
        args.push("--fp16".into());
    }
    if cfg.auto_bgr {
        args.push("--auto-bgr".into());
    } else if !cfg.video_bgr.trim().is_empty() {
        args.push("--video-bgr".into());
        args.push(cfg.video_bgr.clone());
    }
    if !cfg.video_target_bgr.trim().is_empty() {
        args.push("--video-target-bgr".into());
        args.push(cfg.video_target_bgr.clone());
    }
    if cfg.preprocess_alignment {
        args.push("--preprocess-alignment".into());
    }
    if cfg.resize_enabled {
        args.push("--video-resize".into());
        args.push(cfg.resize_width.to_string());
        args.push(cfg.resize_height.to_string());
    }
    args
}

pub fn run_probe_video(paths: &UpdaterPaths, video_src: &str) -> Result<WorkerHandle, String> {
    let python = paths
        .active_python()
        .ok_or_else(|| "Entorno Python no configurado. Ábrelo desde 'Actualizaciones'.".to_owned())?;
    let script = ensure_worker_script(paths)?;
    let args = vec![
        "--mode".into(),
        "info".into(),
        "--video-src".into(),
        video_src.to_owned(),
    ];
    Ok(spawn_python_worker(&python, &script, args))
}

pub fn run_extract_frame(
    paths: &UpdaterPaths,
    video_src: &str,
    frame_idx: u32,
    out_png: &Path,
) -> Result<WorkerHandle, String> {
    let python = paths
        .active_python()
        .ok_or_else(|| "Entorno Python no configurado. Ábrelo desde 'Actualizaciones'.".to_owned())?;
    let script = ensure_worker_script(paths)?;
    let args = vec![
        "--mode".into(),
        "extract".into(),
        "--video-src".into(),
        video_src.to_owned(),
        "--frame-index".into(),
        frame_idx.to_string(),
        "--extract-out".into(),
        out_png.to_string_lossy().into_owned(),
    ];
    Ok(spawn_python_worker(&python, &script, args))
}

pub fn run_preview(paths: &UpdaterPaths, cfg: &MattingConfig) -> Result<WorkerHandle, String> {
    let python = paths
        .active_python()
        .ok_or_else(|| "Entorno Python no configurado. Ábrelo desde 'Actualizaciones'.".to_owned())?;
    if cfg.engine_mode == EngineMode::BgMattingV2 && !paths.is_bgmv2_installed() {
        return Err("BackgroundMattingV2 no está descargado. Instálalo en 'Actualizaciones'.".into());
    }
    let script = ensure_worker_script(paths)?;
    let mut args = build_common_args(cfg, paths);
    args.extend([
        "--mode".into(),
        "preview".into(),
        "--frame-index".into(),
        cfg.preview_frame.to_string(),
        "--preview-layer".into(),
        cfg.preview_layer.as_arg().into(),
    ]);
    Ok(spawn_python_worker(&python, &script, args))
}

pub fn run_export(paths: &UpdaterPaths, cfg: &MattingConfig) -> Result<WorkerHandle, String> {
    let python = paths
        .active_python()
        .ok_or_else(|| "Entorno Python no configurado. Ábrelo desde 'Actualizaciones'.".to_owned())?;
    if cfg.engine_mode == EngineMode::BgMattingV2 && !paths.is_bgmv2_installed() {
        return Err("BackgroundMattingV2 no está descargado. Instálalo en 'Actualizaciones'.".into());
    }
    let script = ensure_worker_script(paths)?;
    let mut args = build_common_args(cfg, paths);
    args.extend([
        "--mode".into(),
        "export".into(),
        "--output-dir".into(),
        cfg.output_dir.clone(),
        "--output-format".into(),
        cfg.output_format.as_arg().into(),
        "--output-types".into(),
    ]);
    for t in cfg.selected_output_types() {
        args.push(t.to_owned());
    }
    Ok(spawn_python_worker(&python, &script, args))
}
