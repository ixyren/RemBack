use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::Receiver;

use egui::{
    pos2, vec2, Align, Color32, ColorImage, CornerRadius, DragValue, Frame, Layout, Margin, Order,
    Rect, RichText, ScrollArea, Sense, Stroke, StrokeKind, TextureHandle, TextureOptions, Ui,
};
use remback_ui::{
    apply_theme, badge_pill, draw_checkerboard, ease_out_cubic, glass_card, paint_logo,
    primary_button, secondary_button, segmented_control, smooth_progress_bar, smooth_slider,
    smooth_toggle, smooth_tooltip, Palette, SplashState,
};

use crate::config::{
    Backbone, BgMode, ComputeDevice, EngineMode, MattingConfig, ModelType, OutputFormat,
    OutputLayer, RefineMode, SimpleQuality, UiMode,
};
use crate::updater::{
    check_bgmv2_updates, default_models_fallback, download_and_launch_app_update,
    download_model_asset, is_newer_version, rvm_model_assets, setup_python_environment,
    update_bgmv2_repo, Bgmv2InstalledState, ModelAssetInfo, RemoteAppRelease, RemoteBgmv2Info,
    UpdaterEvent, UpdaterPaths,
};
use crate::worker::{
    run_export, run_extract_frame, run_preview, run_probe_video, VideoInfo, WorkerEvent,
    WorkerHandle,
};

pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreviewViewMode {
    Result,
    Split,
    Original,
}

#[derive(Debug, Clone)]
struct Toast {
    message: String,
    is_error: bool,
    created_at: f64,
}

pub struct RemBackApp {
    config: MattingConfig,
    paths: UpdaterPaths,
    splash: SplashState,

    video_info: Option<VideoInfo>,
    local_models: Vec<PathBuf>,

    // Preview state
    view_mode: PreviewViewMode,
    split_pos: f32,
    preview_src_tex: Option<TextureHandle>,
    preview_out_tex: Option<TextureHandle>,
    preview_com_tex: Option<TextureHandle>,
    preview_pha_tex: Option<TextureHandle>,
    preview_fgr_tex: Option<TextureHandle>,
    preview_err_tex: Option<TextureHandle>,
    preview_ref_tex: Option<TextureHandle>,
    preview_fade_start: Option<f64>,
    preview_busy: bool,
    preview_progress: f32,
    preview_stage: String,

    // Export state
    export_busy: bool,
    export_stage_progress: f32,
    export_stage: String,
    export_frame: u32,
    export_total: u32,
    export_fps: f32,
    export_eta: f32,
    last_export_dir: Option<String>,

    // Active background worker
    worker: Option<WorkerHandle>,

    // Updater & Environment state
    installed_state: Bgmv2InstalledState,
    remote_info: Option<RemoteBgmv2Info>,
    remote_app_release: Option<RemoteAppRelease>,
    app_update_available: bool,
    update_available: bool,
    checking_updates: bool,
    updater_rx: Option<Receiver<UpdaterEvent>>,
    updater_busy: bool,
    updater_progress: f32,
    updater_status: String,
    updater_logs: Vec<String>,

    // Modals & notifications
    show_updates_modal: bool,
    show_about_modal: bool,
    toast: Option<Toast>,
    theme_initialized: bool,
}

impl RemBackApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_theme(&cc.egui_ctx);
        let paths = UpdaterPaths::detect();
        let mut config = MattingConfig::load_or_default();
        let local_models = paths.list_local_models();

        if config.ui_mode == UiMode::Simple {
            config.preview_layer = OutputLayer::Com;
        }
        if config.model_checkpoint.trim().is_empty() || !Path::new(&config.model_checkpoint).is_file()
        {
            if let Some(best) = local_models
                .iter()
                .find(|p| p.to_string_lossy().contains("resnet101"))
                .or_else(|| {
                    local_models
                        .iter()
                        .find(|p| p.to_string_lossy().contains("resnet50"))
                })
                .or_else(|| local_models.first())
            {
                config.model_checkpoint = best.to_string_lossy().into_owned();
            }
        }

        let installed_state = Bgmv2InstalledState::load(&paths.state_file);

        // Kick off a lightweight background check for BackgroundMattingV2 updates
        let updater_rx = Some(check_bgmv2_updates());

        Self {
            config,
            paths,
            splash: SplashState::new(),
            video_info: None,
            local_models,
            view_mode: PreviewViewMode::Result,
            split_pos: 0.5,
            preview_src_tex: None,
            preview_out_tex: None,
            preview_com_tex: None,
            preview_pha_tex: None,
            preview_fgr_tex: None,
            preview_err_tex: None,
            preview_ref_tex: None,
            preview_fade_start: None,
            preview_busy: false,
            preview_progress: 0.0,
            preview_stage: String::new(),
            export_busy: false,
            export_stage_progress: 0.0,
            export_stage: String::new(),
            export_frame: 0,
            export_total: 0,
            export_fps: 0.0,
            export_eta: 0.0,
            last_export_dir: None,
            worker: None,
            installed_state,
            remote_info: None,
            remote_app_release: None,
            app_update_available: false,
            update_available: false,
            checking_updates: true,
            updater_rx,
            updater_busy: false,
            updater_progress: 0.0,
            updater_status: String::new(),
            updater_logs: Vec::new(),
            show_updates_modal: false,
            show_about_modal: false,
            toast: None,
            theme_initialized: true,
        }
    }

    fn push_toast(&mut self, ctx: &egui::Context, message: impl Into<String>, is_error: bool) {
        let now = ctx.input(|i| i.time);
        self.toast = Some(Toast {
            message: message.into(),
            is_error,
            created_at: now,
        });
    }

    fn load_png_texture(ctx: &egui::Context, name: &str, path: &str) -> Option<TextureHandle> {
        let bytes = std::fs::read(path).ok()?;
        let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
        let size = [img.width() as usize, img.height() as usize];
        let color_image = ColorImage::from_rgba_unmultiplied(size, img.as_raw());
        Some(ctx.load_texture(name, color_image, TextureOptions::LINEAR))
    }

    fn poll_worker(&mut self, ctx: &egui::Context) {
        let Some(handle) = &self.worker else {
            return;
        };
        let mut events = Vec::new();
        while let Ok(ev) = handle.rx.try_recv() {
            events.push(ev);
        }

        if !events.is_empty() {
            ctx.request_repaint();
        }

        let mut finish_worker = false;
        for ev in events {
            match ev {
                WorkerEvent::Info(info) => {
                    self.config.preview_frame = self
                        .config
                        .preview_frame
                        .min(info.frame_count.saturating_sub(1));
                    self.video_info = Some(info);
                    finish_worker = true;
                }
                WorkerEvent::ExtractDone { path } => {
                    self.config.video_bgr = path;
                    self.config.save();
                    self.push_toast(ctx, "Fotograma 0 extraído como imagen de fondo.", false);
                    finish_worker = true;
                }
                WorkerEvent::PreviewDone {
                    frame,
                    width,
                    height,
                    src_path,
                    out_path,
                    com_path,
                    pha_path,
                    fgr_path,
                    err_path,
                    ref_path,
                } => {
                    self.preview_src_tex = Self::load_png_texture(ctx, "remback_src", &src_path);
                    let com_file = if com_path.is_empty() {
                        &out_path
                    } else {
                        &com_path
                    };
                    self.preview_com_tex = Self::load_png_texture(ctx, "remback_com", com_file);
                    self.preview_pha_tex = if pha_path.is_empty() {
                        None
                    } else {
                        Self::load_png_texture(ctx, "remback_pha", &pha_path)
                    };
                    self.preview_fgr_tex = if fgr_path.is_empty() {
                        None
                    } else {
                        Self::load_png_texture(ctx, "remback_fgr", &fgr_path)
                    };
                    self.preview_err_tex = if err_path.is_empty() {
                        None
                    } else {
                        Self::load_png_texture(ctx, "remback_err", &err_path)
                    };
                    self.preview_ref_tex = if ref_path.is_empty() {
                        None
                    } else {
                        Self::load_png_texture(ctx, "remback_ref", &ref_path)
                    };
                    self.preview_out_tex = self.preview_com_tex.clone();
                    self.preview_fade_start = Some(ctx.input(|i| i.time));
                    self.preview_busy = false;
                    self.push_toast(
                        ctx,
                        format!("Vista previa generada: fotograma {frame} ({width}x{height})."),
                        false,
                    );
                    finish_worker = true;
                }
                WorkerEvent::Progress {
                    frame,
                    total,
                    fps,
                    eta,
                    preview_path,
                } => {
                    self.export_frame = frame;
                    self.export_total = total;
                    self.export_fps = fps;
                    self.export_eta = eta;
                    if !preview_path.is_empty() {
                        if let Some(tex) =
                            Self::load_png_texture(ctx, "remback_live", &preview_path)
                        {
                            self.preview_com_tex = Some(tex.clone());
                            self.preview_out_tex = Some(tex);
                            if self.preview_fade_start.is_none() {
                                self.preview_fade_start = Some(ctx.input(|i| i.time));
                            }
                        }
                    }
                }
                WorkerEvent::Stage { progress, message } => {
                    if self.preview_busy {
                        self.preview_progress = progress.clamp(0.0, 1.0);
                        self.preview_stage = message;
                    } else if self.export_busy {
                        self.export_stage_progress = progress.clamp(0.0, 1.0);
                        self.export_stage = message;
                    }
                }
                WorkerEvent::ExportDone {
                    output_dir,
                    frames,
                    elapsed,
                } => {
                    self.export_busy = false;
                    self.last_export_dir = Some(output_dir);
                    self.push_toast(
                        ctx,
                        format!("Exportación completada: {frames} fotogramas en {elapsed:.1} s."),
                        false,
                    );
                    finish_worker = true;
                }
                WorkerEvent::Warn(msg) => {
                    self.push_toast(ctx, msg, false);
                }
                WorkerEvent::Error(msg) => {
                    self.preview_busy = false;
                    self.export_busy = false;
                    self.push_toast(ctx, msg, true);
                    finish_worker = true;
                }
            }
        }

        if finish_worker {
            self.worker = None;
        } else if self.preview_busy || self.export_busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        }
    }

    fn poll_updater(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.updater_rx else {
            return;
        };
        let mut events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }
        if !events.is_empty() {
            ctx.request_repaint();
        }

        let mut finish = false;
        for ev in events {
            match ev {
                UpdaterEvent::Log(line) => {
                    self.updater_logs.push(line);
                    if self.updater_logs.len() > 250 {
                        let excess = self.updater_logs.len() - 250;
                        self.updater_logs.drain(0..excess);
                    }
                }
                UpdaterEvent::Progress(p, msg) => {
                    self.updater_progress = p;
                    self.updater_status = msg;
                }
                UpdaterEvent::CheckFinished(res) => {
                    self.checking_updates = false;
                    finish = true;
                    if let Some(app_rel) = res.app_release {
                        self.app_update_available = is_newer_version(APP_VERSION, &app_rel.version);
                        self.remote_app_release = Some(app_rel);
                    }
                    if let Some(remote) = res.bgmv2 {
                        let outdated = !self.paths.is_bgmv2_installed()
                            || (!self.installed_state.commit_sha.is_empty()
                                && self.installed_state.commit_sha != remote.commit_sha);
                        self.update_available = outdated;
                        self.remote_info = Some(remote);
                    } else if let Some(err) = res.bgmv2_error {
                        if self.show_updates_modal {
                            self.push_toast(ctx, err, true);
                        }
                    }
                }
                UpdaterEvent::AppUpdateReadyToExit => {
                    self.updater_busy = false;
                    finish = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                UpdaterEvent::AppUpdateError(err) => {
                    self.updater_busy = false;
                    finish = true;
                    self.updater_status = err.clone();
                    self.push_toast(ctx, err, true);
                }
                UpdaterEvent::Bgmv2Updated(res) => {
                    self.updater_busy = false;
                    finish = true;
                    match res {
                        Ok(new_state) => {
                            self.installed_state = new_state;
                            self.update_available = false;
                            self.push_toast(
                                ctx,
                                "BackgroundMattingV2 actualizado correctamente.",
                                false,
                            );
                        }
                        Err(e) => {
                            self.updater_status = e.clone();
                            self.push_toast(ctx, e, true);
                        }
                    }
                }
                UpdaterEvent::EnvSetupFinished(res) => {
                    self.updater_busy = false;
                    finish = true;
                    match res {
                        Ok(()) => {
                            self.push_toast(
                                ctx,
                                "Entorno Python y PyTorch instalados correctamente.",
                                false,
                            );
                        }
                        Err(e) => {
                            self.updater_status = e.clone();
                            self.push_toast(ctx, e, true);
                        }
                    }
                }
                UpdaterEvent::ModelDownloaded(res) => {
                    self.updater_busy = false;
                    finish = true;
                    match res {
                        Ok(path) => {
                            self.local_models = self.paths.list_local_models();
                            let fname = path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .into_owned();
                            if fname.ends_with(".torchscript") {
                                self.config.engine_mode = EngineMode::AutoNoRef;
                                self.config.simple_quality = if fname.contains("mobilenet") {
                                    SimpleQuality::Fast
                                } else {
                                    SimpleQuality::Balanced
                                };
                            } else {
                                self.config.model_checkpoint = path.to_string_lossy().into_owned();
                                self.sync_backbone_from_checkpoint();
                            }
                            self.config.save();
                            self.push_toast(
                                ctx,
                                format!("Modelo descargado y activado: {fname}"),
                                false,
                            );
                        }
                        Err(e) => {
                            self.updater_status = e.clone();
                            self.push_toast(ctx, e, true);
                        }
                    }
                }
            }
        }

        if finish {
            self.updater_rx = None;
        } else if self.updater_busy || self.checking_updates {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }

    fn engine_ready(&self) -> bool {
        match self.config.engine_mode {
            EngineMode::AutoNoRef => self.paths.is_venv_ready(),
            EngineMode::BgMattingV2 => {
                self.paths.is_bgmv2_installed() && self.paths.is_venv_ready()
            }
        }
    }

    fn sync_backbone_from_checkpoint(&mut self) {
        if let Some(detected) = Backbone::detect_from_path(&self.config.model_checkpoint) {
            self.config.model_backbone = detected;
        }
    }

    fn active_preview_out_tex(&self) -> Option<&TextureHandle> {
        let preferred = match self.config.preview_layer {
            OutputLayer::Com => self.preview_com_tex.as_ref(),
            OutputLayer::Pha => self.preview_pha_tex.as_ref(),
            OutputLayer::Fgr => self.preview_fgr_tex.as_ref(),
            OutputLayer::Err => self.preview_err_tex.as_ref(),
            OutputLayer::Ref => self.preview_ref_tex.as_ref(),
        };
        preferred
            .or(self.preview_com_tex.as_ref())
            .or(self.preview_out_tex.as_ref())
    }

    fn render_mask_tuning_card(&mut self, ui: &mut Ui) {
        glass_card(ui, "card_mask_tuning", "Ajuste de Máscara y Tolerancia", None, |ui| {
            let mut changed = false;

            let mut tol_pct = (self.config.mask_tolerance * 100.0).round() as u32;
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Tolerancia fondo:")
                        .size(12.5)
                        .color(Palette::TEXT_SECONDARY),
                );
                let sl = smooth_slider(ui, &mut tol_pct, 0..=95, "%");
                if sl.changed() {
                    self.config.mask_tolerance = (tol_pct as f32) / 100.0;
                    changed = true;
                }
                smooth_tooltip(
                    ui,
                    &sl,
                    "Tolerancia al fondo (Limpieza)",
                    "Sube este porcentaje para eliminar halos o restos semitransparentes del fondo alrededor del sujeto (25% recomendado).",
                );
            });

            let mut soft_pct = (self.config.mask_softness * 100.0).round() as u32;
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Suavizado borde:")
                        .size(12.5)
                        .color(Palette::TEXT_SECONDARY),
                );
                let sl = smooth_slider(ui, &mut soft_pct, 5..=100, "%");
                if sl.changed() {
                    self.config.mask_softness = (soft_pct as f32) / 100.0;
                    changed = true;
                }
                smooth_tooltip(
                    ui,
                    &sl,
                    "Suavizado del contorno",
                    "Valores bajos (10–25%) dan un recorte más nítido y firme; valores altos (50–100%) suavizan el borde.",
                );
            });

            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Expansión borde:")
                        .size(12.5)
                        .color(Palette::TEXT_SECONDARY),
                );
                let sl = smooth_slider(ui, &mut self.config.mask_expand, -10..=10, " px");
                if sl.changed() {
                    changed = true;
                }
                smooth_tooltip(
                    ui,
                    &sl,
                    "Expandir o contraer máscara",
                    "Negativo (-1 a -4 px) contrae el borde para quitar halos de color; positivo expande la silueta.",
                );
            });

            let clean_tog = smooth_toggle(
                ui,
                &mut self.config.mask_clean_artifacts,
                "Eliminar artefactos sueltos (manchas/islas)",
            );
            if clean_tog.changed() {
                changed = true;
            }
            smooth_tooltip(
                ui,
                &clean_tog,
                "Limpieza automática de artefactos",
                "Elimina puntos, manchas o partes falsas del fondo desconectadas de la persona principal.",
            );

            ui.horizontal(|ui| {
                if secondary_button(ui, "Restaurar valores", true).clicked() {
                    self.config.mask_tolerance = 0.25;
                    self.config.mask_softness = 0.25;
                    self.config.mask_expand = 0;
                    self.config.mask_clean_artifacts = true;
                    changed = true;
                }
                ui.label(
                    RichText::new("Pulsa 'Vista Previa' para ver el ajuste")
                        .size(11.0)
                        .color(Palette::TEXT_MUTED),
                );
            });

            if changed {
                self.config.save();
            }
        });
    }

    fn start_probe_video(&mut self, ctx: &egui::Context) {
        if self.config.video_src.trim().is_empty() {
            return;
        }
        if !self.paths.is_venv_ready() {
            return;
        }
        if self.worker.is_some() {
            return;
        }
        match run_probe_video(&self.paths, &self.config.video_src) {
            Ok(h) => self.worker = Some(h),
            Err(e) => self.push_toast(ctx, e, true),
        }
    }

    fn start_extract_frame0(&mut self, ctx: &egui::Context) {
        if self.config.video_src.trim().is_empty() || !Path::new(&self.config.video_src).is_file() {
            self.push_toast(ctx, "Selecciona primero el video de origen.", true);
            return;
        }
        if self.worker.is_some() {
            self.push_toast(ctx, "Hay otra operación en curso.", true);
            return;
        }
        let out_png = self.paths.preview_dir.join("extracted_bgr_frame0.png");
        match run_extract_frame(&self.paths, &self.config.video_src, 0, &out_png) {
            Ok(h) => self.worker = Some(h),
            Err(e) => self.push_toast(ctx, e, true),
        }
    }

    fn start_preview(&mut self, ctx: &egui::Context) {
        if self.worker.is_some() {
            return;
        }
        if let Err(e) = self.config.validate(false) {
            self.push_toast(ctx, e, true);
            return;
        }
        self.config.save();
        match run_preview(&self.paths, &self.config) {
            Ok(h) => {
                self.preview_busy = true;
                self.preview_progress = 0.04;
                self.preview_stage = "Iniciando motor...".into();
                self.worker = Some(h);
            }
            Err(e) => self.push_toast(ctx, e, true),
        }
    }

    fn start_export(&mut self, ctx: &egui::Context) {
        if self.worker.is_some() {
            return;
        }
        if let Err(e) = self.config.validate(true) {
            self.push_toast(ctx, e, true);
            return;
        }
        self.config.save();
        match run_export(&self.paths, &self.config) {
            Ok(h) => {
                self.export_busy = true;
                self.export_stage_progress = 0.03;
                self.export_stage = "Iniciando exportación...".into();
                self.export_frame = 0;
                self.export_total = self
                    .video_info
                    .as_ref()
                    .map(|i| i.frame_count)
                    .unwrap_or(100);
                self.export_fps = 0.0;
                self.export_eta = 0.0;
                self.last_export_dir = None;
                self.worker = Some(h);
            }
            Err(e) => self.push_toast(ctx, e, true),
        }
    }

    fn open_folder(path: &str) {
        let _ = Command::new("explorer").arg(path).spawn();
    }

    fn open_url(url: &str) {
        let _ = Command::new("cmd").args(["/C", "start", "", url]).spawn();
    }

    fn render_header(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        let time = ctx.input(|i| i.time) as f32;
        Frame::new()
            .fill(Palette::BG_SURFACE)
            .stroke(Stroke::new(1.0_f32, Palette::BORDER_SUBTLE))
            .inner_margin(Margin::symmetric(16, 10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    // Animated Logo
                    let (logo_rect, logo_resp) =
                        ui.allocate_exact_size(vec2(34.0, 34.0), Sense::hover());
                    let logo_glow = if self.export_busy || self.preview_busy {
                        0.75 + 0.25 * (time * 5.0).sin()
                    } else if logo_resp.hovered() {
                        0.85
                    } else {
                        0.4
                    };
                    paint_logo(
                        ui.painter(),
                        logo_rect.center(),
                        16.0,
                        time * if self.export_busy { 2.2 } else { 0.8 },
                        1.0,
                        logo_glow,
                    );
                    if self.export_busy || self.preview_busy || logo_resp.hovered() {
                        ctx.request_repaint();
                    }
                    smooth_tooltip(
                        ui,
                        &logo_resp,
                        "RemBack",
                        "Interfaz moderna en Rust para PeterL1n/BackgroundMattingV2.",
                    );

                    ui.add_space(6.0);
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 1.0;
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("RemBack")
                                    .size(17.5)
                                    .strong()
                                    .color(Palette::TEXT_PRIMARY),
                            );
                            badge_pill(ui, &format!("v{APP_VERSION}"), Palette::EMERALD_BRIGHT);
                        });
                        ui.label(
                            RichText::new("GUI para BackgroundMattingV2")
                                .size(11.5)
                                .color(Palette::TEXT_MUTED),
                        );
                    });

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let about_resp = secondary_button(ui, "Acerca de", true);
                        smooth_tooltip(
                            ui,
                            &about_resp,
                            "Acerca de RemBack",
                            "Información de versión y enlaces al repositorio oficial de BackgroundMattingV2.",
                        );
                        if about_resp.clicked() {
                            self.show_about_modal = true;
                        }

                        let upd_label = if self.app_update_available
                            || self.update_available
                            || !self.engine_ready()
                        {
                            "● Actualizaciones"
                        } else {
                            "Actualizaciones"
                        };
                        let upd_resp = secondary_button(ui, upd_label, true);
                        smooth_tooltip(
                            ui,
                            &upd_resp,
                            "Actualizador y Entorno",
                            "Busca actualizaciones de RemBack y BackgroundMattingV2 en GitHub, configura PyTorch/CUDA y descarga los modelos.",
                        );
                        if upd_resp.clicked() {
                            self.show_updates_modal = true;
                        }

                        if self.app_update_available {
                            if let Some(rel) = self.remote_app_release.clone() {
                                let app_upd_btn = primary_button(
                                    ui,
                                    &format!("⬆ Actualizar v{}", rel.version),
                                    !self.updater_busy,
                                );
                                smooth_tooltip(
                                    ui,
                                    &app_upd_btn,
                                    "Nueva versión de RemBack disponible",
                                    "Descarga e instala la actualización automáticamente desde GitHub.",
                                );
                                if app_upd_btn.clicked() {
                                    self.show_updates_modal = true;
                                    self.updater_busy = true;
                                    self.updater_progress = 0.03;
                                    self.updater_logs.clear();
                                    self.updater_rx = Some(download_and_launch_app_update(rel));
                                }
                            }
                        }

                        if self.engine_ready() {
                            badge_pill(ui, "Motor Listo", Palette::EMERALD);
                        } else {
                            let warn_resp = ui.add(
                                egui::Button::new(
                                    RichText::new("⚠ Configurar Motor")
                                        .size(12.0)
                                        .strong()
                                        .color(Palette::WARNING),
                                )
                                .fill(Color32::from_rgba_unmultiplied(245, 158, 11, 28))
                                .stroke(Stroke::new(1.0_f32, Palette::WARNING))
                                .corner_radius(CornerRadius::same(10)),
                            );
                            smooth_tooltip(
                                ui,
                                &warn_resp,
                                "Configuración inicial requerida",
                                "Haz clic para descargar BackgroundMattingV2, preparar Python/PyTorch y bajar un modelo .pth.",
                            );
                            if warn_resp.clicked() {
                                self.show_updates_modal = true;
                            }
                        }
                    });
                });
            });
    }

    fn render_video_src_and_bg_target(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        ui.label(
            RichText::new("Video de origen")
                .size(12.5)
                .color(Palette::TEXT_SECONDARY),
        );
        ui.horizontal(|ui| {
            let edit = ui.add(
                egui::TextEdit::singleline(&mut self.config.video_src)
                    .desired_width(ui.available_width() - 88.0)
                    .hint_text("Selecciona un video (.mp4, .mov...)"),
            );
            smooth_tooltip(
                ui,
                &edit,
                "Video de origen (--video-src)",
                "Archivo de video con el sujeto cuyo fondo deseas extraer o reemplazar.",
            );
            let btn = secondary_button(ui, "Elegir...", true);
            smooth_tooltip(
                ui,
                &btn,
                "Seleccionar video",
                "Abre el explorador para elegir el video principal.",
            );
            if btn.clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("Video", &["mp4", "mov", "avi", "mkv", "webm"])
                    .pick_file()
                {
                    self.config.video_src = p.to_string_lossy().into_owned();
                    self.config.save();
                    self.start_probe_video(ctx);
                }
            }
        });

        if let Some(info) = &self.video_info {
            ui.label(
                RichText::new(format!(
                    "{}x{} · {:.1} fps · {} fotogramas",
                    info.width, info.height, info.fps, info.frame_count
                ))
                .size(11.0)
                .color(Palette::CYAN),
            );
        }
    }

    fn render_replacement_and_output_dir(&mut self, ui: &mut Ui) {
        ui.add_space(4.0);
        ui.label(
            RichText::new("Modo de reemplazo")
                .size(12.5)
                .color(Palette::TEXT_SECONDARY),
        );
        let bg_seg = segmented_control(
            ui,
            "bg_mode_seg",
            &mut self.config.bg_mode,
            &[
                (BgMode::GreenScreen, "Verde"),
                (BgMode::Transparent, "Sin fondo"),
                (BgMode::CustomColor, "Color"),
                (BgMode::TargetMedia, "Media"),
            ],
        );
        smooth_tooltip(
            ui,
            &bg_seg,
            "Fondo de salida",
            "Verde: pantalla verde clásica (#78FF9B).\nSin fondo: transparencia alpha (PNG RGBA).\nColor: color sólido personalizado.\nMedia: video o imagen (--video-target-bgr).",
        );

        match self.config.bg_mode {
            BgMode::GreenScreen => {
                ui.label(
                    RichText::new("Pantalla verde estándar RGB(120, 255, 155)")
                        .size(11.0)
                        .color(Palette::TEXT_MUTED),
                );
            }
            BgMode::Transparent => {
                ui.label(
                    RichText::new("Exportará secuencia PNG RGBA con canal alpha real.")
                        .size(11.0)
                        .color(Palette::EMERALD_BRIGHT),
                );
            }
            BgMode::CustomColor => {
                ui.horizontal(|ui| {
                    ui.label("Color sólido:");
                    ui.color_edit_button_srgb(&mut self.config.chroma_color);
                });
            }
            BgMode::TargetMedia => {
                ui.horizontal(|ui| {
                    let edit = ui.add(
                        egui::TextEdit::singleline(&mut self.config.video_target_bgr)
                            .desired_width(ui.available_width() - 88.0)
                            .hint_text("Video o imagen de nuevo fondo"),
                    );
                    smooth_tooltip(
                        ui,
                        &edit,
                        "Fondo destino (--video-target-bgr)",
                        "Video o imagen sobre el cual se compondrá el sujeto extraído.",
                    );
                    let btn = secondary_button(ui, "Elegir...", true);
                    if btn.clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("Media", &["mp4", "mov", "avi", "png", "jpg", "jpeg"])
                            .pick_file()
                        {
                            self.config.video_target_bgr = p.to_string_lossy().into_owned();
                            self.config.save();
                        }
                    }
                });
            }
        }

        ui.add_space(4.0);
        ui.label(
            RichText::new("Carpeta de salida")
                .size(12.5)
                .color(Palette::TEXT_SECONDARY),
        );
        ui.horizontal(|ui| {
            let edit = ui.add(
                egui::TextEdit::singleline(&mut self.config.output_dir)
                    .desired_width(ui.available_width() - 88.0),
            );
            smooth_tooltip(
                ui,
                &edit,
                "Directorio de salida (--output-dir)",
                "Carpeta donde se guardarán los videos o secuencias exportadas.",
            );
            let btn = secondary_button(ui, "Elegir...", true);
            if btn.clicked() {
                if let Some(p) = rfd::FileDialog::new().pick_folder() {
                    self.config.output_dir = p.to_string_lossy().into_owned();
                    self.config.save();
                }
            }
        });
    }

    fn render_bgmv2_bgr_picker(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        ui.label(
            RichText::new("Fondo limpio (referencia)")
                .size(12.5)
                .color(Palette::TEXT_SECONDARY),
        );
        ui.horizontal(|ui| {
            let edit = ui.add(
                egui::TextEdit::singleline(&mut self.config.video_bgr)
                    .desired_width(ui.available_width() - 88.0)
                    .hint_text("Imagen de fondo sin el sujeto"),
            );
            smooth_tooltip(
                ui,
                &edit,
                "Fondo estático (--video-bgr)",
                "Foto del mismo escenario sin la persona/sujeto para lograr precisión milimétrica.",
            );
            let btn = secondary_button(ui, "Elegir...", true);
            if btn.clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("Imagen", &["png", "jpg", "jpeg", "bmp", "webp"])
                    .pick_file()
                {
                    self.config.video_bgr = p.to_string_lossy().into_owned();
                    self.config.save();
                }
            }
        });

        let extract_btn =
            secondary_button(ui, "Extraer Frame 0 del video", !self.config.video_src.is_empty());
        smooth_tooltip(
            ui,
            &extract_btn,
            "Extraer primer fotograma",
            "Extrae automáticamente el fotograma 0 del video por si el inicio está limpio sin el sujeto.",
        );
        if extract_btn.clicked() {
            self.start_extract_frame0(ctx);
        }
    }

    fn render_checkpoint_picker(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("Checkpoint (.pth)")
                .size(12.5)
                .color(Palette::TEXT_SECONDARY),
        );
        if !self.local_models.is_empty() {
            let current_name = Path::new(&self.config.model_checkpoint)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "Elegir descargado...".into());
            let mut changed_ckpt = false;
            egui::ComboBox::from_id_salt("local_model_combo")
                .width(ui.available_width())
                .selected_text(current_name)
                .show_ui(ui, |ui| {
                    for m in &self.local_models {
                        let m_str = m.to_string_lossy().into_owned();
                        let label = m
                            .file_name()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| m_str.clone());
                        if ui
                            .selectable_value(&mut self.config.model_checkpoint, m_str, label)
                            .changed()
                        {
                            changed_ckpt = true;
                        }
                    }
                });
            if changed_ckpt {
                self.sync_backbone_from_checkpoint();
                self.config.save();
            }
        }
        ui.horizontal(|ui| {
            let edit = ui.add(
                egui::TextEdit::singleline(&mut self.config.model_checkpoint)
                    .desired_width(ui.available_width() - 88.0)
                    .hint_text("Ruta al archivo .pth"),
            );
            smooth_tooltip(
                ui,
                &edit,
                "Pesos del modelo (--model-checkpoint)",
                "Archivo .pth preentrenado de BackgroundMattingV2. Puedes descargarlos con 1 clic en Actualizaciones.",
            );
            let btn = secondary_button(ui, "Buscar...", true);
            if btn.clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("PyTorch Checkpoint", &["pth", "pt"])
                    .pick_file()
                {
                    self.config.model_checkpoint = p.to_string_lossy().into_owned();
                    self.sync_backbone_from_checkpoint();
                    self.config.save();
                }
            }
        });
    }

    fn render_left_sidebar(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 10.0;

                // Mode Switcher: Simple vs Avanzado
                let mode_seg = segmented_control(
                    ui,
                    "ui_mode_seg",
                    &mut self.config.ui_mode,
                    &[
                        (UiMode::Simple, "Modo Simple"),
                        (UiMode::Advanced, "Modo Avanzado"),
                    ],
                );
                if mode_seg.changed() {
                    if self.config.ui_mode == UiMode::Simple {
                        self.config.preview_layer = OutputLayer::Com;
                    }
                    self.config.save();
                }
                smooth_tooltip(
                    ui,
                    &mode_seg,
                    "Nivel de controles",
                    "Modo Simple usa ajustes óptimos listos para casi cualquier video.\nModo Avanzado expone todos los parámetros técnicos de BackgroundMattingV2 y RVM.",
                );

                if self.config.ui_mode == UiMode::Simple {
                    glass_card(ui, "card_simple_files", "1. Video y Fondo", None, |ui| {
                        self.render_video_src_and_bg_target(ui, ctx);
                        self.render_replacement_and_output_dir(ui);
                    });

                    glass_card(ui, "card_simple_quality", "2. Modelo y Calidad", None, |ui| {
                        #[derive(Clone, Copy, PartialEq, Eq)]
                        enum SimpleProfile {
                            FastRvm,
                            BalancedRvm,
                            Bgmv2ResNet101,
                        }

                        let mut profile = match self.config.engine_mode {
                            EngineMode::AutoNoRef => match self.config.simple_quality {
                                SimpleQuality::Fast => SimpleProfile::FastRvm,
                                SimpleQuality::Balanced => SimpleProfile::BalancedRvm,
                            },
                            EngineMode::BgMattingV2 => SimpleProfile::Bgmv2ResNet101,
                        };

                        ui.label(
                            RichText::new("Elige el modelo a utilizar")
                                .size(12.5)
                                .color(Palette::TEXT_SECONDARY),
                        );
                        let prof_seg = segmented_control(
                            ui,
                            "simple_profile_seg",
                            &mut profile,
                            &[
                                (SimpleProfile::FastRvm, "MobileNetV3"),
                                (SimpleProfile::BalancedRvm, "ResNet50"),
                                (SimpleProfile::Bgmv2ResNet101, "ResNet101 (BGMv2)"),
                            ],
                        );
                        if prof_seg.changed() {
                            match profile {
                                SimpleProfile::FastRvm => {
                                    self.config.engine_mode = EngineMode::AutoNoRef;
                                    self.config.simple_quality = SimpleQuality::Fast;
                                }
                                SimpleProfile::BalancedRvm => {
                                    self.config.engine_mode = EngineMode::AutoNoRef;
                                    self.config.simple_quality = SimpleQuality::Balanced;
                                }
                                SimpleProfile::Bgmv2ResNet101 => {
                                    self.config.engine_mode = EngineMode::BgMattingV2;
                                    self.config.model_type = ModelType::MattingRefine;
                                    self.config.model_backbone = Backbone::ResNet101;
                                    if let Some(p) = self
                                        .local_models
                                        .iter()
                                        .find(|m| m.to_string_lossy().contains("resnet101"))
                                    {
                                        self.config.model_checkpoint =
                                            p.to_string_lossy().into_owned();
                                    }
                                }
                            }
                            self.config.save();
                        }
                        smooth_tooltip(
                            ui,
                            &prof_seg,
                            "Modelo neuronal",
                            "MobileNetV3 y ResNet50 (RVM): automáticos sin foto de fondo.\nResNet101 (BackgroundMattingV2): usa tu checkpoint .pth descargado (con foto limpia o reconstrucción auto).",
                        );

                        ui.label(
                            RichText::new(format!(
                                "Modelo activo: {}",
                                self.config.active_model_display()
                            ))
                            .size(11.5)
                            .color(Palette::EMERALD_BRIGHT),
                        );

                        if self.config.engine_mode == EngineMode::BgMattingV2 {
                            let auto_tog = smooth_toggle(
                                ui,
                                &mut self.config.auto_bgr,
                                "Sin foto previa (reconstruir fondo auto)",
                            );
                            if auto_tog.changed() {
                                self.config.save();
                            }
                            if !self.config.auto_bgr {
                                self.render_bgmv2_bgr_picker(ui, ctx);
                            }
                            self.render_checkpoint_picker(ui);
                        }

                        ui.add_space(4.0);
                        ui.label(
                            RichText::new("Dispositivo de cómputo")
                                .size(12.5)
                                .color(Palette::TEXT_SECONDARY),
                        );
                        let dev_seg = segmented_control(
                            ui,
                            "simple_dev_seg",
                            &mut self.config.device,
                            &[
                                (ComputeDevice::Cuda, "CUDA (GPU)"),
                                (ComputeDevice::Cpu, "CPU"),
                            ],
                        );
                        if dev_seg.changed() {
                            self.config.save();
                        }
                        smooth_tooltip(
                            ui,
                            &dev_seg,
                            "Aceleración por hardware",
                            "CUDA usa tu tarjeta gráfica NVIDIA para procesar mucho más rápido (FP16 automático).",
                        );
                    });

                    self.render_mask_tuning_card(ui);
                    return;
                }

                // ---------------- MODO AVANZADO ----------------

                // 1. Archivos y Fondo
                glass_card(ui, "card_files", "Archivos y Fondo", None, |ui| {
                    self.render_video_src_and_bg_target(ui, ctx);

                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("Motor neuronal")
                            .size(12.5)
                            .color(Palette::TEXT_SECONDARY),
                    );
                    let eng_seg = segmented_control(
                        ui,
                        "adv_engine_seg",
                        &mut self.config.engine_mode,
                        &[
                            (EngineMode::AutoNoRef, "Auto sin fondo (RVM)"),
                            (EngineMode::BgMattingV2, "BackgroundMattingV2"),
                        ],
                    );
                    if eng_seg.changed() {
                        self.config.save();
                    }
                    smooth_tooltip(
                        ui,
                        &eng_seg,
                        "Motor de segmentación",
                        "Auto sin fondo (RVM): red recurrente TorchScript que no necesita imagen de fondo.\nBackgroundMattingV2: usa foto limpia del fondo o reconstrucción temporal.",
                    );

                    if self.config.engine_mode == EngineMode::BgMattingV2 {
                        ui.add_space(4.0);
                        let auto_tog = smooth_toggle(
                            ui,
                            &mut self.config.auto_bgr,
                            "Sin imagen de referencia (Reconstrucción Auto)",
                        );
                        if auto_tog.changed() {
                            self.config.save();
                        }
                        smooth_tooltip(
                            ui,
                            &auto_tog,
                            "Fondo automático en BackgroundMattingV2",
                            "Reconstruye el fondo limpio mediante muestreo y segmentación temporal cuando no tienes foto vacía.",
                        );

                        if self.config.auto_bgr {
                            ui.label(
                                RichText::new("Se reconstruirá el fondo limpio automáticamente desde el video.")
                                    .size(11.0)
                                    .color(Palette::EMERALD_BRIGHT),
                            );
                        } else {
                            self.render_bgmv2_bgr_picker(ui, ctx);
                        }
                    }

                    self.render_replacement_and_output_dir(ui);
                });

                // 2. Modelo y Hardware
                glass_card(ui, "card_model", "Modelo y Hardware", None, |ui| {
                    if self.config.engine_mode == EngineMode::AutoNoRef {
                        ui.label(
                            RichText::new("Arquitectura RVM")
                                .size(12.5)
                                .color(Palette::TEXT_SECONDARY),
                        );
                        let q_seg = segmented_control(
                            ui,
                            "adv_rvm_q_seg",
                            &mut self.config.simple_quality,
                            &[
                                (SimpleQuality::Balanced, "ResNet50 (Alta Calidad)"),
                                (SimpleQuality::Fast, "MobileNetV3 (Rápido)"),
                            ],
                        );
                        smooth_tooltip(
                            ui,
                            &q_seg,
                            "Modelo TorchScript RVM",
                            "ResNet50 logra máxima precisión en bordes complejos; MobileNetV3 maximiza FPS.",
                        );
                    } else {
                        ui.label(RichText::new("Tipo de modelo").size(12.5).color(Palette::TEXT_SECONDARY));
                        let mt_seg = segmented_control(
                            ui,
                            "model_type_seg",
                            &mut self.config.model_type,
                            &[
                                (ModelType::MattingRefine, "MattingRefine (HD/4K)"),
                                (ModelType::MattingBase, "MattingBase (Rápido)"),
                            ],
                        );
                        smooth_tooltip(
                            ui,
                            &mt_seg,
                            "Tipo de red (--model-type)",
                            "MattingRefine añade refinamiento de alta resolución en bordes y cabello. MattingBase solo ejecuta la red base.",
                        );

                        ui.label(RichText::new("Backbone").size(12.5).color(Palette::TEXT_SECONDARY));
                        let bb_seg = segmented_control(
                            ui,
                            "backbone_seg",
                            &mut self.config.model_backbone,
                            &[
                                (Backbone::MobileNetV2, "MobileNetV2"),
                                (Backbone::ResNet50, "ResNet50"),
                                (Backbone::ResNet101, "ResNet101"),
                            ],
                        );
                        smooth_tooltip(
                            ui,
                            &bb_seg,
                            "Arquitectura (--model-backbone)",
                            "MobileNetV2 es muy rápido; ResNet50 ofrece el mejor balance; ResNet101 da máxima fidelidad. Se sincroniza automáticamente al elegir un .pth.",
                        );

                        self.render_checkpoint_picker(ui);

                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Escala backbone:").size(12.5).color(Palette::TEXT_SECONDARY));
                            let sl = smooth_slider(
                                ui,
                                &mut self.config.model_backbone_scale,
                                0.125..=1.0,
                                "x",
                            );
                            smooth_tooltip(
                                ui,
                                &sl,
                                "Escala del backbone (--model-backbone-scale)",
                                "Factor de resolución de la pasada base (0.25 recomendado para 1080p, 0.125 para 4K). Menor = más rápido.",
                            );
                        });
                    }

                    ui.label(RichText::new("Dispositivo de cómputo").size(12.5).color(Palette::TEXT_SECONDARY));
                    let dev_seg = segmented_control(
                        ui,
                        "device_seg",
                        &mut self.config.device,
                        &[
                            (ComputeDevice::Cuda, "CUDA (GPU)"),
                            (ComputeDevice::Cpu, "CPU"),
                        ],
                    );
                    smooth_tooltip(
                        ui,
                        &dev_seg,
                        "Dispositivo (--device)",
                        "CUDA aprovecha tu GPU NVIDIA para procesar decenas de veces más rápido que CPU.",
                    );

                    let fp16_tog = smooth_toggle(
                        ui,
                        &mut self.config.use_fp16,
                        "Precisión FP16 en GPU (2x más rápido)",
                    );
                    smooth_tooltip(
                        ui,
                        &fp16_tog,
                        "Aceleración media precisión (FP16)",
                        "Ejecuta la inferencia en float16 sobre GPU CUDA, reduciendo a la mitad el uso de VRAM y duplicando la velocidad sin pérdida visual.",
                    );

                    if self.config.engine_mode == EngineMode::BgMattingV2 {
                        let align_tog = smooth_toggle(
                            ui,
                            &mut self.config.preprocess_alignment,
                            "Alineación homográfica (cámara en mano)",
                        );
                        smooth_tooltip(
                            ui,
                            &align_tog,
                            "Preprocesado (--preprocess-alignment)",
                            "Alinea el fondo con la toma si la cámara tiene ligeros movimientos o vibraciones respecto a la foto limpia.",
                        );
                    }

                    let res_tog = smooth_toggle(
                        ui,
                        &mut self.config.resize_enabled,
                        "Redimensionar resolución de salida",
                    );
                    smooth_tooltip(
                        ui,
                        &res_tog,
                        "Resolución (--video-resize)",
                        "Cambia la resolución de entrada/salida antes de inferir para acelerar o estandarizar tamaño.",
                    );

                    let resize_anim = ui.ctx().animate_bool_with_time(
                        egui::Id::new("resize_expand"),
                        self.config.resize_enabled,
                        0.18,
                    );
                    if resize_anim > 0.02 {
                        ui.horizontal(|ui| {
                            ui.add(
                                DragValue::new(&mut self.config.resize_width)
                                    .range(64..=7680)
                                    .prefix("W: "),
                            );
                            ui.add(
                                DragValue::new(&mut self.config.resize_height)
                                    .range(64..=4320)
                                    .prefix("H: "),
                            );
                            if secondary_button(ui, "1080p", true).clicked() {
                                self.config.resize_width = 1920;
                                self.config.resize_height = 1080;
                            }
                            if secondary_button(ui, "720p", true).clicked() {
                                self.config.resize_width = 1280;
                                self.config.resize_height = 720;
                            }
                        });
                    }
                });

                // 3. Refinamiento (solo BackgroundMattingV2 + MattingRefine)
                let is_refine = self.config.engine_mode == EngineMode::BgMattingV2
                    && self.config.model_type == ModelType::MattingRefine;
                if self.config.engine_mode == EngineMode::BgMattingV2 {
                    glass_card(
                        ui,
                        "card_refine",
                        "Refinamiento",
                        Some(if is_refine { "Activo" } else { "Solo Refine" }),
                        |ui| {
                            ui.add_enabled_ui(is_refine, |ui| {
                                ui.label(
                                    RichText::new("Modo de refinamiento")
                                        .size(12.5)
                                        .color(Palette::TEXT_SECONDARY),
                                );
                                let rm_seg = segmented_control(
                                    ui,
                                    "refine_mode_seg",
                                    &mut self.config.refine_mode,
                                    &[
                                        (RefineMode::Sampling, "Sampling"),
                                        (RefineMode::Thresholding, "Threshold"),
                                        (RefineMode::Full, "Full"),
                                    ],
                                );
                                smooth_tooltip(
                                    ui,
                                    &rm_seg,
                                    "Modo (--model-refine-mode)",
                                    "Sampling: refina los N píxeles con mayor error.\nThresholding: refina píxeles por encima del umbral.\nFull: refina toda la imagen (más lento).",
                                );

                                match self.config.refine_mode {
                                    RefineMode::Sampling => {
                                        ui.horizontal(|ui| {
                                            ui.label("Píxeles:");
                                            let sl = smooth_slider(
                                                ui,
                                                &mut self.config.refine_sample_pixels,
                                                10_000..=300_000,
                                                " px",
                                            );
                                            smooth_tooltip(
                                                ui,
                                                &sl,
                                                "Píxeles muestreados (--model-refine-sample-pixels)",
                                                "Cantidad de parches de alta resolución refinados por frame (80,000 recomendado).",
                                            );
                                        });
                                    }
                                    RefineMode::Thresholding => {
                                        ui.horizontal(|ui| {
                                            ui.label("Umbral:");
                                            let sl = smooth_slider(
                                                ui,
                                                &mut self.config.refine_threshold,
                                                0.05..=0.95,
                                                "",
                                            );
                                            smooth_tooltip(
                                                ui,
                                                &sl,
                                                "Umbral de error (--model-refine-threshold)",
                                                "Sensibilidad para seleccionar regiones a refinar (0.7 por defecto).",
                                            );
                                        });
                                    }
                                    RefineMode::Full => {
                                        ui.label(
                                            RichText::new("Refina el 100% de la resolución.")
                                                .size(11.5)
                                                .color(Palette::TEXT_MUTED),
                                        );
                                    }
                                }

                                ui.label(RichText::new("Tamaño de kernel").size(12.5).color(Palette::TEXT_SECONDARY));
                                let k_seg = segmented_control(
                                    ui,
                                    "kernel_seg",
                                    &mut self.config.refine_kernel_size,
                                    &[(1, "Kernel 1x1"), (3, "Kernel 3x3")],
                                );
                                smooth_tooltip(
                                    ui,
                                    &k_seg,
                                    "Kernel (--model-refine-kernel-size)",
                                    "Tamaño de convolución del módulo refinador (3 recomendado en ResNet, 1 opcional en MobileNetV2).",
                                );
                            });
                        },
                    );
                }

                // 4. Ajuste de máscara
                self.render_mask_tuning_card(ui);

                // 5. Salida y Formato
                glass_card(ui, "card_output", "Salidas y Formato", None, |ui| {
                    ui.label(RichText::new("Formato de exportación").size(12.5).color(Palette::TEXT_SECONDARY));
                    let fmt_seg = segmented_control(
                        ui,
                        "out_fmt_seg",
                        &mut self.config.output_format,
                        &[
                            (OutputFormat::Video, "Video (.mp4)"),
                            (OutputFormat::ImageSequences, "Secuencia PNG/JPG"),
                        ],
                    );
                    smooth_tooltip(
                        ui,
                        &fmt_seg,
                        "Formato (--output-format)",
                        "Video genera archivos .mp4 (y PNG RGBA si el fondo es 'Sin fondo'). Secuencia guarda cada fotograma numerado.",
                    );

                    ui.add_space(4.0);
                    ui.label(RichText::new("Capas a exportar (--output-types)").size(12.5).color(Palette::TEXT_SECONDARY));
                    ui.horizontal_wrapped(|ui| {
                        let r1 = smooth_toggle(ui, &mut self.config.out_com, "com");
                        smooth_tooltip(ui, &r1, "Composición (com)", "Sujeto compuesto sobre el fondo elegido o RGBA transparente.");
                        let r2 = smooth_toggle(ui, &mut self.config.out_pha, "pha");
                        smooth_tooltip(ui, &r2, "Máscara Alpha (pha)", "Máscara en escala de grises de transparencia.");
                        let r3 = smooth_toggle(ui, &mut self.config.out_fgr, "fgr");
                        smooth_tooltip(ui, &r3, "Primer Plano (fgr)", "Colores estimados del sujeto sin contaminación del fondo.");
                        if self.config.engine_mode == EngineMode::BgMattingV2 {
                            let r4 = smooth_toggle(ui, &mut self.config.out_err, "err");
                            smooth_tooltip(ui, &r4, "Mapa de Error (err)", "Mapa de incertidumbre de la red base.");
                            ui.add_enabled_ui(is_refine, |ui| {
                                let r5 = smooth_toggle(ui, &mut self.config.out_ref, "ref");
                                smooth_tooltip(ui, &r5, "Mapa de Refinamiento (ref)", "Zonas seleccionadas para refinado HD (solo MattingRefine).");
                            });
                        }
                    });
                });
            });
    }

    fn render_preview_and_export(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        // Top preview toolbar
        Frame::new()
            .fill(Palette::BG_SURFACE)
            .stroke(Stroke::new(1.0_f32, Palette::BORDER_SUBTLE))
            .corner_radius(CornerRadius::same(12))
            .inner_margin(Margin::symmetric(12, 10))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.allocate_ui(vec2(230.0, 30.0), |ui| {
                        let vm = segmented_control(
                            ui,
                            "view_mode_seg",
                            &mut self.view_mode,
                            &[
                                (PreviewViewMode::Result, "Resultado"),
                                (PreviewViewMode::Split, "Dividido"),
                                (PreviewViewMode::Original, "Original"),
                            ],
                        );
                        smooth_tooltip(
                            ui,
                            &vm,
                            "Modo de visualización",
                            "Cambia entre resultado final, comparativa interactiva Antes/Después o video original.",
                        );
                    });

                    ui.add_space(6.0);
                    ui.allocate_ui(vec2(265.0, 30.0), |ui| {
                        let ls = if self.config.engine_mode == EngineMode::AutoNoRef {
                            if matches!(
                                self.config.preview_layer,
                                OutputLayer::Err | OutputLayer::Ref
                            ) {
                                self.config.preview_layer = OutputLayer::Com;
                            }
                            segmented_control(
                                ui,
                                "layer_seg_rvm",
                                &mut self.config.preview_layer,
                                &[
                                    (OutputLayer::Com, "Color"),
                                    (OutputLayer::Pha, "Máscara"),
                                    (OutputLayer::Fgr, "Sujeto"),
                                ],
                            )
                        } else {
                            segmented_control(
                                ui,
                                "layer_seg",
                                &mut self.config.preview_layer,
                                &[
                                    (OutputLayer::Com, "Color"),
                                    (OutputLayer::Pha, "Máscara"),
                                    (OutputLayer::Fgr, "Sujeto"),
                                    (OutputLayer::Err, "Error"),
                                    (OutputLayer::Ref, "Refin."),
                                ],
                            )
                        };
                        if ls.changed() {
                            self.config.save();
                        }
                        smooth_tooltip(
                            ui,
                            &ls,
                            "Capa de vista previa (cambio instantáneo)",
                            "Color: resultado con fondo verde/transparente.\nMáscara: silueta blanco y negro.\nSujeto: colores puros.\nCambia al instante sin recalcular.",
                        );
                    });

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let prev_btn = primary_button(
                            ui,
                            if self.preview_busy {
                                "Generando..."
                            } else {
                                "Vista Previa"
                            },
                            !self.preview_busy && !self.export_busy,
                        );
                        smooth_tooltip(
                            ui,
                            &prev_btn,
                            "Generar vista previa",
                            "Procesa el fotograma seleccionado con BackgroundMattingV2 y muestra el resultado aquí.",
                        );
                        if prev_btn.clicked() {
                            self.start_preview(ctx);
                        }
                    });
                });

                ui.add_space(6.0);
                let max_frame = self
                    .video_info
                    .as_ref()
                    .map(|i| i.frame_count.saturating_sub(1))
                    .unwrap_or(300)
                    .max(1);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Fotograma:").size(12.0).color(Palette::TEXT_MUTED));
                    let f_sl = ui.add(
                        egui::Slider::new(&mut self.config.preview_frame, 0..=max_frame)
                            .trailing_fill(true),
                    );
                    smooth_tooltip(
                        ui,
                        &f_sl,
                        "Selector de fotograma",
                        "Elige el número de fotograma sobre el cual calcular la vista previa.",
                    );
                });
            });

        ui.add_space(10.0);

        // Main preview canvas
        let dock_height = 96.0;
        let canvas_h = (ui.available_height() - dock_height).max(240.0);
        let canvas_w = ui.available_width();
        let (canvas_rect, _) = ui.allocate_exact_size(vec2(canvas_w, canvas_h), Sense::hover());

        let painter = ui.painter();
        painter.rect_filled(canvas_rect, CornerRadius::same(12), Color32::from_rgb(13, 17, 25));
        painter.rect_stroke(
            canvas_rect,
            CornerRadius::same(12),
            Stroke::new(1.0_f32, Palette::BORDER_SUBTLE),
            StrokeKind::Inside,
        );

        let inner_rect = canvas_rect.shrink(10.0);
        let out_tex_opt = self.active_preview_out_tex().cloned();
        let active_tex = out_tex_opt.as_ref().or(self.preview_src_tex.as_ref());

        if let Some(tex) = active_tex {
            let tex_sz = tex.size_vec2();
            let scale = (inner_rect.width() / tex_sz.x).min(inner_rect.height() / tex_sz.y);
            let draw_size = tex_sz * scale;
            let draw_rect = Rect::from_center_size(inner_rect.center(), draw_size);

            draw_checkerboard(painter, draw_rect, 14.0);

            let fade_alpha = if let Some(t0) = self.preview_fade_start {
                let dt = ((ctx.input(|i| i.time) - t0) as f32 / 0.25).clamp(0.0, 1.0);
                if dt < 1.0 {
                    ctx.request_repaint();
                }
                ease_out_cubic(dt)
            } else {
                1.0
            };
            let tint = Color32::from_white_alpha((fade_alpha * 255.0).round() as u8);
            let full_uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));

            match self.view_mode {
                PreviewViewMode::Original => {
                    if let Some(src) = &self.preview_src_tex {
                        painter.image(src.id(), draw_rect, full_uv, tint);
                    } else if let Some(out) = &out_tex_opt {
                        painter.image(out.id(), draw_rect, full_uv, tint);
                    }
                }
                PreviewViewMode::Result => {
                    if let Some(out) = &out_tex_opt {
                        painter.image(out.id(), draw_rect, full_uv, tint);
                    } else if let Some(src) = &self.preview_src_tex {
                        painter.image(src.id(), draw_rect, full_uv, tint);
                    }
                }
                PreviewViewMode::Split => {
                    // Drag anywhere in draw_rect to move divider smoothly
                    let drag_resp = ui.interact(
                        draw_rect,
                        egui::Id::new("split_divider_drag"),
                        Sense::drag(),
                    );
                    if drag_resp.dragged() {
                        if let Some(pos) = ctx.pointer_interact_pos() {
                            self.split_pos =
                                ((pos.x - draw_rect.left()) / draw_rect.width()).clamp(0.05, 0.95);
                        }
                    }

                    let split_x = draw_rect.left() + draw_rect.width() * self.split_pos;
                    if let Some(src) = &self.preview_src_tex {
                        let left_rect = Rect::from_min_max(
                            draw_rect.min,
                            pos2(split_x, draw_rect.bottom()),
                        );
                        let left_uv = Rect::from_min_max(
                            pos2(0.0, 0.0),
                            pos2(self.split_pos, 1.0),
                        );
                        painter.image(src.id(), left_rect, left_uv, tint);
                    }
                    if let Some(out) = &out_tex_opt {
                        let right_rect = Rect::from_min_max(
                            pos2(split_x, draw_rect.top()),
                            draw_rect.max,
                        );
                        let right_uv = Rect::from_min_max(
                            pos2(self.split_pos, 0.0),
                            pos2(1.0, 1.0),
                        );
                        painter.image(out.id(), right_rect, right_uv, tint);
                    }

                    // Glowing divider line + handle
                    painter.line_segment(
                        [pos2(split_x, draw_rect.top()), pos2(split_x, draw_rect.bottom())],
                        Stroke::new(2.0_f32, Palette::EMERALD_BRIGHT),
                    );
                    let knob = pos2(split_x, draw_rect.center().y);
                    painter.circle_filled(knob, 12.0, Palette::BG_SURFACE);
                    painter.circle_stroke(knob, 12.0, Stroke::new(2.0_f32, Palette::EMERALD_BRIGHT));
                    painter.text(
                        knob,
                        egui::Align2::CENTER_CENTER,
                        "↔",
                        egui::FontId::proportional(12.0),
                        Palette::TEXT_PRIMARY,
                    );
                }
            }
        } else {
            // Empty state visual
            let c = inner_rect.center();
            let time = ctx.input(|i| i.time) as f32;
            paint_logo(painter, pos2(c.x, c.y - 44.0), 30.0, time * 0.6, 0.45, 0.25);
            painter.text(
                pos2(c.x, c.y + 8.0),
                egui::Align2::CENTER_CENTER,
                "Vista Previa en Vivo",
                egui::FontId::proportional(17.0),
                Palette::TEXT_PRIMARY,
            );
            let hint = if !self.engine_ready() {
                "Paso 1: Abre 'Actualizaciones' para instalar BackgroundMattingV2 y PyTorch."
            } else {
                "Selecciona tu video y pulsa 'Vista Previa' para comprobar el recorte."
            };
            painter.text(
                pos2(c.x, c.y + 34.0),
                egui::Align2::CENTER_CENTER,
                hint,
                egui::FontId::proportional(12.5),
                Palette::TEXT_MUTED,
            );
        }

        if self.preview_busy {
            let c = inner_rect.center();
            let card_rect = Rect::from_center_size(c, vec2(340.0_f32.min(inner_rect.width() - 24.0), 108.0));
            ui.scope_builder(egui::UiBuilder::new().max_rect(card_rect), |ui| {
                Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(15, 20, 30, 238))
                    .stroke(Stroke::new(1.0_f32, Palette::EMERALD))
                    .corner_radius(CornerRadius::same(12))
                    .inner_margin(Margin::same(14))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        let stage_lbl = if self.preview_stage.is_empty() {
                            "Generando vista previa..."
                        } else {
                            &self.preview_stage
                        };
                        ui.label(
                            RichText::new(stage_lbl)
                                .size(12.5)
                                .strong()
                                .color(Palette::TEXT_PRIMARY),
                        );
                        ui.add_space(6.0);
                        let pct_lbl = format!("{:.0}%", self.preview_progress * 100.0);
                        smooth_progress_bar(
                            ui,
                            "preview_prog",
                            self.preview_progress,
                            true,
                            Some(&pct_lbl),
                        );
                        ui.add_space(6.0);
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if secondary_button(ui, "Cancelar", true).clicked() {
                                if let Some(w) = &self.worker {
                                    w.cancel();
                                }
                            }
                        });
                    });
            });
        }

        ui.add_space(10.0);

        // Bottom Export Dock
        Frame::new()
            .fill(Palette::BG_SURFACE)
            .stroke(Stroke::new(1.0_f32, Palette::BORDER_SUBTLE))
            .corner_radius(CornerRadius::same(12))
            .inner_margin(Margin::symmetric(14, 12))
            .show(ui, |ui| {
                if self.export_busy {
                    let (prog_val, stage_txt, bar_lbl) = if self.export_frame == 0 {
                        let st = if self.export_stage.is_empty() {
                            "Preparando exportación...".to_owned()
                        } else {
                            self.export_stage.clone()
                        };
                        let pct = format!("{:.0}%", self.export_stage_progress * 100.0);
                        (self.export_stage_progress, st, pct)
                    } else {
                        let frac = if self.export_total > 0 {
                            self.export_frame as f32 / self.export_total as f32
                        } else {
                            0.0
                        };
                        let lbl = format!(
                            "{} / {} frames ({:.0}%) · {:.1} fps · ETA {:.0}s",
                            self.export_frame,
                            self.export_total,
                            frac * 100.0,
                            self.export_fps,
                            self.export_eta
                        );
                        (frac, "Exportando video con BackgroundMattingV2...".to_owned(), lbl)
                    };
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.set_width((ui.available_width() - 120.0).max(160.0));
                            ui.label(
                                RichText::new(&stage_txt)
                                    .size(12.0)
                                    .color(Palette::TEXT_SECONDARY),
                            );
                            ui.add_space(4.0);
                            smooth_progress_bar(ui, "export_prog", prog_val, true, Some(&bar_lbl));
                        });
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let cancel_btn = secondary_button(ui, "Cancelar", true);
                            smooth_tooltip(
                                ui,
                                &cancel_btn,
                                "Cancelar exportación",
                                "Detiene el procesamiento del video de forma inmediata.",
                            );
                            if cancel_btn.clicked() {
                                if let Some(w) = &self.worker {
                                    w.cancel();
                                }
                            }
                        });
                    });
                } else {
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            let layers = self.config.selected_output_types().join(", ");
                            let bg_str = match self.config.bg_mode {
                                BgMode::GreenScreen => "Pantalla Verde",
                                BgMode::Transparent => "Transparente RGBA",
                                BgMode::CustomColor => "Color sólido",
                                BgMode::TargetMedia => "Fondo personalizado",
                            };
                            let engine_lbl = match self.config.engine_mode {
                                EngineMode::AutoNoRef => format!(
                                    "Auto ({})",
                                    self.config.simple_quality.rvm_backbone_arg()
                                ),
                                EngineMode::BgMattingV2 => {
                                    format!("BGMv2 {}", self.config.model_type.as_arg())
                                }
                            };
                            ui.label(
                                RichText::new(format!(
                                    "{engine_lbl} · {bg_str} · Capas: [{layers}]"
                                ))
                                .size(12.5)
                                .color(Palette::TEXT_SECONDARY),
                            );
                            ui.label(
                                RichText::new(format!("Destino: {}", self.config.output_dir))
                                    .size(11.0)
                                    .color(Palette::TEXT_MUTED),
                            );
                        });

                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let exp_btn = primary_button(
                                ui,
                                "Exportar Video",
                                !self.preview_busy && !self.export_busy,
                            );
                            smooth_tooltip(
                                ui,
                                &exp_btn,
                                "Exportar resultado",
                                "Procesa todos los fotogramas del video con BackgroundMattingV2 y guarda los archivos en la carpeta de salida.",
                            );
                            if exp_btn.clicked() {
                                self.start_export(ctx);
                            }

                            if let Some(done_dir) = self.last_export_dir.clone() {
                                let open_btn = secondary_button(ui, "Abrir Carpeta", true);
                                smooth_tooltip(
                                    ui,
                                    &open_btn,
                                    "Abrir carpeta exportada",
                                    "Abre el Explorador de Windows en la carpeta con los resultados.",
                                );
                                if open_btn.clicked() {
                                    Self::open_folder(&done_dir);
                                }
                            }
                        });
                    });
                }
            });
    }

    fn render_updates_modal(&mut self, ctx: &egui::Context) {
        let open_t = ctx.animate_bool_with_time(
            egui::Id::new("updates_modal_anim"),
            self.show_updates_modal,
            0.18,
        );
        if open_t <= 0.01 {
            return;
        }
        let alpha = ease_out_cubic(open_t);
        let screen = ctx.screen_rect();

        egui::Area::new(egui::Id::new("updates_modal_backdrop"))
            .order(Order::Middle)
            .fixed_pos(screen.min)
            .interactable(true)
            .show(ctx, |ui| {
                ui.allocate_rect(screen, Sense::click());
                ui.painter().rect_filled(
                    screen,
                    CornerRadius::ZERO,
                    Color32::from_black_alpha((175.0 * alpha).round() as u8),
                );
            });

        let modal_w = 580.0_f32.min(screen.width() - 40.0);
        let modal_pos = pos2(
            screen.center().x - modal_w * 0.5,
            screen.center().y - 255.0 + (1.0 - alpha) * 18.0,
        );

        egui::Area::new(egui::Id::new("updates_modal_window"))
            .order(Order::Foreground)
            .fixed_pos(modal_pos)
            .show(ctx, |ui| {
                ui.set_width(modal_w);
                Frame::new()
                    .fill(Palette::BG_SURFACE)
                    .stroke(Stroke::new(1.0_f32, Palette::BORDER_FOCUS))
                    .corner_radius(CornerRadius::same(14))
                    .inner_margin(Margin::same(18))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("Actualizaciones y Motor BackgroundMattingV2")
                                    .size(16.0)
                                    .strong()
                                    .color(Palette::TEXT_PRIMARY),
                            );
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if secondary_button(ui, "Cerrar", !self.updater_busy).clicked() {
                                    self.show_updates_modal = false;
                                }
                            });
                        });
                        ui.add_space(10.0);

                        ScrollArea::vertical().max_height(430.0).show(ui, |ui| {
                            // Section 0: RemBack Self-Updater
                            glass_card(ui, "upd_app", "0. Actualizaciones de RemBack (GitHub)", None, |ui| {
                                ui.horizontal(|ui| {
                                    badge_pill(ui, &format!("Actual: v{APP_VERSION}"), Palette::CYAN);
                                    if let Some(rel) = &self.remote_app_release {
                                        if self.app_update_available {
                                            badge_pill(
                                                ui,
                                                &format!("Disponible: v{}", rel.version),
                                                Palette::EMERALD_BRIGHT,
                                            );
                                        } else {
                                            badge_pill(ui, "Al día", Palette::EMERALD);
                                        }
                                    }
                                });
                                if let Some(rel) = self.remote_app_release.clone() {
                                    if self.app_update_available {
                                        ui.label(
                                            RichText::new(format!(
                                                "Hay una nueva versión RemBack v{} lista para instalar automáticamente.",
                                                rel.version
                                            ))
                                            .size(12.0)
                                            .color(Palette::EMERALD_BRIGHT),
                                        );
                                        if primary_button(
                                            ui,
                                            &format!("Actualizar RemBack a v{}", rel.version),
                                            !self.updater_busy,
                                        )
                                        .clicked()
                                        {
                                            self.updater_busy = true;
                                            self.updater_progress = 0.03;
                                            self.updater_logs.clear();
                                            self.updater_rx =
                                                Some(download_and_launch_app_update(rel));
                                        }
                                    }
                                } else {
                                    ui.label(
                                        RichText::new("Comprueba si hay nuevas versiones en ixyren/RemBack.")
                                            .size(11.5)
                                            .color(Palette::TEXT_MUTED),
                                    );
                                }
                            });

                            ui.add_space(8.0);

                            // Section 1: BackgroundMattingV2 Repo Updater
                            glass_card(ui, "upd_repo", "1. Código Fuente (PeterL1n/BackgroundMattingV2)", None, |ui| {
                                let installed = self.paths.is_bgmv2_installed();
                                ui.horizontal(|ui| {
                                    if installed {
                                        badge_pill(ui, "Instalado", Palette::EMERALD);
                                        let short_sha = if self.installed_state.commit_sha.len() >= 7 {
                                            &self.installed_state.commit_sha[..7]
                                        } else if self.installed_state.commit_sha.is_empty() {
                                            "local"
                                        } else {
                                            &self.installed_state.commit_sha
                                        };
                                        ui.label(
                                            RichText::new(format!("Commit actual: {short_sha}"))
                                                .size(12.0)
                                                .color(Palette::TEXT_MUTED),
                                        );
                                    } else {
                                        badge_pill(ui, "No instalado", Palette::WARNING);
                                    }
                                });

                                if let Some(remote) = &self.remote_info {
                                    let rsha = &remote.commit_sha[..remote.commit_sha.len().min(7)];
                                    let status_txt = if self.update_available {
                                        format!("Nueva versión disponible en GitHub: {rsha} ({})", remote.commit_message)
                                    } else {
                                        format!("Al día con GitHub ({rsha}) · {}", remote.commit_message)
                                    };
                                    ui.label(
                                        RichText::new(status_txt)
                                            .size(12.0)
                                            .color(if self.update_available {
                                                Palette::EMERALD_BRIGHT
                                            } else {
                                                Palette::TEXT_SECONDARY
                                            }),
                                    );
                                }

                                ui.horizontal(|ui| {
                                    let chk_btn = secondary_button(
                                        ui,
                                        if self.checking_updates {
                                            "Buscando..."
                                        } else {
                                            "Buscar Actualizaciones"
                                        },
                                        !self.updater_busy && !self.checking_updates,
                                    );
                                    if chk_btn.clicked() {
                                        self.checking_updates = true;
                                        self.updater_rx = Some(check_bgmv2_updates());
                                    }

                                    let upd_lbl = if !installed {
                                        "Instalar BackgroundMattingV2"
                                    } else if self.update_available {
                                        "Actualizar Ahora"
                                    } else {
                                        "Reinstalar / Sincronizar"
                                    };
                                    let upd_btn = primary_button(
                                        ui,
                                        upd_lbl,
                                        !self.updater_busy && !self.checking_updates,
                                    );
                                    if upd_btn.clicked() {
                                        self.updater_busy = true;
                                        self.updater_progress = 0.05;
                                        self.updater_logs.clear();
                                        self.updater_rx = Some(update_bgmv2_repo(
                                            self.paths.clone(),
                                            self.remote_info.clone(),
                                        ));
                                    }
                                });
                            });

                            ui.add_space(8.0);

                            // Section 2: Python + PyTorch Environment
                            glass_card(ui, "upd_env", "2. Entorno Python y PyTorch (CUDA / CPU)", None, |ui| {
                                let ready = self.paths.is_venv_ready();
                                ui.horizontal(|ui| {
                                    if ready {
                                        badge_pill(ui, "Venv Listo", Palette::EMERALD);
                                    } else {
                                        badge_pill(ui, "Pendiente", Palette::WARNING);
                                    }
                                    ui.label(
                                        RichText::new(self.paths.venv_dir.to_string_lossy().as_ref())
                                            .size(11.0)
                                            .color(Palette::TEXT_MUTED),
                                    );
                                });
                                let btn_lbl = if ready {
                                    "Actualizar / Reparar Entorno Python"
                                } else {
                                    "Instalar Entorno Python + PyTorch"
                                };
                                if primary_button(ui, btn_lbl, !self.updater_busy).clicked() {
                                    self.updater_busy = true;
                                    self.updater_progress = 0.05;
                                    self.updater_logs.clear();
                                    self.updater_rx =
                                        Some(setup_python_environment(self.paths.clone(), false));
                                }
                            });

                            ui.add_space(8.0);

                            // Section 3: Pretrained Models (RVM + BackgroundMattingV2)
                            glass_card(ui, "upd_models", "3. Modelos Descargables (RVM y BGMv2)", None, |ui| {
                                let mut assets: Vec<ModelAssetInfo> = rvm_model_assets();
                                let bgm_assets: Vec<ModelAssetInfo> = self
                                    .remote_info
                                    .as_ref()
                                    .map(|r| r.models.clone())
                                    .unwrap_or_else(default_models_fallback);
                                assets.extend(bgm_assets);

                                for asset in assets {
                                    let local_path = self.paths.models_dir.join(&asset.name);
                                    let exists = local_path.is_file();
                                    let mb = asset.size_bytes as f32 / 1_048_576.0;
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new(format!("{} ({:.0} MB)", asset.name, mb))
                                                .size(12.5)
                                                .color(Palette::TEXT_PRIMARY),
                                        );
                                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                            if exists {
                                                badge_pill(ui, "Descargado", Palette::EMERALD);
                                                if secondary_button(ui, "Usar", true).clicked() {
                                                    if asset.name.ends_with(".torchscript") {
                                                        self.config.engine_mode =
                                                            EngineMode::AutoNoRef;
                                                        self.config.simple_quality =
                                                            if asset.name.contains("mobilenet") {
                                                                SimpleQuality::Fast
                                                            } else {
                                                                SimpleQuality::Balanced
                                                            };
                                                    } else {
                                                        self.config.engine_mode =
                                                            EngineMode::BgMattingV2;
                                                        self.config.model_checkpoint = local_path
                                                            .to_string_lossy()
                                                            .into_owned();
                                                        self.sync_backbone_from_checkpoint();
                                                    }
                                                    self.config.save();
                                                    self.push_toast(
                                                        ctx,
                                                        format!("Modelo activo: {}", asset.name),
                                                        false,
                                                    );
                                                }
                                            } else if secondary_button(
                                                ui,
                                                "Descargar",
                                                !self.updater_busy,
                                            )
                                            .clicked()
                                            {
                                                self.updater_busy = true;
                                                self.updater_progress = 0.02;
                                                self.updater_logs.clear();
                                                self.updater_rx = Some(download_model_asset(
                                                    self.paths.clone(),
                                                    asset.clone(),
                                                ));
                                            }
                                        });
                                    });
                                }
                            });

                            if self.updater_busy || !self.updater_status.is_empty() {
                                ui.add_space(10.0);
                                smooth_progress_bar(
                                    ui,
                                    "upd_prog_bar",
                                    self.updater_progress,
                                    self.updater_busy,
                                    Some(&self.updater_status),
                                );
                            }
                            if !self.updater_logs.is_empty() {
                                ui.add_space(6.0);
                                Frame::new()
                                    .fill(Palette::BG_BASE)
                                    .corner_radius(CornerRadius::same(8))
                                    .inner_margin(Margin::same(8))
                                    .show(ui, |ui| {
                                        ScrollArea::vertical()
                                            .max_height(90.0)
                                            .stick_to_bottom(true)
                                            .show(ui, |ui| {
                                                for line in &self.updater_logs {
                                                    ui.label(
                                                        RichText::new(line)
                                                            .monospace()
                                                            .size(10.5)
                                                            .color(Palette::TEXT_MUTED),
                                                    );
                                                }
                                            });
                                    });
                            }
                        });
                    });
            });
    }

    fn render_about_modal(&mut self, ctx: &egui::Context) {
        let open_t = ctx.animate_bool_with_time(
            egui::Id::new("about_modal_anim"),
            self.show_about_modal,
            0.18,
        );
        if open_t <= 0.01 {
            return;
        }
        let alpha = ease_out_cubic(open_t);
        let screen = ctx.screen_rect();

        egui::Area::new(egui::Id::new("about_modal_backdrop"))
            .order(Order::Middle)
            .fixed_pos(screen.min)
            .interactable(true)
            .show(ctx, |ui| {
                let resp = ui.allocate_rect(screen, Sense::click());
                ui.painter().rect_filled(
                    screen,
                    CornerRadius::ZERO,
                    Color32::from_black_alpha((165.0 * alpha).round() as u8),
                );
                if resp.clicked() {
                    self.show_about_modal = false;
                }
            });

        let w = 420.0_f32.min(screen.width() - 40.0);
        let pos = pos2(
            screen.center().x - w * 0.5,
            screen.center().y - 140.0 + (1.0 - alpha) * 14.0,
        );
        egui::Area::new(egui::Id::new("about_modal_win"))
            .order(Order::Foreground)
            .fixed_pos(pos)
            .show(ctx, |ui| {
                ui.set_width(w);
                Frame::new()
                    .fill(Palette::BG_SURFACE)
                    .stroke(Stroke::new(1.0_f32, Palette::BORDER_FOCUS))
                    .corner_radius(CornerRadius::same(14))
                    .inner_margin(Margin::same(20))
                    .show(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            let (r, _) = ui.allocate_exact_size(vec2(56.0, 56.0), Sense::hover());
                            paint_logo(ui.painter(), r.center(), 26.0, ctx.input(|i| i.time) as f32, 1.0, 0.7);
                            ui.add_space(6.0);
                            ui.label(
                                RichText::new(format!("RemBack v{APP_VERSION}"))
                                    .size(18.0)
                                    .strong()
                                    .color(Palette::TEXT_PRIMARY),
                            );
                            ui.label(
                                RichText::new("Interfaz gráfica moderna escrita en Rust para PeterL1n/BackgroundMattingV2.")
                                    .size(12.5)
                                    .color(Palette::TEXT_SECONDARY),
                            );
                            ui.add_space(10.0);
                            ui.horizontal(|ui| {
                                if secondary_button(ui, "GitHub: RemBack", true).clicked() {
                                    Self::open_url("https://github.com/ixyren/RemBack");
                                }
                                if secondary_button(ui, "GitHub: BackgroundMattingV2", true).clicked() {
                                    Self::open_url("https://github.com/PeterL1n/BackgroundMattingV2");
                                }
                            });
                            ui.add_space(10.0);
                            if primary_button(ui, "Entendido", true).clicked() {
                                self.show_about_modal = false;
                            }
                        });
                    });
            });
    }

    fn render_toast(&mut self, ctx: &egui::Context) {
        let Some(toast) = self.toast.clone() else {
            return;
        };
        let now = ctx.input(|i| i.time);
        let age = (now - toast.created_at) as f32;
        if age > 5.0 {
            self.toast = None;
            return;
        }
        let fade_in = (age / 0.18).clamp(0.0, 1.0);
        let fade_out = ((5.0 - age) / 0.35).clamp(0.0, 1.0);
        let alpha = ease_out_cubic(fade_in.min(fade_out));
        ctx.request_repaint();

        let screen = ctx.screen_rect();
        let accent = if toast.is_error {
            Palette::DANGER
        } else {
            Palette::EMERALD
        };

        egui::Area::new(egui::Id::new("remback_toast"))
            .order(Order::Tooltip)
            .fixed_pos(pos2(screen.center().x - 190.0, screen.top() + 58.0 + (1.0 - alpha) * -8.0))
            .show(ctx, |ui| {
                ui.set_width(380.0);
                Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(19, 24, 35, (245.0 * alpha) as u8))
                    .stroke(Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), (220.0 * alpha) as u8)))
                    .corner_radius(CornerRadius::same(10))
                    .inner_margin(Margin::symmetric(14, 10))
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(&toast.message)
                                .size(12.5)
                                .color(Color32::from_rgba_unmultiplied(241, 245, 249, (255.0 * alpha) as u8)),
                        );
                    });
            });
    }
}

impl eframe::App for RemBackApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_initialized {
            apply_theme(ctx);
            self.theme_initialized = true;
        }

        self.poll_worker(ctx);
        self.poll_updater(ctx);

        egui::TopBottomPanel::top("remback_header")
            .frame(Frame::NONE)
            .show(ctx, |ui| {
                self.render_header(ui, ctx);
            });

        egui::SidePanel::left("remback_controls")
            .resizable(true)
            .default_width(410.0)
            .width_range(360.0..=500.0)
            .frame(
                Frame::new()
                    .fill(Palette::BG_BASE)
                    .inner_margin(Margin::same(12)),
            )
            .show(ctx, |ui| {
                self.render_left_sidebar(ui, ctx);
            });

        egui::CentralPanel::default()
            .frame(
                Frame::new()
                    .fill(Palette::BG_BASE)
                    .inner_margin(Margin::same(12)),
            )
            .show(ctx, |ui| {
                self.render_preview_and_export(ui, ctx);
            });

        self.render_updates_modal(ctx);
        self.render_about_modal(ctx);
        self.render_toast(ctx);

        // Smooth startup splash reveal
        self.splash
            .show(ctx, APP_VERSION, "GUI para BackgroundMattingV2");
    }
}
