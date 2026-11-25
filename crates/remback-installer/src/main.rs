#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver};
use std::thread;

use directories::{BaseDirs, UserDirs};
use egui::{vec2, Align, CornerRadius, Frame, Layout, Margin, RichText, Sense, Stroke};
use flate2::read::ZlibDecoder;
use remback_ui::{
    apply_theme, badge_pill, glass_card, load_window_icon, paint_logo, primary_button,
    secondary_button, smooth_progress_bar, smooth_toggle, smooth_tooltip, Palette, SplashState,
};

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
const ICON_PNG: &[u8] = include_bytes!("../../../assets/remback.png");
const EMBEDDED_APP_ZLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/remback.exe.zlib"));

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Configure,
    Installing,
    Done,
    UninstallConfirm,
    Uninstalling,
    UninstallDone,
    Error,
}

enum TaskEvent {
    Progress(f32, String),
    Finished(Result<(), String>),
}

struct ExistingInstall {
    version: String,
    install_dir: PathBuf,
}

fn query_reg_sz(key: &str, value_name: &str) -> Option<String> {
    let out = hidden_cmd("reg")
        .args(["query", key, "/v", value_name])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if line.contains(value_name) && line.contains("REG_SZ") {
            if let Some((_, rhs)) = line.split_once("REG_SZ") {
                let val = rhs.trim();
                if !val.is_empty() {
                    return Some(val.to_owned());
                }
            }
        }
    }
    None
}

fn detect_existing_install() -> Option<ExistingInstall> {
    let key = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\RemBack";
    let version = query_reg_sz(key, "DisplayVersion")?;
    let install_dir = query_reg_sz(key, "InstallLocation")
        .map(PathBuf::from)
        .or_else(|| BaseDirs::new().map(|d| d.data_local_dir().join("RemBack")))?;
    Some(ExistingInstall {
        version,
        install_dir,
    })
}

struct InstallerApp {
    stage: Stage,
    splash: SplashState,
    existing: Option<ExistingInstall>,
    silent_update: bool,
    install_dir: String,
    desktop_shortcut: bool,
    start_menu_shortcut: bool,
    launch_after: bool,
    progress: f32,
    status_text: String,
    error_message: String,
    rx: Option<Receiver<TaskEvent>>,
}

impl InstallerApp {
    fn new(cc: &eframe::CreationContext<'_>, uninstall_mode: bool, silent_update: bool) -> Self {
        apply_theme(&cc.egui_ctx);

        let existing = detect_existing_install();
        let default_dir = existing
            .as_ref()
            .map(|e| e.install_dir.clone())
            .or_else(|| BaseDirs::new().map(|d| d.data_local_dir().join("RemBack")))
            .unwrap_or_else(|| PathBuf::from("RemBack"));

        let mut app = Self {
            stage: if uninstall_mode {
                Stage::UninstallConfirm
            } else {
                Stage::Configure
            },
            splash: SplashState::new(),
            existing,
            silent_update,
            install_dir: default_dir.to_string_lossy().into_owned(),
            desktop_shortcut: true,
            start_menu_shortcut: true,
            launch_after: true,
            progress: 0.0,
            status_text: String::new(),
            error_message: String::new(),
            rx: None,
        };
        if silent_update && !uninstall_mode {
            app.splash.finished = true;
            app.start_install();
        }
        app
    }

    fn start_install(&mut self) {
        let dir = PathBuf::from(self.install_dir.trim());
        let desktop = self.desktop_shortcut;
        let start_menu = self.start_menu_shortcut;
        let (tx, rx) = mpsc::channel();

        self.stage = Stage::Installing;
        self.progress = 0.05;
        self.status_text = "Preparando carpeta de instalación...".into();
        self.rx = Some(rx);

        thread::spawn(move || {
            let res = run_install_steps(&dir, desktop, start_menu, &tx);
            let _ = tx.send(TaskEvent::Finished(res));
        });
    }

    fn start_uninstall(&mut self) {
        let dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from(&self.install_dir));
        let (tx, rx) = mpsc::channel();

        self.stage = Stage::Uninstalling;
        self.progress = 0.1;
        self.status_text = "Eliminando accesos directos y registro...".into();
        self.rx = Some(rx);

        thread::spawn(move || {
            let res = run_uninstall_steps(&dir, &tx);
            let _ = tx.send(TaskEvent::Finished(res));
        });
    }

    fn poll_task(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.rx else {
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
                TaskEvent::Progress(p, msg) => {
                    self.progress = p;
                    self.status_text = msg;
                }
                TaskEvent::Finished(res) => {
                    finish = true;
                    match res {
                        Ok(()) => {
                            self.progress = 1.0;
                            if self.stage == Stage::Installing {
                                if self.silent_update {
                                    let exe = PathBuf::from(&self.install_dir).join("remback.exe");
                                    let _ = Command::new(exe).spawn();
                                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                                    return;
                                }
                                self.stage = Stage::Done;
                            } else if self.stage == Stage::Uninstalling {
                                self.stage = Stage::UninstallDone;
                            }
                        }
                        Err(err) => {
                            self.error_message = err;
                            self.stage = Stage::Error;
                        }
                    }
                }
            }
        }
        if finish {
            self.rx = None;
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        }
    }
}

fn hidden_cmd(program: &str) -> Command {
    let mut cmd = Command::new(program);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

fn escape_ps(s: &str) -> String {
    s.replace('\'', "''")
}

fn create_shortcut(lnk_path: &Path, target: &Path, working_dir: &Path) -> Result<(), String> {
    if let Some(parent) = lnk_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let ps_script = format!(
        "$ws = New-Object -ComObject WScript.Shell; \
         $s = $ws.CreateShortcut('{}'); \
         $s.TargetPath = '{}'; \
         $s.WorkingDirectory = '{}'; \
         $s.IconLocation = '{}'; \
         $s.Description = 'RemBack - BackgroundMattingV2 GUI'; \
         $s.Save()",
        escape_ps(&lnk_path.to_string_lossy()),
        escape_ps(&target.to_string_lossy()),
        escape_ps(&working_dir.to_string_lossy()),
        escape_ps(&target.to_string_lossy()),
    );
    let out = hidden_cmd("powershell")
        .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", &ps_script])
        .output()
        .map_err(|e| format!("Error creando acceso directo: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(())
}

fn desktop_shortcut_path() -> Option<PathBuf> {
    UserDirs::new()
        .and_then(|d| d.desktop_dir().map(|p| p.join("RemBack.lnk")))
        .or_else(|| BaseDirs::new().map(|d| d.home_dir().join("Desktop").join("RemBack.lnk")))
}

fn start_menu_shortcut_path() -> Option<PathBuf> {
    BaseDirs::new().map(|d| {
        d.data_dir()
            .join("Microsoft")
            .join("Windows")
            .join("Start Menu")
            .join("Programs")
            .join("RemBack")
            .join("RemBack.lnk")
    })
}

fn register_uninstall_key(install_dir: &Path, exe_path: &Path, uninstall_exe: &Path, size_kb: u64) {
    let key = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\RemBack";
    let uninst_cmd = format!("\"{}\" --uninstall", uninstall_exe.display());
    let entries = [
        ("DisplayName", "REG_SZ", "RemBack".to_owned()),
        ("DisplayVersion", "REG_SZ", APP_VERSION.to_owned()),
        ("Publisher", "REG_SZ", "ixyren".to_owned()),
        ("InstallLocation", "REG_SZ", install_dir.to_string_lossy().into_owned()),
        ("DisplayIcon", "REG_SZ", exe_path.to_string_lossy().into_owned()),
        ("UninstallString", "REG_SZ", uninst_cmd),
        ("URLInfoAbout", "REG_SZ", "https://github.com/ixyren/RemBack".to_owned()),
        ("NoModify", "REG_DWORD", "1".to_owned()),
        ("NoRepair", "REG_DWORD", "1".to_owned()),
        ("EstimatedSize", "REG_DWORD", size_kb.to_string()),
    ];
    for (name, ty, val) in entries {
        let _ = hidden_cmd("reg")
            .args(["add", key, "/v", name, "/t", ty, "/d", &val, "/f"])
            .output();
    }
}

fn run_install_steps(
    install_dir: &Path,
    desktop: bool,
    start_menu: bool,
    tx: &mpsc::Sender<TaskEvent>,
) -> Result<(), String> {
    if EMBEDDED_APP_ZLIB.len() < 16 {
        return Err("El instalador fue compilado sin remback.exe embebido.".into());
    }

    let _ = tx.send(TaskEvent::Progress(
        0.08,
        "Comprobando si RemBack está abierto...".into(),
    ));
    let _ = hidden_cmd("taskkill")
        .args(["/IM", "remback.exe", "/F"])
        .output();
    thread::sleep(std::time::Duration::from_millis(250));

    let _ = tx.send(TaskEvent::Progress(
        0.18,
        "Creando directorio de instalación...".into(),
    ));
    std::fs::create_dir_all(install_dir)
        .map_err(|e| format!("No se pudo crear {}: {e}", install_dir.display()))?;

    let _ = tx.send(TaskEvent::Progress(
        0.35,
        "Descomprimiendo RemBack.exe...".into(),
    ));
    let mut decoder = ZlibDecoder::new(EMBEDDED_APP_ZLIB);
    let mut exe_bytes = Vec::new();
    decoder
        .read_to_end(&mut exe_bytes)
        .map_err(|e| format!("Error al descomprimir binario: {e}"))?;

    if exe_bytes.is_empty() {
        return Err("El binario embebido está vacío.".into());
    }

    let target_exe = install_dir.join("remback.exe");
    let mut file_handle = None;
    let mut last_err = None;
    for _ in 0..12 {
        match File::create(&target_exe) {
            Ok(f) => {
                file_handle = Some(f);
                break;
            }
            Err(e) => {
                last_err = Some(e);
                thread::sleep(std::time::Duration::from_millis(250));
            }
        }
    }
    let mut f = file_handle.ok_or_else(|| {
        format!(
            "No se pudo escribir {} (cierra RemBack e inténtalo de nuevo): {}",
            target_exe.display(),
            last_err.map(|e| e.to_string()).unwrap_or_default()
        )
    })?;
    f.write_all(&exe_bytes)
        .map_err(|e| format!("Error escribiendo binario: {e}"))?;
    f.flush().ok();
    drop(f);

    let _ = tx.send(TaskEvent::Progress(
        0.60,
        "Instalando desinstalador...".into(),
    ));
    let uninstall_exe = install_dir.join("uninstall.exe");
    if let Ok(self_exe) = std::env::current_exe() {
        if self_exe != uninstall_exe {
            for _ in 0..5 {
                if std::fs::copy(&self_exe, &uninstall_exe).is_ok() {
                    break;
                }
                thread::sleep(std::time::Duration::from_millis(150));
            }
        }
    }

    if desktop {
        let _ = tx.send(TaskEvent::Progress(
            0.75,
            "Creando acceso directo en el Escritorio...".into(),
        ));
        if let Some(lnk) = desktop_shortcut_path() {
            create_shortcut(&lnk, &target_exe, install_dir)?;
        }
    }

    if start_menu {
        let _ = tx.send(TaskEvent::Progress(
            0.88,
            "Creando acceso directo en el Menú Inicio...".into(),
        ));
        if let Some(lnk) = start_menu_shortcut_path() {
            create_shortcut(&lnk, &target_exe, install_dir)?;
        }
    }

    let _ = tx.send(TaskEvent::Progress(
        0.95,
        "Registrando aplicación en Windows...".into(),
    ));
    let size_kb = (exe_bytes.len() as u64) / 1024;
    register_uninstall_key(install_dir, &target_exe, &uninstall_exe, size_kb);

    let _ = tx.send(TaskEvent::Progress(1.0, "¡Instalación completada!".into()));
    Ok(())
}

fn run_uninstall_steps(install_dir: &Path, tx: &mpsc::Sender<TaskEvent>) -> Result<(), String> {
    let _ = tx.send(TaskEvent::Progress(0.25, "Eliminando accesos directos...".into()));
    if let Some(lnk) = desktop_shortcut_path() {
        let _ = std::fs::remove_file(lnk);
    }
    if let Some(lnk) = start_menu_shortcut_path() {
        if let Some(dir) = lnk.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    let _ = tx.send(TaskEvent::Progress(0.55, "Eliminando registro de Windows...".into()));
    let _ = hidden_cmd("reg")
        .args([
            "delete",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\RemBack",
            "/f",
        ])
        .output();

    let _ = tx.send(TaskEvent::Progress(0.80, "Eliminando archivos de RemBack...".into()));
    let app_exe = install_dir.join("remback.exe");
    if app_exe.exists() {
        let _ = std::fs::remove_file(app_exe);
    }

    // Schedule removal of running uninstall.exe after exit
    if let Ok(self_exe) = std::env::current_exe() {
        let script = format!(
            "ping 127.0.0.1 -n 3 > nul & del /f /q \"{}\"",
            self_exe.display()
        );
        let _ = hidden_cmd("cmd").args(["/C", &script]).spawn();
    }

    let _ = tx.send(TaskEvent::Progress(1.0, "RemBack desinstalado.".into()));
    Ok(())
}

impl eframe::App for InstallerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_task(ctx);
        let time = ctx.input(|i| i.time) as f32;

        egui::CentralPanel::default()
            .frame(
                Frame::new()
                    .fill(Palette::BG_BASE)
                    .inner_margin(Margin::same(18)),
            )
            .show(ctx, |ui| {
                // Header
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(vec2(44.0, 44.0), Sense::hover());
                    paint_logo(ui.painter(), r.center(), 20.0, time * 0.9, 1.0, 0.7);
                    ctx.request_repaint();
                    ui.add_space(6.0);
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 2.0;
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("RemBack Setup")
                                    .size(19.0)
                                    .strong()
                                    .color(Palette::TEXT_PRIMARY),
                            );
                            badge_pill(ui, &format!("v{APP_VERSION}"), Palette::EMERALD_BRIGHT);
                            if let Some(ex) = &self.existing {
                                let txt = if ex.version == APP_VERSION {
                                    format!("Instalada: v{}", ex.version)
                                } else {
                                    format!("Actualización: v{} → v{APP_VERSION}", ex.version)
                                };
                                badge_pill(ui, &txt, Palette::CYAN);
                            }
                        });
                        ui.label(
                            RichText::new("GUI moderna en Rust para BackgroundMattingV2")
                                .size(12.0)
                                .color(Palette::TEXT_MUTED),
                        );
                    });
                });

                ui.add_space(14.0);

                match self.stage {
                    Stage::Configure => {
                        glass_card(ui, "inst_dir_card", "Ubicación de instalación", None, |ui| {
                            ui.horizontal(|ui| {
                                let ed = ui.add(
                                    egui::TextEdit::singleline(&mut self.install_dir)
                                        .desired_width(ui.available_width() - 90.0),
                                );
                                smooth_tooltip(
                                    ui,
                                    &ed,
                                    "Carpeta de destino",
                                    "Ruta donde se instalará remback.exe y su desinstalador.",
                                );
                                let btn = secondary_button(ui, "Cambiar...", true);
                                if btn.clicked() {
                                    if let Some(p) = rfd::FileDialog::new().pick_folder() {
                                        self.install_dir = p.to_string_lossy().into_owned();
                                    }
                                }
                            });
                        });

                        ui.add_space(10.0);

                        glass_card(ui, "inst_opts_card", "Opciones", None, |ui| {
                            let d_t = smooth_toggle(
                                ui,
                                &mut self.desktop_shortcut,
                                "Crear acceso directo en el Escritorio",
                            );
                            smooth_tooltip(
                                ui,
                                &d_t,
                                "Acceso directo en Escritorio",
                                "Añade un icono de acceso rápido en tu Escritorio.",
                            );
                            let s_t = smooth_toggle(
                                ui,
                                &mut self.start_menu_shortcut,
                                "Crear acceso directo en el Menú Inicio",
                            );
                            smooth_tooltip(
                                ui,
                                &s_t,
                                "Acceso en Menú Inicio",
                                "Permite abrir RemBack buscando desde la barra de Windows.",
                            );
                            let l_t = smooth_toggle(
                                ui,
                                &mut self.launch_after,
                                "Abrir RemBack al terminar",
                            );
                            smooth_tooltip(
                                ui,
                                &l_t,
                                "Iniciar automáticamente",
                                "Abre la aplicación inmediatamente cuando finalice la instalación.",
                            );
                        });

                        ui.add_space(18.0);
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let primary_lbl = if self.existing.is_some() {
                                "Actualizar RemBack"
                            } else {
                                "Instalar RemBack"
                            };
                            let inst_btn = primary_button(ui, primary_lbl, true);
                            if inst_btn.clicked() {
                                self.start_install();
                            }
                            if secondary_button(ui, "Cancelar", true).clicked() {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                        });
                    }

                    Stage::Installing | Stage::Uninstalling => {
                        glass_card(
                            ui,
                            "inst_prog_card",
                            if self.stage == Stage::Installing {
                                "Instalando RemBack..."
                            } else {
                                "Desinstalando RemBack..."
                            },
                            None,
                            |ui| {
                                ui.add_space(6.0);
                                ui.label(
                                    RichText::new(&self.status_text)
                                        .size(13.0)
                                        .color(Palette::TEXT_SECONDARY),
                                );
                                ui.add_space(10.0);
                                smooth_progress_bar(
                                    ui,
                                    "inst_bar",
                                    self.progress,
                                    true,
                                    Some(&format!("{:.0}%", self.progress * 100.0)),
                                );
                                ui.add_space(8.0);
                            },
                        );
                    }

                    Stage::Done => {
                        glass_card(ui, "inst_done_card", "Instalación completada", None, |ui| {
                            ui.label(
                                RichText::new("RemBack se ha instalado correctamente.")
                                    .size(14.0)
                                    .color(Palette::EMERALD_BRIGHT),
                            );
                            ui.add_space(6.0);
                            ui.label(
                                RichText::new(
                                    "Al abrir RemBack por primera vez, entra a 'Actualizaciones' para sincronizar BackgroundMattingV2, preparar PyTorch y bajar un modelo .pth.",
                                )
                                .size(12.5)
                                .color(Palette::TEXT_SECONDARY),
                            );
                        });
                        ui.add_space(18.0);
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if primary_button(ui, "Finalizar", true).clicked() {
                                if self.launch_after {
                                    let exe = PathBuf::from(&self.install_dir).join("remback.exe");
                                    let _ = Command::new(exe).spawn();
                                }
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                        });
                    }

                    Stage::UninstallConfirm => {
                        glass_card(ui, "uninst_conf", "Desinstalar RemBack", None, |ui| {
                            ui.label(
                                RichText::new("¿Seguro que deseas eliminar RemBack de este equipo?")
                                    .size(13.5)
                                    .color(Palette::TEXT_PRIMARY),
                            );
                            ui.add_space(4.0);
                            ui.label(
                                RichText::new("Se quitarán los accesos directos, la entrada en Windows y el ejecutable principal.")
                                    .size(12.0)
                                    .color(Palette::TEXT_MUTED),
                            );
                        });
                        ui.add_space(18.0);
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if primary_button(ui, "Desinstalar", true).clicked() {
                                self.start_uninstall();
                            }
                            if secondary_button(ui, "Cancelar", true).clicked() {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                        });
                    }

                    Stage::UninstallDone => {
                        glass_card(ui, "uninst_done", "Desinstalación completada", None, |ui| {
                            ui.label(
                                RichText::new("RemBack ha sido eliminado de tu sistema.")
                                    .size(13.5)
                                    .color(Palette::EMERALD_BRIGHT),
                            );
                        });
                        ui.add_space(18.0);
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if primary_button(ui, "Cerrar", true).clicked() {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                        });
                    }

                    Stage::Error => {
                        Frame::new()
                            .fill(Palette::BG_SURFACE)
                            .stroke(Stroke::new(1.0_f32, Palette::DANGER))
                            .corner_radius(CornerRadius::same(12))
                            .inner_margin(Margin::same(14))
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new("Error durante la operación")
                                        .size(14.0)
                                        .strong()
                                        .color(Palette::DANGER),
                                );
                                ui.add_space(6.0);
                                ui.label(
                                    RichText::new(&self.error_message)
                                        .size(12.5)
                                        .color(Palette::TEXT_SECONDARY),
                                );
                            });
                        ui.add_space(18.0);
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if secondary_button(ui, "Cerrar", true).clicked() {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                        });
                    }
                }
            });

        self.splash
            .show(ctx, APP_VERSION, "Instalador Oficial · BackgroundMattingV2 GUI");
    }
}

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("RemBack Setup {APP_VERSION}");
        return Ok(());
    }
    let uninstall_mode = args.iter().any(|a| a == "--uninstall");
    let silent_update = args.iter().any(|a| a == "--silent-update");

    let mut viewport = egui::ViewportBuilder::default()
        .with_title(format!("RemBack v{APP_VERSION} Setup"))
        .with_inner_size([540.0, 440.0])
        .with_resizable(false);

    if let Some(icon) = load_window_icon(ICON_PNG) {
        viewport = viewport.with_icon(icon);
    }

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "RemBack Setup",
        options,
        Box::new(move |cc| Ok(Box::new(InstallerApp::new(cc, uninstall_mode, silent_update)))),
    )
}
