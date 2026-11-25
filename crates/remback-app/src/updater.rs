use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;

use serde::{Deserialize, Serialize};

use crate::config::MattingConfig;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

const BGMV2_COMMIT_API: &str =
    "https://api.github.com/repos/PeterL1n/BackgroundMattingV2/commits/master";
const BGMV2_RELEASE_API: &str =
    "https://api.github.com/repos/PeterL1n/BackgroundMattingV2/releases/latest";
const BGMV2_ZIP_URL: &str =
    "https://github.com/PeterL1n/BackgroundMattingV2/archive/refs/heads/master.zip";

#[derive(Debug, Clone)]
pub struct UpdaterPaths {
    pub root_dir: PathBuf,
    pub bgmv2_dir: PathBuf,
    pub venv_dir: PathBuf,
    pub models_dir: PathBuf,
    pub backend_dir: PathBuf,
    pub preview_dir: PathBuf,
    pub state_file: PathBuf,
}

impl UpdaterPaths {
    pub fn detect() -> Self {
        let root_dir = MattingConfig::app_data_dir();
        Self {
            bgmv2_dir: root_dir.join("BackgroundMattingV2"),
            venv_dir: root_dir.join("venv"),
            models_dir: root_dir.join("models"),
            backend_dir: root_dir.join("backend"),
            preview_dir: root_dir.join("preview_cache"),
            state_file: root_dir.join("bgmv2_state.json"),
            root_dir,
        }
    }

    pub fn venv_python(&self) -> PathBuf {
        #[cfg(target_os = "windows")]
        {
            self.venv_dir.join("Scripts").join("python.exe")
        }
        #[cfg(not(target_os = "windows"))]
        {
            self.venv_dir.join("bin").join("python")
        }
    }

    pub fn is_bgmv2_installed(&self) -> bool {
        self.bgmv2_dir.join("model").join("model.py").is_file()
    }

    pub fn is_venv_ready(&self) -> bool {
        self.venv_python().is_file()
    }

    /// Return isolated venv python if installed, otherwise check if system python is usable.
    pub fn active_python(&self) -> Option<PathBuf> {
        let vpy = self.venv_python();
        if vpy.is_file() {
            return Some(vpy);
        }
        None
    }

    /// List downloaded `.pth` checkpoint files in `models_dir`.
    pub fn list_local_models(&self) -> Vec<PathBuf> {
        let mut items = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.models_dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("pth") && p.is_file() {
                    items.push(p);
                }
            }
        }
        items.sort();
        items
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Bgmv2InstalledState {
    pub commit_sha: String,
    pub commit_date: String,
    pub commit_message: String,
    pub release_tag: String,
}

impl Bgmv2InstalledState {
    pub fn load(path: &Path) -> Self {
        if let Ok(raw) = std::fs::read_to_string(path) {
            if let Ok(s) = serde_json::from_str(&raw) {
                return s;
            }
        }
        Self::default()
    }

    pub fn save(&self, path: &Path) {
        if let Some(p) = path.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelAssetInfo {
    pub name: String,
    pub url: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct RemoteBgmv2Info {
    pub commit_sha: String,
    pub commit_date: String,
    pub commit_message: String,
    pub release_tag: String,
    pub models: Vec<ModelAssetInfo>,
}

#[derive(Debug, Clone)]
pub enum UpdaterEvent {
    Log(String),
    Progress(f32, String),
    CheckFinished(Result<RemoteBgmv2Info, String>),
    Bgmv2Updated(Result<Bgmv2InstalledState, String>),
    EnvSetupFinished(Result<(), String>),
    ModelDownloaded(Result<PathBuf, String>),
}

#[derive(Deserialize)]
struct GhCommitResp {
    sha: String,
    commit: GhCommitInner,
}
#[derive(Deserialize)]
struct GhCommitInner {
    message: String,
    committer: GhCommitAuthor,
}
#[derive(Deserialize)]
struct GhCommitAuthor {
    date: String,
}

#[derive(Deserialize)]
struct GhReleaseResp {
    tag_name: String,
    assets: Vec<GhAsset>,
}
#[derive(Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

fn default_models_fallback() -> Vec<ModelAssetInfo> {
    vec![
        ModelAssetInfo {
            name: "pytorch_mobilenetv2.pth".into(),
            url: "https://github.com/PeterL1n/BackgroundMattingV2/releases/download/v1.0.0/pytorch_mobilenetv2.pth".into(),
            size_bytes: 18_700_000,
        },
        ModelAssetInfo {
            name: "pytorch_resnet50.pth".into(),
            url: "https://github.com/PeterL1n/BackgroundMattingV2/releases/download/v1.0.0/pytorch_resnet50.pth".into(),
            size_bytes: 158_000_000,
        },
        ModelAssetInfo {
            name: "pytorch_resnet101.pth".into(),
            url: "https://github.com/PeterL1n/BackgroundMattingV2/releases/download/v1.0.0/pytorch_resnet101.pth".into(),
            size_bytes: 233_000_000,
        },
    ]
}

fn http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(60))
        .build()
}

/// Query GitHub API for latest BackgroundMattingV2 commit and release assets.
pub fn check_bgmv2_updates() -> Receiver<UpdaterEvent> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let agent = http_agent();
        let commit_res = agent
            .get(BGMV2_COMMIT_API)
            .set("User-Agent", "RemBack-App")
            .set("Accept", "application/vnd.github+json")
            .call();

        let commit_data: GhCommitResp = match commit_res {
            Ok(resp) => match resp.into_json() {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx.send(UpdaterEvent::CheckFinished(Err(format!(
                        "Error al leer respuesta de GitHub: {e}"
                    ))));
                    return;
                }
            },
            Err(e) => {
                let _ = tx.send(UpdaterEvent::CheckFinished(Err(format!(
                    "No se pudo consultar GitHub (PeterL1n/BackgroundMattingV2): {e}"
                ))));
                return;
            }
        };

        let mut release_tag = "v1.0.0".to_owned();
        let mut models = Vec::new();
        if let Ok(resp) = agent
            .get(BGMV2_RELEASE_API)
            .set("User-Agent", "RemBack-App")
            .set("Accept", "application/vnd.github+json")
            .call()
        {
            if let Ok(rel) = resp.into_json::<GhReleaseResp>() {
                release_tag = rel.tag_name;
                for asset in rel.assets {
                    if asset.name.starts_with("pytorch_") && asset.name.ends_with(".pth") {
                        models.push(ModelAssetInfo {
                            name: asset.name,
                            url: asset.browser_download_url,
                            size_bytes: asset.size,
                        });
                    }
                }
            }
        }
        if models.is_empty() {
            models = default_models_fallback();
        }

        let first_line = commit_data
            .commit
            .message
            .lines()
            .next()
            .unwrap_or("")
            .to_owned();

        let _ = tx.send(UpdaterEvent::CheckFinished(Ok(RemoteBgmv2Info {
            commit_sha: commit_data.sha,
            commit_date: commit_data.commit.committer.date,
            commit_message: first_line,
            release_tag,
            models,
        })));
    });
    rx
}

/// Download and extract BackgroundMattingV2 master branch zip cleanly into `paths.bgmv2_dir`.
pub fn update_bgmv2_repo(
    paths: UpdaterPaths,
    remote_info: Option<RemoteBgmv2Info>,
) -> Receiver<UpdaterEvent> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(UpdaterEvent::Progress(0.05, "Descargando BackgroundMattingV2 desde GitHub...".into()));
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(180))
            .build();

        let resp = match agent
            .get(BGMV2_ZIP_URL)
            .set("User-Agent", "RemBack-App")
            .call()
        {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(UpdaterEvent::Bgmv2Updated(Err(format!(
                    "Error al descargar repositorio: {e}"
                ))));
                return;
            }
        };

        let total_len: u64 = resp
            .header("Content-Length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let mut reader = resp.into_reader();
        let mut bytes = Vec::new();
        let mut chunk = [0u8; 32768];
        let mut downloaded: u64 = 0;

        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    bytes.extend_from_slice(&chunk[..n]);
                    downloaded += n as u64;
                    if total_len > 0 {
                        let p = 0.05 + 0.55 * (downloaded as f32 / total_len as f32);
                        let _ = tx.send(UpdaterEvent::Progress(
                            p,
                            format!("Descargando código fuente ({:.1} MB)...", downloaded as f32 / 1_048_576.0),
                        ));
                    }
                }
                Err(e) => {
                    let _ = tx.send(UpdaterEvent::Bgmv2Updated(Err(format!(
                        "Error durante la descarga: {e}"
                    ))));
                    return;
                }
            }
        }

        let _ = tx.send(UpdaterEvent::Progress(0.65, "Extrayendo archivos...".into()));
        let cursor = std::io::Cursor::new(bytes);
        let mut archive = match zip::ZipArchive::new(cursor) {
            Ok(a) => a,
            Err(e) => {
                let _ = tx.send(UpdaterEvent::Bgmv2Updated(Err(format!(
                    "Archivo zip inválido: {e}"
                ))));
                return;
            }
        };

        let staging_dir = paths.root_dir.join("BackgroundMattingV2_staging");
        if staging_dir.exists() {
            let _ = std::fs::remove_dir_all(&staging_dir);
        }
        if let Err(e) = std::fs::create_dir_all(&staging_dir) {
            let _ = tx.send(UpdaterEvent::Bgmv2Updated(Err(format!(
                "No se pudo crear carpeta temporal: {e}"
            ))));
            return;
        }

        let total_files = archive.len().max(1);
        for i in 0..archive.len() {
            let mut file = match archive.by_index(i) {
                Ok(f) => f,
                Err(e) => {
                    let _ = tx.send(UpdaterEvent::Bgmv2Updated(Err(format!("Error zip: {e}"))));
                    return;
                }
            };
            let rel_name = file.name().to_owned();
            // Strip leading `BackgroundMattingV2-master/` component
            let stripped = match rel_name.split_once('/') {
                Some((_, rest)) if !rest.is_empty() => rest,
                _ => continue,
            };
            let out_path = staging_dir.join(stripped);
            if file.is_dir() {
                let _ = std::fs::create_dir_all(&out_path);
            } else {
                if let Some(parent) = out_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if let Ok(mut out_f) = std::fs::File::create(&out_path) {
                    let _ = std::io::copy(&mut file, &mut out_f);
                }
            }
            if i % 15 == 0 {
                let p = 0.65 + 0.30 * (i as f32 / total_files as f32);
                let _ = tx.send(UpdaterEvent::Progress(p, "Instalando módulos...".into()));
            }
        }

        if paths.bgmv2_dir.exists() {
            let _ = std::fs::remove_dir_all(&paths.bgmv2_dir);
        }
        if let Err(e) = std::fs::rename(&staging_dir, &paths.bgmv2_dir) {
            let _ = tx.send(UpdaterEvent::Bgmv2Updated(Err(format!(
                "No se pudo mover carpeta final: {e}"
            ))));
            return;
        }

        let new_state = if let Some(info) = remote_info {
            Bgmv2InstalledState {
                commit_sha: info.commit_sha,
                commit_date: info.commit_date,
                commit_message: info.commit_message,
                release_tag: info.release_tag,
            }
        } else {
            Bgmv2InstalledState {
                commit_sha: "master-latest".into(),
                commit_date: String::new(),
                commit_message: "Actualizado desde GitHub".into(),
                release_tag: "v1.0.0".into(),
            }
        };
        new_state.save(&paths.state_file);

        let _ = tx.send(UpdaterEvent::Progress(1.0, "BackgroundMattingV2 actualizado.".into()));
        let _ = tx.send(UpdaterEvent::Bgmv2Updated(Ok(new_state)));
    });
    rx
}

/// Download a `.pth` checkpoint model with live progress.
pub fn download_model_asset(paths: UpdaterPaths, asset: ModelAssetInfo) -> Receiver<UpdaterEvent> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        if let Err(e) = std::fs::create_dir_all(&paths.models_dir) {
            let _ = tx.send(UpdaterEvent::ModelDownloaded(Err(format!(
                "No se pudo crear carpeta models: {e}"
            ))));
            return;
        }

        let _ = tx.send(UpdaterEvent::Progress(
            0.02,
            format!("Conectando para descargar {}...", asset.name),
        ));

        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(600))
            .build();

        let resp = match agent.get(&asset.url).set("User-Agent", "RemBack-App").call() {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(UpdaterEvent::ModelDownloaded(Err(format!(
                    "Error al descargar {}: {e}",
                    asset.name
                ))));
                return;
            }
        };

        let total_len: u64 = resp
            .header("Content-Length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(asset.size_bytes.max(1));

        let final_path = paths.models_dir.join(&asset.name);
        let tmp_path = paths.models_dir.join(format!("{}.part", asset.name));
        let mut out_f = match std::fs::File::create(&tmp_path) {
            Ok(f) => f,
            Err(e) => {
                let _ = tx.send(UpdaterEvent::ModelDownloaded(Err(format!(
                    "No se pudo crear archivo {}: {e}",
                    tmp_path.display()
                ))));
                return;
            }
        };

        let mut reader = resp.into_reader();
        let mut buf = [0u8; 65536];
        let mut downloaded: u64 = 0;
        let mut last_pct = -1.0_f32;

        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if let Err(e) = out_f.write_all(&buf[..n]) {
                        let _ = tx.send(UpdaterEvent::ModelDownloaded(Err(format!(
                            "Error al escribir archivo: {e}"
                        ))));
                        return;
                    }
                    downloaded += n as u64;
                    let pct = (downloaded as f32 / total_len as f32).clamp(0.0, 1.0);
                    if pct - last_pct >= 0.01 {
                        last_pct = pct;
                        let mb_done = downloaded as f32 / 1_048_576.0;
                        let mb_tot = total_len as f32 / 1_048_576.0;
                        let _ = tx.send(UpdaterEvent::Progress(
                            pct,
                            format!("Descargando {} ({:.1} / {:.1} MB)", asset.name, mb_done, mb_tot),
                        ));
                    }
                }
                Err(e) => {
                    let _ = tx.send(UpdaterEvent::ModelDownloaded(Err(format!(
                        "Error de red al descargar {}: {e}",
                        asset.name
                    ))));
                    return;
                }
            }
        }
        let _ = out_f.flush();
        drop(out_f);

        if final_path.exists() {
            let _ = std::fs::remove_file(&final_path);
        }
        if let Err(e) = std::fs::rename(&tmp_path, &final_path) {
            let _ = tx.send(UpdaterEvent::ModelDownloaded(Err(format!(
                "No se pudo guardar {}: {e}",
                final_path.display()
            ))));
            return;
        }

        let _ = tx.send(UpdaterEvent::Progress(1.0, format!("{} listo.", asset.name)));
        let _ = tx.send(UpdaterEvent::ModelDownloaded(Ok(final_path)));
    });
    rx
}

fn find_system_python() -> Option<(String, Vec<String>)> {
    for (exe, prefix_args) in [("python", vec![]), ("py", vec!["-3".to_owned()])] {
        let mut cmd = Command::new(exe);
        cmd.args(&prefix_args).arg("--version");
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        if let Ok(out) = cmd.output() {
            if out.status.success() {
                return Some((exe.to_owned(), prefix_args));
            }
        }
    }
    None
}

fn detect_nvidia_gpu() -> bool {
    let mut cmd = Command::new("nvidia-smi");
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd.output().map(|o| o.status.success()).unwrap_or(false)
}

fn run_logged_cmd(
    exe: &Path,
    args: &[&str],
    tx: &mpsc::Sender<UpdaterEvent>,
) -> Result<(), String> {
    let mut cmd = Command::new(exe);
    cmd.args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Error al ejecutar {}: {e}", exe.display()))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let tx_err = tx.clone();
    let err_thread = stderr.map(|p| {
        thread::spawn(move || {
            for line in BufReader::new(p).lines().map_while(Result::ok) {
                let _ = tx_err.send(UpdaterEvent::Log(line));
            }
        })
    });

    if let Some(p) = stdout {
        for line in BufReader::new(p).lines().map_while(Result::ok) {
            let _ = tx.send(UpdaterEvent::Log(line));
        }
    }
    if let Some(t) = err_thread {
        let _ = t.join();
    }

    let status = child
        .wait()
        .map_err(|e| format!("Error esperando proceso: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "El comando {} falló con código {:?}",
            exe.display(),
            status.code()
        ))
    }
}

/// Create/repair isolated Python venv and install PyTorch + OpenCV + Pillow + tqdm.
pub fn setup_python_environment(paths: UpdaterPaths, force_reinstall: bool) -> Receiver<UpdaterEvent> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(UpdaterEvent::Progress(0.05, "Buscando Python del sistema...".into()));
        let Some((sys_py, prefix)) = find_system_python() else {
            let _ = tx.send(UpdaterEvent::EnvSetupFinished(Err(
                "No se encontró Python 3 en el sistema. Instala Python 3.10–3.12 desde python.org y reintenta.".into(),
            )));
            return;
        };

        if !paths.is_venv_ready() || force_reinstall {
            let _ = tx.send(UpdaterEvent::Progress(
                0.15,
                "Creando entorno virtual aislado en LocalAppData...".into(),
            ));
            let _ = tx.send(UpdaterEvent::Log(format!("Creando venv en {}", paths.venv_dir.display())));
            let mut venv_args: Vec<&str> = prefix.iter().map(|s| s.as_str()).collect();
            let venv_str = paths.venv_dir.to_string_lossy().into_owned();
            venv_args.extend(["-m", "venv", &venv_str]);
            if let Err(e) = run_logged_cmd(Path::new(&sys_py), &venv_args, &tx) {
                let _ = tx.send(UpdaterEvent::EnvSetupFinished(Err(e)));
                return;
            }
        }

        let vpy = paths.venv_python();
        let _ = tx.send(UpdaterEvent::Progress(0.30, "Actualizando pip...".into()));
        let _ = run_logged_cmd(&vpy, &["-m", "pip", "install", "--upgrade", "pip"], &tx);

        let has_cuda = detect_nvidia_gpu();
        if has_cuda {
            let _ = tx.send(UpdaterEvent::Progress(
                0.45,
                "GPU NVIDIA detectada: instalando PyTorch con soporte CUDA...".into(),
            ));
            let mut cuda_ok = false;
            for idx_url in [
                "https://download.pytorch.org/whl/cu128",
                "https://download.pytorch.org/whl/cu126",
            ] {
                if run_logged_cmd(
                    &vpy,
                    &[
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "torch",
                        "torchvision",
                        "--index-url",
                        idx_url,
                    ],
                    &tx,
                )
                .is_ok()
                {
                    cuda_ok = true;
                    break;
                }
            }
            if !cuda_ok {
                let _ = tx.send(UpdaterEvent::Log(
                    "Aviso CUDA index: intentando instalación estándar de PyTorch...".into(),
                ));
                if let Err(e2) = run_logged_cmd(
                    &vpy,
                    &["-m", "pip", "install", "--upgrade", "torch", "torchvision"],
                    &tx,
                ) {
                    let _ = tx.send(UpdaterEvent::EnvSetupFinished(Err(e2)));
                    return;
                }
            }
        } else {
            let _ = tx.send(UpdaterEvent::Progress(
                0.45,
                "Instalando PyTorch (CPU)...".into(),
            ));
            if let Err(e) = run_logged_cmd(
                &vpy,
                &["-m", "pip", "install", "--upgrade", "torch", "torchvision"],
                &tx,
            ) {
                let _ = tx.send(UpdaterEvent::EnvSetupFinished(Err(e)));
                return;
            }
        }

        let _ = tx.send(UpdaterEvent::Progress(
            0.80,
            "Instalando OpenCV, Pillow y dependencias de video...".into(),
        ));
        if let Err(e) = run_logged_cmd(
            &vpy,
            &[
                "-m",
                "pip",
                "install",
                "--upgrade",
                "opencv-python",
                "pillow",
                "tqdm",
            ],
            &tx,
        ) {
            let _ = tx.send(UpdaterEvent::EnvSetupFinished(Err(e)));
            return;
        }

        let _ = tx.send(UpdaterEvent::Progress(1.0, "Entorno Python listo.".into()));
        let _ = tx.send(UpdaterEvent::EnvSetupFinished(Ok(())));
    });
    rx
}
