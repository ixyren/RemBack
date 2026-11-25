use std::path::{Path, PathBuf};

use directories::{BaseDirs, UserDirs};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    MattingRefine,
    MattingBase,
}

impl ModelType {
    pub fn as_arg(&self) -> &'static str {
        match self {
            ModelType::MattingRefine => "mattingrefine",
            ModelType::MattingBase => "mattingbase",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backbone {
    MobileNetV2,
    ResNet50,
    ResNet101,
}

impl Backbone {
    pub fn as_arg(&self) -> &'static str {
        match self {
            Backbone::MobileNetV2 => "mobilenetv2",
            Backbone::ResNet50 => "resnet50",
            Backbone::ResNet101 => "resnet101",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefineMode {
    Sampling,
    Thresholding,
    Full,
}

impl RefineMode {
    pub fn as_arg(&self) -> &'static str {
        match self {
            RefineMode::Sampling => "sampling",
            RefineMode::Thresholding => "thresholding",
            RefineMode::Full => "full",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeDevice {
    Cuda,
    Cpu,
}

impl ComputeDevice {
    pub fn as_arg(&self) -> &'static str {
        match self {
            ComputeDevice::Cuda => "cuda",
            ComputeDevice::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BgMode {
    GreenScreen,
    Transparent,
    CustomColor,
    TargetMedia,
}

impl BgMode {
    pub fn as_arg(&self) -> &'static str {
        match self {
            BgMode::GreenScreen => "green",
            BgMode::Transparent => "transparent",
            BgMode::CustomColor => "custom_color",
            BgMode::TargetMedia => "target_media",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Video,
    ImageSequences,
}

impl OutputFormat {
    pub fn as_arg(&self) -> &'static str {
        match self {
            OutputFormat::Video => "video",
            OutputFormat::ImageSequences => "image_sequences",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputLayer {
    Com,
    Pha,
    Fgr,
    Err,
    Ref,
}

impl OutputLayer {
    pub fn as_arg(&self) -> &'static str {
        match self {
            OutputLayer::Com => "com",
            OutputLayer::Pha => "pha",
            OutputLayer::Fgr => "fgr",
            OutputLayer::Err => "err",
            OutputLayer::Ref => "ref",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MattingConfig {
    pub video_src: String,
    pub auto_bgr: bool,
    pub video_bgr: String,
    pub bg_mode: BgMode,
    pub chroma_color: [u8; 3],
    pub video_target_bgr: String,
    pub output_dir: String,

    pub model_type: ModelType,
    pub model_backbone: Backbone,
    pub model_backbone_scale: f32,
    pub model_checkpoint: String,

    pub refine_mode: RefineMode,
    pub refine_sample_pixels: u32,
    pub refine_threshold: f32,
    pub refine_kernel_size: u8,

    pub device: ComputeDevice,
    pub preprocess_alignment: bool,
    pub resize_enabled: bool,
    pub resize_width: u32,
    pub resize_height: u32,

    pub output_format: OutputFormat,
    pub out_com: bool,
    pub out_pha: bool,
    pub out_fgr: bool,
    pub out_err: bool,
    pub out_ref: bool,

    pub preview_layer: OutputLayer,
    pub preview_frame: u32,
}

impl Default for MattingConfig {
    fn default() -> Self {
        let default_out = UserDirs::new()
            .and_then(|d| {
                d.video_dir()
                    .map(|p| p.join("RemBack_Output").to_string_lossy().into_owned())
            })
            .or_else(|| {
                BaseDirs::new()
                    .map(|b| b.home_dir().join("Videos").join("RemBack_Output").to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "RemBack_Output".to_owned());

        Self {
            video_src: String::new(),
            auto_bgr: true,
            video_bgr: String::new(),
            bg_mode: BgMode::GreenScreen,
            chroma_color: [120, 255, 155],
            video_target_bgr: String::new(),
            output_dir: default_out,

            model_type: ModelType::MattingRefine,
            model_backbone: Backbone::ResNet50,
            model_backbone_scale: 0.25,
            model_checkpoint: String::new(),

            refine_mode: RefineMode::Sampling,
            refine_sample_pixels: 80_000,
            refine_threshold: 0.7,
            refine_kernel_size: 3,

            device: ComputeDevice::Cuda,
            preprocess_alignment: false,
            resize_enabled: false,
            resize_width: 1920,
            resize_height: 1080,

            output_format: OutputFormat::Video,
            out_com: true,
            out_pha: false,
            out_fgr: false,
            out_err: false,
            out_ref: false,

            preview_layer: OutputLayer::Com,
            preview_frame: 0,
        }
    }
}

impl MattingConfig {
    pub fn app_data_dir() -> PathBuf {
        if let Some(dirs) = BaseDirs::new() {
            dirs.data_local_dir().join("RemBack")
        } else {
            PathBuf::from("RemBackData")
        }
    }

    pub fn config_path() -> PathBuf {
        Self::app_data_dir().join("config.json")
    }

    pub fn load_or_default() -> Self {
        let path = Self::config_path();
        if let Ok(raw) = std::fs::read_to_string(path) {
            if let Ok(cfg) = serde_json::from_str::<Self>(&raw) {
                return cfg;
            }
        }
        Self::default()
    }

    pub fn save(&self) {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }

    /// Return the list of selected `--output-types` respecting model capabilities.
    pub fn selected_output_types(&self) -> Vec<&'static str> {
        let mut types = Vec::new();
        if self.out_com {
            types.push("com");
        }
        if self.out_pha {
            types.push("pha");
        }
        if self.out_fgr {
            types.push("fgr");
        }
        if self.out_err {
            types.push("err");
        }
        if self.out_ref && self.model_type == ModelType::MattingRefine {
            types.push("ref");
        }
        if types.is_empty() {
            types.push("com");
        }
        types
    }

    /// Effective RGB background color to composite over when using solid/chroma mode.
    pub fn effective_bg_rgb(&self) -> [u8; 3] {
        match self.bg_mode {
            BgMode::GreenScreen => [120, 255, 155],
            BgMode::CustomColor => self.chroma_color,
            BgMode::Transparent | BgMode::TargetMedia => [120, 255, 155],
        }
    }

    /// Validate parameters before starting preview or export.
    pub fn validate(&self, require_output_dir: bool) -> Result<(), String> {
        if self.video_src.trim().is_empty() || !Path::new(&self.video_src).is_file() {
            return Err("Selecciona un archivo de video de origen válido (--video-src).".into());
        }
        if !self.auto_bgr
            && (self.video_bgr.trim().is_empty() || !Path::new(&self.video_bgr).is_file())
        {
            return Err(
                "Selecciona una imagen de fondo limpia (--video-bgr), usa 'Extraer Frame 0' o activa 'Sin imagen de referencia (Auto)'.".into(),
            );
        }
        if self.model_checkpoint.trim().is_empty() || !Path::new(&self.model_checkpoint).is_file() {
            return Err(
                "Selecciona un modelo (.pth) o descárgalo desde 'Actualizaciones'.".into(),
            );
        }
        if self.bg_mode == BgMode::TargetMedia
            && (self.video_target_bgr.trim().is_empty()
                || !Path::new(&self.video_target_bgr).is_file())
        {
            return Err(
                "Has elegido 'Fondo personalizado': selecciona un video o imagen de destino.".into(),
            );
        }
        if self.resize_enabled && (self.resize_width < 64 || self.resize_height < 64) {
            return Err("La resolución personalizada debe ser al menos 64x64.".into());
        }
        if require_output_dir && self.output_dir.trim().is_empty() {
            return Err("Indica una carpeta de salida válida (--output-dir).".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_bgmv2_defaults() {
        let cfg = MattingConfig::default();
        assert_eq!(cfg.model_type, ModelType::MattingRefine);
        assert_eq!(cfg.model_backbone, Backbone::ResNet50);
        assert!((cfg.model_backbone_scale - 0.25).abs() < 1e-6);
        assert_eq!(cfg.refine_mode, RefineMode::Sampling);
        assert_eq!(cfg.refine_sample_pixels, 80_000);
        assert!((cfg.refine_threshold - 0.7).abs() < 1e-6);
        assert_eq!(cfg.refine_kernel_size, 3);
        assert_eq!(cfg.chroma_color, [120, 255, 155]);
        assert!(cfg.auto_bgr);
    }

    #[test]
    fn auto_bgr_skips_missing_video_bgr_validation() {
        let tmp_dir = std::env::temp_dir().join("remback_cfg_test");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let vid = tmp_dir.join("dummy.mp4");
        let ckpt = tmp_dir.join("dummy.pth");
        std::fs::write(&vid, b"0").unwrap();
        std::fs::write(&ckpt, b"0").unwrap();

        let mut cfg = MattingConfig::default();
        cfg.video_src = vid.to_string_lossy().into_owned();
        cfg.model_checkpoint = ckpt.to_string_lossy().into_owned();
        cfg.auto_bgr = true;
        cfg.video_bgr.clear();
        assert!(cfg.validate(false).is_ok());

        cfg.auto_bgr = false;
        assert!(cfg.validate(false).is_err());

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn ref_output_disabled_for_mattingbase() {
        let mut cfg = MattingConfig::default();
        cfg.model_type = ModelType::MattingBase;
        cfg.out_com = true;
        cfg.out_ref = true;
        let types = cfg.selected_output_types();
        assert_eq!(types, vec!["com"]);
    }

    #[test]
    fn roundtrip_serde() {
        let mut cfg = MattingConfig::default();
        cfg.video_src = "clip.mp4".into();
        cfg.bg_mode = BgMode::Transparent;
        cfg.preprocess_alignment = true;
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: MattingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.video_src, "clip.mp4");
        assert_eq!(parsed.bg_mode, BgMode::Transparent);
        assert!(parsed.preprocess_alignment);
    }
}
