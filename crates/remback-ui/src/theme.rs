use std::sync::Arc;

use egui::{Color32, CornerRadius, FontId, Margin, Shadow, Stroke, TextStyle, Visuals};

/// Obsidian + Emerald/Cyan dark theme palette shared by RemBack and its installer.
pub struct Palette;

impl Palette {
    pub const BG_BASE: Color32 = Color32::from_rgb(11, 14, 20);
    pub const BG_SURFACE: Color32 = Color32::from_rgb(19, 24, 35);
    pub const BG_ELEVATED: Color32 = Color32::from_rgb(27, 34, 50);
    pub const BG_HOVER: Color32 = Color32::from_rgb(35, 44, 64);

    pub const BORDER_SUBTLE: Color32 = Color32::from_rgb(37, 46, 66);
    pub const BORDER_FOCUS: Color32 = Color32::from_rgb(56, 74, 106);

    pub const EMERALD: Color32 = Color32::from_rgb(16, 185, 129);
    pub const EMERALD_BRIGHT: Color32 = Color32::from_rgb(52, 211, 153);
    pub const CYAN: Color32 = Color32::from_rgb(6, 182, 212);
    pub const VIOLET: Color32 = Color32::from_rgb(139, 92, 246);

    pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(241, 245, 249);
    pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(203, 213, 225);
    pub const TEXT_MUTED: Color32 = Color32::from_rgb(148, 163, 184);

    pub const DANGER: Color32 = Color32::from_rgb(239, 68, 68);
    pub const WARNING: Color32 = Color32::from_rgb(245, 158, 11);
}

/// Apply the unified RemBack obsidian dark theme and smooth animation timings.
pub fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.animation_time = 0.16;

    let mut visuals = Visuals::dark();
    visuals.dark_mode = true;
    visuals.panel_fill = Palette::BG_BASE;
    visuals.window_fill = Palette::BG_SURFACE;
    visuals.extreme_bg_color = Color32::from_rgb(8, 10, 15);
    visuals.faint_bg_color = Palette::BG_ELEVATED;
    visuals.code_bg_color = Palette::BG_ELEVATED;

    visuals.window_corner_radius = CornerRadius::same(14);
    visuals.menu_corner_radius = CornerRadius::same(10);
    visuals.window_stroke = Stroke::new(1.0_f32, Palette::BORDER_SUBTLE);
    visuals.window_shadow = Shadow {
        offset: [0, 10],
        blur: 28,
        spread: 0,
        color: Color32::from_black_alpha(150),
    };
    visuals.popup_shadow = Shadow {
        offset: [0, 6],
        blur: 18,
        spread: 0,
        color: Color32::from_black_alpha(140),
    };

    visuals.selection.bg_fill = Color32::from_rgba_unmultiplied(16, 185, 129, 70);
    visuals.selection.stroke = Stroke::new(1.0_f32, Palette::EMERALD);

    visuals.widgets.noninteractive.bg_fill = Palette::BG_SURFACE;
    visuals.widgets.noninteractive.weak_bg_fill = Palette::BG_SURFACE;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, Palette::BORDER_SUBTLE);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, Palette::TEXT_SECONDARY);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(8);

    visuals.widgets.inactive.bg_fill = Palette::BG_ELEVATED;
    visuals.widgets.inactive.weak_bg_fill = Palette::BG_ELEVATED;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, Palette::BORDER_SUBTLE);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, Palette::TEXT_PRIMARY);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(8);

    visuals.widgets.hovered.bg_fill = Palette::BG_HOVER;
    visuals.widgets.hovered.weak_bg_fill = Palette::BG_HOVER;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.2_f32, Palette::EMERALD);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.2_f32, Palette::TEXT_PRIMARY);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(8);
    visuals.widgets.hovered.expansion = 1.0;

    visuals.widgets.active.bg_fill = Color32::from_rgb(42, 54, 78);
    visuals.widgets.active.weak_bg_fill = Color32::from_rgb(42, 54, 78);
    visuals.widgets.active.bg_stroke = Stroke::new(1.5_f32, Palette::CYAN);
    visuals.widgets.active.fg_stroke = Stroke::new(1.5_f32, Color32::WHITE);
    visuals.widgets.active.corner_radius = CornerRadius::same(8);

    visuals.widgets.open.bg_fill = Palette::BG_HOVER;
    visuals.widgets.open.bg_stroke = Stroke::new(1.0_f32, Palette::EMERALD);
    visuals.widgets.open.corner_radius = CornerRadius::same(8);

    style.visuals = visuals;

    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(12.0, 7.0);
    style.spacing.window_margin = Margin::same(14);
    style.spacing.slider_width = 140.0;
    style.spacing.scroll.bar_width = 7.0;
    style.spacing.scroll.floating = true;

    style.text_styles.insert(TextStyle::Heading, FontId::proportional(19.0));
    style.text_styles.insert(TextStyle::Body, FontId::proportional(13.5));
    style.text_styles.insert(TextStyle::Button, FontId::proportional(13.5));
    style.text_styles.insert(TextStyle::Small, FontId::proportional(11.5));
    style.text_styles.insert(TextStyle::Monospace, FontId::monospace(12.0));

    ctx.set_style(style);
}

/// Decode embedded PNG bytes into an eframe/egui window IconData.
pub fn load_window_icon(png_bytes: &[u8]) -> Option<Arc<egui::IconData>> {
    let img = image::load_from_memory(png_bytes).ok()?.to_rgba8();
    let (width, height) = img.dimensions();
    Some(Arc::new(egui::IconData {
        rgba: img.into_raw(),
        width,
        height,
    }))
}
