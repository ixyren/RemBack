#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod config;
mod updater;
mod worker;

use app::{RemBackApp, APP_VERSION};
use remback_ui::load_window_icon;

const ICON_PNG: &[u8] = include_bytes!("../../../assets/remback.png");

fn main() -> eframe::Result<()> {
    if std::env::args().any(|a| a == "--version" || a == "-V") {
        println!("RemBack {APP_VERSION}");
        return Ok(());
    }

    let mut viewport = egui::ViewportBuilder::default()
        .with_title(format!("RemBack v{APP_VERSION} — BackgroundMattingV2 GUI"))
        .with_inner_size([1180.0, 760.0])
        .with_min_inner_size([920.0, 620.0]);

    if let Some(icon) = load_window_icon(ICON_PNG) {
        viewport = viewport.with_icon(icon);
    }

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "RemBack",
        options,
        Box::new(|cc| Ok(Box::new(RemBackApp::new(cc)))),
    )
}
