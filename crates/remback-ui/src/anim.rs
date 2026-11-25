use egui::{pos2, Align2, Color32, CornerRadius, FontId, Order, Rect, Stroke, StrokeKind};

use crate::{logo::paint_logo, theme::Palette};

pub fn ease_out_cubic(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    1.0 - (1.0 - t).powi(3)
}

pub fn ease_in_out_cubic(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}

pub fn ease_out_back(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    let c1 = 1.70158;
    let c3 = c1 + 1.0;
    1.0 + c3 * (t - 1.0).powi(3) + c1 * (t - 1.0).powi(2)
}

pub fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let inv = 1.0 - t;
    Color32::from_rgba_unmultiplied(
        (a.r() as f32 * inv + b.r() as f32 * t).round() as u8,
        (a.g() as f32 * inv + b.g() as f32 * t).round() as u8,
        (a.b() as f32 * inv + b.b() as f32 * t).round() as u8,
        (a.a() as f32 * inv + b.a() as f32 * t).round() as u8,
    )
}

/// Startup splash animation controller with smooth logo reveal and cross-fade transition.
pub struct SplashState {
    start_time: Option<f64>,
    hold_duration: f32,
    fade_duration: f32,
    pub finished: bool,
    skipped_at: Option<f32>,
}

impl SplashState {
    pub fn new() -> Self {
        Self {
            start_time: None,
            hold_duration: 1.25,
            fade_duration: 0.38,
            finished: false,
            skipped_at: None,
        }
    }

    /// Update and render the intro splash overlay if still active.
    /// Returns true while the splash is still animating or visible.
    pub fn show(&mut self, ctx: &egui::Context, version: &str, subtitle: &str) -> bool {
        if self.finished {
            return false;
        }

        let now = ctx.input(|i| i.time);
        let start = *self.start_time.get_or_insert(now);
        let elapsed = (now - start) as f32;

        // Allow skipping smoothly by clicking or pressing Space/Enter/Esc after 0.2s
        if elapsed > 0.2 && self.skipped_at.is_none() {
            let skip = ctx.input(|i| {
                i.pointer.any_click()
                    || i.key_pressed(egui::Key::Space)
                    || i.key_pressed(egui::Key::Enter)
                    || i.key_pressed(egui::Key::Escape)
            });
            if skip {
                self.skipped_at = Some(elapsed);
            }
        }

        let fade_start = self.skipped_at.unwrap_or(self.hold_duration);
        let fade_t = ((elapsed - fade_start) / self.fade_duration).clamp(0.0, 1.0);
        let overlay_alpha = 1.0 - ease_in_out_cubic(fade_t);

        if fade_t >= 1.0 {
            self.finished = true;
            return false;
        }

        ctx.request_repaint();

        let screen = ctx.screen_rect();
        egui::Area::new(egui::Id::new("remback_intro_splash"))
            .order(Order::Foreground)
            .fixed_pos(screen.min)
            .interactable(overlay_alpha > 0.2)
            .show(ctx, |ui| {
                ui.allocate_rect(screen, egui::Sense::hover());
                let painter = ui.painter();

                // Fullscreen obsidian backdrop
                painter.rect_filled(
                    screen,
                    CornerRadius::ZERO,
                    Color32::from_rgba_unmultiplied(
                        Palette::BG_BASE.r(),
                        Palette::BG_BASE.g(),
                        Palette::BG_BASE.b(),
                        (255.0 * overlay_alpha).round() as u8,
                    ),
                );

                // Expanding ambient emerald ring ripple
                let ripple_t = ease_out_cubic((elapsed / 1.1).clamp(0.0, 1.0));
                let ripple_a = (1.0 - ripple_t) * 0.25 * overlay_alpha;
                if ripple_a > 0.005 {
                    painter.circle_stroke(
                        screen.center(),
                        32.0 + ripple_t * 170.0,
                        Stroke::new(
                            2.0_f32,
                            Color32::from_rgba_unmultiplied(
                                Palette::EMERALD.r(),
                                Palette::EMERALD.g(),
                                Palette::EMERALD.b(),
                                (ripple_a * 255.0).round() as u8,
                            ),
                        ),
                    );
                }

                // Animated logo entrance
                let logo_t = ease_out_back((elapsed / 0.55).clamp(0.0, 1.0));
                let logo_radius = 44.0 * logo_t.max(0.01);
                let logo_y_offset = -28.0 + (1.0 - ease_out_cubic((elapsed / 0.55).clamp(0.0, 1.0))) * 14.0;
                let logo_center = pos2(screen.center().x, screen.center().y + logo_y_offset);
                let logo_alpha = (elapsed / 0.22).clamp(0.0, 1.0) * overlay_alpha;
                let glow = 0.65 + 0.35 * (elapsed * 3.5).sin();
                paint_logo(painter, logo_center, logo_radius, elapsed * 1.4, logo_alpha, glow);

                // Title slide-up + fade-in
                let title_t = ease_out_cubic(((elapsed - 0.18) / 0.45).clamp(0.0, 1.0));
                if title_t > 0.01 {
                    let title_y = screen.center().y + 38.0 + (1.0 - title_t) * 14.0;
                    let title_alpha = (title_t * overlay_alpha * 255.0).round() as u8;
                    painter.text(
                        pos2(screen.center().x, title_y),
                        Align2::CENTER_CENTER,
                        "RemBack",
                        FontId::proportional(28.0),
                        Color32::from_rgba_unmultiplied(
                            Palette::TEXT_PRIMARY.r(),
                            Palette::TEXT_PRIMARY.g(),
                            Palette::TEXT_PRIMARY.b(),
                            title_alpha,
                        ),
                    );
                }

                // Subtitle + version pill fade-in
                let sub_t = ease_out_cubic(((elapsed - 0.36) / 0.45).clamp(0.0, 1.0));
                if sub_t > 0.01 {
                    let sub_y = screen.center().y + 68.0 + (1.0 - sub_t) * 10.0;
                    let sub_alpha = (sub_t * overlay_alpha * 255.0).round() as u8;
                    painter.text(
                        pos2(screen.center().x, sub_y),
                        Align2::CENTER_CENTER,
                        subtitle,
                        FontId::proportional(13.5),
                        Color32::from_rgba_unmultiplied(
                            Palette::TEXT_MUTED.r(),
                            Palette::TEXT_MUTED.g(),
                            Palette::TEXT_MUTED.b(),
                            sub_alpha,
                        ),
                    );

                    // Version pill
                    let pill_center = pos2(screen.center().x, sub_y + 28.0);
                    let pill_rect = Rect::from_center_size(pill_center, egui::vec2(68.0, 22.0));
                    painter.rect_filled(
                        pill_rect,
                        CornerRadius::same(11),
                        Color32::from_rgba_unmultiplied(
                            Palette::EMERALD.r(),
                            Palette::EMERALD.g(),
                            Palette::EMERALD.b(),
                            ((sub_t * overlay_alpha) * 40.0).round() as u8,
                        ),
                    );
                    painter.rect_stroke(
                        pill_rect,
                        CornerRadius::same(11),
                        Stroke::new(
                            1.0_f32,
                            Color32::from_rgba_unmultiplied(
                                Palette::EMERALD.r(),
                                Palette::EMERALD.g(),
                                Palette::EMERALD.b(),
                                ((sub_t * overlay_alpha) * 160.0).round() as u8,
                            ),
                        ),
                        StrokeKind::Inside,
                    );
                    painter.text(
                        pill_center,
                        Align2::CENTER_CENTER,
                        format!("v{version}"),
                        FontId::proportional(11.5),
                        Color32::from_rgba_unmultiplied(
                            Palette::EMERALD_BRIGHT.r(),
                            Palette::EMERALD_BRIGHT.g(),
                            Palette::EMERALD_BRIGHT.b(),
                            sub_alpha,
                        ),
                    );
                }
            });

        true
    }
}

impl Default for SplashState {
    fn default() -> Self {
        Self::new()
    }
}
