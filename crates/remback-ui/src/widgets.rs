use std::hash::Hash;

use egui::{
    pos2, vec2, Align, Color32, CornerRadius, Frame, Layout, Margin, Order, Pos2, Rect,
    Response, RichText, Sense, Shadow, Stroke, StrokeKind, Ui,
};

use crate::{
    anim::{ease_out_cubic, lerp_color},
    theme::Palette,
};

fn with_alpha(c: Color32, alpha: f32) -> Color32 {
    let a = ((c.a() as f32) * alpha.clamp(0.0, 1.0)).round() as u8;
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

#[derive(Clone, Default)]
struct TooltipHoverState {
    hover_since: Option<f64>,
    last_pos: Option<Pos2>,
}

/// Smooth hover-delayed floating tooltip with fade-in and subtle slide-up motion.
pub fn smooth_tooltip(ui: &Ui, response: &Response, title: &str, body: &str) {
    let ctx = ui.ctx();
    let now = ctx.input(|i| i.time);
    let hover_id = response.id.with("remback_tip_state");
    let anim_id = response.id.with("remback_tip_anim");

    let mut state = ctx.data_mut(|d| d.get_temp::<TooltipHoverState>(hover_id).unwrap_or_default());
    let hovered = response.hovered();

    if hovered {
        if state.hover_since.is_none() {
            state.hover_since = Some(now);
        }
        if let Some(pointer) = ctx.pointer_interact_pos().or_else(|| ctx.pointer_hover_pos()) {
            state.last_pos = Some(pointer);
        } else {
            state.last_pos = Some(pos2(response.rect.center().x, response.rect.bottom()));
        }
    } else {
        state.hover_since = None;
    }
    ctx.data_mut(|d| d.insert_temp(hover_id, state.clone()));

    let delay_elapsed = state
        .hover_since
        .map(|t| (now - t) as f32 >= 0.16)
        .unwrap_or(false);

    // Keep repainting during the 160ms hover delay so the tooltip appears right on time.
    if hovered && !delay_elapsed {
        ctx.request_repaint();
    }

    let target = if hovered && delay_elapsed { 1.0 } else { 0.0 };
    let anim = ctx.animate_value_with_time(anim_id, target, 0.16);

    if anim <= 0.01 {
        return;
    }

    let alpha = ease_out_cubic(anim);
    let slide_y = (1.0 - alpha) * 7.0;
    let anchor = state
        .last_pos
        .unwrap_or_else(|| pos2(response.rect.center().x, response.rect.bottom()));

    let screen = ctx.screen_rect();
    let tip_w = 272.0_f32.min(screen.width() - 24.0);
    let desired_x = (anchor.x + 14.0).min(screen.right() - tip_w - 12.0).max(screen.left() + 12.0);
    let mut desired_y = anchor.y + 18.0 + slide_y;
    if desired_y + 92.0 > screen.bottom() - 8.0 {
        desired_y = (response.rect.top() - 86.0 - slide_y).max(screen.top() + 8.0);
    }

    let frame_fill = with_alpha(Color32::from_rgb(20, 26, 38), alpha * 0.97);
    let border_col = with_alpha(Palette::BORDER_FOCUS, alpha);

    egui::Area::new(response.id.with("remback_tip_area"))
        .order(Order::Tooltip)
        .interactable(false)
        .fixed_pos(pos2(desired_x, desired_y))
        .show(ctx, |ui| {
            ui.set_max_width(tip_w);
            let resp = Frame::new()
                .fill(frame_fill)
                .stroke(Stroke::new(1.0_f32, border_col))
                .corner_radius(CornerRadius::same(10))
                .inner_margin(Margin::symmetric(12, 9))
                .shadow(Shadow {
                    offset: [0, 8],
                    blur: 20,
                    spread: 0,
                    color: Color32::from_black_alpha((150.0 * alpha).round() as u8),
                })
                .show(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 3.0;
                        ui.label(
                            RichText::new(title)
                                .size(12.5)
                                .strong()
                                .color(with_alpha(Palette::EMERALD_BRIGHT, alpha)),
                        );
                        ui.label(
                            RichText::new(body)
                                .size(11.8)
                                .color(with_alpha(Palette::TEXT_SECONDARY, alpha)),
                        );
                    });
                });

            // Subtle emerald left accent strip
            let bar = Rect::from_min_size(
                pos2(resp.response.rect.left() + 1.0, resp.response.rect.top() + 7.0),
                vec2(3.0, (resp.response.rect.height() - 14.0).max(6.0)),
            );
            ui.painter()
                .rect_filled(bar, CornerRadius::same(2), with_alpha(Palette::EMERALD, alpha));
        });
}

/// Animated glassmorphic dark card container with optional header badge.
pub fn glass_card(
    ui: &mut Ui,
    id_salt: impl Hash,
    title: &str,
    badge: Option<&str>,
    add_contents: impl FnOnce(&mut Ui),
) {
    let card_id = ui.make_persistent_id(id_salt);
    let hover_anim = ui.ctx().animate_bool_with_time(card_id, false, 0.18);
    let border = lerp_color(Palette::BORDER_SUBTLE, Palette::BORDER_FOCUS, hover_anim);

    let frame = Frame::new()
        .fill(Palette::BG_SURFACE)
        .stroke(Stroke::new(1.0_f32, border))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::same(14));

    let resp = frame.show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(title)
                    .size(14.0)
                    .strong()
                    .color(Palette::TEXT_PRIMARY),
            );
            if let Some(b) = badge {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    badge_pill(ui, b, Palette::CYAN);
                });
            }
        });
        ui.add_space(6.0);
        add_contents(ui);
    });

    // Feed actual hover state back for next frame's smooth border glow
    let hovered = ui.rect_contains_pointer(resp.response.rect);
    ui.ctx().animate_bool_with_time(card_id, hovered, 0.18);
}

/// Crisp rounded pill badge.
pub fn badge_pill(ui: &mut Ui, text: &str, color: Color32) {
    Frame::new()
        .fill(with_alpha(color, 0.15))
        .stroke(Stroke::new(1.0_f32, with_alpha(color, 0.55)))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::symmetric(8, 2))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(11.0).strong().color(color));
        });
}

/// Smooth pill toggle switch with sliding knob and color interpolation.
pub fn smooth_toggle(ui: &mut Ui, value: &mut bool, label: &str) -> Response {
    ui.horizontal(|ui| {
        let desired_size = vec2(40.0, 22.0);
        let (rect, mut response) = ui.allocate_exact_size(desired_size, Sense::click());
        if response.clicked() {
            *value = !*value;
            response.mark_changed();
        }

        let t = ui.ctx().animate_bool_with_time(response.id, *value, 0.16);
        let hover = ui
            .ctx()
            .animate_bool_with_time(response.id.with("h"), response.hovered(), 0.14);

        let bg = lerp_color(
            lerp_color(Palette::BG_ELEVATED, Palette::BG_HOVER, hover),
            Palette::EMERALD,
            t,
        );
        let border = lerp_color(Palette::BORDER_FOCUS, Palette::EMERALD_BRIGHT, t.max(hover * 0.6));

        let painter = ui.painter();
        let radius = rect.height() * 0.5;
        painter.rect_filled(rect, CornerRadius::same(radius as u8), bg);
        painter.rect_stroke(
            rect,
            CornerRadius::same(radius as u8),
            Stroke::new(1.0_f32, border),
            StrokeKind::Inside,
        );

        let knob_r = radius - 3.5;
        let knob_x = egui::lerp((rect.left() + radius)..=(rect.right() - radius), t);
        let knob_center = pos2(knob_x, rect.center().y);
        painter.circle_filled(knob_center, knob_r + hover * 0.7, Color32::WHITE);

        if !label.is_empty() {
            let lbl_resp = ui.add(
                egui::Label::new(RichText::new(label).size(13.0).color(Palette::TEXT_PRIMARY))
                    .sense(Sense::click()),
            );
            if lbl_resp.clicked() {
                *value = !*value;
                response.mark_changed();
            }
            response = response.union(lbl_resp);
        }

        response
    })
    .inner
}

/// Smooth segmented control with an animated sliding active pill.
pub fn segmented_control<T: PartialEq + Copy>(
    ui: &mut Ui,
    id_salt: impl Hash,
    current: &mut T,
    options: &[(T, &str)],
) -> Response {
    let base_id = ui.make_persistent_id(id_salt);
    let count = options.len().max(1);
    let total_w = ui.available_width().max(120.0);
    let seg_h = 30.0;

    let (outer_rect, mut response) = ui.allocate_exact_size(vec2(total_w, seg_h), Sense::hover());
    let painter = ui.painter();

    painter.rect_filled(outer_rect, CornerRadius::same(9), Palette::BG_BASE);
    painter.rect_stroke(
        outer_rect,
        CornerRadius::same(9),
        Stroke::new(1.0_f32, Palette::BORDER_SUBTLE),
        StrokeKind::Inside,
    );

    let pad = 3.0;
    let inner_rect = outer_rect.shrink(pad);
    let seg_w = inner_rect.width() / (count as f32);

    let active_idx = options
        .iter()
        .position(|(val, _)| *val == *current)
        .unwrap_or(0);

    let anim_idx = ui
        .ctx()
        .animate_value_with_time(base_id.with("idx"), active_idx as f32, 0.18);

    let pill_rect = Rect::from_min_size(
        pos2(inner_rect.left() + anim_idx * seg_w, inner_rect.top()),
        vec2(seg_w, inner_rect.height()),
    );
    painter.rect_filled(
        pill_rect,
        CornerRadius::same(7),
        with_alpha(Palette::EMERALD, 0.24),
    );
    painter.rect_stroke(
        pill_rect,
        CornerRadius::same(7),
        Stroke::new(1.0_f32, with_alpha(Palette::EMERALD, 0.75)),
        StrokeKind::Inside,
    );

    for (idx, (val, label)) in options.iter().enumerate() {
        let seg_rect = Rect::from_min_size(
            pos2(inner_rect.left() + (idx as f32) * seg_w, inner_rect.top()),
            vec2(seg_w, inner_rect.height()),
        );
        let seg_resp = ui.interact(seg_rect, base_id.with(idx), Sense::click());
        if seg_resp.clicked() && *current != *val {
            *current = *val;
            response.mark_changed();
        }
        let is_active = *current == *val;
        let text_col = if is_active {
            Palette::TEXT_PRIMARY
        } else if seg_resp.hovered() {
            Palette::TEXT_SECONDARY
        } else {
            Palette::TEXT_MUTED
        };
        painter.text(
            seg_rect.center(),
            egui::Align2::CENTER_CENTER,
            *label,
            egui::FontId::proportional(12.5),
            text_col,
        );
        response = response.union(seg_resp);
    }

    response
}

/// Styled slider with smooth value tracking.
pub fn smooth_slider<Num: egui::emath::Numeric>(
    ui: &mut Ui,
    value: &mut Num,
    range: std::ops::RangeInclusive<Num>,
    suffix: &str,
) -> Response {
    ui.add(
        egui::Slider::new(value, range)
            .suffix(suffix)
            .trailing_fill(true),
    )
}

/// Primary glowing emerald-cyan button with smooth hover interpolation.
pub fn primary_button(ui: &mut Ui, text: &str, enabled: bool) -> Response {
    let desired_h = 36.0;
    let galley = ui.painter().layout_no_wrap(
        text.to_owned(),
        egui::FontId::proportional(13.5),
        Color32::WHITE,
    );
    let width = (galley.size().x + 32.0).max(110.0);
    let (rect, response) = ui.allocate_exact_size(
        vec2(width, desired_h),
        if enabled { Sense::click() } else { Sense::hover() },
    );

    let hover = ui
        .ctx()
        .animate_bool_with_time(response.id.with("h"), enabled && response.hovered(), 0.14);
    let press = ui
        .ctx()
        .animate_bool_with_time(response.id.with("p"), enabled && response.is_pointer_button_down_on(), 0.08);

    let base_col = if enabled {
        lerp_color(Palette::EMERALD, Palette::EMERALD_BRIGHT, hover)
    } else {
        Palette::BG_ELEVATED
    };
    let fill = lerp_color(base_col, Palette::CYAN, press * 0.45);
    let painter = ui.painter();

    if enabled && hover > 0.02 {
        painter.rect_filled(
            rect.expand(2.5 * hover),
            CornerRadius::same(11),
            with_alpha(Palette::EMERALD, 0.18 * hover),
        );
    }

    painter.rect_filled(rect, CornerRadius::same(9), fill);
    if !enabled {
        painter.rect_stroke(
            rect,
            CornerRadius::same(9),
            Stroke::new(1.0_f32, Palette::BORDER_SUBTLE),
            StrokeKind::Inside,
        );
    }

    let text_col = if enabled {
        Color32::from_rgb(8, 16, 14)
    } else {
        Palette::TEXT_MUTED
    };
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(13.5),
        text_col,
    );

    response
}

/// Secondary subtle elevated button with smooth hover highlight.
pub fn secondary_button(ui: &mut Ui, text: &str, enabled: bool) -> Response {
    let desired_h = 32.0;
    let galley = ui.painter().layout_no_wrap(
        text.to_owned(),
        egui::FontId::proportional(12.5),
        Palette::TEXT_PRIMARY,
    );
    let width = galley.size().x + 24.0;
    let (rect, response) = ui.allocate_exact_size(
        vec2(width, desired_h),
        if enabled { Sense::click() } else { Sense::hover() },
    );

    let hover = ui
        .ctx()
        .animate_bool_with_time(response.id.with("sh"), enabled && response.hovered(), 0.14);
    let fill = lerp_color(Palette::BG_ELEVATED, Palette::BG_HOVER, hover);
    let border = lerp_color(Palette::BORDER_SUBTLE, Palette::EMERALD, hover * 0.85);

    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(8), fill);
    painter.rect_stroke(rect, CornerRadius::same(8), Stroke::new(1.0_f32, border), StrokeKind::Inside);
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(12.5),
        if enabled {
            Palette::TEXT_PRIMARY
        } else {
            Palette::TEXT_MUTED
        },
    );

    response
}

/// Animated progress bar with smooth fill interpolation and shimmer sweep.
pub fn smooth_progress_bar(ui: &mut Ui, id_salt: impl Hash, progress: f32, active: bool, label: Option<&str>) {
    let target = progress.clamp(0.0, 1.0);
    let anim_id = ui.make_persistent_id(id_salt);
    let smooth_p = ui.ctx().animate_value_with_time(anim_id, target, 0.22);

    let w = ui.available_width();
    let h = 14.0;
    let (rect, _) = ui.allocate_exact_size(vec2(w, h), Sense::hover());
    let painter = ui.painter();

    painter.rect_filled(rect, CornerRadius::same(7), Palette::BG_BASE);
    painter.rect_stroke(
        rect,
        CornerRadius::same(7),
        Stroke::new(1.0_f32, Palette::BORDER_SUBTLE),
        StrokeKind::Inside,
    );

    let fill_w = (rect.width() * smooth_p).max(if smooth_p > 0.005 { 10.0 } else { 0.0 });
    if fill_w > 0.0 {
        let fill_rect = Rect::from_min_size(rect.min, vec2(fill_w, rect.height()));
        let time = ui.ctx().input(|i| i.time) as f32;
        let pulse = if active {
            0.5 + 0.5 * (time * 3.0).sin()
        } else {
            0.3
        };
        let fill_col = lerp_color(Palette::EMERALD, Palette::CYAN, pulse * 0.55);
        painter.rect_filled(fill_rect, CornerRadius::same(7), fill_col);

        // Moving highlight shimmer while active
        if active && fill_w > 24.0 {
            ui.ctx().request_repaint();
            let sweep = ((time * 0.9) % 1.4) - 0.2;
            let shimmer_x = fill_rect.left() + sweep * fill_rect.width();
            let shimmer_rect = Rect::from_center_size(
                pos2(shimmer_x, fill_rect.center().y),
                vec2(34.0, fill_rect.height()),
            )
            .intersect(fill_rect);
            if shimmer_rect.width() > 1.0 {
                painter.rect_filled(
                    shimmer_rect,
                    CornerRadius::same(7),
                    Color32::from_rgba_unmultiplied(255, 255, 255, 48),
                );
            }
        }
    }

    if let Some(txt) = label {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            txt,
            egui::FontId::proportional(10.5),
            Palette::TEXT_PRIMARY,
        );
    }
}

/// Draw a dark checkerboard background inside `rect` for visualizing alpha transparency.
pub fn draw_checkerboard(painter: &egui::Painter, rect: Rect, cell_size: f32) {
    let c_dark = Color32::from_rgb(18, 22, 31);
    let c_light = Color32::from_rgb(26, 32, 44);
    painter.rect_filled(rect, CornerRadius::same(8), c_dark);

    let cols = (rect.width() / cell_size).ceil() as i32;
    let rows = (rect.height() / cell_size).ceil() as i32;
    let clipped = painter.with_clip_rect(rect);

    for r in 0..rows {
        for c in 0..cols {
            if (r + c) % 2 == 0 {
                let cell = Rect::from_min_size(
                    pos2(rect.left() + c as f32 * cell_size, rect.top() + r as f32 * cell_size),
                    vec2(cell_size, cell_size),
                );
                clipped.rect_filled(cell, CornerRadius::ZERO, c_light);
            }
        }
    }
}
