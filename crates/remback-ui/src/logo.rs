use std::f32::consts::{PI, TAU};

use egui::{pos2, vec2, Color32, CornerRadius, Painter, Pos2, Rect, Shape, Stroke, StrokeKind};

use crate::{anim::lerp_color, theme::Palette};

fn with_alpha(c: Color32, alpha: f32) -> Color32 {
    let a = ((c.a() as f32) * alpha.clamp(0.0, 1.0)).round() as u8;
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

/// Draw the RemBack vector emblem with animated emerald/cyan aperture ring and silhouette cutout.
pub fn paint_logo(painter: &Painter, center: Pos2, radius: f32, time: f32, alpha: f32, glow: f32) {
    if alpha <= 0.005 || radius <= 1.0 {
        return;
    }

    // Soft ambient radial glow behind the squircle
    if glow > 0.01 {
        for i in 0..3 {
            let frac = 1.0 + (i as f32) * 0.24;
            let a = alpha * glow * (0.14 - (i as f32) * 0.04);
            painter.circle_filled(center, radius * 1.18 * frac, with_alpha(Palette::EMERALD, a));
        }
    }

    // Outer rounded squircle container
    let box_rect = Rect::from_center_size(center, vec2(radius * 2.0, radius * 2.0));
    let rounding = CornerRadius::same((radius * 0.38).round() as u8);
    painter.rect_filled(box_rect, rounding, with_alpha(Color32::from_rgb(15, 20, 30), alpha));
    painter.rect_stroke(
        box_rect,
        rounding,
        Stroke::new((radius * 0.04).max(1.0), with_alpha(Palette::BORDER_FOCUS, alpha * 0.85)),
        StrokeKind::Inside,
    );

    // Animated rotating dual-gradient aperture ring
    let ring_r = radius * 0.54;
    let ring_w = (radius * 0.14).max(1.5);
    let segments = 40;
    let spin = time * 0.85;

    for i in 0..segments {
        let t0 = (i as f32) / (segments as f32);
        let t1 = ((i + 1) as f32) / (segments as f32);
        let a0 = spin + t0 * TAU;
        let a1 = spin + t1 * TAU;
        let mid = 0.5 + 0.5 * (a0 - spin).sin();
        let col = lerp_color(Palette::EMERALD, Palette::CYAN, mid);

        let p0 = pos2(center.x + ring_r * a0.cos(), center.y + ring_r * a0.sin());
        let p1 = pos2(center.x + ring_r * a1.cos(), center.y + ring_r * a1.sin());
        painter.line_segment([p0, p1], Stroke::new(ring_w, with_alpha(col, alpha)));
    }

    // Orbiting highlight spark on the ring
    let spark_angle = spin * 1.35 - 0.65;
    let spark_pos = pos2(
        center.x + ring_r * spark_angle.cos(),
        center.y + ring_r * spark_angle.sin(),
    );
    painter.circle_filled(
        spark_pos,
        (ring_w * 0.72).max(1.8),
        with_alpha(Color32::from_rgb(167, 243, 208), alpha),
    );

    // Inner dark disc
    let inner_r = radius * 0.41;
    painter.circle_filled(center, inner_r, with_alpha(Palette::BG_BASE, alpha));

    // Stylized foreground silhouette (head + curved shoulders clipped cleanly inside inner circle)
    let sil_col = with_alpha(Palette::TEXT_PRIMARY, alpha * 0.96);
    let head_center = pos2(center.x, center.y - radius * 0.12);
    painter.circle_filled(head_center, radius * 0.145, sil_col);

    // Curved shoulders arc approximated via convex polygon clamped within inner_r
    let mut shoulder_pts = Vec::with_capacity(20);
    let sh_steps = 16;
    for i in 0..=sh_steps {
        let f = (i as f32) / (sh_steps as f32);
        let ang = PI + 0.32 + f * (PI - 0.64); // lower arc inside inner_r
        let px = center.x + inner_r * 0.86 * ang.cos();
        let py = center.y - inner_r * 0.86 * ang.sin();
        shoulder_pts.push(pos2(px, py));
    }
    // Top shoulder curve
    let arch_steps = 12;
    for i in (0..=arch_steps).rev() {
        let f = (i as f32) / (arch_steps as f32);
        let x_norm = -0.68 + f * 1.36;
        let px = center.x + x_norm * inner_r;
        let py = center.y + radius * 0.08 + (x_norm * x_norm) * radius * 0.16;
        shoulder_pts.push(pos2(px, py));
    }
    painter.add(Shape::convex_polygon(shoulder_pts, sil_col, Stroke::NONE));
}
