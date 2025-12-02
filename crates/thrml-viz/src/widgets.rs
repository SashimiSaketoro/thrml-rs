//! GUI widgets for the visualizer.

use eframe::egui::{Color32, RichText, Ui};

/// Color mode for point coloring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    Prominence,
    Radius,
}

/// Statistics about the loaded data.
#[derive(Clone, Debug, Default)]
pub struct DataStats {
    pub n_points: usize,
    pub embedding_dim: usize,
    pub prominence_range: (f32, f32),
    pub radius_range: (f32, f32),
    pub has_sphere_coords: bool,
    pub last_step: usize,
}

impl DataStats {
    pub fn from_data(
        n_points: usize,
        embedding_dim: usize,
        prominence: Option<&[f32]>,
        has_sphere_coords: bool,
        last_step: usize,
    ) -> Self {
        let prominence_range = prominence
            .map(|p| {
                let min = p.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            })
            .unwrap_or((0.0, 1.0));

        Self {
            n_points,
            embedding_dim,
            prominence_range,
            radius_range: (0.0, 500.0), // Default
            has_sphere_coords,
            last_step,
        }
    }
}

/// Draw the control panel sidebar.
pub fn draw_control_panel(
    ui: &mut Ui,
    stats: &DataStats,
    progress: f32,
    show_points: &mut bool,
    show_tree: &mut bool,
    show_shells: &mut bool,
    color_mode: &mut ColorMode,
    point_size: &mut f32,
    auto_refresh: &mut bool,
) {
    ui.heading("TheSphere Visualizer");
    ui.separator();

    // Progress bar during ingestion
    if progress > 0.0 && progress < 1.0 {
        ui.horizontal(|ui| {
            ui.label("Ingesting:");
            ui.add(
                eframe::egui::ProgressBar::new(progress)
                    .text(format!("{:.0}%", progress * 100.0))
                    .animate(true),
            );
        });
        ui.separator();
    }

    // Data statistics
    ui.collapsing("Data", |ui| {
        ui.horizontal(|ui| {
            ui.label("Points:");
            ui.label(
                RichText::new(format!("{}", stats.n_points))
                    .strong()
                    .color(Color32::LIGHT_BLUE),
            );
        });

        if stats.embedding_dim > 0 {
            ui.horizontal(|ui| {
                ui.label("Dimensions:");
                ui.label(format!("{}", stats.embedding_dim));
            });
        }

        ui.horizontal(|ui| {
            ui.label("View:");
            if stats.has_sphere_coords {
                ui.label(
                    RichText::new("Sphere")
                        .color(Color32::LIGHT_GREEN),
                );
            } else {
                ui.label(
                    RichText::new("PCA")
                        .color(Color32::YELLOW),
                );
            }
        });

        if stats.last_step > 0 {
            ui.horizontal(|ui| {
                ui.label("Step:");
                ui.label(format!("{}", stats.last_step));
            });
        }

        ui.horizontal(|ui| {
            ui.label("Prominence:");
            ui.label(format!(
                "[{:.2}, {:.2}]",
                stats.prominence_range.0, stats.prominence_range.1
            ));
        });
    });

    ui.separator();

    // Layer visibility
    ui.collapsing("Layers", |ui| {
        ui.checkbox(show_points, "Points");
        ui.checkbox(show_tree, "Hierarchy");
        ui.checkbox(show_shells, "Shells");
    });

    ui.separator();

    // Rendering options
    ui.collapsing("Rendering", |ui| {
        ui.horizontal(|ui| {
            ui.label("Color by:");
        });
        ui.radio_value(color_mode, ColorMode::Prominence, "Prominence");
        ui.radio_value(color_mode, ColorMode::Radius, "Radius");

        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label("Point size:");
            ui.add(eframe::egui::Slider::new(point_size, 1.0..=20.0));
        });
    });

    ui.separator();

    // Monitoring controls
    ui.collapsing("Monitoring", |ui| {
        ui.checkbox(auto_refresh, "Auto-refresh");

        if ui.button("Manual Refresh").clicked() {
            // Handled by caller
        }

        if ui.button("Reset Camera").clicked() {
            // Handled by caller
        }
    });
}

/// Draw a help overlay.
pub fn draw_help_overlay(ui: &mut Ui) {
    eframe::egui::Frame::popup(ui.style()).show(ui, |ui| {
        ui.heading("Controls");
        ui.separator();
        ui.label("Left-click drag: Orbit");
        ui.label("Right-click drag: Pan");
        ui.label("Scroll: Zoom");
        ui.separator();
        ui.label("Press H to hide this help");
    });
}

/// Draw FPS counter.
pub fn draw_fps(ui: &mut Ui, fps: f32) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("{:.0} FPS", fps))
                .small()
                .color(Color32::GRAY),
        );
    });
}
