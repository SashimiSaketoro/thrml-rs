//! Main application shell for the visualizer.

use crate::camera::OrbitalCamera;
use crate::monitor::{CheckpointNotify, IpcMonitor, MonitoredData};
use crate::renderer::SphereRenderer;
use crate::widgets::{self, ColorMode, DataStats};

use anyhow::Result;
use eframe::egui;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Configuration for the visualizer.
#[derive(Clone, Debug)]
pub struct VizConfig {
    /// Input file path (standalone mode)
    pub input: Option<PathBuf>,

    /// Session ID for live monitoring
    pub monitor_session: Option<String>,

    /// Whether to show ROOTS hierarchy
    pub show_tree: bool,

    /// Number of ROOTS partitions
    pub partitions: usize,

    /// Initial window size
    pub window_size: (u32, u32),
}

impl Default for VizConfig {
    fn default() -> Self {
        Self {
            input: None,
            monitor_session: None,
            show_tree: false,
            partitions: 32,
            window_size: (1280, 800),
        }
    }
}

/// Main visualizer application.
pub struct VizApp {
    // Data
    data: MonitoredData,
    stats: DataStats,

    // Monitoring
    monitor: Option<IpcMonitor>,
    progress: f32,
    auto_refresh: bool,

    // Camera
    camera: OrbitalCamera,

    // UI state
    show_points: bool,
    show_tree: bool,
    show_shells: bool,
    color_mode: ColorMode,
    point_size: f32,
    show_help: bool,

    // Rendering
    renderer: Option<Arc<Mutex<SphereRenderer>>>,
    needs_upload: bool,

    // Performance
    last_frame: Instant,
    fps: f32,
}

impl VizApp {
    /// Create a new visualizer app.
    pub fn new(cc: &eframe::CreationContext<'_>, config: VizConfig) -> Self {
        // Initialize renderer if we have wgpu access
        let renderer = cc.wgpu_render_state.as_ref().map(|render_state| {
            let device = &render_state.device;
            let format = render_state.target_format;
            Arc::new(Mutex::new(SphereRenderer::new(device, format)))
        });

        // Set up monitoring if requested
        let monitor = config.monitor_session.as_ref().and_then(|session| {
            match IpcMonitor::start(session) {
                Ok(m) => {
                    log::info!("Started monitoring session: {}", session);
                    Some(m)
                }
                Err(e) => {
                    log::error!("Failed to start monitor: {}", e);
                    None
                }
            }
        });

        // Load initial data if provided
        let mut data = MonitoredData::new();
        if let Some(ref path) = config.input {
            // Try NPZ first, fall back to SafeTensors
            if path.extension().map(|e| e == "npz").unwrap_or(false) {
                match MonitoredData::load_npz(path) {
                    Ok(d) => data = d,
                    Err(e) => log::error!("Failed to load NPZ: {}", e),
                }
            } else {
                match MonitoredData::load_safetensors(path) {
                    Ok(d) => data = d,
                    Err(e) => log::error!("Failed to load SafeTensors: {}", e),
                }
            }
        }

        let stats = DataStats::from_data(
            data.n_points(),
            data.embedding_dim(),
            data.prominence(),
            data.has_sphere_coords(),
            data.last_step(),
        );

        // Auto-center camera on data
        let mut camera = OrbitalCamera::default();
        if let Some(positions) = data.positions() {
            if !positions.is_empty() {
                // Find center and extent of data
                let mut min = [f32::INFINITY; 3];
                let mut max = [f32::NEG_INFINITY; 3];
                for pos in positions {
                    for i in 0..3 {
                        min[i] = min[i].min(pos[i]);
                        max[i] = max[i].max(pos[i]);
                    }
                }
                let center = [
                    (min[0] + max[0]) / 2.0,
                    (min[1] + max[1]) / 2.0,
                    (min[2] + max[2]) / 2.0,
                ];
                let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
                let radius = (extent[0].powi(2) + extent[1].powi(2) + extent[2].powi(2)).sqrt() / 2.0;
                
                camera.target = center;
                camera.distance = radius * 2.5;
                log::info!("Auto-centered camera: center={:?}, radius={:.1}", center, radius);
            }
        }

        Self {
            data,
            stats,
            monitor,
            progress: 0.0,
            auto_refresh: true,
            camera,
            show_points: true,
            show_tree: config.show_tree,
            show_shells: true,
            color_mode: ColorMode::Prominence,
            point_size: 4.0,
            show_help: true,
            renderer,
            needs_upload: true,
            last_frame: Instant::now(),
            fps: 60.0,
        }
    }

    /// Update data from a checkpoint notification.
    fn update_from_checkpoint(&mut self, notify: &CheckpointNotify) {
        self.progress = notify.progress();

        if self.auto_refresh {
            match self.data.update(notify) {
                Ok(true) => {
                    log::info!("Reloaded data at step {}", notify.step);
                    self.needs_upload = true;
                    self.stats = DataStats::from_data(
                        self.data.n_points(),
                        self.data.embedding_dim(),
                        self.data.prominence(),
                        self.data.has_sphere_coords(),
                        self.data.last_step(),
                    );
                }
                Ok(false) => {}
                Err(e) => {
                    log::error!("Failed to reload data: {}", e);
                }
            }
        }
    }

    /// Upload point data to GPU.
    fn upload_points(&mut self, device: &wgpu::Device) {
        if let Some(ref renderer) = self.renderer {
            if let Ok(mut renderer) = renderer.lock() {
                if let (Some(positions), Some(prominence)) =
                    (self.data.positions(), self.data.prominence())
                {
                    let colors: Vec<f32> = match self.color_mode {
                        ColorMode::Prominence => prominence.to_vec(),
                        ColorMode::Radius => positions
                            .iter()
                            .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
                            .collect(),
                    };

                    renderer.point_cloud.upload_from_arrays(device, positions, &colors);

                    // Update color range
                    let min = colors.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = colors.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    renderer.set_color_range(min, max);
                }
            }
        }
        self.needs_upload = false;
    }
}

impl eframe::App for VizApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Calculate FPS
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt);
        self.last_frame = now;

        // Check for new checkpoint
        if let Some(ref monitor) = self.monitor {
            if let Some(notify) = monitor.poll() {
                self.update_from_checkpoint(&notify);
            }
        }

        // Upload data if needed
        if self.needs_upload {
            if let Some(render_state) = frame.wgpu_render_state() {
                self.upload_points(&render_state.device);
            }
        }

        // Control panel sidebar
        egui::SidePanel::left("controls")
            .default_width(200.0)
            .show(ctx, |ui| {
                widgets::draw_control_panel(
                    ui,
                    &self.stats,
                    self.progress,
                    &mut self.show_points,
                    &mut self.show_tree,
                    &mut self.show_shells,
                    &mut self.color_mode,
                    &mut self.point_size,
                    &mut self.auto_refresh,
                );

                ui.separator();
                widgets::draw_fps(ui, self.fps);
            });

        // Main 3D viewport
        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let (response, painter) = ui.allocate_painter(available, egui::Sense::drag());
            let rect = response.rect;

            // Update camera aspect ratio
            self.camera.set_aspect(rect.width(), rect.height());

            // Handle camera input
            self.camera.handle_input(&response);

            // Draw background
            painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(20, 20, 30));

            // Draw points using GPU renderer
            if self.show_points {
                if let Some(ref renderer) = self.renderer {
                    if let Ok(mut r) = renderer.lock() {
                        // Update camera uniforms
                        let view_proj = self.camera.view_projection_matrix();
                        let camera_pos = self.camera.position();
                        
                        r.set_view_proj(view_proj);
                        r.set_camera_pos(camera_pos);
                        r.set_point_size(self.point_size);
                    }

                    // Add paint callback
                    ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(
                        rect,
                        crate::renderer::SphereRenderCallback {
                            renderer: renderer.clone(),
                        },
                    ));
                }
            }

            // Draw status text if no data
            if !self.data.is_loaded() {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    if self.monitor.is_some() {
                        "Waiting for ingestion data..."
                    } else {
                        "No data loaded"
                    },
                    egui::FontId::proportional(24.0),
                    egui::Color32::GRAY,
                );
            } else {
                // Draw point count
                painter.text(
                    egui::pos2(rect.left() + 10.0, rect.top() + 10.0),
                    egui::Align2::LEFT_TOP,
                    format!("{} points", self.data.n_points()),
                    egui::FontId::proportional(14.0),
                    egui::Color32::WHITE,
                );
            }

            // Show help overlay
            if self.show_help {
                egui::Window::new("Help")
                    .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
                    .collapsible(false)
                    .resizable(false)
                    .show(ctx, |ui| {
                        widgets::draw_help_overlay(ui);
                        if ui.button("Close").clicked() {
                            self.show_help = false;
                        }
                    });
            }

            // Toggle help with H key
            if ctx.input(|i| i.key_pressed(egui::Key::H)) {
                self.show_help = !self.show_help;
            }

            // Request repaint for animation
            ctx.request_repaint();
        });
    }
}

/// Viridis colormap approximation
fn viridis_color(t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    
    // Simplified viridis: purple -> blue -> green -> yellow
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (68.0 + s * 20.0, 1.0 + s * 50.0, 84.0 + s * 70.0)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (88.0 - s * 50.0, 51.0 + s * 100.0, 154.0 - s * 30.0)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (38.0 + s * 80.0, 151.0 + s * 50.0, 124.0 - s * 80.0)
    } else {
        let s = (t - 0.75) / 0.25;
        (118.0 + s * 135.0, 201.0 + s * 50.0, 44.0 + s * 50.0)
    };
    
    egui::Color32::from_rgb(r as u8, g as u8, b as u8)
}

/// Run the visualizer application.
pub fn run(config: VizConfig) -> Result<()> {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([config.window_size.0 as f32, config.window_size.1 as f32])
            .with_title("TheSphere Visualizer"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "TheSphere Visualizer",
        options,
        Box::new(move |cc| Ok(Box::new(VizApp::new(cc, config)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {}", e))
}
