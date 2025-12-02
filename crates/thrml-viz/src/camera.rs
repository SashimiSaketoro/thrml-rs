//! Orbital camera for 3D navigation.
//!
//! Provides orbit, pan, and zoom controls using mouse input.

use std::f32::consts::PI;

/// Orbital camera that rotates around a target point.
pub struct OrbitalCamera {
    /// Target point to orbit around
    pub target: [f32; 3],

    /// Distance from target
    pub distance: f32,

    /// Azimuthal angle (rotation around Y axis)
    pub azimuth: f32,

    /// Polar angle (rotation from Y axis)
    pub elevation: f32,

    /// Field of view in radians
    pub fov: f32,

    /// Near clipping plane
    pub near: f32,

    /// Far clipping plane
    pub far: f32,

    /// Aspect ratio (width / height)
    pub aspect: f32,

    /// Orbit sensitivity
    pub orbit_sensitivity: f32,

    /// Zoom sensitivity
    pub zoom_sensitivity: f32,

    /// Pan sensitivity
    pub pan_sensitivity: f32,

    /// Minimum distance
    pub min_distance: f32,

    /// Maximum distance
    pub max_distance: f32,
}

impl Default for OrbitalCamera {
    fn default() -> Self {
        Self {
            target: [0.0, 0.0, 0.0],
            distance: 500.0,
            azimuth: 0.0,
            elevation: PI / 4.0, // 45 degrees
            fov: PI / 4.0,       // 45 degrees
            near: 1.0,
            far: 10000.0,
            aspect: 1.0,
            orbit_sensitivity: 0.005,
            zoom_sensitivity: 0.1,
            pan_sensitivity: 0.5,
            min_distance: 10.0,
            max_distance: 5000.0,
        }
    }
}

impl OrbitalCamera {
    /// Create a new orbital camera.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the aspect ratio.
    pub fn set_aspect(&mut self, width: f32, height: f32) {
        if height > 0.0 {
            self.aspect = width / height;
        }
    }

    /// Get the camera position in world space.
    pub fn position(&self) -> [f32; 3] {
        let x = self.target[0] + self.distance * self.elevation.sin() * self.azimuth.sin();
        let y = self.target[1] + self.distance * self.elevation.cos();
        let z = self.target[2] + self.distance * self.elevation.sin() * self.azimuth.cos();
        [x, y, z]
    }

    /// Get the view matrix (camera transform).
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let pos = self.position();
        let target = self.target;

        // Look-at matrix
        let forward = normalize([
            target[0] - pos[0],
            target[1] - pos[1],
            target[2] - pos[2],
        ]);
        let up = [0.0, 1.0, 0.0];
        let right = normalize(cross(forward, up));
        let up = cross(right, forward);

        [
            [right[0], up[0], -forward[0], 0.0],
            [right[1], up[1], -forward[1], 0.0],
            [right[2], up[2], -forward[2], 0.0],
            [
                -dot(right, pos),
                -dot(up, pos),
                dot(forward, pos),
                1.0,
            ],
        ]
    }

    /// Get the projection matrix (perspective).
    pub fn projection_matrix(&self) -> [[f32; 4]; 4] {
        let f = 1.0 / (self.fov / 2.0).tan();
        let nf = 1.0 / (self.near - self.far);

        [
            [f / self.aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (self.far + self.near) * nf, -1.0],
            [0.0, 0.0, 2.0 * self.far * self.near * nf, 0.0],
        ]
    }

    /// Get the combined view-projection matrix.
    pub fn view_projection_matrix(&self) -> [[f32; 4]; 4] {
        let view = self.view_matrix();
        let proj = self.projection_matrix();
        mat4_mul(proj, view)
    }

    /// Handle mouse drag for orbiting.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.azimuth -= dx * self.orbit_sensitivity;
        self.elevation -= dy * self.orbit_sensitivity;

        // Clamp elevation to avoid gimbal lock
        self.elevation = self.elevation.clamp(0.01, PI - 0.01);

        // Wrap azimuth
        if self.azimuth > PI * 2.0 {
            self.azimuth -= PI * 2.0;
        } else if self.azimuth < 0.0 {
            self.azimuth += PI * 2.0;
        }
    }

    /// Handle scroll for zooming.
    pub fn zoom(&mut self, delta: f32) {
        self.distance *= 1.0 - delta * self.zoom_sensitivity;
        self.distance = self.distance.clamp(self.min_distance, self.max_distance);
    }

    /// Handle right-click drag for panning.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        // Get camera right and up vectors
        let pos = self.position();
        let forward = normalize([
            self.target[0] - pos[0],
            self.target[1] - pos[1],
            self.target[2] - pos[2],
        ]);
        let up = [0.0, 1.0, 0.0];
        let right = normalize(cross(forward, up));
        let up = cross(right, forward);

        // Move target
        let scale = self.distance * self.pan_sensitivity * 0.001;
        self.target[0] -= right[0] * dx * scale + up[0] * dy * scale;
        self.target[1] -= right[1] * dx * scale + up[1] * dy * scale;
        self.target[2] -= right[2] * dx * scale + up[2] * dy * scale;
    }

    /// Handle egui input.
    pub fn handle_input(&mut self, response: &eframe::egui::Response) {
        // Orbit with left-click drag
        if response.dragged_by(eframe::egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            self.orbit(delta.x, delta.y);
        }

        // Pan with right-click drag
        if response.dragged_by(eframe::egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            self.pan(delta.x, delta.y);
        }

        // Zoom with scroll
        if response.hovered() {
            let scroll = response.ctx.input(|i| i.raw_scroll_delta.y);
            if scroll.abs() > 0.0 {
                self.zoom(scroll * 0.01);
            }
        }
    }

    /// Reset camera to default view.
    pub fn reset(&mut self) {
        self.target = [0.0, 0.0, 0.0];
        self.distance = 500.0;
        self.azimuth = 0.0;
        self.elevation = PI / 4.0;
    }

    /// Focus on a bounding box.
    pub fn focus_on(&mut self, center: [f32; 3], radius: f32) {
        self.target = center;
        self.distance = radius * 2.5;
        self.distance = self.distance.clamp(self.min_distance, self.max_distance);
    }
}

// Vector math helpers
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0001 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_position() {
        let camera = OrbitalCamera::default();
        let pos = camera.position();

        // Should be at distance from origin
        let dist = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        assert!((dist - camera.distance).abs() < 0.1);
    }

    #[test]
    fn test_view_projection() {
        let camera = OrbitalCamera::default();
        let vp = camera.view_projection_matrix();

        // Should be a valid 4x4 matrix (non-zero determinant)
        let det = vp[0][0] * vp[1][1] - vp[0][1] * vp[1][0];
        assert!(det.abs() > 0.0001);
    }
}
