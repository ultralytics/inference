// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Constant-velocity Kalman filter of a box, the `KalmanFilterXYAH` and `KalmanFilterXYWH` of
//! Ultralytics.

use super::{Mat8, State, Track, TrackerType};

/// Kalman filter position noise relative to the box size.
const STD_POS: f64 = 1.0 / 20.0;
/// Kalman filter velocity noise relative to the box size.
const STD_VEL: f64 = 1.0 / 160.0;

impl TrackerType {
    /// Kalman state and covariance of a new track at the center box `xywh`.
    pub(super) fn initiate(self, xywh: [f64; 4]) -> ([f64; 8], Mat8) {
        let z = self.measure(xywh);
        let pos = self.std(&z, 2.0 * STD_POS, 1e-2);
        let vel = self.std(&z, 10.0 * STD_VEL, 1e-5);
        let mut mean = [0.0; 8];
        let mut cov = [[0.0; 8]; 8];
        for i in 0..4 {
            mean[i] = z[i];
            cov[i][i] = pos[i] * pos[i];
            cov[i + 4][i + 4] = vel[i] * vel[i];
        }
        (mean, cov)
    }

    /// Kalman measurement of a center box.
    pub(super) fn measure(self, [x, y, w, h]: [f64; 4]) -> [f64; 4] {
        if self.xyah() {
            [x, y, w / h, h]
        } else {
            [x, y, w, h]
        }
    }

    /// Standard deviations of the measurement `m`, `weight` times the box size. The aspect
    /// ratio of an `xyah` state uses the fixed `aspect`.
    pub(super) fn std(self, m: &[f64], weight: f64, aspect: f64) -> [f64; 4] {
        if self.xyah() {
            [weight * m[3], weight * m[3], aspect, weight * m[3]]
        } else {
            [m[2], m[3], m[2], m[3]].map(|s| weight * s)
        }
    }
}

impl Track {
    /// Kalman prediction with a constant velocity.
    pub(super) fn kf_predict(&mut self) {
        let pos = self.kind.std(&self.mean, STD_POS, 1e-2);
        let vel = self.kind.std(&self.mean, STD_VEL, 1e-5);
        // mean = F mean and cov = F cov F^T + Q, for F = [[I, I], [0, I]].
        for i in 0..4 {
            self.mean[i] += self.mean[i + 4];
            for j in 0..8 {
                self.cov[i][j] += self.cov[i + 4][j];
            }
        }
        for row in &mut self.cov {
            for j in 0..4 {
                row[j] += row[j + 4];
            }
        }
        for i in 0..4 {
            self.cov[i][i] += pos[i] * pos[i];
            self.cov[i + 4][i + 4] += vel[i] * vel[i];
        }
    }

    /// Kalman prediction, where a track that is not tracked keeps its size.
    pub(super) fn predict(&mut self) {
        if self.state != State::Tracked {
            self.mean[7] = 0.0;
            if !self.kind.xyah() {
                self.mean[6] = 0.0;
            }
        }
        self.kf_predict();
    }

    /// Kalman correction with measurement `z`. A `confidence` scales the measurement noise by
    /// `max(1 - confidence, 0.05)` (NSA Kalman).
    pub(super) fn kf_update(&mut self, z: [f64; 4], confidence: Option<f32>) {
        let std = self.kind.std(&self.mean, STD_POS, 1e-1);
        let scale = confidence.map_or(1.0, |c| f64::from((1.0 - c).max(0.05)));
        // Innovation covariance S = H cov H^T + R, then K^T = S^-1 H cov.
        let mut s = [[0.0; 4]; 4];
        let mut kt = [[0.0; 8]; 4];
        for i in 0..4 {
            s[i].copy_from_slice(&self.cov[i][..4]);
            s[i][i] += std[i] * std[i] * scale;
            kt[i] = self.cov[i];
        }
        let h_cov = kt;
        solve(&mut s, &mut kt);
        // mean += K (z - H mean) and cov -= K S K^T, which is K H cov.
        let innovation: [f64; 4] = std::array::from_fn(|k| z[k] - self.mean[k]);
        for i in 0..8 {
            for k in 0..4 {
                self.mean[i] += kt[k][i] * innovation[k];
                for j in 0..8 {
                    self.cov[i][j] -= kt[k][i] * h_cov[k][j];
                }
            }
        }
    }
}

/// Solve `s x = b` in place for the symmetric positive definite `s`, leaving `x` in `b`.
/// Gauss-Jordan elimination needs no pivoting for such a matrix.
fn solve(s: &mut [[f64; 4]; 4], b: &mut [[f64; 8]; 4]) {
    for i in 0..4 {
        let p = s[i][i];
        s[i].iter_mut().for_each(|v| *v /= p);
        b[i].iter_mut().for_each(|v| *v /= p);
        for r in (0..4).filter(|&r| r != i) {
            let f = s[r][i];
            for j in 0..4 {
                s[r][j] -= f * s[i][j];
            }
            for j in 0..8 {
                b[r][j] -= f * b[i][j];
            }
        }
    }
}
