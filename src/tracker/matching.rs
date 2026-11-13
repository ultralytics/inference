// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Cost matrices and optimal assignment, the `matching` utilities of Ultralytics.

use ndarray::Array2;

use super::Track;

/// `f(track, detection)` for every pair of arena indices.
pub(super) fn matrix(
    tracks: &[usize],
    dets: &[usize],
    f: impl Fn(usize, usize) -> f64,
) -> Array2<f64> {
    Array2::from_shape_fn((tracks.len(), dets.len()), |(i, j)| f(tracks[i], dets[j]))
}

/// `1 - IoU * score` from the `IoU` cost when `fuse_score`, so confident detections match first.
pub(super) fn fuse(
    iou: &Array2<f64>,
    fuse_score: bool,
    all: &[Track],
    dets: &[usize],
) -> Array2<f64> {
    let mut cost = iou.clone();
    if fuse_score {
        for ((_, j), v) in cost.indexed_iter_mut() {
            *v = f64::from(1.0 - (1.0 - *v as f32) * all[dets[j]].score);
        }
    }
    cost
}

/// Optimal matching of rows to columns that pairs only costs up to `thresh`, the
/// `lap.lapjv(cost, extend_cost=True, cost_limit=thresh)` of Ultralytics. Like it, this runs
/// the Jonker-Volgenant solver of `lap` (BSD-2, Tomas Kazmar) on the cost extended with
/// unmatched rows and columns at `thresh / 2`, so equal costs resolve the same way.
///
/// Returns the `(row, col)` matches in row order, then the unmatched rows and columns.
pub(super) fn assign(
    cost: &Array2<f64>,
    thresh: f64,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let (rows, cols) = cost.dim();
    if rows == 0 || cols == 0 {
        return (Vec::new(), (0..rows).collect(), (0..cols).collect());
    }
    let (x, y) = lapjv(rows + cols, |i, j| match (i < rows, j < cols) {
        (true, true) => cost[[i, j]],
        (false, false) => 0.0,
        _ => thresh / 2.0,
    });
    let matches = (0..rows)
        .filter(|&i| x[i] < cols)
        .map(|i| (i, x[i]))
        .collect();
    let u_rows = (0..rows).filter(|&i| x[i] >= cols).collect();
    let u_cols = (0..cols).filter(|&j| y[j] >= rows).collect();
    (matches, u_rows, u_cols)
}

/// `lapjv_internal` of `lap`: the column of each row and the row of each column of the
/// minimum-cost assignment of an `n x n` cost.
// Kept as one function with `lap`'s exact float ties and fixed scan range so results match.
#[allow(clippy::too_many_lines, clippy::float_cmp, clippy::mut_range_bound)]
fn lapjv(n: usize, cost: impl Fn(usize, usize) -> f64) -> (Vec<usize>, Vec<usize>) {
    const LARGE: f64 = 1e6;
    let (mut x, mut y, mut v) = (vec![-1isize; n], vec![0isize; n], vec![LARGE; n]);
    // Column reduction and reduction transfer.
    for i in 0..n {
        for j in 0..n {
            if cost(i, j) < v[j] {
                (v[j], y[j]) = (cost(i, j), i as isize);
            }
        }
    }
    let mut unique = vec![true; n];
    for j in (0..n).rev() {
        let i = y[j] as usize;
        if x[i] < 0 {
            x[i] = j as isize;
        } else {
            (unique[i], y[j]) = (false, -1);
        }
    }
    let mut free = Vec::new();
    for i in 0..n {
        if x[i] < 0 {
            free.push(i);
        } else if unique[i] {
            let j = x[i] as usize;
            let min = (0..n).filter(|&k| k != j).map(|k| cost(i, k) - v[k]);
            v[j] -= min.fold(LARGE, |m, c| if c < m { c } else { m });
        }
    }
    // Augmenting row reduction, twice.
    for _ in 0..2 {
        if free.is_empty() {
            break;
        }
        let (mut current, mut kept, mut rounds) = (0, 0, 0);
        let n_free = free.len();
        while current < n_free {
            rounds += 1;
            let i = free[current];
            current += 1;
            let (mut j1, mut v1, mut j2, mut v2) = (0, cost(i, 0) - v[0], -1isize, LARGE);
            for j in 1..n {
                let c = cost(i, j) - v[j];
                if c < v2 {
                    if c >= v1 {
                        (v2, j2) = (c, j as isize);
                    } else {
                        (v2, v1, j2, j1) = (v1, c, j1 as isize, j);
                    }
                }
            }
            let mut i0 = y[j1];
            let v1_new = v[j1] - (v2 - v1);
            let lowers = v1_new < v[j1];
            if rounds < current * n {
                if lowers {
                    v[j1] = v1_new;
                } else if i0 >= 0 && j2 >= 0 {
                    j1 = j2 as usize;
                    i0 = y[j1];
                }
                if i0 >= 0 {
                    if lowers {
                        current -= 1;
                        free[current] = i0 as usize;
                    } else {
                        free[kept] = i0 as usize;
                        kept += 1;
                    }
                }
            } else if i0 >= 0 {
                free[kept] = i0 as usize;
                kept += 1;
            }
            (x[i], y[j1]) = (j1 as isize, i as isize);
        }
        free.truncate(kept);
    }
    // Augment along shortest paths (Dijkstra) from each row still free.
    let mut pred = vec![0; n];
    for &start in &free {
        let mut cols: Vec<usize> = (0..n).collect();
        let mut d: Vec<f64> = (0..n).map(|j| cost(start, j) - v[j]).collect();
        pred.fill(start);
        let (mut lo, mut hi, mut ready) = (0, 0, 0);
        let mut end = None;
        while end.is_none() {
            if lo == hi {
                // Columns at the minimum distance go next.
                ready = lo;
                hi = lo + 1;
                let mut mind = d[cols[lo]];
                for k in lo + 1..n {
                    let j = cols[k];
                    if d[j] <= mind {
                        if d[j] < mind {
                            (hi, mind) = (lo, d[j]);
                        }
                        (cols[k], cols[hi]) = (cols[hi], j);
                        hi += 1;
                    }
                }
                end = cols[lo..hi].iter().rev().find(|&&j| y[j] < 0).copied();
            }
            if end.is_none() {
                // Relax the remaining columns through the next scanned one.
                let (mut l, mut h) = (lo, hi);
                'scan: while l != h {
                    let j = cols[l];
                    l += 1;
                    let (i, mind) = (y[j] as usize, d[j]);
                    let shift = cost(i, j) - v[j] - mind;
                    for k in h..n {
                        let j = cols[k];
                        let reduced = cost(i, j) - v[j] - shift;
                        if reduced < d[j] {
                            (d[j], pred[j]) = (reduced, i);
                            if reduced == mind {
                                if y[j] < 0 {
                                    end = Some(j);
                                    break 'scan;
                                }
                                (cols[k], cols[h]) = (cols[h], j);
                                h += 1;
                            }
                        }
                    }
                }
                if end.is_none() {
                    (lo, hi) = (l, h);
                }
            }
        }
        let mind = d[cols[lo]];
        for &j in &cols[..ready] {
            v[j] += d[j] - mind;
        }
        let mut j = end.unwrap_or_default();
        loop {
            let i = pred[j];
            y[j] = i as isize;
            let next = x[i] as usize;
            x[i] = j as isize;
            j = next;
            if i == start {
                break;
            }
        }
    }
    (
        x.into_iter().map(|j| j as usize).collect(),
        y.into_iter().map(|i| i as usize).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assign_matches_below_threshold() {
        let cost = ndarray::array![[0.1, 0.9], [0.2, 0.3], [0.95, 0.99]];
        let (matches, u_rows, u_cols) = assign(&cost, 0.8);
        assert_eq!(matches, vec![(0, 0), (1, 1)]);
        assert_eq!((u_rows, u_cols), (vec![2], vec![]));
        // Ties resolve like `lap.lapjv`.
        assert_eq!(
            assign(&ndarray::array![[0.067, 0.067]], 0.5).0,
            vec![(0, 1)]
        );
        assert_eq!(
            assign(&ndarray::array![[0.26, 0.26, 0.26]], 0.5).0,
            vec![(0, 0)]
        );
    }
}
