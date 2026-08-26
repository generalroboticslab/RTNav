/**
 * Scene Graph Acceleration Module
 * 
 * High-performance C++ implementation of hot-path operations in object tracking:
 * - 3D bounding box IoU
 * - Feature cosine similarity
 * - Batch association scoring
 * 
 * Uses pybind11 for Python bindings.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include <string>

namespace py = pybind11;


/**
 * Score candidate pairs passed from Python (with KDTree pre-filtering).
 * This avoids O(N×M) by only scoring the candidates Python found nearby.
 * 
 * @param det_indices Which detection each candidate belongs to
 * @param node_indices Which node each candidate pairs with
 * @param det_bboxes All detection bboxes (N, 2, 3)
 * @param det_features All detection features (N, D)
 * @param det_labels All detection labels
 * @param det_centroids All detection centroids (N, 3)
 * @param node_bboxes All node bboxes (M, 2, 3)
 * @param node_features All node features (M, D)
 * @param node_labels All node labels
 * @param node_centroids All node centroids (M, 3)
 * @return scores for each candidate pair
 */
py::array_t<double> score_candidates(
    py::array_t<int> det_indices,
    py::array_t<int> node_indices,
    py::array_t<double> det_bboxes,
    py::array_t<double> det_features,
    py::array_t<int> det_labels,
    py::array_t<double> det_centroids,
    py::array_t<double> node_bboxes,
    py::array_t<double> node_features,
    py::array_t<int> node_labels,
    py::array_t<double> node_centroids
) {
    auto di = det_indices.unchecked<1>();
    auto ni = node_indices.unchecked<1>();
    auto db = det_bboxes.unchecked<3>();
    auto df = det_features.unchecked<2>();
    auto dl = det_labels.unchecked<1>();
    auto dc = det_centroids.unchecked<2>();
    auto nb = node_bboxes.unchecked<3>();
    auto nf = node_features.unchecked<2>();
    auto nl = node_labels.unchecked<1>();
    auto nc = node_centroids.unchecked<2>();
    
    ssize_t num_candidates = di.shape(0);
    ssize_t feat_dim = df.shape(1);
    
    py::array_t<double> scores(num_candidates);
    auto s = scores.mutable_unchecked<1>();
    
    for (ssize_t c = 0; c < num_candidates; ++c) {
        int d = di(c);
        int n = ni(c);
        
        // Detection data
        double det_min_x = db(d, 0, 0), det_min_y = db(d, 0, 1), det_min_z = db(d, 0, 2);
        double det_max_x = db(d, 1, 0), det_max_y = db(d, 1, 1), det_max_z = db(d, 1, 2);
        double det_size = std::sqrt(
            (det_max_x - det_min_x) * (det_max_x - det_min_x) +
            (det_max_y - det_min_y) * (det_max_y - det_min_y) +
            (det_max_z - det_min_z) * (det_max_z - det_min_z)
        );
        
        // Node data
        double node_min_x = nb(n, 0, 0), node_min_y = nb(n, 0, 1), node_min_z = nb(n, 0, 2);
        double node_max_x = nb(n, 1, 0), node_max_y = nb(n, 1, 1), node_max_z = nb(n, 1, 2);
        double node_size = std::sqrt(
            (node_max_x - node_min_x) * (node_max_x - node_min_x) +
            (node_max_y - node_min_y) * (node_max_y - node_min_y) +
            (node_max_z - node_min_z) * (node_max_z - node_min_z)
        );
        
        bool labels_match = (dl(d) == nl(n));
        
        // Margin check
        double margin = labels_match ? 
            1.0 * std::max({det_size, node_size, 1.0}) :
            0.3 * std::max({det_size, node_size, 1.0});
        
        if (det_max_x < node_min_x - margin || det_min_x > node_max_x + margin ||
            det_max_y < node_min_y - margin || det_min_y > node_max_y + margin ||
            det_max_z < node_min_z - margin || det_min_z > node_max_z + margin) {
            s(c) = -1.0;  // Invalid candidate
            continue;
        }
        
        // Geometric IoU
        double int_min_x = std::max(det_min_x, node_min_x);
        double int_min_y = std::max(det_min_y, node_min_y);
        double int_min_z = std::max(det_min_z, node_min_z);
        double int_max_x = std::min(det_max_x, node_max_x);
        double int_max_y = std::min(det_max_y, node_max_y);
        double int_max_z = std::min(det_max_z, node_max_z);
        
        double geo_sim = 0.0;
        if (int_min_x < int_max_x && int_min_y < int_max_y && int_min_z < int_max_z) {
            double int_vol = (int_max_x - int_min_x) * (int_max_y - int_min_y) * (int_max_z - int_min_z);
            double vol1 = (det_max_x - det_min_x) * (det_max_y - det_min_y) * (det_max_z - det_min_z);
            double vol2 = (node_max_x - node_min_x) * (node_max_y - node_min_y) * (node_max_z - node_min_z);
            double union_vol = vol1 + vol2 - int_vol;
            if (union_vol > 1e-8) {
                geo_sim = int_vol / union_vol;
            }
        }
        
        // Distance
        double dx = nc(n, 0) - dc(d, 0);
        double dy = nc(n, 1) - dc(d, 1);
        double dz = nc(n, 2) - dc(d, 2);
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        double avg_size = std::max({det_size, node_size, 0.5});
        double dist_score = labels_match ?
            std::exp(-dist / (avg_size * 3.0)) :
            std::exp(-dist / (avg_size * 1.5));
        
        // Semantic similarity
        double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
        for (ssize_t f = 0; f < feat_dim; ++f) {
            dot += df(d, f) * nf(n, f);
            norm1 += df(d, f) * df(d, f);
            norm2 += nf(n, f) * nf(n, f);
        }
        norm1 = std::sqrt(norm1) + 1e-8;
        norm2 = std::sqrt(norm2) + 1e-8;
        double sem_sim = (dot / (norm1 * norm2) + 1.0) / 2.0;
        
        // Point cloud estimate
        double pc_sim = labels_match ?
            0.5 * geo_sim + 0.5 * dist_score :
            (geo_sim > 0.3 ? 0.7 : 0.0);
        
        // Combined score
        double score;
        if (labels_match) {
            score = 0.35 + 0.05 * geo_sim + 0.10 * dist_score + 0.30 * sem_sim + 0.20 * pc_sim;
        } else {
            score = 0.25 * geo_sim + 0.20 * dist_score + 0.35 * sem_sim + 0.20 * pc_sim;
        }
        
        // Threshold
        double min_score = labels_match ? 0.20 : 0.50;
        s(c) = (score >= min_score) ? score : -1.0;
    }
    
    return scores;
}

// ============================================================================
// NEW OPTIMIZATIONS: Voxel IoU, Merge Candidates, Batch Node Update
// ============================================================================


/**
 * Batch compute voxel IoU for multiple pairs.
 * 
 * @param all_voxel_keys List of voxel key arrays for each node
 * @param pairs Nx2 array of (node_i, node_j) pairs to compute IoU for
 * @return Array of IoU values for each pair
 */
py::array_t<double> batch_voxel_iou(
    py::list all_voxel_keys,
    py::array_t<int> pairs
) {
    auto p = pairs.unchecked<2>();
    ssize_t num_pairs = p.shape(0);
    
    py::array_t<double> result(num_pairs);
    auto r = result.mutable_unchecked<1>();
    
    // Pre-build hash sets for all nodes
    std::vector<std::unordered_set<int64_t>> node_sets;
    node_sets.reserve(all_voxel_keys.size());
    
    for (size_t i = 0; i < all_voxel_keys.size(); ++i) {
        py::array_t<int> keys = all_voxel_keys[i].cast<py::array_t<int>>();
        auto k = keys.unchecked<2>();
        ssize_t n = k.shape(0);
        
        std::unordered_set<int64_t> node_set;
        node_set.reserve(n);
        for (ssize_t j = 0; j < n; ++j) {
            int64_t key = (int64_t)k(j, 0) + 
                          (int64_t)k(j, 1) * 10000LL + 
                          (int64_t)k(j, 2) * 100000000LL;
            node_set.insert(key);
        }
        node_sets.push_back(std::move(node_set));
    }
    
    // Compute IoU for each pair
    for (ssize_t i = 0; i < num_pairs; ++i) {
        int idx1 = p(i, 0);
        int idx2 = p(i, 1);
        
        if (idx1 < 0 || idx1 >= (int)node_sets.size() ||
            idx2 < 0 || idx2 >= (int)node_sets.size()) {
            r(i) = 0.0;
            continue;
        }
        
        const auto& set1 = node_sets[idx1];
        const auto& set2 = node_sets[idx2];
        
        if (set1.empty() || set2.empty()) {
            r(i) = 0.0;
            continue;
        }
        
        // Count intersection
        int64_t intersection = 0;
        const auto& smaller = (set1.size() < set2.size()) ? set1 : set2;
        const auto& larger = (set1.size() < set2.size()) ? set2 : set1;
        
        for (const auto& key : smaller) {
            if (larger.count(key)) {
                intersection++;
            }
        }
        
        int64_t union_size = (int64_t)set1.size() + (int64_t)set2.size() - intersection;
        r(i) = (union_size > 0) ? (double)intersection / (double)union_size : 0.0;
    }
    
    return result;
}

/**
 * Find merge candidates efficiently using spatial indexing.
 * Returns pairs of nodes that:
 * 1. Have the same label
 * 2. Are within max_distance of each other
 * 
 * Uses simple grid-based spatial hashing for O(N) instead of O(N²).
 * 
 * @param node_centroids Node centroids (N, 3)
 * @param node_labels Node labels as indices
 * @param max_distance Maximum centroid distance to consider
 * @return Pairs of (node_i, node_j) candidates
 */
py::array_t<int> find_merge_candidates(
    py::array_t<double> node_centroids,
    py::array_t<int> node_labels,
    double max_distance
) {
    auto nc = node_centroids.unchecked<2>();
    auto nl = node_labels.unchecked<1>();
    
    ssize_t num_nodes = nc.shape(0);
    
    // Grid cell size should be max_distance to ensure nearby objects are in adjacent cells
    double cell_size = max_distance;
    
    // Build spatial hash map: cell_key -> list of node indices
    std::unordered_map<int64_t, std::vector<int>> spatial_hash;
    
    auto cell_key = [cell_size](double x, double y, double z) -> int64_t {
        int cx = (int)std::floor(x / cell_size);
        int cy = (int)std::floor(y / cell_size);
        int cz = (int)std::floor(z / cell_size);
        return (int64_t)cx + (int64_t)cy * 10000LL + (int64_t)cz * 100000000LL;
    };
    
    // Insert all nodes into spatial hash
    for (ssize_t i = 0; i < num_nodes; ++i) {
        int64_t key = cell_key(nc(i, 0), nc(i, 1), nc(i, 2));
        spatial_hash[key].push_back(i);
    }
    
    // Find candidate pairs
    std::vector<std::pair<int, int>> candidates;
    double max_dist_sq = max_distance * max_distance;
    
    for (ssize_t i = 0; i < num_nodes; ++i) {
        int label_i = nl(i);
        double cx = nc(i, 0), cy = nc(i, 1), cz = nc(i, 2);
        
        // Check this cell and all 26 neighboring cells
        int base_cx = (int)std::floor(cx / cell_size);
        int base_cy = (int)std::floor(cy / cell_size);
        int base_cz = (int)std::floor(cz / cell_size);
        
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    int64_t neighbor_key = (int64_t)(base_cx + dx) + 
                                           (int64_t)(base_cy + dy) * 10000LL + 
                                           (int64_t)(base_cz + dz) * 100000000LL;
                    
                    auto it = spatial_hash.find(neighbor_key);
                    if (it == spatial_hash.end()) continue;
                    
                    for (int j : it->second) {
                        // Only check j > i to avoid duplicates
                        if (j <= (int)i) continue;
                        
                        // Same label check
                        if (nl(j) != label_i) continue;
                        
                        // Distance check
                        double dx2 = nc(j, 0) - cx;
                        double dy2 = nc(j, 1) - cy;
                        double dz2 = nc(j, 2) - cz;
                        double dist_sq = dx2*dx2 + dy2*dy2 + dz2*dz2;
                        
                        if (dist_sq <= max_dist_sq) {
                            candidates.push_back({(int)i, j});
                        }
                    }
                }
            }
        }
    }
    
    // Convert to numpy array
    py::array_t<int> result({(ssize_t)candidates.size(), (ssize_t)2});
    auto r = result.mutable_unchecked<2>();
    
    for (size_t i = 0; i < candidates.size(); ++i) {
        r(i, 0) = candidates[i].first;
        r(i, 1) = candidates[i].second;
    }
    
    return result;
}


/**
 * Batch compute bbox IoU with containment for merge candidates.
 * 
 * @param all_bboxes All node bboxes (N, 2, 3)
 * @param pairs Candidate pairs (M, 2)
 * @return IoU values (M,)
 */
py::array_t<double> batch_bbox_iou_containment(
    py::array_t<double> all_bboxes,
    py::array_t<int> pairs
) {
    auto b = all_bboxes.unchecked<3>();
    auto p = pairs.unchecked<2>();
    
    ssize_t num_pairs = p.shape(0);
    
    py::array_t<double> result(num_pairs);
    auto r = result.mutable_unchecked<1>();
    
    for (ssize_t i = 0; i < num_pairs; ++i) {
        int idx1 = p(i, 0);
        int idx2 = p(i, 1);
        
        // Extract bboxes
        double min1_x = b(idx1, 0, 0), min1_y = b(idx1, 0, 1), min1_z = b(idx1, 0, 2);
        double max1_x = b(idx1, 1, 0), max1_y = b(idx1, 1, 1), max1_z = b(idx1, 1, 2);
        double min2_x = b(idx2, 0, 0), min2_y = b(idx2, 0, 1), min2_z = b(idx2, 0, 2);
        double max2_x = b(idx2, 1, 0), max2_y = b(idx2, 1, 1), max2_z = b(idx2, 1, 2);
        
        // Intersection
        double int_min_x = std::max(min1_x, min2_x);
        double int_min_y = std::max(min1_y, min2_y);
        double int_min_z = std::max(min1_z, min2_z);
        double int_max_x = std::min(max1_x, max2_x);
        double int_max_y = std::min(max1_y, max2_y);
        double int_max_z = std::min(max1_z, max2_z);
        
        if (int_min_x >= int_max_x || int_min_y >= int_max_y || int_min_z >= int_max_z) {
            r(i) = 0.0;
            continue;
        }
        
        double int_vol = (int_max_x - int_min_x) * (int_max_y - int_min_y) * (int_max_z - int_min_z);
        double vol1 = (max1_x - min1_x) * (max1_y - min1_y) * (max1_z - min1_z);
        double vol2 = (max2_x - min2_x) * (max2_y - min2_y) * (max2_z - min2_z);
        double union_vol = vol1 + vol2 - int_vol;
        
        if (union_vol < 1e-8) {
            r(i) = 0.0;
            continue;
        }
        
        double iou = int_vol / union_vol;
        
        // Containment bonus
        double smaller_vol = std::min(vol1, vol2);
        if (smaller_vol > 1e-8) {
            double containment = int_vol / smaller_vol;
            if (containment > 0.7) {
                iou = std::min(1.0, iou + 0.3 * (containment - 0.7));
            }
        }
        
        r(i) = iou;
    }
    
    return result;
}


/**
 * Project 2D detections to 3D using depth map.
 *
 * This is the main det3d acceleration - handles a single camera with all detections.
 *
 * @param depth Depth image (H, W) float32
 * @param rgb RGB image (H, W, 3) uint8
 * @param bboxes 2D bounding boxes (N, 4) as [x1, y1, x2, y2]
 * @param intrinsics Camera intrinsics (3, 3)
 * @param rotation Rotation matrix from camera to world (3, 3)
 * @param translation Translation vector from camera to world (3,)
 * @param max_points Maximum points per detection
 * @return Tuple of (valid_mask, centroids, bbox_mins, bbox_maxs, point_clouds, colors)
 */
std::tuple<
    py::array_t<bool>,                // valid_mask (N,)
    py::array_t<double>,              // centroids (N, 3)
    py::array_t<double>,              // bbox_mins (N, 3)
    py::array_t<double>,              // bbox_maxs (N, 3)
    std::vector<py::array_t<double>>, // point_clouds list of (K, 3)
    std::vector<py::array_t<uint8_t>> // colors list of (K, 3)
    >
project_detections_to_3d(
    py::array_t<float> depth,
    py::array_t<uint8_t> rgb,
    py::array_t<double> bboxes,
    py::array_t<double> intrinsics,
    py::array_t<double> rotation,
    py::array_t<double> translation,
    int max_points = 1000)
{
    auto d = depth.unchecked<2>();
    auto r = rgb.unchecked<3>();
    auto b = bboxes.unchecked<2>();
    auto K = intrinsics.unchecked<2>();
    auto R = rotation.unchecked<2>();
    auto t = translation.unchecked<1>();

    int H = d.shape(0);
    int W = d.shape(1);
    ssize_t num_dets = b.shape(0);

    // Camera intrinsics
    double fx = K(0, 0), fy = K(1, 1);
    double cx = K(0, 2), cy = K(1, 2);

    // Rotation matrix elements
    double R00 = R(0, 0), R01 = R(0, 1), R02 = R(0, 2);
    double R10 = R(1, 0), R11 = R(1, 1), R12 = R(1, 2);
    double R20 = R(2, 0), R21 = R(2, 1), R22 = R(2, 2);
    double tx = t(0), ty = t(1), tz = t(2);

    // Pre-compute normalized pixel coordinates
    std::vector<double> u_norm(W), v_norm(H);
    for (int u = 0; u < W; ++u)
    {
        u_norm[u] = (u - cx) / fx;
    }
    for (int v = 0; v < H; ++v)
    {
        v_norm[v] = (v - cy) / fy;
    }

    // Output arrays
    py::array_t<bool> valid_mask(num_dets);
    py::array_t<double> centroids({num_dets, ssize_t(3)});
    py::array_t<double> bbox_mins({num_dets, ssize_t(3)});
    py::array_t<double> bbox_maxs({num_dets, ssize_t(3)});
    std::vector<py::array_t<double>> point_clouds;
    std::vector<py::array_t<uint8_t>> colors_out;

    auto vm = valid_mask.mutable_unchecked<1>();
    auto cent = centroids.mutable_unchecked<2>();
    auto bmin = bbox_mins.mutable_unchecked<2>();
    auto bmax = bbox_maxs.mutable_unchecked<2>();

    // Random number generator for downsampling
    std::mt19937 rng(42);

    for (ssize_t det_idx = 0; det_idx < num_dets; ++det_idx)
    {
        // Get bbox
        int x1 = std::max(0, (int)b(det_idx, 0));
        int y1 = std::max(0, (int)b(det_idx, 1));
        int x2 = std::min(W, (int)b(det_idx, 2));
        int y2 = std::min(H, (int)b(det_idx, 3));

        if (x2 <= x1 || y2 <= y1)
        {
            vm(det_idx) = false;
            for (int i = 0; i < 3; ++i)
            {
                cent(det_idx, i) = 0.0;
                bmin(det_idx, i) = 0.0;
                bmax(det_idx, i) = 0.0;
            }
            point_clouds.push_back(py::array_t<double>(std::vector<ssize_t>{0, 3}));
            colors_out.push_back(py::array_t<uint8_t>(std::vector<ssize_t>{0, 3}));
            continue;
        }

        int ph = y2 - y1;
        int pw = x2 - x1;

        // Compute reference depth from center region
        int cy_l = y1 + ph / 2;
        int cx_l = x1 + pw / 2;
        int sh = std::max(3, ph / 10);
        int sw = std::max(3, pw / 10);

        int cy_start = std::max(y1, cy_l - sh);
        int cy_end = std::min(y2, cy_l + sh);
        int cx_start = std::max(x1, cx_l - sw);
        int cx_end = std::min(x2, cx_l + sw);

        std::vector<float> center_depths;
        for (int v = cy_start; v < cy_end; ++v)
        {
            for (int u = cx_start; u < cx_end; ++u)
            {
                float z = d(v, u);
                if (z > 0)
                {
                    center_depths.push_back(z);
                }
            }
        }

        if (center_depths.empty())
        {
            vm(det_idx) = false;
            for (int i = 0; i < 3; ++i)
            {
                cent(det_idx, i) = 0.0;
                bmin(det_idx, i) = 0.0;
                bmax(det_idx, i) = 0.0;
            }
            point_clouds.push_back(py::array_t<double>(std::vector<ssize_t>{0, 3}));
            colors_out.push_back(py::array_t<uint8_t>(std::vector<ssize_t>{0, 3}));
            continue;
        }

        // Compute median for reference depth
        size_t mid = center_depths.size() / 2;
        std::nth_element(center_depths.begin(), center_depths.begin() + mid, center_depths.end());
        double ref_depth = center_depths[mid];

        // Compute std for threshold
        double sum = 0.0, sq_sum = 0.0;
        for (float z : center_depths)
        {
            sum += z;
            sq_sum += z * z;
        }
        double mean = sum / center_depths.size();
        double variance = sq_sum / center_depths.size() - mean * mean;
        double std_dev = (center_depths.size() > 1) ? std::sqrt(std::max(0.0, variance)) : 0.1;
        double thresh = 2.0 * std_dev + 0.15;

        // Collect valid 3D points
        std::vector<double> points_x, points_y, points_z;
        std::vector<uint8_t> cols_r, cols_g, cols_b;

        for (int v = y1; v < y2; ++v)
        {
            for (int u = x1; u < x2; ++u)
            {
                float z = d(v, u);
                if (z <= 0 || std::abs(z - ref_depth) >= thresh)
                {
                    continue;
                }

                // Camera coordinates
                double x_cam = u_norm[u] * z;
                double y_cam = v_norm[v] * z;

                // World coordinates
                double wx = R00 * x_cam + R01 * y_cam + R02 * z + tx;
                double wy = R10 * x_cam + R11 * y_cam + R12 * z + ty;
                double wz = R20 * x_cam + R21 * y_cam + R22 * z + tz;

                points_x.push_back(wx);
                points_y.push_back(wy);
                points_z.push_back(wz);
                cols_r.push_back(r(v, u, 0));
                cols_g.push_back(r(v, u, 1));
                cols_b.push_back(r(v, u, 2));
            }
        }

        if (points_x.size() < 10)
        {
            vm(det_idx) = false;
            for (int i = 0; i < 3; ++i)
            {
                cent(det_idx, i) = 0.0;
                bmin(det_idx, i) = 0.0;
                bmax(det_idx, i) = 0.0;
            }
            point_clouds.push_back(py::array_t<double>(std::vector<ssize_t>{0, 3}));
            colors_out.push_back(py::array_t<uint8_t>(std::vector<ssize_t>{0, 3}));
            continue;
        }

        // Compute bbox
        double min_x = points_x[0], max_x = points_x[0];
        double min_y = points_y[0], max_y = points_y[0];
        double min_z = points_z[0], max_z = points_z[0];
        double sum_x = 0, sum_y = 0, sum_z = 0;

        for (size_t i = 0; i < points_x.size(); ++i)
        {
            min_x = std::min(min_x, points_x[i]);
            max_x = std::max(max_x, points_x[i]);
            min_y = std::min(min_y, points_y[i]);
            max_y = std::max(max_y, points_y[i]);
            min_z = std::min(min_z, points_z[i]);
            max_z = std::max(max_z, points_z[i]);
            sum_x += points_x[i];
            sum_y += points_y[i];
            sum_z += points_z[i];
        }

        // Downsample if needed
        std::vector<size_t> indices(points_x.size());
        for (size_t i = 0; i < indices.size(); ++i)
        {
            indices[i] = i;
        }

        if ((int)points_x.size() > max_points)
        {
            std::shuffle(indices.begin(), indices.end(), rng);
            indices.resize(max_points);
        }

        // Create output arrays
        ssize_t n_pts = indices.size();
        py::array_t<double> pc({n_pts, ssize_t(3)});
        py::array_t<uint8_t> cols({n_pts, ssize_t(3)});
        auto pc_m = pc.mutable_unchecked<2>();
        auto cols_m = cols.mutable_unchecked<2>();

        for (ssize_t i = 0; i < n_pts; ++i)
        {
            size_t idx = indices[i];
            pc_m(i, 0) = points_x[idx];
            pc_m(i, 1) = points_y[idx];
            pc_m(i, 2) = points_z[idx];
            cols_m(i, 0) = cols_r[idx];
            cols_m(i, 1) = cols_g[idx];
            cols_m(i, 2) = cols_b[idx];
        }

        vm(det_idx) = true;
        cent(det_idx, 0) = sum_x / points_x.size();
        cent(det_idx, 1) = sum_y / points_y.size();
        cent(det_idx, 2) = sum_z / points_z.size();
        bmin(det_idx, 0) = min_x;
        bmin(det_idx, 1) = min_y;
        bmin(det_idx, 2) = min_z;
        bmax(det_idx, 0) = max_x;
        bmax(det_idx, 1) = max_y;
        bmax(det_idx, 2) = max_z;

        point_clouds.push_back(pc);
        colors_out.push_back(cols);
    }

    return std::make_tuple(valid_mask, centroids, bbox_mins, bbox_maxs, point_clouds, colors_out);
}


PYBIND11_MODULE(sg_accel, m) {
    m.def("project_detections_to_3d", &project_detections_to_3d,
          "Project 2D detections to 3D using depth map",
          py::arg("depth"), py::arg("rgb"), py::arg("bboxes"),
          py::arg("intrinsics"), py::arg("rotation"), py::arg("translation"),
          py::arg("max_points") = 1000);

    m.doc() = "Scene Graph Acceleration Module - C++ optimized association";
    
    
    m.def("score_candidates", &score_candidates,
          "Score pre-filtered candidate pairs from KDTree",
          py::arg("det_indices"),
          py::arg("node_indices"),
          py::arg("det_bboxes"),
          py::arg("det_features"),
          py::arg("det_labels"),
          py::arg("det_centroids"),
          py::arg("node_bboxes"),
          py::arg("node_features"),
          py::arg("node_labels"),
          py::arg("node_centroids"));
    
    // New optimized functions
    
    m.def("batch_voxel_iou", &batch_voxel_iou,
          "Batch compute voxel IoU for multiple pairs",
          py::arg("all_voxel_keys"), py::arg("pairs"));
    
    m.def("find_merge_candidates", &find_merge_candidates,
          "Find merge candidates using spatial hashing",
          py::arg("node_centroids"), py::arg("node_labels"), py::arg("max_distance"));
    
    
    m.def("batch_bbox_iou_containment", &batch_bbox_iou_containment,
          "Batch compute bbox IoU with containment for pairs",
          py::arg("all_bboxes"), py::arg("pairs"));
    
}
