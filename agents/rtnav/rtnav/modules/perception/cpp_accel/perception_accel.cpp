/**
 * Perception acceleration (pybind11): detection 2D→3D projection + post-processing.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace py = pybind11;

/**
 * Batch post-process detector output for all cameras.
 *
 * For each detection:
 *   1. Rotate bbox back to original coords (rot_k)
 *   2. Clamp to image bounds (img_w, img_h)
 *   3. Skip if degenerate (w<=0 or h<=0)
 *   4. Project bbox center to world XY via depth + intrinsics + T_world_cam
 *
 * Returns:
 *   adjusted_bboxes  (N, 4) float64 — adjusted x1,y1,x2,y2
 *   valid_mask        (N,)   bool    — True if detection survives filtering
 *   world_xy          (N, 2) float64 — world x,y (0,0 if invalid or no depth)
 *   has_world_pos     (N,)   bool    — True if world position is valid
 */
std::tuple<
    py::array_t<double>, // adjusted_bboxes (N, 4)
    py::array_t<bool>,   // valid_mask (N,)
    py::array_t<double>, // world_xy (N, 2)
    py::array_t<bool>    // has_world_pos (N,)
    >
batch_postprocess_detections(
    py::array_t<double> bboxes,      // (N, 4) from detector [x1,y1,x2,y2] in cropped coords
    int rot_k,                       // rotation: 0, 1 (CCW90), -1 (CW90), 2 (180)
    int img_w,                       // original image width
    int img_h,                       // original image height
    py::array_t<float> depth,        // (D_H, D_W) depth map (unrotated)
    py::array_t<double> intrinsics,  // (3, 3) camera intrinsics
    py::array_t<double> T_world_cam, // (4, 4) camera-to-world transform
    double min_depth_m = 0.0,        // valid depth lower bound (metres); endpoints excluded
    double max_depth_m = 10.0        // valid depth upper bound (metres); endpoints excluded
)
{
    auto bb = bboxes.unchecked<2>();
    auto dep = depth.unchecked<2>();
    auto K = intrinsics.unchecked<2>();
    auto T = T_world_cam.unchecked<2>();

    ssize_t N = bb.shape(0);
    int dep_h = dep.shape(0), dep_w = dep.shape(1);

    // Habitat (with normalize_depth=True) clips out-of-range rays to the
    // interval endpoints, so decoded depths of *exactly* min_depth_m /
    // max_depth_m are sentinel values, not real measurements. Including
    // them plants phantom targets at exactly those radii (e.g. a "fake
    // target" ring at exactly 5 m forward on HM3D where rays exit through
    // doorways/windows). We exclude both ends with a small epsilon.
    const float eps = 1e-3f;
    const float z_lo = std::max((float)min_depth_m, 0.0f) + eps;
    const float z_hi = (float)max_depth_m - eps;

    // Rotated dimensions for rotate-back
    int rot_w, rot_h;
    if (std::abs(rot_k) == 1)
    {
        rot_h = img_w;
        rot_w = img_h;
    }
    else
    {
        rot_h = img_h;
        rot_w = img_w;
    }

    double fx = K(0, 0), fy = K(1, 1), ppx = K(0, 2), ppy = K(1, 2);

    // Output arrays
    py::array_t<double> adj_bboxes({N, ssize_t(4)});
    py::array_t<bool> valid_mask(N);
    py::array_t<double> world_xy({N, ssize_t(2)});
    py::array_t<bool> has_world(N);

    auto ab = adj_bboxes.mutable_unchecked<2>();
    auto vm = valid_mask.mutable_unchecked<1>();
    auto wxy = world_xy.mutable_unchecked<2>();
    auto hw = has_world.mutable_unchecked<1>();

    for (ssize_t i = 0; i < N; ++i)
    {
        double x1 = bb(i, 0);
        double y1 = bb(i, 1);
        double x2 = bb(i, 2);
        double y2 = bb(i, 3);

        // --- rotate back ---
        if (rot_k == 1)
        { // 90 CCW
            double nx1 = rot_h - 1.0 - y2, nx2 = rot_h - 1.0 - y1;
            double ny1 = x1, ny2 = x2;
            x1 = nx1;
            y1 = ny1;
            x2 = nx2;
            y2 = ny2;
        }
        else if (rot_k == -1)
        { // 90 CW
            double nx1 = y1, nx2 = y2;
            double ny1 = rot_w - 1.0 - x2, ny2 = rot_w - 1.0 - x1;
            x1 = nx1;
            y1 = ny1;
            x2 = nx2;
            y2 = ny2;
        }
        else if (rot_k == 2)
        { // 180
            double nx1 = rot_w - 1.0 - x2, nx2 = rot_w - 1.0 - x1;
            double ny1 = rot_h - 1.0 - y2, ny2 = rot_h - 1.0 - y1;
            x1 = nx1;
            y1 = ny1;
            x2 = nx2;
            y2 = ny2;
        }
        // Ensure order
        if (x1 > x2)
            std::swap(x1, x2);
        if (y1 > y2)
            std::swap(y1, y2);

        // --- clamp ---
        x1 = std::max(0.0, std::min((double)img_w, x1));
        y1 = std::max(0.0, std::min((double)img_h, y1));
        x2 = std::max(0.0, std::min((double)img_w, x2));
        y2 = std::max(0.0, std::min((double)img_h, y2));

        ab(i, 0) = x1;
        ab(i, 1) = y1;
        ab(i, 2) = x2;
        ab(i, 3) = y2;

        // --- degenerate check ---
        if (x2 <= x1 || y2 <= y1)
        {
            vm(i) = false;
            wxy(i, 0) = 0;
            wxy(i, 1) = 0;
            hw(i) = false;
            continue;
        }

        vm(i) = true;

        // --- world position from depth ---
        // Scale bbox to depth coords if sizes differ
        double sx = (double)dep_w / img_w;
        double sy = (double)dep_h / img_h;
        double dx1 = x1 * sx, dy1 = y1 * sy, dx2 = x2 * sx, dy2 = y2 * sy;

        int cx = std::max(0, std::min(dep_w - 1, (int)((dx1 + dx2) / 2)));
        int cy = std::max(0, std::min(dep_h - 1, (int)((dy1 + dy2) / 2)));

        float z = dep(cy, cx);

        // Try center first, fall back to median of bbox region. Exclude
        // both endpoints (z_lo / z_hi) since those are Habitat's clip-to-
        // min/max sentinels, not real depth measurements.
        if (z <= z_lo || z >= z_hi)
        {
            int bx1 = std::max(0, (int)dx1);
            int by1 = std::max(0, (int)dy1);
            int bx2 = std::min(dep_w, (int)dx2);
            int by2 = std::min(dep_h, (int)dy2);
            if (bx2 > bx1 && by2 > by1)
            {
                std::vector<float> valid_depths;
                for (int v = by1; v < by2; ++v)
                {
                    for (int u = bx1; u < bx2; ++u)
                    {
                        float dv = dep(v, u);
                        if (dv > z_lo && dv < z_hi)
                            valid_depths.push_back(dv);
                    }
                }
                if (!valid_depths.empty())
                {
                    size_t mid = valid_depths.size() / 2;
                    std::nth_element(valid_depths.begin(), valid_depths.begin() + mid, valid_depths.end());
                    z = valid_depths[mid];
                }
                else
                {
                    z = -1;
                }
            }
            else
            {
                z = -1;
            }
        }

        if (z <= z_lo)
        {
            hw(i) = false;
            wxy(i, 0) = 0;
            wxy(i, 1) = 0;
        }
        else
        {
            // Back-project to camera coords
            double cam_x = (cx - ppx) * z / fx;
            double cam_y = (cy - ppy) * z / fy;
            double cam_z = z;

            // Transform to world coords
            double wx = T(0, 0) * cam_x + T(0, 1) * cam_y + T(0, 2) * cam_z + T(0, 3);
            double wy = T(1, 0) * cam_x + T(1, 1) * cam_y + T(1, 2) * cam_z + T(1, 3);

            hw(i) = true;
            wxy(i, 0) = wx;
            wxy(i, 1) = wy;
        }
    }

    return std::make_tuple(adj_bboxes, valid_mask, world_xy, has_world);
}

// ── fill_small_holes: fill zero-holes in a depth image (from VLFM) ──
py::array_t<float> fill_small_holes(py::array_t<float, py::array::c_style | py::array::forcecast> depth,
                                    int area_thresh) {
    py::buffer_info bi = depth.request();
    if (bi.ndim != 2) {
        throw std::invalid_argument("depth must be a 2D float array");
    }
    int H = static_cast<int>(bi.shape[0]);
    int W = static_cast<int>(bi.shape[1]);

    const float* src = static_cast<const float*>(bi.ptr);

    // binary mask: 1 where depth == 0, else 0 (uint8 0/1, matches VLFM exactly)
    cv::Mat binary(H, W, CV_8UC1);
    for (int y = 0; y < H; ++y) {
        const float* sr = src + y * W;
        uint8_t* br = binary.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) br[x] = (sr[x] == 0.0f) ? 1 : 0;
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat filled_holes = cv::Mat::zeros(H, W, CV_8UC1);
    for (size_t i = 0; i < contours.size(); ++i) {
        if (cv::contourArea(contours[i]) < area_thresh) {
            cv::drawContours(filled_holes, contours, static_cast<int>(i),
                             cv::Scalar(1), cv::FILLED);
        }
    }

    py::array_t<float> out({static_cast<py::ssize_t>(H), static_cast<py::ssize_t>(W)});
    py::buffer_info bo = out.request();
    float* dst = static_cast<float*>(bo.ptr);
    for (int y = 0; y < H; ++y) {
        const float* sr = src + y * W;
        const uint8_t* mr = filled_holes.ptr<uint8_t>(y);
        float* dr = dst + y * W;
        for (int x = 0; x < W; ++x) dr[x] = (mr[x] == 1) ? 1.0f : sr[x];
    }
    return out;
}


PYBIND11_MODULE(perception_accel, m)
{
    m.doc() = "Perception acceleration: detection 2D->3D projection + post-processing";

    m.def("fill_small_holes", &fill_small_holes,
          "Fill zero-holes smaller than area_thresh in a depth image",
          py::arg("depth"), py::arg("area_thresh"));

    m.def("batch_postprocess_detections", &batch_postprocess_detections,
          "Batch post-process detector output: rotate-back, clamp, world projection",
          py::arg("bboxes"),
          py::arg("rot_k"),
          py::arg("img_w"),
          py::arg("img_h"),
          py::arg("depth"),
          py::arg("intrinsics"),
          py::arg("T_world_cam"),
          py::arg("min_depth_m") = 0.0,
          py::arg("max_depth_m") = 10.0);

}
