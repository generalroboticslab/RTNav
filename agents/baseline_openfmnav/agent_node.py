#!/usr/bin/env python3
"""OpenFMNav ROS2 agent node for the shared env-agent evaluation pipeline."""

import json
import hashlib
import math
import os
import random
import sys
import time
from copy import deepcopy
from types import SimpleNamespace
from typing import Optional, Tuple

import cv2
import numpy as np
import skimage.morphology
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import measure
from std_msgs.msg import String
from torchvision import transforms

from base_agent import BaseAgentNode, run_agent


OPENFMNAV_ROOT = os.environ.get("OPENFMNAV_ROOT", "/opt/openfmnav")
if OPENFMNAV_ROOT in sys.path:
    sys.path.remove(OPENFMNAV_ROOT)
sys.path.insert(0, OPENFMNAV_ROOT)
os.chdir(OPENFMNAV_ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _ensure_api_key_file():
    if os.environ.get("OPENFMNAV_DUMMY_LLM", "").strip() == "1":
        return
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    key_path = os.path.join(OPENFMNAV_ROOT, "apikey.txt")
    if not api_key and not os.path.exists(key_path):
        raise RuntimeError(
            "OpenFMNav requires an OpenAI API key. Set OPENAI_API_KEY or create "
            f"{key_path} before starting the agent."
        )


_ensure_api_key_file()

from agents.llm import LLM  # noqa: E402
from constants import object_category  # noqa: E402
from Grounded_SAM.gsam import GSAM, convert_SAM  # noqa: E402
from envs.utils.fmm_planner import FMMPlanner  # noqa: E402
import envs.utils.pose as pu  # noqa: E402
from model import Semantic_Mapping  # noqa: E402
import utils.visualization as vu  # noqa: E402
from vl_prompt.p_manager import object_query_constructor  # noqa: E402

tmp_llm = LLM(None, None)


_GOAL_ALIASES = {
    "chair": "chair",
    "bed": "bed",
    "plant": "plant",
    "potted plant": "plant",
    "toilet": "toilet",
    "tv": "tv",
    "television": "tv",
    "tv monitor": "tv",
    "tv_monitor": "tv",
    "sofa": "couch",
    "couch": "couch",
}

_SENSOR_DEFAULTS = {
    "hm3d": {
        "camera_height": "0.88",
        "hfov": "79.0",
        "env_frame_width": "640",
        "env_frame_height": "480",
        "frame_width": "160",
        "frame_height": "120",
    },
    "ovon": {
        "camera_height": "1.31",
        "hfov": "42.0",
        "env_frame_width": "360",
        "env_frame_height": "640",
        "frame_width": "180",
        "frame_height": "320",
    },
}


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _l2(x1: float, x2: float, y1: float, y2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _relative_pose(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2
    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = _l2(x1, x2, y1, y2)
    return [dist * np.cos(theta), dist * np.sin(theta), _angle_diff(o2, o1)]


def _find_big_connect(image):
    img_label, _ = measure.label(image, connectivity=2, return_num=True)
    props = measure.regionprops(img_label)
    res_matrix = np.zeros(img_label.shape)
    tmp_area = 0
    for i in range(0, len(props)):
        if props[i].area > tmp_area:
            tmp = (img_label == i + 1).astype(np.uint8)
            res_matrix = tmp
            tmp_area = props[i].area
    return res_matrix


def _hash_array(arr):
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _array_signature(arr):
    a = np.asarray(arr)
    if a.size == 0:
        return {"shape": list(a.shape), "dtype": str(a.dtype), "sum": None, "nnz": 0, "hash": _hash_array(a)}
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sum": float(np.sum(a, dtype=np.float64)),
        "nnz": int(np.count_nonzero(a)),
        "hash": _hash_array(a),
    }


def _trace_enabled():
    return bool(os.environ.get("OPENFMNAV_AGENT_TRACE_PATH", "").strip())


def _tensor_signature(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return _array_signature(x)


def _sam_signature(pred):
    try:
        masks, boxes_filt, pred_phrases = pred
    except Exception:
        return {"is_none": True}
    return {
        "is_none": False,
        "masks": _tensor_signature(masks),
        "boxes": _tensor_signature(boxes_filt),
        "pred_phrases": [str(x) for x in pred_phrases],
    }


class OpenFMNavPlanner:
    """OpenFMNav Sem_Exp planner/preprocessor without the Habitat env base class."""

    def __init__(self, args, object_category, device, node):
        self.args = args
        self.object_category = deepcopy(object_category)
        self.device = device
        self.node = node

        self.res = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (args.frame_height, args.frame_width),
                    interpolation=Image.NEAREST,
                ),
            ]
        )

        text_threshold = args.text_threshold
        use_ram = args.tag_freq > 0
        gsam_device = "cuda" if torch.cuda.is_available() else "cpu"
        while True:
            try:
                self.GSAM = GSAM(
                    self.object_category[:-1],
                    text_threshold=text_threshold,
                    device=gsam_device,
                    use_ram=use_ram,
                )
                break
            except Exception as ex:
                self.node.get_logger().error(f"GSAM init failed: {ex}; retrying in 20s")
                time.sleep(20)

        self.selem = skimage.morphology.disk(3)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        if args.visualize or args.print_images:
            self.vis_image = None
            self.rgb_vis = None
            self.semantics_vis = None
            self.set_legend()

        self.obs = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None

        self.replan_count = 0
        self.collision_n = 0
        self.eve_angle = 0
        self.eve_angle_old = 0
        self._last_sem_debug = None
        self.fail_case = {"collision": 0, "success": 0, "detection": 0, "exploration": 0}

    def set_legend(self):
        return
        # if os.environ.get("OPENFMNAV_DISABLE_LEGEND", "").strip() == "1":
        #     return
        vu.save_legend(self.object_category)
        self.legend = cv2.imread("img/legend.png")
        h, w = self.legend.shape[0], self.legend.shape[1]
        self.legend = cv2.resize(
            self.legend,
            (int(w * 980 / h), 980),
            interpolation=cv2.INTER_NEAREST,
        )
        lx, ly = self.legend.shape[0], self.legend.shape[1]
        if self.vis_image is not None:
            self.vis_image[50:, 1165 + 500:, :] = 255
            try:
                self.vis_image[50:50 + lx, 1165 + 500:1165 + 500 + ly, :] = self.legend
            except:
                print("====> legend error")

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        exp_pred = np.rint(planner_inputs['exp_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        # if args.visualize or args.print_images:
            # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self.visited_vis[gx1:gx2, gy1:gy2] = \
            vu.draw_line(last_start, start,
                            self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1 and not planner_inputs["new_goal"]:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                self.collision_n += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, replan, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window)

        if replan:
            self.replan_count += 1
            print("replan_count: ", self.replan_count)
        else:
            self.replan_count = 0

        # Deterministic Local Policy
        if (stop and planner_inputs['found_goal'] == 1) or self.replan_count > 26:
            action = 0  # Stop
            decision_branch = "stop_or_replan"
            angle_st_goal = None
            angle_agent = None
            relative_angle = None
            eve_start_x = None
            eve_start_y = None
            eve_probe = None
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            angle_eps = 1e-4
            if abs(relative_angle - 180.0) <= angle_eps:
                relative_angle = 180.0
            elif relative_angle > 180:
                relative_angle -= 360

            ## add the evelution angle
            eve_start_x = int(5 * math.sin(angle_st_goal) + start[0])
            eve_start_y = int(5 * math.cos(angle_st_goal) + start[1])
            if eve_start_x > map_pred.shape[0]: eve_start_x = map_pred.shape[0]
            if eve_start_y > map_pred.shape[0]: eve_start_y = map_pred.shape[0]
            if eve_start_x < 0: eve_start_x = 0
            if eve_start_y < 0: eve_start_y = 0
            turn_threshold = self.args.turn_angle / 2.0
            if exp_pred[eve_start_x, eve_start_y] == 0 and self.eve_angle > -60:
                action = 5
                self.eve_angle -= 30
                decision_branch = "look_down"
            elif exp_pred[eve_start_x, eve_start_y] == 1 and self.eve_angle < 0:
                action = 4
                self.eve_angle += 30
                decision_branch = "look_up"
            elif relative_angle > turn_threshold + angle_eps:
                action = 3  # Right
                decision_branch = "turn_right"
            elif relative_angle < -turn_threshold - angle_eps:
                action = 2  # Left
                decision_branch = "turn_left"
            else:
                action = 1  # Forward
                decision_branch = "forward"
            eve_probe = int(exp_pred[eve_start_x, eve_start_y])

        self._last_plan_debug = {
            "start": [int(start[0]), int(start[1])],
            "stg": [int(stg[0]), int(stg[1])],
            "replan": bool(replan),
            "stop": bool(stop),
            "start_o": float(start_o),
            "angle_st_goal": None if angle_st_goal is None else float(angle_st_goal),
            "angle_agent": None if angle_agent is None else float(angle_agent),
            "relative_angle": None if relative_angle is None else float(relative_angle),
            "eve_start": None if eve_start_x is None else [int(eve_start_x), int(eve_start_y)],
            "eve_probe": eve_probe,
            "decision_branch": decision_branch,
            "action": int(action),
        }

        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[cv2.dilate(self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2], self.kernel) == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), replan, stop

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        # print("obs: ", obs)
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]
        semantic = obs[:,:,4:5].squeeze()
        sem_debug = None
        if _trace_enabled():
            sem_debug = {
                "rgb": _array_signature(rgb),
                "depth_raw": _array_signature(depth),
                "object_category_len_before": int(len(self.object_category)),
                "object_category_before": [str(x) for x in self.object_category],
            }
        # BGR to RGB
        self.rgb_vis = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                                  (640, 480), interpolation=cv2.INTER_NEAREST)
        # print("obs: ", semantic.shape)
        if args.use_gtsem:
            self.semantics_vis = self.rgb_vis
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))
            for i in range(16):
                sem_seg_pred[:,:,i][semantic == i+1] = 1
        else:
            semantic_output = self._get_sem_pred(
                rgb.astype(np.uint8), depth, use_seg=use_seg)
            sam_semantic_pred = semantic_output['sam_semantic_pred']
            sam_all_cls = convert_SAM(sam_semantic_pred, self.object_category, rgb.shape[:2])
            sem_seg_pred = sam_all_cls
            if sem_debug is not None:
                sem_debug.update(semantic_output.get("sem_debug") or {})
                sem_debug["sem_seg_pred"] = _array_signature(sem_seg_pred)
                sem_debug["object_category_len_after"] = int(len(self.object_category))
                sem_debug["object_category_after"] = [str(x) for x in self.object_category]

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)
        if sem_debug is not None:
            sem_debug["depth_processed"] = _array_signature(depth)
            self._last_sem_debug = sem_debug

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]
            if sem_debug is not None:
                sem_debug["sem_seg_pred_downsampled"] = _array_signature(sem_seg_pred)

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        # depth = min_d * 100.0 + depth * max_d * 100.0
        depth = min_d * 100.0 + depth * (max_d-min_d) * 100.0
        # depth = depth*1000.

        return depth

    def _get_sem_pred(self, rgb, depth, use_seg=True):
        if use_seg:
            # # save rgb and depth
            # skimage.io.imsave("current_rgb.png", rgb)
            # skimage.io.imsave("current_depth.png", (np.repeat(depth, 3, axis=2) * 255).astype(np.uint8))

            self.semantics_vis = None
            image = torch.from_numpy(rgb).to(self.device).unsqueeze_(0).float()
            depth = torch.from_numpy(depth).to(self.device).unsqueeze_(0).float()
            sem_debug = {"tag_freq": int(self.args.tag_freq)} if _trace_enabled() else None
            with torch.no_grad():
                # print(image.shape, depth.shape) # torch.Size([1, 480, 640, 3]) torch.Size([1, 480, 640, 1])
                try:
                    rgb_Image = Image.fromarray(rgb).convert('RGB')
                    if self.args.tag_freq > 0:
                        import random
                        random_num = random.randint(1, self.args.tag_freq)
                        if sem_debug is not None:
                            sem_debug["tag_random_num"] = int(random_num)
                            sem_debug["tag_fired"] = bool(random_num == 1)
                        if random_num == 1:
                            # tag and update
                            print("========> [DiscoverVLM]: tagging...")
                            ndo_list = tmp_llm.discover_objects(rgb_Image, self.object_category)
                            if sem_debug is not None:
                                sem_debug["tag_objects"] = [str(x) for x in ndo_list]
                            self.object_category = self.object_category[:-2] + ndo_list + self.object_category[-2:]
                            self.GSAM.add_text(ndo_list)
                            self.set_legend()

                    sam_semantic_pred = self.GSAM.predict(rgb_Image) # (N, 1, 480, 640), we need (480, 640, 16)
                    if sem_debug is not None:
                        sem_debug["sam_pred"] = _sam_signature(sam_semantic_pred)
                    self.semantics_vis = self.GSAM.get_vis(rgb_Image, sam_semantic_pred)
                except Exception as ex:
                    print(f"========> [SAM]: no object detected: {ex}")
                    if sem_debug is not None:
                        sem_debug["exception"] = str(ex)
                    sam_semantic_pred = None

                if self.semantics_vis is None:
                    self.semantics_vis = self.rgb_vis
        else:
            raise NotImplementedError
        outputs = {
            "sam_semantic_pred": sam_semantic_pred,
            "sem_debug": sem_debug,
        }
        return outputs


class OpenFMNavPolicy:
    """Single-environment OpenFMNav policy state machine."""

    def __init__(self, node: BaseAgentNode):
        self.node = node
        self.device = node._device
        self.args = self._make_args()
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        self.num_scenes = 1
        self.base_object_category = deepcopy(object_category)
        self.object_category = deepcopy(object_category)
        self.nc = len(self.object_category) + 4
        self.map_size = self.args.map_size_cm // self.args.map_resolution
        self.full_w = self.full_h = self.map_size
        self.local_w = int(self.full_w / self.args.global_downscaling)
        self.local_h = int(self.full_h / self.args.global_downscaling)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.tv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        self.sem_map_module = Semantic_Mapping(self.args).to(self.device)
        self.sem_map_module.eval()
        self.planner_agent = self._make_planner_agent()
        self.goal_rng = torch.Generator()
        self.goal_rng.manual_seed(self.args.seed)
        self._allocate_maps(self.nc)
        self.global_goals = [[0, 0] for _ in range(self.num_scenes)]

        self.prev_sim_location = None
        self.goal_cat_id = 0
        self.goal_name = "chair"
        self.found_goal = [0]
        self._pending_clear_flag = 0
        self.reply_list = [None]
        self.loop_step = 0
        self.decision_step = 0
        self._official_first_action_done = False
        self._zero_obs_countdown = 0
        self.initialized = False
        self.llm_retry_sleep = float(os.environ.get("OPENFMNAV_LLM_RETRY_SLEEP", "20"))
        self.trace_path = os.environ.get("OPENFMNAV_AGENT_TRACE_PATH", "").strip()
        self.trace = None
        self.trace_episodes = []
        self._last_goal_debug = None
        self._last_state_debug = None
        self._last_llm_debug = None

    def _make_args(self):
        benchmark = os.environ.get("BENCHMARK", "hm3d").strip().lower()
        sensor_defaults = _SENSOR_DEFAULTS.get(benchmark, _SENSOR_DEFAULTS["hm3d"])
        return SimpleNamespace(
            seed=int(os.environ.get("HABITAT_SEED", "1")),
            cuda=torch.cuda.is_available(),
            device=self.device,
            eval=1,
            task_config=os.environ.get("OPENFMNAV_TASK_CONFIG", "tasks/objectnav_hm3d.yaml"),
            split=os.environ.get("EVAL_SPLIT", "val"),
            version=os.environ.get("OPENFMNAV_VERSION", "v1.1"),
            dump_location=os.environ.get("OPENFMNAV_DUMP_LOCATION", "./nav_res"),
            exp_name=os.environ.get("OPENFMNAV_EXP_NAME", "ros_openfmnav"),
            visualize=int(os.environ.get("OPENFMNAV_VISUALIZE", "0")),
            print_images=int(os.environ.get("OPENFMNAV_PRINT_IMAGES", "0")),
            env_frame_width=int(os.environ.get("OPENFMNAV_ENV_FRAME_WIDTH", sensor_defaults["env_frame_width"])),
            env_frame_height=int(os.environ.get("OPENFMNAV_ENV_FRAME_HEIGHT", sensor_defaults["env_frame_height"])),
            frame_width=int(os.environ.get("OPENFMNAV_FRAME_WIDTH", sensor_defaults["frame_width"])),
            frame_height=int(os.environ.get("OPENFMNAV_FRAME_HEIGHT", sensor_defaults["frame_height"])),
            max_episode_length=int(os.environ.get("OPENFMNAV_MAX_EPISODE_LENGTH", "500")),
            camera_height=float(os.environ.get("OPENFMNAV_CAMERA_HEIGHT", sensor_defaults["camera_height"])),
            hfov=float(os.environ.get("OPENFMNAV_HFOV", sensor_defaults["hfov"])),
            turn_angle=float(os.environ.get("OPENFMNAV_TURN_ANGLE", "30")),
            min_depth=float(os.environ.get("OPENFMNAV_MIN_DEPTH", "0.5")),
            max_depth=float(os.environ.get("OPENFMNAV_MAX_DEPTH", "5.0")),
            floor_thr=50,
            num_processes=1,
            num_local_steps=int(os.environ.get("OPENFMNAV_NUM_LOCAL_STEPS", "20")),
            num_global_steps=int(os.environ.get("OPENFMNAV_NUM_GLOBAL_STEPS", "20")),
            num_sem_categories=len(object_category),
            sem_pred_prob_thr=float(os.environ.get("OPENFMNAV_SEM_PRED_PROB_THR", "0.9")),
            global_downscaling=int(os.environ.get("OPENFMNAV_GLOBAL_DOWNSCALING", "2")),
            vision_range=int(os.environ.get("OPENFMNAV_VISION_RANGE", "100")),
            map_resolution=int(os.environ.get("OPENFMNAV_MAP_RESOLUTION", "5")),
            du_scale=int(os.environ.get("OPENFMNAV_DU_SCALE", "1")),
            map_size_cm=int(os.environ.get("OPENFMNAV_MAP_SIZE_CM", "4800")),
            cat_pred_threshold=float(os.environ.get("OPENFMNAV_CAT_PRED_THRESHOLD", "5.0")),
            map_pred_threshold=float(os.environ.get("OPENFMNAV_MAP_PRED_THRESHOLD", "1.0")),
            exp_pred_threshold=float(os.environ.get("OPENFMNAV_EXP_PRED_THRESHOLD", "1.0")),
            collision_threshold=float(os.environ.get("OPENFMNAV_COLLISION_THRESHOLD", "0.10")),
            use_gtsem=int(os.environ.get("OPENFMNAV_USE_GTSEM", "0")),
            sem_gpu_id=int(os.environ.get("OPENFMNAV_SEM_GPU_ID", "0" if torch.cuda.is_available() else "-2")),
            boundary_coeff=int(os.environ.get("OPENFMNAV_BOUNDARY_COEFF", "12")),
            text_threshold=float(os.environ.get("OPENFMNAV_TEXT_THRESHOLD", "0.55")),
            prompt_type=os.environ.get("OPENFMNAV_PROMPT_TYPE", "scoring"),
            tag_freq=int(os.environ.get("OPENFMNAV_TAG_FREQ", "0")),
            random_sample=False,
        )

    def _make_planner_agent(self):
        return OpenFMNavPlanner(self.args, self.object_category, self.device, self.node)

    def _allocate_maps(self, nc: int):
        self.full_map = torch.zeros(self.num_scenes, nc, self.full_w, self.full_h).float().to(self.device)
        self.local_map = torch.zeros(self.num_scenes, nc, self.local_w, self.local_h).float().to(self.device)
        self.local_ob_map = np.zeros((self.num_scenes, self.local_w, self.local_h))
        self.local_ex_map = np.zeros((self.num_scenes, self.local_w, self.local_h))
        self.target_edge_map = np.zeros((self.num_scenes, self.local_w, self.local_h))
        self.target_point_map = np.zeros((self.num_scenes, self.local_w, self.local_h))
        self.full_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
        self.local_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
        self.origins = np.zeros((self.num_scenes, 3))
        self.lmb = np.zeros((self.num_scenes, 4)).astype(int)
        self.planner_pose_inputs = np.zeros((self.num_scenes, 7))
        self.step_masks = torch.zeros(self.num_scenes).float().to(self.device)
        self.stair_flag = np.zeros(self.num_scenes)
        self.clear_flag = np.zeros(self.num_scenes)
        self.wait_env = np.zeros(self.num_scenes)
        self.finished = np.zeros(self.num_scenes)

    def reset(self, obs_dict: dict, target_name: Optional[str] = None):
        self.object_category = deepcopy(self.base_object_category)
        normalized_target = self._normalize_target_name(target_name)
        if normalized_target:
            self._add_episode_goal_category(normalized_target)
        self.nc = len(self.object_category) + 4
        if self.local_map.shape[1] != self.nc:
            self._allocate_maps(self.nc)
        object_category_changed = len(self.planner_agent.object_category) != len(self.object_category)
        self.planner_agent.object_category = deepcopy(self.object_category)
        self.planner_agent.GSAM.set_text(self.object_category[:-1])
        if object_category_changed:
            self.planner_agent.set_legend()

        self.prev_sim_location = None
        self.goal_cat_id, self.goal_name = self._resolve_goal(obs_dict, normalized_target)
        hidden_wait_steps = 0
        if self._official_first_action_done:
            next_loop_step = self._next_episode_loop_step(self.loop_step)
            hidden_wait_steps = next_loop_step - self.loop_step
            self._zero_obs_countdown = int(hidden_wait_steps > 0)
            self.loop_step = next_loop_step
        else:
            self.loop_step = 0
            self._zero_obs_countdown = 0
        if not self._official_first_action_done:
            self._pending_clear_flag = 0
        self.found_goal = [0]
        self.reply_list = [None]
        self.wait_env.fill(0)
        self.finished.fill(0)
        self._init_map_and_pose_for_env(0)
        self.decision_step = 0
        self._last_goal_debug = None
        self._last_state_debug = None
        self._last_llm_debug = None
        if self.trace_path and self.trace is not None:
            self.trace_episodes.append(self.trace)
        self.trace = None

        p = self.planner_agent
        p.replan_count = 0
        p.collision_n = 0
        p.obs = None
        p.obs_shape = None
        p.collision_map = np.zeros((self.map_size, self.map_size))
        p.visited = np.zeros((self.map_size, self.map_size))
        p.visited_vis = np.zeros((self.map_size, self.map_size))
        p.col_width = 1
        p.count_forward_actions = 0
        p.curr_loc = [
            self.args.map_size_cm / 100.0 / 2.0,
            self.args.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        p.last_loc = p.curr_loc
        p.last_action = None
        p.eve_angle = 0
        p.eve_angle_old = 0
        p.fail_case = {"collision": 0, "success": 0, "detection": 0, "exploration": 0}
        p.info = self._info([0.0, 0.0, 0.0])
        if self.args.visualize or self.args.print_images:
            p.vis_image = vu.init_vis_image(self.goal_name, p.legend)
        self.initialized = self._official_first_action_done
        if hidden_wait_steps:
            self._advance_wait_steps(hidden_wait_steps, obs_dict)
        if self.trace_path:
            os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
            self.trace = {
                "source": "ros_openfmnav_policy",
                "episode": {
                    "goal_name": self.goal_name,
                    "goal_cat_id": int(self.goal_cat_id),
                    "start_gps": np.asarray(obs_dict.get("gps", [0.0, 0.0]), dtype=np.float32).tolist(),
                    "start_compass": float(np.asarray(obs_dict.get("compass", [0.0])).flat[0]),
                },
                "actions": [],
                "steps": [],
            }
            self._write_trace()

    def act(self, obs_dict: dict) -> Tuple[int, dict]:
        if not self.initialized:
            action = self._first_action(obs_dict)
            self._official_first_action_done = True
            self.initialized = True
            return action, self._policy_info(action)

        sim_location = self._sim_location(obs_dict)
        sensor_pose = self._sensor_pose(obs_dict)
        if sensor_pose is None:
            if self.prev_sim_location is None:
                sensor_pose = [0.0, 0.0, 0.0]
            else:
                sensor_pose = _relative_pose(sim_location, self.prev_sim_location)
        self.prev_sim_location = sim_location
        info = self._info(sensor_pose)
        obs = self._preprocess_ros_obs(obs_dict)
        if self._zero_obs_countdown > 0:
            obs = torch.zeros_like(obs)
            self._zero_obs_countdown -= 1
        main_step = self.loop_step
        l_step = main_step % self.args.num_local_steps
        action = self._main_loop_action(obs, info)
        info["clear_flag"] = self._update_pending_clear_flag(
            new_goal=l_step == self.args.num_local_steps - 1
        )
        if self._obs_step_id(obs_dict) == self.args.max_episode_length - 2:
            action = 0
            self.planner_agent.last_action = action
        self._append_trace(action, info, sim_location, main_step, l_step)
        return action, self._policy_info(action)

    def _first_action(self, obs_dict: dict) -> int:
        self.prev_sim_location = self._sim_location(obs_dict)
        info = self._info(self._sensor_pose(obs_dict) or [0.0, 0.0, 0.0])
        obs = self._preprocess_ros_obs(obs_dict)
        self._expand_maps_if_needed()

        poses = torch.from_numpy(np.asarray([info["sensor_pose"]])).float().to(self.device)
        eve_angle = np.asarray([info["eve_angle"]])
        _, self.local_map, self.local_map_stair, self.local_pose = self.sem_map_module(
            obs, poses, self.local_map, self.local_pose, eve_angle, self.object_category
        )
        if self.local_map.shape[1] > 5:
            self.local_map[:, 0, :, :][self.local_map[:, -2, :, :] > 0] = 0

        self._sample_random_global_goals()
        goal_maps = self._global_goal_maps()

        planner_input = {
            "map_pred": self.local_map[0, 0, :, :].cpu().numpy(),
            "exp_pred": self.local_map[0, 1, :, :].cpu().numpy(),
            "pose_pred": self.planner_pose_inputs[0],
            "goal": goal_maps[0],
            "map_target": self.target_point_map[0],
            "new_goal": 1,
            "found_goal": 0,
            "wait": 0,
        }
        action = self.planner_agent._plan(planner_input)
        info["clear_flag"] = self._update_pending_clear_flag(new_goal=True)
        self.planner_agent.obs = obs[0].cpu().numpy()
        self.planner_agent.obs_shape = self.planner_agent.obs.shape
        self.planner_agent.last_action = action
        self._last_goal_debug = {
            "global_item": 0,
            "goal_source": "random_global",
            "global_goal": [int(self.global_goals[0][0]), int(self.global_goals[0][1])],
            "found_goal": 0,
            "reply": None,
        }
        if self.trace is not None:
            self._last_state_debug = self._state_debug(goal_maps[0])
        self._last_llm_debug = None
        self._append_trace(int(action), info, self.prev_sim_location, -1, -1)
        return int(action)

    def _write_trace(self):
        if not self.trace_path:
            return
        with open(self.trace_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source": "ros_openfmnav_policy",
                    "episodes": self.trace_episodes,
                    "current_episode": self.trace,
                },
                f,
                indent=2,
            )

    def _append_trace(self, action: int, info: dict, sim_location, main_step: int, l_step: int):
        if not self.trace_path or self.trace is None:
            return
        rec = {
            "decision_step": int(self.decision_step + 1),
            "main_step": int(main_step),
            "l_step": int(l_step),
            "action": int(action),
            "sensor_pose": [float(x) for x in info.get("sensor_pose", [0.0, 0.0, 0.0])],
            "sim_location": [float(x) for x in sim_location],
            "eve_angle": float(getattr(self.planner_agent, "eve_angle", 0.0)),
            "clear_flag": int(info.get("clear_flag", 0)),
            "replan_count": int(getattr(self.planner_agent, "replan_count", 0)),
            "collision_n": int(getattr(self.planner_agent, "collision_n", 0)),
            "found_goal": int(self.found_goal[0]) if len(self.found_goal) > 0 else 0,
            "goal_debug": self._last_goal_debug or {},
            "plan_debug": getattr(self.planner_agent, "_last_plan_debug", {}),
            "state_debug": self._last_state_debug,
            "sem_debug": getattr(self.planner_agent, "_last_sem_debug", None),
            "llm_debug": self._last_llm_debug,
        }
        self.trace["actions"].append(int(action))
        self.trace["steps"].append(rec)
        self.decision_step += 1
        self._write_trace()

    def _next_episode_loop_step(self, step: int) -> int:
        l_step = step % self.args.num_local_steps
        target_l_step = self.args.num_local_steps - 1
        if l_step == target_l_step:
            return step
        return step + (target_l_step - l_step)

    def _zero_obs_tensor(self):
        return torch.zeros(
            1,
            self.nc,
            self.args.frame_height,
            self.args.frame_width,
            dtype=torch.float32,
            device=self.device,
        )

    def _advance_wait_steps(self, count: int, obs_dict: dict):
        obs = self._preprocess_ros_obs(obs_dict)
        poses = torch.zeros((self.num_scenes, 3), dtype=torch.float32, device=self.device)
        eve_angle = np.zeros(self.num_scenes)
        for i in range(count):
            self._expand_maps_if_needed()
            _, self.local_map, self.local_map_stair, self.local_pose = self.sem_map_module(
                obs, poses, self.local_map, self.local_pose, eve_angle, self.object_category
            )
            if i == 0:
                obs = self._zero_obs_tensor()
            locs = self.local_pose.cpu().numpy()
            self.planner_pose_inputs[:, :3] = locs + self.origins
            self.local_map[:, 2, :, :].fill_(0.0)
            for e in range(self.num_scenes):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [
                    int(r * 100.0 / self.args.map_resolution),
                    int(c * 100.0 / self.args.map_resolution),
                ]
                self.local_map[e, 2:4, loc_r - 2 : loc_r + 3, loc_c - 2 : loc_c + 3] = 1.0

    def _state_debug(self, local_goal_map, target_semantic_map=None):
        target_semantic = None
        if target_semantic_map is not None:
            target_semantic = _array_signature(target_semantic_map)
        return {
            "map_obstacle": _array_signature(self.local_map[0, 0].detach().cpu().numpy()),
            "map_explored": _array_signature(self.local_map[0, 1].detach().cpu().numpy()),
            "target_edge": _array_signature(self.target_edge_map[0]),
            "target_point": _array_signature(self.target_point_map[0]),
            "target_semantic": target_semantic,
            "local_goal": _array_signature(local_goal_map),
            "local_pose": [float(x) for x in self.local_pose[0].detach().cpu().numpy().reshape(-1)[:3]],
            "planner_pose": [float(x) for x in np.asarray(self.planner_pose_inputs[0]).reshape(-1)[:7]],
        }

    def _main_loop_action(self, obs: torch.Tensor, info: dict) -> int:
        step = self.loop_step
        l_step = step % self.args.num_local_steps
        cn = self.goal_cat_id + 4
        cname = self.goal_name
        self._last_llm_debug = None

        poses = torch.from_numpy(np.asarray([info["sensor_pose"]])).float().to(self.device)
        eve_angle = np.asarray([info["eve_angle"]])
        self._expand_maps_if_needed()
        _, self.local_map, self.local_map_stair, self.local_pose = self.sem_map_module(
            obs, poses, self.local_map, self.local_pose, eve_angle, self.object_category
        )

        locs = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs + self.origins
        self.local_map[:, 2, :, :].fill_(0.0)
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / self.args.map_resolution),
                int(c * 100.0 / self.args.map_resolution),
            ]
            self.local_map[e, 2:4, loc_r - 2 : loc_r + 3, loc_c - 2 : loc_c + 3] = 1.0
            loc_r = min(max(loc_r, 0), self.local_w - 1)
            loc_c = min(max(loc_c, 0), self.local_h - 1)
            if info["clear_flag"] or self.local_map[e, -2, loc_r, loc_c] > 0.5:
                self.stair_flag[e] = 1
            if self.stair_flag[e]:
                if torch.any(self.local_map[e, -2, :, :] > 0.5):
                    self.local_map[e, 0, :, :] = self.local_map_stair[e, 0, :, :]
                self.local_map[e, 0, :, :] = self.local_map_stair[e, 0, :, :]

        if l_step == self.args.num_local_steps - 1:
            self._update_frontiers(info, cname)
            self._sample_random_global_goals()

        local_goal_maps = [np.zeros((self.local_w, self.local_h)) for _ in range(self.num_scenes)]
        found_goal = [0 for _ in range(self.num_scenes)]
        for e in range(self.num_scenes):
            global_item = 0
            if self.reply_list[e]:
                global_item = self.reply_list[e]

            if np.any(self.target_point_map[e] == global_item + 1):
                local_goal_maps[e][self.target_point_map[e] == global_item + 1] = 1
                goal_source = "frontier"
            else:
                x, y = self.global_goals[e]
                local_goal_maps[e][x, y] = 1
                goal_source = "random_global"

            if cn < self.local_map.shape[1] and self.local_map[e, cn, :, :].sum() != 0.0:
                self.node.get_logger().info("OpenFMNav found goal in semantic map.")
                cat_semantic_scores = self.local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores[cat_semantic_scores > 0] = 1.0
                if "tv" in cname:
                    cat_semantic_scores = cv2.dilate(cat_semantic_scores, self.tv_kernel)
                local_goal_maps[e] = _find_big_connect(cat_semantic_scores)
                found_goal[e] = 1
                goal_source = "found_goal"

            if e == 0:
                self._last_goal_debug = {
                    "global_item": int(global_item),
                    "goal_source": goal_source,
                    "global_goal": [int(self.global_goals[e][0]), int(self.global_goals[e][1])],
                    "found_goal": int(found_goal[e]),
                    "reply": None if self.reply_list[e] is None else int(self.reply_list[e]),
                }
                if self.trace is not None:
                    target_semantic = self.local_map[e, cn].detach().cpu().numpy()
                    self._last_state_debug = self._state_debug(local_goal_maps[e], target_semantic)

        planner_input = {
            "map_pred": self.local_map[0, 0, :, :].cpu().numpy(),
            "exp_pred": self.local_map[0, 1, :, :].cpu().numpy(),
            "pose_pred": self.planner_pose_inputs[0],
            "goal": local_goal_maps[0],
            "map_target": self.target_point_map[0],
            "new_goal": l_step == self.args.num_local_steps - 1,
            "found_goal": found_goal[0],
            "wait": 0,
        }
        action = int(self.planner_agent._plan(planner_input))
        self.planner_agent.obs = obs[0].cpu().numpy()
        self.planner_agent.obs_shape = self.planner_agent.obs.shape
        self.planner_agent.last_action = action
        self.found_goal = found_goal
        self.loop_step += 1
        return action

    def _sample_random_global_goals(self):
        actions = torch.randn(self.num_scenes, 2, generator=self.goal_rng) * 6
        cpu_actions = nn.Sigmoid()(actions).numpy()
        self.global_goals = [
            [
                min(int(action[0] * self.local_w), int(self.local_w - 1)),
                min(int(action[1] * self.local_h), int(self.local_h - 1)),
            ]
            for action in cpu_actions
        ]

    def _global_goal_maps(self):
        goal_maps = [np.zeros((self.local_w, self.local_h)) for _ in range(self.num_scenes)]
        for e, (x, y) in enumerate(self.global_goals):
            goal_maps[e][x, y] = 1
        return goal_maps

    def _update_frontiers(self, info: dict, cname: str):
        for e in range(self.num_scenes):
            self.step_masks[e] += 1
            self.full_map[e, :, self.lmb[e, 0] : self.lmb[e, 1], self.lmb[e, 2] : self.lmb[e, 3]] = self.local_map[e]
            self.full_pose[e] = self.local_pose[e] + torch.from_numpy(self.origins[e]).to(self.device).float()
            locs = self.full_pose[e].cpu().numpy()
            r, c = locs[1], locs[0]
            loc_r, loc_c = [
                int(r * 100.0 / self.args.map_resolution),
                int(c * 100.0 / self.args.map_resolution),
            ]
            self.lmb[e] = self._get_local_map_boundaries(
                (loc_r, loc_c), (self.local_w, self.local_h), (self.full_w, self.full_h)
            )
            self.planner_pose_inputs[e, 3:] = self.lmb[e]
            self.origins[e] = [
                self.lmb[e][2] * self.args.map_resolution / 100.0,
                self.lmb[e][0] * self.args.map_resolution / 100.0,
                0.0,
            ]
            self.local_map[e] = self.full_map[
                e, :, self.lmb[e, 0] : self.lmb[e, 1], self.lmb[e, 2] : self.lmb[e, 3]
            ]
            self.local_pose[e] = self.full_pose[e] - torch.from_numpy(self.origins[e]).to(self.device).float()
            if info["clear_flag"]:
                self.clear_flag[e] = 1
            if self.clear_flag[e]:
                self.local_map[e].fill_(0.0)
                self.clear_flag[e] = 0

        for e in range(self.num_scenes):
            _local_ob_map = self.local_map[e][0].cpu().numpy()
            self.local_ob_map[e] = cv2.dilate(_local_ob_map, self.kernel)
            show_ex = cv2.inRange(self.local_map[e][1].cpu().numpy(), 0.1, 1)
            self.kernel = np.ones((5, 5), dtype=np.uint8)
            free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, self.kernel)
            contours, _ = cv2.findContours(free_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(self.local_ex_map[e], contour, -1, 1, 1)

            self.local_ex_map[e, 0:2, 0 : self.local_w] = 0.0
            self.local_ex_map[e, self.local_w - 2 : self.local_w, 0 : self.local_w - 1] = 0.0
            self.local_ex_map[e, 0 : self.local_w, 0:2] = 0.0
            self.local_ex_map[e, 0 : self.local_w, self.local_w - 2 : self.local_w] = 0.0

            target_edge = self.local_ex_map[e] - self.local_ob_map[e]
            target_edge[target_edge > 0.8] = 1.0
            target_edge[target_edge != 1.0] = 0.0
            local_pose_map = [
                self.local_pose[e][1] * 100 / self.args.map_resolution,
                self.local_pose[e][0] * 100 / self.args.map_resolution,
            ]
            self.target_edge_map[e], self.target_point_map[e], _ = self._remove_small_points(
                _local_ob_map, target_edge, 4, local_pose_map
            )

            self.local_ob_map[e] = np.zeros((self.local_w, self.local_h))
            self.local_ex_map[e] = np.zeros((self.local_w, self.local_h))
            self._choose_frontier_with_llm(e, cname)

    def _choose_frontier_with_llm(self, e: int, cname: str):
        lm = LLM(cname, self.args.prompt_type)
        self.reply_list[e] = None
        tpm = len(list(set(self.target_point_map[e].ravel()))) - 1
        frontier_desc_list = []
        for lay in range(tpm):
            f_pos = np.argwhere(self.target_point_map[e] == lay + 1)
            fmb = self._get_frontier_boundaries(
                (f_pos[0][0], f_pos[0][1]),
                (self.local_w / self.args.boundary_coeff, self.local_h / self.args.boundary_coeff),
                (self.local_w, self.local_h),
            )
            objs_list = []
            for se_cn in range(len(self.object_category) - 1):
                if self.local_map[e][se_cn + 4, fmb[0] : fmb[1], fmb[2] : fmb[3]].sum() != 0.0:
                    objs_list.append(self.object_category[se_cn])

            if len(objs_list) > 0:
                frontier_desc_list.append(object_query_constructor(objs_list))
            else:
                frontier_desc_list.append("This area contains nothing.")

        query = f"Goal: {cname}\n\n"
        for idx, desc in enumerate(frontier_desc_list):
            query += f"- Description {idx}: {desc}\n\n"
        if len(frontier_desc_list) == 0:
            query += "No current frontiers in this map.\n\n"

        self.node.get_logger().info(f"OpenFMNav LLM query:\n{query}")
        while True:
            try:
                answer, reply = lm.choose_frontier(query)
                break
            except Exception as ex:
                self.node.get_logger().error(
                    f"OpenFMNav LLM inference failed: {ex}; sleeping {self.llm_retry_sleep:g}s"
                )
                time.sleep(self.llm_retry_sleep)

        self.node.get_logger().info(f"OpenFMNav LLM output:\n{reply}")
        if answer == -1:
            try:
                answer = random.randint(0, tpm - 1)
            except ValueError:
                answer = 0
        self._last_llm_debug = {
            "query": query,
            "reply": reply,
            "answer": int(answer),
            "frontier_count": int(tpm),
            "frontier_desc_list": frontier_desc_list,
        }
        self.reply_list[e] = answer

    def _remove_small_points(self, local_ob_map, image, threshold_point, pose):
        selem = skimage.morphology.disk(1)
        traversible = skimage.morphology.binary_dilation(local_ob_map, selem) != True
        planner = FMMPlanner(traversible)
        goal_pose_map = np.zeros((local_ob_map.shape))
        pose_x = int(pose[0].cpu()) if int(pose[0].cpu()) < self.local_w - 1 else self.local_w - 1
        pose_y = int(pose[1].cpu()) if int(pose[1].cpu()) < self.local_w - 1 else self.local_w - 1
        goal_pose_map[pose_x, pose_y] = 1
        planner.set_multi_goal(goal_pose_map)

        img_label, _ = measure.label(image, connectivity=2, return_num=True)
        props = measure.regionprops(img_label)
        goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
        goal_point = np.zeros(img_label.shape)
        goal_score = []

        dict_cost = {}
        for i in range(1, len(props)):
            dist = planner.fmm_dist[int(props[i].centroid[0]), int(props[i].centroid[1])] * 5
            dist_s = 8 if dist < 300 else 0
            cost = props[i].area + dist_s
            if props[i].area > threshold_point and 50 < dist < 500:
                dict_cost[i] = cost

        if dict_cost:
            dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)
            for i, (key, value) in enumerate(dict_cost):
                goal_edge[img_label == key + 1] = 1
                goal_point[int(props[key].centroid[0]), int(props[key].centroid[1])] = i + 1
                goal_score.append(value)
                if i == 3:
                    break

        return goal_edge, goal_point, goal_score

    def _preprocess_ros_obs(self, obs_dict: dict) -> torch.Tensor:
        rgb = obs_dict["rgb"].astype(np.uint8)
        depth = obs_dict["depth"]
        semantic = np.zeros(depth.shape[:2] + (1,), dtype=np.uint8)
        state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)
        obs = self.planner_agent._preprocess_obs(state)
        self._sync_dynamic_categories()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        return obs_tensor

    def _sync_dynamic_categories(self):
        if self.planner_agent.object_category != self.object_category:
            self.object_category = deepcopy(self.planner_agent.object_category)
            self._expand_maps_if_needed()

    def _expand_maps_if_needed(self):
        expected_nc = len(self.object_category) + 4
        if self.local_map.shape[1] == expected_nc:
            return
        if self.local_map.shape[1] > expected_nc:
            self._allocate_maps(expected_nc)
            self._init_map_and_pose_for_env(0)
            self.nc = expected_nc
            return
        num_new_obj = expected_nc - self.local_map.shape[1]
        new_l_obj_map = torch.zeros(self.num_scenes, num_new_obj, self.local_w, self.local_h).float().to(self.device)
        new_f_obj_map = torch.zeros(self.num_scenes, num_new_obj, self.full_w, self.full_h).float().to(self.device)
        self.local_map = torch.cat((self.local_map[:, :-2, :, :], new_l_obj_map, self.local_map[:, -2:, :, :]), dim=1)
        self.full_map = torch.cat((self.full_map[:, :-2, :, :], new_f_obj_map, self.full_map[:, -2:, :, :]), dim=1)
        self.nc = expected_nc

    def _sim_location(self, obs_dict: dict):
        gps = np.asarray(obs_dict.get("gps", [0.0, 0.0]), dtype=np.float32)
        heading = float(np.asarray(obs_dict.get("compass", [0.0])).flat[0])
        return float(gps[0]), float(-gps[1]), heading

    @staticmethod
    def _sensor_pose(obs_dict: dict):
        if "sensor_pose" not in obs_dict:
            return None
        sensor_pose = np.asarray(obs_dict["sensor_pose"], dtype=np.float32).reshape(-1)
        if sensor_pose.size < 3 or not np.isfinite(sensor_pose[:3]).all():
            return None
        return [float(x) for x in sensor_pose[:3]]

    @staticmethod
    def _obs_step_id(obs_dict: dict) -> int:
        return int(np.asarray(obs_dict.get("step_id", [-1])).flat[0])

    def _info(self, sensor_pose):
        return {
            "sensor_pose": sensor_pose,
            "eve_angle": getattr(self.planner_agent, "eve_angle", 0),
            "goal_cat_id": self.goal_cat_id,
            "goal_name": self.goal_name,
            "clear_flag": int(self._pending_clear_flag),
        }

    def _update_pending_clear_flag(self, new_goal=False):
        clear_flag = int(
            getattr(self.planner_agent, "collision_n", 0) > 20
            or getattr(self.planner_agent, "replan_count", 0) > 26
        )
        if clear_flag:
            self.planner_agent.collision_n = 0
            self._pending_clear_flag = 1
        elif new_goal:
            self._pending_clear_flag = 0
        return int(self._pending_clear_flag)

    def _normalize_target_name(self, target_name: Optional[str]):
        name = self._clean_target_name(target_name)
        if not name:
            return None
        return _GOAL_ALIASES.get(name, name)

    @staticmethod
    def _clean_target_name(target_name: Optional[str]):
        if not target_name:
            return None
        name = target_name.strip().lower().replace("_", " ").replace("-", " ")
        name = " ".join(name.split())
        if not name or name == "unknown object":
            return None
        return name

    def _add_episode_goal_category(self, goal_name: str):
        if goal_name in self.object_category:
            return
        if self.object_category[-2:] == ["stairs", "void"]:
            self.object_category = self.object_category[:-2] + [goal_name] + self.object_category[-2:]
        elif self.object_category[-1:] == ["void"]:
            self.object_category = self.object_category[:-1] + [goal_name] + self.object_category[-1:]
        else:
            self.object_category.append(goal_name)

    def _resolve_goal(self, obs_dict: dict, target_name: Optional[str]):
        if target_name:
            goal_name = self._normalize_target_name(target_name)
            if goal_name in self.object_category:
                return self.object_category.index(goal_name), goal_name
            raise RuntimeError(f"OpenFMNav target {target_name!r} was not added to the open-set category list.")
        if "objectgoal" in obs_dict:
            objectgoal = int(np.asarray(obs_dict["objectgoal"]).flat[0])
            if 0 <= objectgoal < 6:
                goal_name = self.object_category[objectgoal]
                return objectgoal, goal_name
            raise RuntimeError(f"OpenFMNav received invalid closed-set objectgoal id {objectgoal}.")
        raise RuntimeError(
            "OpenFMNav requires an episode target for open-set OVON episodes. "
            "No target was received from reset metadata or /episode_info_json."
        )

    def _init_map_and_pose_for_env(self, e: int):
        self.full_map[e].fill_(0.0)
        self.full_pose[e].fill_(0.0)
        self.local_ob_map[e] = np.zeros((self.local_w, self.local_h))
        self.local_ex_map[e] = np.zeros((self.local_w, self.local_h))
        self.target_edge_map[e] = np.zeros((self.local_w, self.local_h))
        self.target_point_map[e] = np.zeros((self.local_w, self.local_h))
        self.step_masks[e] = 0
        self.stair_flag[e] = 0
        self.clear_flag[e] = 0
        self.full_pose[e, :2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose[e].cpu().numpy()
        self.planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [
            int(r * 100.0 / self.args.map_resolution),
            int(c * 100.0 / self.args.map_resolution),
        ]
        self.full_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0
        self.lmb[e] = self._get_local_map_boundaries(
            (loc_r, loc_c), (self.local_w, self.local_h), (self.full_w, self.full_h)
        )
        self.planner_pose_inputs[e, 3:] = self.lmb[e]
        self.origins[e] = [
            self.lmb[e][2] * self.args.map_resolution / 100.0,
            self.lmb[e][0] * self.args.map_resolution / 100.0,
            0.0,
        ]
        self.local_map[e] = self.full_map[
            e, :, self.lmb[e, 0] : self.lmb[e, 1], self.lmb[e, 2] : self.lmb[e, 3]
        ]
        self.local_pose[e] = self.full_pose[e] - torch.from_numpy(self.origins[e]).to(self.device).float()

    def _get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes
        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w
            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h
        return [gx1, gx2, gy1, gy2]

    def _get_frontier_boundaries(self, frontier_loc, frontier_sizes, map_sizes):
        loc_r, loc_c = frontier_loc
        local_w, local_h = frontier_sizes
        full_w, full_h = map_sizes
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w
        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
        return [int(gx1), int(gx2), int(gy1), int(gy2)]

    def _policy_info(self, action: int):
        info = {
            "action": int(action),
            "goal_name": self.goal_name,
            "target_object": self.goal_name,
            "goal_cat_id": int(self.goal_cat_id),
            "found_goal": int(self.found_goal[0]),
            "num_categories": len(self.object_category),
        }
        if os.environ.get("AGENT_VIDEO_MODE", "none") in ("display", "save"):
            info.update(self._video_maps())
        return info

    def _video_maps(self):
        if self.local_map is None or self.target_point_map is None:
            return {}
        local = self.local_map[0].detach().cpu().numpy()
        frontiers = np.asarray(self.target_point_map[0])
        reply = self.reply_list[0] if self.reply_list else None
        return {
            "obstacle_map": self._obstacle_video_map(local),
            "frontier_map": self._frontier_video_map(frontiers, None),
            "value_map": self._frontier_video_map(frontiers, reply),
        }

    @staticmethod
    def _obstacle_video_map(local: np.ndarray) -> np.ndarray:
        obstacle = local[0] > 0.1
        explored = local[1] > 0.1
        out = np.ones((*obstacle.shape, 3), dtype=np.uint8) * 245
        out[explored] = (210, 230, 255)
        out[obstacle] = (40, 40, 40)
        return out

    @staticmethod
    def _frontier_video_map(frontiers: np.ndarray, selected) -> np.ndarray:
        labels = frontiers.astype(np.int32)
        out = np.ones((*labels.shape, 3), dtype=np.uint8) * 245
        colors = np.array(
            [
                [230, 80, 70],
                [70, 150, 230],
                [70, 180, 100],
                [220, 160, 60],
                [160, 100, 220],
                [70, 190, 190],
            ],
            dtype=np.uint8,
        )
        for label in np.unique(labels):
            if label <= 0:
                continue
            out[labels == label] = colors[(label - 1) % len(colors)]
        if selected is not None:
            selected_label = int(selected) + 1
            mask = labels == selected_label
            out[mask] = (255, 240, 0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (0, 0, 0), 2)
        return out


class OpenFMNavAgentNode(BaseAgentNode):
    """OpenFMNav HM3D ObjectNav agent."""

    def __init__(self, mode: str = "sync"):
        super().__init__("openfmnav_agent_node", mode)
        self._policy: Optional[OpenFMNavPolicy] = None
        self._episode_targets = {}
        self.create_subscription(String, "/episode_info_json", self._on_episode_info_json, 10)

    def _load_policy(self):
        self._policy = OpenFMNavPolicy(self)
        self.get_logger().info("OpenFMNav policy loaded.")

    def _reset_policy_state(self):
        pass

    def _on_episode_start(self, reset_resp, initial_obs: dict):
        ep_hash = int(reset_resp.episode_id_hash)
        target = self._target_from_reset_response(reset_resp) or self._episode_targets.get(ep_hash) or None
        if target is None and "objectgoal" not in initial_obs:
            wait_s = float(os.environ.get("OPENFMNAV_TARGET_WAIT_SECONDS", "5.0"))
            deadline = time.time() + wait_s
            while target is None and time.time() < deadline:
                time.sleep(0.01)
                target = self._episode_targets.get(ep_hash) or None
        self._policy.reset(initial_obs, target)
        self.get_logger().info(f"OpenFMNav target: {self._policy.goal_name}")

    def _compute_action(self, obs_dict: dict, step: int) -> Tuple[int, dict]:
        with torch.no_grad():
            return self._policy.act(obs_dict)

    def _on_episode_info_json(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if "episode_id_hash" in data:
            self._episode_targets[int(data["episode_id_hash"])] = data.get("target", "")

    @staticmethod
    def _target_from_reset_response(reset_resp):
        info_json = getattr(getattr(reset_resp, "observation", None), "info_json", "")
        if not info_json:
            return None
        try:
            data = json.loads(info_json)
        except json.JSONDecodeError:
            return None
        return data.get("target") or None


if __name__ == "__main__":
    run_agent(OpenFMNavAgentNode, env_prefix="OPENFMNAV_")
