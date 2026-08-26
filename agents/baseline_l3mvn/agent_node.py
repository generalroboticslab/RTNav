#!/usr/bin/env python3
"""L3MVN ROS2 agent node for the shared env-agent evaluation pipeline.

The upstream L3MVN implementation owns both Habitat stepping and policy logic in
main_llm_zeroshot.py.  This wrapper keeps the upstream mapping, semantic
prediction, LLM frontier scoring, and deterministic local planner, but drives it
from ROS observations and returns Habitat discrete actions to /sync_step.
"""

import json
import hashlib
import math
import os
import sys
import time
from collections import deque
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
from transformers import RobertaForMaskedLM, RobertaTokenizer

from base_agent import BaseAgentNode, run_agent


L3MVN_ROOT = os.environ.get("L3MVN_ROOT", "/opt/l3mvn")
if L3MVN_ROOT in sys.path:
    sys.path.remove(L3MVN_ROOT)
sys.path.insert(0, L3MVN_ROOT)

from agents.utils.semantic_prediction import SemanticPredMaskRCNN  # noqa: E402
from constants import category_to_id, hm3d_category, mp_categories_mapping  # noqa: E402
from model import Semantic_Mapping  # noqa: E402
from RedNet.RedNet_model import load_rednet  # noqa: E402
from envs.utils.fmm_planner import FMMPlanner  # noqa: E402
import agents.utils.visualization as vu  # noqa: E402
import envs.utils.pose as pu  # noqa: E402


STOP, FORWARD, LEFT, RIGHT, LOOK_UP, LOOK_DOWN = 0, 1, 2, 3, 4, 5
_COCO_OBJECTGOAL_TO_L3MVN = [0, 3, 2, 4, 5, 1]
_GOAL_ALIASES = {
    "chair": "chair",
    "bed": "bed",
    "plant": "plant",
    "potted plant": "plant",
    "toilet": "toilet",
    "tv": "tv_monitor",
    "television": "tv_monitor",
    "tv monitor": "tv_monitor",
    "tv_monitor": "tv_monitor",
    "sofa": "sofa",
    "couch": "sofa",
}


def _l2(x1: float, x2: float, y1: float, y2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _relative_pose(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2
    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = _l2(x1, x2, y1, y2)
    return [
        dist * np.cos(theta),
        dist * np.sin(theta),
        o2 - o1,
    ]


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


def _hash_bytes(arr):
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _array_signature(arr):
    a = np.asarray(arr)
    if a.size == 0:
        return {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "sum": None,
            "nnz": 0,
            "hash": _hash_bytes(a),
        }
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sum": float(np.sum(a, dtype=np.float64)),
        "nnz": int(np.count_nonzero(a)),
        "hash": _hash_bytes(a),
    }


def _state_debug_payload(local_map_e, target_edge_e, target_point_e, local_goal_e, local_pose_e, planner_pose_e):
    return {
        "map_obstacle": _array_signature(local_map_e[0]),
        "map_explored": _array_signature(local_map_e[1]),
        "target_edge": _array_signature(target_edge_e),
        "target_point": _array_signature(np.asarray(target_point_e).astype(np.int32)),
        "local_goal": _array_signature(local_goal_e),
        "local_pose": [float(x) for x in np.asarray(local_pose_e).reshape(-1)[:3]],
        "planner_pose": [float(x) for x in np.asarray(planner_pose_e).reshape(-1)[:7]],
    }


class L3MVNLocalPlanner:
    """Habitat-free copy of the upstream Sem_Exp_Env_Agent policy surface."""

    def __init__(self, args, device, rednet_path):
        self.args = args
        self.rank = 0
        self.device = device
        self.res = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (args.frame_height, args.frame_width),
                    interpolation=Image.NEAREST,
                ),
            ]
        )
        self.sem_pred = SemanticPredMaskRCNN(args)
        self.red_sem_pred = load_rednet(device, ckpt=rednet_path, resize=True)
        self.red_sem_pred.eval()
        self.selem = skimage.morphology.disk(3)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.legend = None
        self.rgb_vis = None

    def _plan(self, planner_inputs):
        args = self.args

        self.last_loc = self.curr_loc

        map_pred = np.rint(planner_inputs["map_pred"])
        exp_pred = np.rint(planner_inputs["exp_pred"])
        goal = planner_inputs["goal"]

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / args.map_resolution - gx1),
            int(c * 100.0 / args.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1] = 1

        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [
            int(r * 100.0 / args.map_resolution - gx1),
            int(c * 100.0 / args.map_resolution - gy1),
        ]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self.visited_vis[gx1:gx2, gy1:gy2] = vu.draw_line(
            last_start, start, self.visited_vis[gx1:gx2, gy1:gy2]
        )

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
            if dist < args.collision_threshold:
                self.collision_n += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * (
                            (i + buf) * np.cos(np.deg2rad(t1))
                            + (j - width // 2) * np.sin(np.deg2rad(t1))
                        )
                        wy = y1 + 0.05 * (
                            (i + buf) * np.sin(np.deg2rad(t1))
                            - (j - width // 2) * np.cos(np.deg2rad(t1))
                        )
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, replan, stop = self._get_stg(map_pred, start, np.copy(goal), planning_window)

        if replan:
            self.replan_count += 1
            print("false: ", self.replan_count)
        else:
            self.replan_count = 0

        if (stop and planner_inputs["found_goal"] == 1) or self.replan_count > 26:
            action = 0
            decision_branch = "stop_or_replan"
            angle_st_goal = None
            angle_agent = None
            relative_angle = None
            eve_start_x = None
            eve_start_y = None
            eve_probe = None
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0], stg_y - start[1]))
            angle_agent = start_o % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            angle_eps = 1e-4
            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if abs(relative_angle - 180.0) <= angle_eps:
                relative_angle = 180.0
            elif relative_angle > 180:
                relative_angle -= 360

            eve_start_x = int(5 * math.sin(angle_st_goal) + start[0])
            eve_start_y = int(5 * math.cos(angle_st_goal) + start[1])
            if eve_start_x > map_pred.shape[0]:
                eve_start_x = map_pred.shape[0]
            if eve_start_y > map_pred.shape[0]:
                eve_start_y = map_pred.shape[0]
            if eve_start_x < 0:
                eve_start_x = 0
            if eve_start_y < 0:
                eve_start_y = 0
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
                action = 3
                decision_branch = "turn_right"
            elif relative_angle < -turn_threshold - angle_eps:
                action = 2
                decision_branch = "turn_left"
            else:
                action = 1
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
        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1 = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1 : h + 1, 1 : w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(grid[x1:x2, y1:y2], self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[
            cv2.dilate(self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2], self.kernel) == 1
        ] = 1

        traversible[
            int(start[0] - x1) - 1 : int(start[0] - x1) + 2,
            int(start[1] - y1) - 1 : int(start[1] - y1) + 2,
        ] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(goal, selem) != True
        goal = 1 - goal * 1.0
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), replan, stop

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]
        semantic = obs[:, :, 4:5].squeeze()
        if args.use_gtsem:
            self.rgb_vis = rgb
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))
            for i in range(16):
                sem_seg_pred[:, :, i][semantic == i + 1] = 1
        else:
            red_semantic_pred, semantic_pred = self._get_sem_pred(
                rgb.astype(np.uint8), depth, use_seg=use_seg
            )

            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))
            for i in range(0, 15):
                sem_seg_pred[:, :, i][red_semantic_pred == mp_categories_mapping[i]] = 1

            sem_seg_pred[:, :, 0][semantic_pred[:, :, 0] == 0] = 0
            sem_seg_pred[:, :, 1][semantic_pred[:, :, 1] == 0] = 0
            sem_seg_pred[:, :, 3][semantic_pred[:, :, 3] == 0] = 0
            sem_seg_pred[:, :, 4][semantic_pred[:, :, 4] == 1] = 1
            sem_seg_pred[:, :, 5][semantic_pred[:, :, 5] == 1] = 1

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2 :: ds, ds // 2 :: ds]
            sem_seg_pred = sem_seg_pred[ds // 2 :: ds, ds // 2 :: ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.0] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.0

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * (max_d - min_d) * 100.0

        return depth

    def _get_sem_pred(self, rgb, depth, use_seg=True):
        if use_seg:
            image = torch.from_numpy(rgb).to(self.device).unsqueeze_(0).float()
            depth = torch.from_numpy(depth).to(self.device).unsqueeze_(0).float()
            with torch.no_grad():
                red_semantic_pred = self.red_sem_pred(image, depth)
                red_semantic_pred = red_semantic_pred.squeeze().cpu().detach().numpy()
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            red_semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1]))
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb[:, :, ::-1]
        return red_semantic_pred, semantic_pred


class L3MVNPolicy:
    """Single-environment L3MVN zero-shot policy state machine."""

    def __init__(self, node: BaseAgentNode):
        self.node = node
        self.device = node._device
        self.args = self._make_args()
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        self.num_scenes = 1
        self.nc = int(self.args.num_sem_categories) + 4
        self.map_size = self.args.map_size_cm // self.args.map_resolution
        self.full_w = self.full_h = self.map_size
        self.local_w = int(self.full_w / self.args.global_downscaling)
        self.local_h = int(self.full_h / self.args.global_downscaling)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.tv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        self.scoring_fxn = self._configure_lm()
        self.sem_map_module = Semantic_Mapping(self.args).to(self.device)
        self.sem_map_module.eval()
        self.goal_rng = torch.Generator()
        self.goal_rng.set_state(torch.get_rng_state())

        self.planner_agent = self._make_planner_agent()

        self.full_map = torch.zeros(
            self.num_scenes, self.nc, self.full_w, self.full_h
        ).float().to(self.device)
        self.local_map = torch.zeros(
            self.num_scenes, self.nc, self.local_w, self.local_h
        ).float().to(self.device)
        self.local_ob_map = np.zeros((self.num_scenes, self.local_w, self.local_h))
        self.local_ex_map = np.zeros((self.num_scenes, self.local_w, self.local_h))
        self.target_edge_map = np.zeros((self.num_scenes, self.local_w, self.local_h))
        self.target_point_map = np.zeros((self.num_scenes, self.local_w, self.local_h))
        self.full_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
        self.local_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
        self.origins = np.zeros((self.num_scenes, 3))
        self.lmb = np.zeros((self.num_scenes, 4)).astype(int)
        self.planner_pose_inputs = np.zeros((self.num_scenes, 7))
        self.frontier_score_list = [deque(maxlen=10) for _ in range(self.num_scenes)]
        self.global_goals = [[0, 0] for _ in range(self.num_scenes)]

        self.step_masks = torch.zeros(self.num_scenes).float().to(self.device)
        self.stair_flag = np.zeros(self.num_scenes)
        self.clear_flag = np.zeros(self.num_scenes)
        self.wait_env = np.zeros(self.num_scenes)
        self.finished = np.zeros(self.num_scenes)

        self.prev_sim_location = None
        self.goal_cat_id = 0
        self.goal_name = "chair"
        self.found_goal = [0]
        self._pending_clear_flag = 0
        self.loop_step = 0
        self.decision_step = 0
        self._official_first_action_done = False
        self._zero_obs_countdown = 0
        self.initialized = False
        self.trace_path = os.environ.get("L3MVN_AGENT_TRACE_PATH", "").strip()
        self.trace = None
        self.trace_episodes = []
        self._last_state_debug = None

    def _make_args(self):
        return SimpleNamespace(
            seed=int(os.environ.get("HABITAT_SEED", "1")),
            cuda=torch.cuda.is_available(),
            device=self.device,
            eval=1,
            task_config=os.environ.get("L3MVN_TASK_CONFIG", "tasks/objectnav_hm3d.yaml"),
            split=os.environ.get("EVAL_SPLIT", "val"),
            version=os.environ.get("L3MVN_VERSION", "v1"),
            dump_location=os.environ.get("L3MVN_DUMP_LOCATION", "./tmp"),
            exp_name=os.environ.get("L3MVN_EXP_NAME", "ros_l3mvn_zero"),
            visualize=int(os.environ.get("L3MVN_VISUALIZE", "0")),
            print_images=int(os.environ.get("L3MVN_PRINT_IMAGES", "0")),
            env_frame_width=int(os.environ.get("L3MVN_ENV_FRAME_WIDTH", "640")),
            env_frame_height=int(os.environ.get("L3MVN_ENV_FRAME_HEIGHT", "480")),
            frame_width=int(os.environ.get("L3MVN_FRAME_WIDTH", "160")),
            frame_height=int(os.environ.get("L3MVN_FRAME_HEIGHT", "120")),
            max_episode_length=int(os.environ.get("L3MVN_MAX_EPISODE_LENGTH", "500")),
            camera_height=float(os.environ.get("L3MVN_CAMERA_HEIGHT", "0.88")),
            hfov=float(os.environ.get("L3MVN_HFOV", "79.0")),
            turn_angle=float(os.environ.get("L3MVN_TURN_ANGLE", "30")),
            min_depth=float(os.environ.get("L3MVN_MIN_DEPTH", "0.5")),
            max_depth=float(os.environ.get("L3MVN_MAX_DEPTH", "5.0")),
            floor_thr=50,
            num_processes=1,
            num_local_steps=int(os.environ.get("L3MVN_NUM_LOCAL_STEPS", "10")),
            num_global_steps=int(os.environ.get("L3MVN_NUM_GLOBAL_STEPS", "20")),
            num_sem_categories=int(os.environ.get("L3MVN_NUM_SEM_CATEGORIES", "16")),
            sem_pred_prob_thr=float(os.environ.get("L3MVN_SEM_PRED_PROB_THR", "0.9")),
            global_downscaling=int(os.environ.get("L3MVN_GLOBAL_DOWNSCALING", "2")),
            vision_range=int(os.environ.get("L3MVN_VISION_RANGE", "100")),
            map_resolution=int(os.environ.get("L3MVN_MAP_RESOLUTION", "5")),
            du_scale=int(os.environ.get("L3MVN_DU_SCALE", "1")),
            map_size_cm=int(os.environ.get("L3MVN_MAP_SIZE_CM", "4800")),
            cat_pred_threshold=float(os.environ.get("L3MVN_CAT_PRED_THRESHOLD", "5.0")),
            map_pred_threshold=float(os.environ.get("L3MVN_MAP_PRED_THRESHOLD", "1.0")),
            exp_pred_threshold=float(os.environ.get("L3MVN_EXP_PRED_THRESHOLD", "1.0")),
            collision_threshold=float(os.environ.get("L3MVN_COLLISION_THRESHOLD", "0.10")),
            use_gtsem=int(os.environ.get("L3MVN_USE_GTSEM", "0")),
            sem_gpu_id=int(os.environ.get("L3MVN_SEM_GPU_ID", "0" if torch.cuda.is_available() else "-2")),
        )

    def _make_planner_agent(self):
        rednet_path = os.environ.get(
            "L3MVN_REDNET_PATH", "RedNet/model/rednet_semmap_mp3d_40.pth"
        )
        return L3MVNLocalPlanner(self.args, self.device, rednet_path)

    def _configure_lm(self):
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        lm_model = RobertaForMaskedLM.from_pretrained("roberta-large")
        mask_id = tokenizer.mask_token_id
        lm_model.eval()
        lm_model = lm_model.to(self.device)

        def scoring_fxn(text, mean=True):
            text = "<s> " + text.capitalize() + " </s>"
            tokenized_text = tokenizer.tokenize(text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor(indexed_tokens, device=self.device)
            num_masks = tokens_tensor.shape[0] - 3
            masked_tokens_tensor = tokens_tensor.repeat(num_masks, 1)
            mask_rows = torch.arange(num_masks, device=self.device)
            mask_positions = torch.arange(1, num_masks + 1, device=self.device)
            masked_tokens_tensor[mask_rows, mask_positions] = mask_id

            with torch.no_grad():
                logits = lm_model(masked_tokens_tensor).logits
                mask_scores = logits[mask_rows, mask_positions]
                token_scores = mask_scores[mask_rows, tokens_tensor[mask_positions]]
                total = (token_scores - torch.logsumexp(mask_scores, dim=1)).sum()

            return total / num_masks if mean else total

        return scoring_fxn

    def reset(self, obs_dict: dict, target_name: Optional[str] = None):
        self.prev_sim_location = None
        self.goal_cat_id, self.goal_name = self._resolve_goal(obs_dict, target_name)
        hidden_wait_steps = 0
        if self._official_first_action_done:
            next_loop_step = self._next_episode_loop_step(self.loop_step)
            hidden_wait_steps = next_loop_step - self.loop_step
            self._zero_obs_countdown = int(hidden_wait_steps > 0)
            self.loop_step = next_loop_step
        else:
            self.loop_step = 0
            self._zero_obs_countdown = 0
        self.decision_step = 0
        self.found_goal = [0]
        if not self._official_first_action_done:
            self._pending_clear_flag = 0
        self.wait_env.fill(0)
        self.finished.fill(0)
        self._init_map_and_pose_for_env(0)

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
        self.initialized = self._official_first_action_done
        self._last_state_debug = [None for _ in range(self.num_scenes)]
        if hidden_wait_steps:
            self._advance_wait_steps(hidden_wait_steps, obs_dict)
        if self.trace_path and self.trace is not None:
            self.trace_episodes.append(self.trace)
        self.trace = None
        if self.trace_path:
            os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
            self.trace = {
                "source": "ros_l3mvn_policy",
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
        info["wrapped_do"] = float(sensor_pose[2])
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
        self._append_trace(action, info, sim_location, main_step, l_step)
        return action, self._policy_info(action)

    def _first_action(self, obs_dict: dict) -> int:
        self.prev_sim_location = self._sim_location(obs_dict)
        info = self._info(self._sensor_pose(obs_dict) or [0.0, 0.0, 0.0])
        obs = self._preprocess_ros_obs(obs_dict)

        poses = torch.from_numpy(np.asarray([info["sensor_pose"]])).float().to(self.device)
        eve_angle = np.asarray([info["eve_angle"]])
        _, self.local_map, self.local_map_stair, self.local_pose = self.sem_map_module(
            obs, poses, self.local_map, self.local_pose, eve_angle
        )
        self.local_map[:, 0, :, :][self.local_map[:, 13, :, :] > 0] = 0

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
        self._append_trace(int(action), info, self.prev_sim_location, -1, -1)
        return int(action)

    def _append_trace(self, action: int, info: dict, sim_location, main_step: int, l_step: int):
        if not self.trace_path or self.trace is None:
            return
        frontier_scores = list(self.frontier_score_list[0]) if len(self.frontier_score_list) > 0 else []
        rec = {
            "decision_step": int(self.decision_step + 1),
            "main_step": int(main_step),
            "l_step": int(l_step),
            "action": int(action),
            "sensor_pose": [float(x) for x in info.get("sensor_pose", [0.0, 0.0, 0.0])],
            "raw_do": float(info.get("sensor_pose", [0.0, 0.0, 0.0])[2]),
            "wrapped_do": float(info.get("wrapped_do", info.get("sensor_pose", [0.0, 0.0, 0.0])[2])),
            "sim_location": [float(x) for x in sim_location],
            "eve_angle": float(getattr(self.planner_agent, "eve_angle", 0.0)),
            "clear_flag": int(info.get("clear_flag", 0)),
            "replan_count": int(getattr(self.planner_agent, "replan_count", 0)),
            "collision_n": int(getattr(self.planner_agent, "collision_n", 0)),
            "found_goal": int(self.found_goal[0]) if len(self.found_goal) > 0 else 0,
            "global_goal": [int(self.global_goals[0][0]), int(self.global_goals[0][1])],
            "frontier_count": int(len(frontier_scores)),
            "frontier_max": float(max(frontier_scores)) if frontier_scores else None,
            "goal_debug": (self._last_goal_debug[0] if getattr(self, "_last_goal_debug", None) else {}),
            "plan_debug": getattr(self.planner_agent, "_last_plan_debug", {}),
            "state_debug": (self._last_state_debug[0] if self._last_state_debug else None),
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
            _, self.local_map, self.local_map_stair, self.local_pose = self.sem_map_module(
                obs, poses, self.local_map, self.local_pose, eve_angle
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

    def _write_trace(self):
        if not self.trace_path:
            return
        with open(self.trace_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source": "ros_l3mvn_policy",
                    "episodes": self.trace_episodes,
                    "current_episode": self.trace,
                },
                f,
                indent=2,
            )

    def _main_loop_action(self, obs: torch.Tensor, info: dict) -> int:
        step = self.loop_step
        l_step = step % self.args.num_local_steps

        poses = torch.from_numpy(np.asarray([info["sensor_pose"]])).float().to(self.device)
        eve_angle = np.asarray([info["eve_angle"]])
        _, self.local_map, self.local_map_stair, self.local_pose = self.sem_map_module(
            obs, poses, self.local_map, self.local_pose, eve_angle
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
            if loc_r > self.local_w:
                loc_r = self.local_w - 1
            if loc_c > self.local_h:
                loc_c = self.local_h - 1
            if info["clear_flag"] or self.local_map[e, 18, loc_r, loc_c] > 0.5:
                self.stair_flag[e] = 1
            if self.stair_flag[e]:
                if torch.any(self.local_map[e, 18, :, :] > 0.5):
                    self.local_map[e, 0, :, :] = self.local_map_stair[e, 0, :, :]
                self.local_map[e, 0, :, :] = self.local_map_stair[e, 0, :, :]

        if l_step == self.args.num_local_steps - 1:
            self._update_frontiers(info)
            self._sample_random_global_goals()

        local_goal_maps = [np.zeros((self.local_w, self.local_h)) for _ in range(self.num_scenes)]
        found_goal = [0 for _ in range(self.num_scenes)]
        goal_debug = []
        state_debug = [None for _ in range(self.num_scenes)]
        for e in range(self.num_scenes):
            global_item = self._select_frontier(e)
            if np.any(self.target_point_map[e] == global_item + 1):
                local_goal_maps[e][self.target_point_map[e] == global_item + 1] = 1
                goal_source = "frontier"
            else:
                x, y = self.global_goals[e]
                local_goal_maps[e][x, y] = 1
                goal_source = "random_global"
            goal_debug.append(
                {
                    "global_item": int(global_item),
                    "goal_source": goal_source,
                    "global_goal": [int(self.global_goals[e][0]), int(self.global_goals[e][1])],
                }
            )

            cn = self.goal_cat_id + 4
            if self.local_map[e, cn, :, :].sum() != 0.0:
                cat_semantic_scores = self.local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores[cat_semantic_scores > 0] = 1.0
                if cn == 9:
                    cat_semantic_scores = cv2.dilate(cat_semantic_scores, self.tv_kernel)
                local_goal_maps[e] = _find_big_connect(cat_semantic_scores)
                found_goal[e] = 1

        if self.trace is not None and l_step == self.args.num_local_steps - 1:
            for e in range(self.num_scenes):
                state_debug[e] = _state_debug_payload(
                    self.local_map[e].detach().cpu().numpy(),
                    self.target_edge_map[e],
                    self.target_point_map[e],
                    local_goal_maps[e],
                    self.local_pose[e].detach().cpu().numpy(),
                    self.planner_pose_inputs[e],
                )

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
        self._last_goal_debug = goal_debug
        self._last_state_debug = state_debug
        self.loop_step += 1
        return action

    def _update_frontiers(self, info: dict):
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
            self.target_edge_map[e], self.target_point_map[e], goal_score = self._remove_small_points(
                _local_ob_map, target_edge, 4, local_pose_map
            )

            self.local_ob_map[e] = np.zeros((self.local_w, self.local_h))
            self.local_ex_map[e] = np.zeros((self.local_w, self.local_h))

            self.frontier_score_list[e] = []
            tpm = len(list(set(self.target_point_map[e].ravel()))) - 1
            for lay in range(tpm):
                f_pos = np.argwhere(self.target_point_map[e] == lay + 1)
                fmb = self._get_frontier_boundaries(
                    (f_pos[0][0], f_pos[0][1]),
                    (self.local_w / 6, self.local_h / 6),
                    (self.local_w, self.local_h),
                )
                objs_list = []
                for se_cn in range(self.args.num_sem_categories - 1):
                    if self.local_map[e][se_cn + 4, fmb[0] : fmb[1], fmb[2] : fmb[3]].sum() != 0.0:
                        objs_list.append(hm3d_category[se_cn])

                if len(objs_list) > 0 and self.found_goal[e] == 0:
                    ref_dist = F.softmax(self._construct_dist(objs_list), dim=0).to(self.device)
                    self.frontier_score_list[e].append(float(ref_dist[category_to_id.index(self.goal_name)].item()))
                else:
                    fallback = 0.1
                    if goal_score:
                        fallback = goal_score[lay] / max(goal_score) * 0.1 + 0.1
                    self.frontier_score_list[e].append(fallback)

    def _select_frontier(self, e: int) -> int:
        global_item = 0
        scores = self.frontier_score_list[e]
        if len(scores) > 0:
            if max(scores) > 0.3:
                global_item = scores.index(max(scores))
            elif max(scores) > 0.1:
                for f_score in scores:
                    if f_score > 0.1:
                        break
                    global_item += 1
        return global_item

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

    def _construct_dist(self, objs):
        query_str = "A room containing "
        for ob in objs:
            query_str += ob + ", "
        query_str += "and"

        temp = []
        for label in category_to_id:
            temp_str = query_str + " " + label + "."
            score = self.scoring_fxn(temp_str)
            temp.append(score)
        return torch.stack(temp)

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
        return torch.from_numpy(obs).unsqueeze(0).float().to(self.device)

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

    def _resolve_goal(self, obs_dict: dict, target_name: Optional[str]):
        if "objectgoal" in obs_dict:
            objectgoal = int(np.asarray(obs_dict["objectgoal"]).flat[0])
            if 0 <= objectgoal < len(category_to_id):
                return _COCO_OBJECTGOAL_TO_L3MVN[objectgoal], category_to_id[objectgoal]
        if target_name:
            alias = _GOAL_ALIASES.get(target_name.strip().lower())
            if alias in category_to_id:
                return _COCO_OBJECTGOAL_TO_L3MVN[category_to_id.index(alias)], alias
        self.node.get_logger().warning(
            f"L3MVN target {target_name!r} is outside the original six-category vocabulary; defaulting to chair."
        )
        return 0, "chair"

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
        return {
            "action": int(action),
            "goal_name": self.goal_name,
            "goal_cat_id": int(self.goal_cat_id),
            "found_goal": int(self.found_goal[0]),
        }


class L3MVNAgentNode(BaseAgentNode):
    """L3MVN zero-shot ObjectNav agent."""

    def __init__(self, mode: str = "sync"):
        super().__init__("l3mvn_agent_node", mode)
        self._policy: Optional[L3MVNPolicy] = None
        self._episode_targets = {}
        self.create_subscription(String, "/episode_info_json", self._on_episode_info_json, 10)
        os.chdir(L3MVN_ROOT)

    def _load_policy(self):
        self._policy = L3MVNPolicy(self)
        self.get_logger().info("L3MVN zero-shot policy loaded.")

    def _reset_policy_state(self):
        pass

    def _on_episode_start(self, reset_resp, initial_obs: dict):
        ep_hash = int(reset_resp.episode_id_hash)
        target = self._episode_targets.get(ep_hash)
        if target is None and "objectgoal" not in initial_obs:
            deadline = time.time() + 0.5
            while target is None and time.time() < deadline:
                time.sleep(0.01)
                target = self._episode_targets.get(ep_hash)
        self._policy.reset(initial_obs, target)
        self.get_logger().info(f"L3MVN target: {self._policy.goal_name}")

    def _compute_action(self, obs_dict: dict, step: int) -> Tuple[int, dict]:
        # L3MVN keeps map tensors as mutable episode state. torch.inference_mode()
        # marks outputs as inference tensors, which later rejects in-place map
        # resets. no_grad keeps the original eval behavior without that state
        # mutation constraint.
        with torch.no_grad():
            return self._policy.act(obs_dict)

    def _on_episode_info_json(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if "episode_id_hash" in data:
            self._episode_targets[int(data["episode_id_hash"])] = data.get("target", "")


if __name__ == "__main__":
    run_agent(L3MVNAgentNode, env_prefix="L3MVN_")
