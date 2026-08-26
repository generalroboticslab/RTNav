"""Fast tests for the fixed behavior choices; no model weights required."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rtnav.config.environments.config_loader import load_config
from rtnav.config.modules.decision_cfg import DecisionConfig
from rtnav.config.modules.frontier_cfg import FrontierConfig
from rtnav.core.data_types import FrontierMapSnapshot
from rtnav.core.inference_gate import InferenceGate
from rtnav.core.shared_state import SharedState
from rtnav.modules.decision.decision_thread import DecisionThread, NavUpdate
from rtnav.modules.decision.frontier_image_vlm import FrontierImageEntry, FrontierImageVLM
from rtnav.modules.decision.frontier_strategy import FrontierStrategy
from rtnav.modules.decision.target_strategy import TargetStrategy
from rtnav.modules.decision.target_verifier import (
    TargetCandidate,
    TargetVerifier,
    VerificationResult,
    _mask_too_offset,
)
from rtnav.modules.frontier.vlfm_frontier_detector import _warm_frontier_jit
from rtnav.modules.mapping.obstacle_map.utils import find_navigable_frontier_goal
from rtnav.modules.mapping.obstacle_map_habitat import HabitatObstacleMap, filter_obstacle_points
from rtnav.modules.mapping.utils import cast_rays_batch
from rtnav.modules.perception.detection_policy import (
    NMS_IOU_THRESHOLD,
    PROMPT_TEMPLATES,
    TOP_K_LABELS,
    average_prompt_logits,
    canonical_top_k,
    prompt_queries,
    square_padding,
)
from rtnav.modules.perception.detector_thread import OpenVocabDetectorThread
from rtnav.modules.scenegraph.merge_policy import same_label_pairs
from rtnav.modules.scenegraph.scene_graph import ObjectDetection, ObjectNode
from rtnav.task.task_parser.task_parser import TaskParser
from rtnav.tools.visualization.detection_visualizer_thread import (
    WebDetectionVisualizerThread,
    _as_rgb3,
)
from rtnav.tools.visualization.obstacle_map_renderer import render_region
from rtnav.tools.visualization.vlm_decision_logger import VLMDecisionLogger


class NoAliases:
    def filter_candidates(self, target, candidates, threshold):
        return []


def detection(label, confidence, xy=(0.0, 0.0), yaw=0.0, top_k=()):
    return ObjectDetection(
        label=label,
        bbox_2d=np.zeros(4),
        bbox_3d=np.zeros((2, 3)),
        centroid=np.zeros(3),
        point_cloud=np.zeros((1, 3)),
        colors=np.zeros((1, 3)),
        owl_features=np.ones(2),
        confidence=confidence,
        observed_from_xy=xy,
        observed_from_yaw=yaw,
        top_k_label_probs=list(top_k),
    )


class CanonicalBehaviorTest(unittest.TestCase):
    def test_map_visualization_uses_vlfm_explored_mask(self):
        class FakeObstacleMap:
            size = 400
            ppm = 1.0
            origin_px = np.array([200, 200])
            _last_robot_pose = (0.0, 0.0, 0.0)

            def __init__(self):
                self.height_map = np.full((self.size, self.size), -np.inf)
                self.height_map[100, 100] = 1.0
                self.traversability = np.ones((self.size, self.size))

            def xy_to_px(self, points):
                pixels = np.empty_like(points)
                pixels[:, 0] = self.origin_px[0] + points[:, 0] * self.ppm
                pixels[:, 1] = self.origin_px[1] - points[:, 1] * self.ppm
                return pixels.astype(int)

        obstacle_map = FakeObstacleMap()
        explored = np.zeros((400, 400), dtype=bool)
        # VLFM pixel for world (x=1, y=2): (198, 201).
        explored[201, 198] = True
        snapshot = FrontierMapSnapshot(
            explored=explored,
            pixels_per_meter=1.0,
            episode_pixel_origin=(200.0, 200.0),
        )

        rendered, bounds = render_region(obstacle_map, 15.0, snapshot)
        self.assertEqual(bounds, (0, 0, 400, 400))
        self.assertTrue(np.array_equal(rendered[198, 201], (170, 170, 170)))
        self.assertTrue(np.array_equal(rendered[100, 100], (220, 220, 220)))

        no_snapshot, _ = render_region(obstacle_map, 15.0)
        self.assertTrue(np.array_equal(no_snapshot[100, 100], (220, 220, 220)))

    def test_detector_waits_only_for_task_ready(self):
        detector = object.__new__(OpenVocabDetectorThread)
        detector.shared_state = SharedState()
        self.assertEqual(detector._should_skip(), (True, "task_not_ready"))
        detector.shared_state.task_ready.set()
        self.assertEqual(detector._should_skip(), (False, ""))

    def test_float_detection_rgb_is_not_quantized_to_black(self):
        image = _as_rgb3(np.full((4, 5, 3), 0.5, dtype=np.float32))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(float(image.mean()), 100.0)

        visualizer = object.__new__(WebDetectionVisualizerThread)
        visualizer.display_height = 0
        camera = SimpleNamespace(
            rgb_image_detector_input=np.full((8, 10, 3), 0.5, dtype=np.float32),
            detections_detector_input=[],
            rgb_image=None,
            detections=[],
        )
        result = SimpleNamespace(camera_results={"camera": camera})
        panel = visualizer._create_detection_grid(result)
        self.assertGreater(float(panel[:8].mean()), 100.0)

    def test_detection_visualization_uses_blue_boxes_and_red_targets(self):
        visualizer = object.__new__(WebDetectionVisualizerThread)
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        chair = SimpleNamespace(name="chair", confidence=0.75, bbox=(20, 30, 80, 90))

        rendered = visualizer._draw_detections_on_image(image, [chair])
        blue = rendered[60, 20]
        self.assertGreater(int(blue[0]), int(blue[2]))

        rendered = visualizer._draw_detections_on_image(image, [chair], {"chair"})
        red = rendered[60, 20]
        self.assertGreater(int(red[2]), int(red[0]))

    def test_vlm_decision_logger_writes_images_and_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = VLMDecisionLogger()
            logger.new_episode("scene", "episode", output_dir=directory)
            image = np.full((12, 16, 3), 180, dtype=np.uint8)
            logger.record_frontier(
                image,
                image,
                {
                    "selected": {"frontier_idx": 2},
                    "frontiers": [{"direction": 1, "logprob": -0.1}],
                    "top_tokens": [
                        {"token": "1", "probability": 0.75},
                        {"token": " chair", "probability": 0.2},
                    ],
                },
                rejected_rgb=image,
            )
            logger.record_verification(image, {"accepted": True, "yes_normalized": 0.95})
            logger.save_episode()
            logger.shutdown()

            output = Path(directory) / "vlm_decisions"
            self.assertTrue((output / "frontier/0001_input.jpg").is_file())
            self.assertTrue((output / "frontier/0001_selected.jpg").is_file())
            self.assertTrue((output / "frontier/0001_rejected.jpg").is_file())
            self.assertTrue((output / "verification/0001_yes.png").is_file())
            events = [
                json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual([event["type"] for event in events], ["frontier", "verification"])
            token_log = (output / "frontier/logprobs.txt").read_text()
            self.assertIn("token='1' probability=0.750000", token_log)
            self.assertIn("token=' chair' probability=0.200000", token_log)
            self.assertNotIn("logprob=", token_log)

    def test_frontier_log_uses_top_ten_token_probabilities(self):
        top_logprobs = [
            SimpleNamespace(token=f"token_{i}", logprob=-float(i)) for i in range(12)
        ]
        choice = SimpleNamespace(
            logprobs=SimpleNamespace(
                top_logprobs=None,
                content=[SimpleNamespace(top_logprobs=reversed(top_logprobs))],
            )
        )

        probabilities = FrontierImageVLM._top_token_probabilities(choice)

        self.assertEqual(len(probabilities), 10)
        self.assertEqual(probabilities[0], ("token_0", 1.0))
        self.assertEqual(probabilities[-1][0], "token_9")
        self.assertAlmostEqual(probabilities[1][1], math.exp(-1.0))

    def test_detection_threshold_is_fixed_for_supported_benchmarks(self):
        for env in ("hm3d_v1", "hm3d_v2", "ovon"):
            config = load_config(env)
            self.assertEqual(config.detection.threshold, 0.25)
            self.assertEqual(config.detection.inference_res, 960)

    def test_owlv2_policy_is_prompt_averaged_square_padded_and_top_k(self):
        self.assertEqual(len(PROMPT_TEMPLATES), 7)
        self.assertEqual(len(prompt_queries(["chair", "bed"])), 14)
        logits = np.arange(14, dtype=float).reshape(1, 1, 14)
        averaged = average_prompt_logits(logits, num_classes=2)
        np.testing.assert_allclose(averaged, [[[3.0, 10.0]]])
        self.assertEqual(square_padding(3, 5), (0, 0, 0, 2))
        self.assertEqual(square_padding(5, 3), (0, 2, 0, 0))
        self.assertEqual(NMS_IOU_THRESHOLD, 0.6)
        self.assertEqual(TOP_K_LABELS, 3)

        top_k = canonical_top_k(
            ["sofa", "couch", "chair", "bed"],
            [0.7, 0.9, 0.8, 0.6],
            {"sofa": "couch"},
        )
        self.assertEqual(top_k, [("couch", 0.9), ("chair", 0.8), ("bed", 0.6)])

    def test_task_parser_accepts_only_one_exact_find_target(self):
        parser = TaskParser(dummy=True, synonym_model=NoAliases())
        expanded = parser.parse_task("Find the potted plant.")
        self.assertEqual(expanded, {"potted plant": ["potted plant"]})
        with self.assertRaises(ValueError):
            parser.parse_task("Find a chair and then a bed")

    def test_confirmation_counts_distinct_camera_viewpoints(self):
        node = ObjectNode(node_id=1, number_to_be_confirmed=3)
        node.update_from_detection(detection("chair", 0.8), 1)
        node.update_from_detection(detection("chair", 0.8), 2)
        self.assertEqual(node.view_count, 1)
        self.assertFalse(node.is_confirmed)

        node.update_from_detection(detection("chair", 0.8, yaw=0.1), 3)
        node.update_from_detection(detection("chair", 0.8, xy=(0.2, 0.0)), 4)
        self.assertEqual(node.view_count, 3)
        self.assertTrue(node.is_confirmed)

    def test_label_voting_and_top_k_history(self):
        node = ObjectNode(node_id=1, voting_window=5)
        node.label_history = [
            ("chair", 0.4, 1, []),
            ("stool", 0.7, 2, []),
            ("chair", 0.4, 3, []),
        ]
        self.assertEqual(node.get_consensus_label(), "chair")

        node.label = "chair"
        node.is_confirmed = True
        node.label_history = [
            *[("chair", 1.0, i, []) for i in range(5)],
            *[("couch", 1.0, i + 5, []) for i in range(7)],
        ]
        self.assertEqual(node.get_consensus_label(), "couch")

        node.update_from_detection(
            detection("couch", 0.9, top_k=(("couch", 0.9), ("chair", 0.1))),
            20,
        )
        self.assertEqual(node.label_prob_count, 1)
        self.assertEqual(node.label_prob_sums, {"couch": 0.9, "chair": 0.1})

    def test_scene_graph_merge_pairs_are_same_label_only(self):
        nodes = {
            1: SimpleNamespace(label="chair"),
            2: SimpleNamespace(label="chair"),
            3: SimpleNamespace(label="couch"),
        }
        self.assertEqual(same_label_pairs([(1, 2), (1, 3)], nodes), [(1, 2)])

    def test_vllm_gate_pauses_detector_without_touching_navigation(self):
        state = SharedState()
        state.nav.status = "navigating"
        state.nav.goal_xy = (1.0, 2.0)
        gate = InferenceGate()
        self.assertTrue(gate.pause_for_vllm(timeout=0.0))
        self.assertTrue(gate.detector_should_pause())
        self.assertEqual((state.nav.status, state.nav.goal_xy), ("navigating", (1.0, 2.0)))
        with patch("time.sleep"):
            gate.resume_after_vllm()
        self.assertFalse(gate.detector_should_pause())

    def test_fog_of_war_rays_use_individual_depth_ranges(self):
        endpoints = cast_rays_batch(
            5,
            5,
            np.array([0.0, math.pi / 2]),
            np.array([2.0, 4.0]),
            np.zeros((12, 12), dtype=np.uint8),
            12,
            12,
        )
        np.testing.assert_array_equal(endpoints, [[5, 5], [7, 5], [5, 1]])

    def test_target_commit_completes_the_single_episode_goal(self):
        state = SharedState()
        target = TargetStrategy(state, DecisionConfig(), SimpleNamespace())
        target._current_target = {"node_id": 7}
        target._current_goal = ((1.0, 2.0), None, {})
        self.assertTrue(target._commit("chair", 1.0, 2.0, 0.2))
        self.assertTrue(state.task.complete)
        self.assertIsNone(state.target.current_target_node_id)

    def test_target_commit_radius_matches_objectnav_success_distance(self):
        target = TargetStrategy(SharedState(), DecisionConfig(), SimpleNamespace())
        target._current_target = {
            "label": "chair",
            "surface_x": 0.25,
            "surface_y": 0.0,
        }
        self.assertTrue(target.commit_if_close(np.array([0.0, 0.0])))

        target = TargetStrategy(SharedState(), DecisionConfig(), SimpleNamespace())
        target._current_target = {
            "label": "chair",
            "surface_x": 0.301,
            "surface_y": 0.0,
        }
        self.assertFalse(target.commit_if_close(np.array([0.0, 0.0])))

    def test_policy_stop_commits_within_point_seven_five_metres(self):
        state = SharedState()
        state.sensor.latest_odom = (0.0, 0.0, 0.0)
        target = TargetStrategy(state, DecisionConfig(), SimpleNamespace())
        target._current_target = {
            "label": "chair",
            "surface_x": 0.65,
            "surface_y": 0.0,
        }
        target._current_goal = ((0.65, 0.0), None, {})

        self.assertFalse(target.commit_if_close(np.array([0.0, 0.0])))
        target.on_failed("policy_stop")
        self.assertIsNone(target.current_target)
        self.assertTrue(state.task.complete)

    def test_temporary_target_finishes_within_point_two_metres(self):
        state = SharedState()
        temporary = {
            "label": "chair",
            "canonical": "chair",
            "surface_x": 0.2,
            "surface_y": 0.0,
            "x": 0.2,
            "y": 0.0,
            "temporary_target": True,
        }
        state.target.target_goals = [temporary]
        target = TargetStrategy(state, DecisionConfig(), SimpleNamespace())
        target._current_target = dict(temporary)

        self.assertFalse(target.finish_temporary_if_close(np.array([-0.001, 0.0])))
        self.assertTrue(target.finish_temporary_if_close(np.array([0.0, 0.0])))
        self.assertIsNone(target.current_target)
        self.assertEqual(state.target.target_goals, [])
        self.assertFalse(state.task.complete)

    def test_final_confirmation_uses_current_not_observation_distance(self):
        state = SharedState()
        state.sensor.latest_odom = (0.0, 0.0, 0.0)
        verifier = TargetVerifier(state)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[180:300, 250:350] = 1
        verifier._mobile_sam = SimpleNamespace(segment_bbox=lambda *_args: mask)
        cam = SimpleNamespace(
            rgb_image=np.zeros((480, 640, 3), dtype=np.uint8),
            T_world_camera=np.eye(4),
        )
        det = SimpleNamespace(bbox=(250.0, 180.0, 350.0, 300.0))
        candidate = TargetCandidate(
            signature=(0, "hm3d", "chair", (250, 180, 350, 300)),
            camera_name="hm3d",
            label="chair",
            canonical="chair",
            bbox=det.bbox,
            confidence=0.9,
            timestamp=1.0,
            boxed_rgb=cam.rgb_image,
            camera_result=cam,
            detection_entity=det,
            object_detection=SimpleNamespace(robot_distance=3.01),
            node_id=7,
            x=2.9,
            y=0.0,
            surface_x=2.8,
            surface_y=0.0,
            distance=2.9,
            temporary_target=False,
        )
        projected = detection("chair", 0.9)
        projected.robot_distance = 3.01
        projected.centroid = np.array([2.9, 0.0, 0.5])
        projected.point_cloud = np.array([[2.8, 0.0, 0.5], [2.9, 0.0, 0.5]])

        with patch(
            "rtnav.modules.scenegraph.det3d.project_mask_detection_to_3d",
            return_value=projected,
        ):
            refined = verifier._refine_confirmed_candidate(candidate)

        self.assertFalse(refined.temporary_target)
        self.assertIs(refined.object_detection, projected)

    def test_over_five_metre_target_keeps_projected_goal(self):
        state = SharedState()
        verifier = TargetVerifier(state)
        cam = SimpleNamespace(rgb_image=np.zeros((480, 640, 3), dtype=np.uint8))
        det = SimpleNamespace(bbox=(250.0, 200.0, 350.0, 300.0))
        candidate = TargetCandidate(
            signature=(0, "hm3d", "chair", (250, 200, 350, 300)),
            camera_name="hm3d",
            label="chair",
            canonical="chair",
            bbox=det.bbox,
            confidence=0.9,
            timestamp=1.0,
            boxed_rgb=cam.rgb_image,
            camera_result=cam,
            detection_entity=det,
            object_detection=SimpleNamespace(robot_distance=4.9),
            node_id=7,
            x=5.01,
            y=0.2,
            surface_x=3.8,
            surface_y=0.2,
            distance=5.01,
            temporary_target=False,
        )

        refined = verifier._refine_confirmed_candidate(candidate)

        self.assertTrue(refined.temporary_target)
        self.assertIsNone(refined.object_detection)
        self.assertIsNone(refined.node_id)
        np.testing.assert_allclose(
            (refined.x, refined.y, refined.surface_x, refined.surface_y),
            (5.01, 0.2, 3.8, 0.2),
        )

    def test_unknown_depth_temporary_target_is_clamped_to_four_point_nine_five_metres(self):
        rotation = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        transform = np.eye(4)
        transform[:3, :3] = rotation
        cam = SimpleNamespace(
            rgb_image=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_image=np.zeros((4, 4), dtype=np.float32),
            intrinsics_rgb=np.eye(3),
            T_world_camera=transform,
        )

        goal = TargetVerifier._temporary_boundary_xy(cam, (0.0, 0.0, 0.0, 0.0))

        np.testing.assert_allclose(goal, (4.95, 0.0))

    def test_final_confirmation_can_interrupt_at_five_metres(self):
        candidate = TargetCandidate(
            signature=(0, "hm3d", "chair", (250, 200, 350, 300)),
            camera_name="hm3d",
            label="chair",
            canonical="chair",
            bbox=(250.0, 200.0, 350.0, 300.0),
            confidence=0.9,
            timestamp=1.0,
            boxed_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            camera_result=None,
            detection_entity=None,
            object_detection=SimpleNamespace(),
            node_id=7,
            x=5.0,
            y=0.0,
            surface_x=5.0,
            surface_y=0.0,
            distance=5.0,
            temporary_target=False,
        )
        result = SimpleNamespace(candidate=candidate)
        finished = []
        decision = object.__new__(DecisionThread)
        decision._target_vlm_running = False
        decision._temporary_close_vlm_signatures = set()
        decision.target = SimpleNamespace(
            current_target={"temporary_target": True},
            finish_temporary_target=lambda **kwargs: finished.append(kwargs),
        )
        decision.target_verifier = SimpleNamespace(
            collect_candidates=lambda *_args, **_kwargs: [candidate],
            target_name=lambda name: name,
            verify_batch=lambda *_args: [result],
            apply_results=lambda _results: [{"temporary_target": False}],
        )
        decision.shared_state = SharedState()
        decision.shared_state.task_ready.set()
        decision.shared_state.inference = SimpleNamespace(
            pause_for_vllm=lambda: None,
            resume_after_vllm=lambda: None,
        )
        decision._log_verification = lambda *_args, **_kwargs: None

        snap = SimpleNamespace(reset_counter=decision.shared_state.system.episode_index)
        self.assertTrue(decision._maybe_verify_close_target(snap, "chair"))
        self.assertEqual(finished, [{"reached": False, "reason": "replaced_by_confirmed_target"}])

    def test_three_low_vlm_scores_permanently_blacklist_node(self):
        state = SharedState()
        verifier = TargetVerifier(state)

        def result(node_id, distance, yes_norm=0.0):
            candidate = TargetCandidate(
                signature=(0, "hm3d", "toilet", (1, 2, 3, 4)),
                camera_name="hm3d",
                label="toilet",
                canonical="toilet",
                bbox=(1.0, 2.0, 3.0, 4.0),
                confidence=0.9,
                timestamp=1.0,
                boxed_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                camera_result=None,
                detection_entity=None,
                object_detection=SimpleNamespace(),
                node_id=node_id,
                x=1.0,
                y=0.0,
                surface_x=0.9,
                surface_y=0.0,
                distance=distance,
                temporary_target=False,
            )
            return VerificationResult(candidate, yes_norm, 1.0 - yes_norm, yes_norm, 0.1)

        low_score_rejection = result(7, 10.0, yes_norm=0.5)
        verifier.apply_results([low_score_rejection])
        verifier.apply_results([low_score_rejection])
        self.assertNotIn(7, state.target.target_node_blacklist_ids)
        verifier.apply_results([low_score_rejection])
        self.assertIn(7, state.target.target_node_blacklist_ids)

        higher_score_rejection = result(8, 2.0, yes_norm=0.51)
        for _ in range(3):
            verifier.apply_results([higher_score_rejection])
        self.assertNotIn(8, state.target.target_node_blacklist_ids)

        with patch.object(verifier, "_refine_confirmed_candidate", side_effect=lambda item: item):
            accepted = verifier.apply_results([result(7, 2.0, yes_norm=0.99)])
        self.assertEqual(accepted, [])
        self.assertIn(7, state.target.target_node_blacklist_ids)

        state.task.goal_category = "toilet"
        state.task.synonym_to_canonical = {"toilet": "toilet"}
        state.sensor.latest_odom = (0.0, 0.0, 0.0)
        state.scenegraph.scene_graph = SimpleNamespace(
            nodes={7: SimpleNamespace(node_id=7, centroid=np.array([1.0, 0.0, 0.5]))}
        )
        detection = SimpleNamespace(
            name="toilet",
            bbox=(1.0, 2.0, 3.0, 4.0),
            confidence=0.9,
            top_k_label_probs=[],
        )
        camera = SimpleNamespace(
            detections=[detection],
            rgb_image=np.zeros((4, 4, 3), dtype=np.uint8),
        )
        state.perception.detection_result = SimpleNamespace(
            episode_index=state.system.episode_index,
            timestamp=2.0,
            camera_results={"hm3d": camera},
        )
        projected = SimpleNamespace(
            centroid=np.array([1.0, 0.0, 0.5]),
            point_cloud=np.array([[0.9, 0.0, 0.5], [1.0, 0.0, 0.5]]),
            robot_distance=1.0,
        )
        with patch.object(verifier, "_project_detection", return_value=projected):
            self.assertEqual(verifier.collect_candidates("toilet", include_queried=True), [])

    def test_vlfm_offset_mask_is_provisional_but_bottom_crop_is_not(self):
        right = np.zeros((480, 640), dtype=np.uint8)
        right[280:480, 500:640] = 1
        self.assertTrue(_mask_too_offset(right))

        bottom = np.zeros((480, 640), dtype=np.uint8)
        bottom[380:480, 240:400] = 1
        self.assertFalse(_mask_too_offset(bottom))

        right_not_at_edge = np.zeros((480, 640), dtype=np.uint8)
        right_not_at_edge[280:450, 500:580] = 1
        self.assertFalse(_mask_too_offset(right_not_at_edge))

    def test_offset_confirmation_uses_mobilesam_geometry_as_temporary(self):
        state = SharedState()
        state.sensor.latest_odom = (0.0, 0.0, 0.0)
        verifier = TargetVerifier(state)

        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[300:480, 500:640] = 1
        calls = []
        verifier._mobile_sam = SimpleNamespace(
            segment_bbox=lambda *_: calls.append(True) or mask
        )
        cam = SimpleNamespace(rgb_image=np.zeros((480, 640, 3), dtype=np.uint8))
        cam.T_world_camera = np.eye(4)
        det = SimpleNamespace(bbox=(497.75, 312.0, 640.0, 480.0))
        preliminary = SimpleNamespace(robot_distance=2.0)
        candidate = TargetCandidate(
            signature=(0, "hm3d", "toilet", (498, 312, 640, 480)),
            camera_name="hm3d",
            label="toilet",
            canonical="toilet",
            bbox=det.bbox,
            confidence=0.9,
            timestamp=1.0,
            boxed_rgb=cam.rgb_image,
            camera_result=cam,
            detection_entity=det,
            object_detection=preliminary,
            node_id=7,
            x=1.8,
            y=0.4,
            surface_x=1.6,
            surface_y=0.4,
            distance=2.0,
            temporary_target=False,
        )
        projected = detection("toilet", 0.9)
        projected.robot_distance = 2.0
        projected.centroid = np.array([2.0, 0.5, 0.5])
        projected.point_cloud = np.array([[1.4, 0.5, 0.5], [2.0, 0.5, 0.5]])

        with patch(
            "rtnav.modules.scenegraph.det3d.project_mask_detection_to_3d",
            return_value=projected,
        ):
            refined = verifier._refine_confirmed_candidate(candidate)

        self.assertTrue(calls)
        self.assertTrue(refined.temporary_target)
        self.assertIs(refined.sam_mask, mask)
        self.assertIsNone(refined.object_detection)
        self.assertIsNone(refined.node_id)
        np.testing.assert_allclose((refined.x, refined.y), (2.0, 0.5))
        np.testing.assert_allclose((refined.surface_x, refined.surface_y), (1.4, 0.5))

    def test_mask_projection_failure_keeps_detector_geometry(self):
        state = SharedState()
        verifier = TargetVerifier(state)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[200:300, 250:350] = 1
        verifier._mobile_sam = SimpleNamespace(segment_bbox=lambda *_: mask)
        cam = SimpleNamespace(rgb_image=np.zeros((480, 640, 3), dtype=np.uint8))
        det = SimpleNamespace(bbox=(250.0, 200.0, 350.0, 300.0))
        candidate = TargetCandidate(
            signature=(0, "hm3d", "toilet", (250, 200, 350, 300)),
            camera_name="hm3d",
            label="toilet",
            canonical="toilet",
            bbox=det.bbox,
            confidence=0.9,
            timestamp=1.0,
            boxed_rgb=cam.rgb_image,
            camera_result=cam,
            detection_entity=det,
            object_detection=SimpleNamespace(robot_distance=2.0),
            node_id=7,
            x=1.8,
            y=0.4,
            surface_x=1.6,
            surface_y=0.4,
            distance=2.0,
            temporary_target=False,
        )

        with patch(
            "rtnav.modules.scenegraph.det3d.project_mask_detection_to_3d",
            return_value=None,
        ):
            refined = verifier._refine_confirmed_candidate(candidate)

        self.assertTrue(refined.temporary_target)
        self.assertIsNone(refined.object_detection)
        self.assertIsNone(refined.node_id)
        np.testing.assert_allclose(
            (refined.x, refined.y, refined.surface_x, refined.surface_y),
            (1.8, 0.4, 1.6, 0.4),
        )

    def test_confirmed_geometry_is_preferred_over_temporary_geometry(self):
        state = SharedState()
        state.target.target_goals = [
            {
                "x": 1.0,
                "y": 0.0,
                "surface_x": 1.0,
                "surface_y": 0.0,
                "label": "toilet",
                "canonical": "toilet",
                "vlm_confirmed": True,
                "vlm_yes_norm": 0.99,
                "temporary_target": True,
                "node_id": None,
            },
            {
                "x": 2.0,
                "y": 0.0,
                "surface_x": 2.0,
                "surface_y": 0.0,
                "label": "toilet",
                "canonical": "toilet",
                "vlm_confirmed": True,
                "vlm_yes_norm": 0.91,
                "temporary_target": False,
                "node_id": None,
            },
        ]

        class Map:
            @staticmethod
            def xy_to_px(points):
                return np.asarray(points)

        target = TargetStrategy(state, DecisionConfig(), SimpleNamespace())
        goal = target.select_goal("toilet", np.array([0.0, 0.0]), Map())
        np.testing.assert_allclose(goal[0], (2.0, 0.0))
        self.assertFalse(target.current_target["temporary_target"])

    def test_frontier_snaps_to_a_navigable_approach_cell(self):
        class Map:
            size = 11
            ppm = 1
            navigable = np.zeros((11, 11), dtype=np.uint8)

            @staticmethod
            def xy_to_px(xy):
                return np.asarray(xy, dtype=int)

            @staticmethod
            def px_to_xy(px):
                return np.asarray(px, dtype=float)

            def is_navigable_px(self, x, y):
                return 0 <= x < self.size and 0 <= y < self.size and bool(self.navigable[y, x])

        obstacle_map = Map()
        obstacle_map.navigable[5, 6] = 1
        goal = find_navigable_frontier_goal(obstacle_map, np.array([5, 5]))
        self.assertEqual(tuple(goal[1]), (6, 5))

        obstacle_map.navigable.fill(0)
        self.assertIsNone(find_navigable_frontier_goal(obstacle_map, np.array([5, 5])))
        self.assertIsNone(find_navigable_frontier_goal(obstacle_map, np.array([-50, 5])))

    def test_frontier_reached_and_two_strike_failure_radii(self):
        self.assertAlmostEqual(
            FrontierImageVLM.MAX_CAMERA_ANGLE_RAD,
            math.radians(30),
        )
        strategy = FrontierStrategy(SharedState(), FrontierConfig(), DecisionConfig())
        strategy.on_goal_failed((0.0, 0.0), "no_progress")
        self.assertFalse(strategy._is_banned((0.5, 0.0)))
        strategy.on_goal_failed((0.5, 0.0), "path_unreachable")
        self.assertTrue(strategy._is_banned((0.5, 0.0)))
        self.assertFalse(strategy._is_banned((1.6, 0.0)))

        strategy._reached_frontiers = [(0.0, 0.0)]
        self.assertTrue(strategy._was_reached((0.29, 0.0)))
        self.assertFalse(strategy._was_reached((0.31, 0.0)))

    def test_frontier_selection_prefers_primary_then_falls_back_to_secondary(self):
        selector = FrontierImageVLM(SharedState())
        self.assertEqual(selector.PRIMARY_IMAGE_RADIUS_M, 4.0)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        selector._current_camera_xy = (0.0, 0.0)
        selector._entries = [
            FrontierImageEntry(1, 0, (0.0, 0.0), image, image_camera_xy=(-4.1, 0.0)),
            FrontierImageEntry(2, 1, (1.0, 0.0), image, image_camera_xy=(0.5, 0.0)),
            FrontierImageEntry(3, 2, (2.0, 0.0), image, image_camera_xy=(-4.5, 0.0)),
        ]

        self.assertEqual(
            [entry.frontier_idx for entry in selector._selection_entries()],
            [1],
        )
        self.assertEqual(selector.selection_scope(), "primary")
        self.assertEqual(selector.ready_to_query(), (True, ""))
        self.assertEqual(selector.single_candidate_frontier_idx(), 1)
        bypass_events = []
        selector.shared_state.system.vlm_logger = SimpleNamespace(
            record_frontier=lambda _input, _selected, event, *_: bypass_events.append(event)
        )

        class Map:
            @staticmethod
            def px_to_xy(points):
                return np.asarray(points, dtype=float)

            @staticmethod
            def find_navigable_frontier_goal(centroid):
                centroid = np.asarray(centroid)
                return centroid.astype(float), centroid.astype(int)

        strategy = FrontierStrategy(SharedState(), FrontierConfig(), DecisionConfig())
        strategy.image_vlm = selector
        frontier_output = SimpleNamespace(
            frontier_clusters=[np.array([[i, 0]]) for i in range(3)],
            frontier_centroids=np.array([[0, 0], [1, 0], [2, 0]]),
            frontier_unexplored_directions=[],
            timestamp=1.0,
        )
        context = {
            "frontier_output": frontier_output,
            "robot_pose": (0.0, 0.0, 0.0),
            "obstacle_map": Map(),
        }
        with patch.object(strategy, "_sync_images"), patch.object(selector, "choose") as choose:
            goal = strategy.select_goal(context)
        choose.assert_not_called()
        np.testing.assert_allclose(goal[0], (1.0, 0.0))
        self.assertTrue(bypass_events[0]["bypassed_vlm"])
        self.assertEqual(
            bypass_events[0]["excluded_frontier_indices"],
            [0, 2],
        )
        self.assertEqual(
            [entry["tier"] for entry in bypass_events[0]["excluded_frontiers"]],
            ["secondary", "secondary"],
        )

        selector._current_camera_xy = (10.0, 0.0)
        self.assertEqual(
            [entry.frontier_idx for entry in selector._selection_entries()],
            [0, 1, 2],
        )
        self.assertEqual(selector.selection_scope(), "secondary")
        self.assertIsNone(selector.single_candidate_frontier_idx())

        selector._entries = [FrontierImageEntry(4, 7, (3.0, 0.0), None)]
        self.assertEqual(selector.single_candidate_frontier_idx(), 7)

    def test_temporary_target_is_pursued_then_biases_frontier_along_extended_ray(self):
        state = SharedState()

        class Map:
            @staticmethod
            def xy_to_px(points):
                return np.asarray(points)

            @staticmethod
            def px_to_xy(points):
                return np.asarray(points, dtype=float)

            @staticmethod
            def find_navigable_frontier_goal(centroid):
                centroid = np.asarray(centroid)
                return centroid.astype(float), centroid.astype(int)

        state.target.target_goals = [
            {
                "x": 2.0,
                "y": 0.0,
                "surface_x": 2.0,
                "surface_y": 0.0,
                "label": "chair",
                "canonical": "chair",
                "vlm_confirmed": True,
                "temporary_target": True,
                "vlm_yes_norm": 0.8,
            }
        ]
        decision = object.__new__(DecisionThread)
        decision.shared_state = state
        self.assertTrue(decision._has_pending_target("chair"))

        target = TargetStrategy(state, DecisionConfig(), SimpleNamespace())
        goal = target.select_goal("chair", np.array([0.0, 0.0]), Map())
        np.testing.assert_allclose(goal[0], (2.0, 0.0))
        self.assertTrue(target.current_target["temporary_target"])
        target.finish_temporary_target(reached=True)
        self.assertFalse(decision._has_pending_target("chair"))
        self.assertEqual(state.target.last_temporary_target_ray, ((0.0, 0.0), (2.0, 0.0)))

        strategy = FrontierStrategy(state, FrontierConfig(), DecisionConfig())
        frontier_output = SimpleNamespace(
            frontier_clusters=[np.array([[-1, 0]]), np.array([[6, 1]])],
            frontier_centroids=np.array([[-1, 0], [6, 1]]),
            frontier_unexplored_directions=[],
            timestamp=1.0,
        )
        context = {
            "frontier_output": frontier_output,
            "robot_pose": (2.0, 0.0, 0.0),
            "obstacle_map": Map(),
        }
        goal = strategy.select_goal(context, allow_image_vlm=False)
        np.testing.assert_allclose(goal[0], (6.0, 1.0))

        state.target.reset()
        self.assertIsNone(state.target.last_temporary_target_ray)

    def test_frontier_image_tier_uses_saved_camera_position(self):
        selector = FrontierImageVLM(SharedState())

        def camera(timestamp, angle_deg, xy=(0.0, 0.0)):
            angle = math.radians(angle_deg)
            transform = np.eye(4)
            transform[:2, 0] = (math.cos(angle), -math.sin(angle))
            transform[:2, 2] = (math.sin(angle), math.cos(angle))
            transform[:2, 3] = xy
            return SimpleNamespace(
                rgb=np.zeros((40, 60, 3), dtype=np.uint8),
                depth=None,
                intrinsics=np.array([[30.0, 0.0, 30.0], [0.0, 30.0, 20.0], [0, 0, 1]]),
                T_world_cam=transform,
                timestamp=timestamp,
            )

        cameras = {
            "older_aligned": camera(1.0, 0.0),
            "newer": camera(2.0, 20.0),
            "newest_ineligible": camera(100.0, 40.0, (4.1, 0.0)),
        }
        entries = selector.sync([(0.0, 2.0)], cameras=cameras)
        self.assertEqual(entries[0].camera_name, "older_aligned")
        self.assertEqual(entries[0].image_camera_xy, (0.0, 0.0))
        self.assertAlmostEqual(selector._image_distance_m(entries[0]), 4.1)
        self.assertEqual(selector.selection_scope(), "secondary")

        cameras["nearby_less_aligned"] = camera(101.0, -40.0, (4.1, 0.0))
        entries = selector.sync([(0.0, 2.0)], cameras=cameras)
        self.assertEqual(entries[0].camera_name, "nearby_less_aligned")
        self.assertEqual(entries[0].image_camera_xy, (4.1, 0.0))
        self.assertAlmostEqual(selector._image_distance_m(entries[0]), 0.0)
        self.assertEqual(selector.selection_scope(), "primary")

    def test_frontier_image_prefers_centering_when_viewpoints_match(self):
        selector = FrontierImageVLM(SharedState())

        def camera(timestamp, angle_deg):
            angle = math.radians(angle_deg)
            transform = np.eye(4)
            transform[:2, 0] = (math.cos(angle), -math.sin(angle))
            transform[:2, 2] = (math.sin(angle), math.cos(angle))
            return SimpleNamespace(
                rgb=np.zeros((40, 60, 3), dtype=np.uint8),
                intrinsics=np.array([[30.0, 0.0, 30.0], [0.0, 30.0, 20.0], [0, 0, 1]]),
                T_world_cam=transform,
                timestamp=timestamp,
            )

        cameras = {
            "older_centered": camera(1.0, 0.0),
            "newest_visible": camera(2.0, 25.0),
        }
        entries = selector.sync([(0.0, 2.0)], cameras=cameras)

        self.assertEqual(entries[0].camera_name, "older_centered")
        self.assertEqual(entries[0].image_timestamp, 1.0)

    def test_vlfm_marks_points_below_starting_floor_as_obstacles(self):
        points = np.array(
            [
                [0.0, 0.0, -0.201],
                [1.0, 0.0, -0.2],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.61],
                [4.0, 0.0, 0.88],
                [5.0, 0.0, 0.881],
            ]
        )
        obstacles = filter_obstacle_points(points, 0.61, 0.88, -0.2)
        np.testing.assert_array_equal(obstacles[:, 0], [0.0, 3.0, 4.0])

    def test_vlfm_frontier_jit_is_warmed_for_runtime_array_layouts(self):
        from frontier_exploration.frontier_detection import (
            contour_to_frontiers,
            get_frontier_midpoint,
        )

        _warm_frontier_jit()

        self.assertTrue({"C", "F"} <= {sig[0].layout for sig in contour_to_frontiers.signatures})
        self.assertTrue({"A", "C"} <= {sig[0].layout for sig in get_frontier_midpoint.signatures})

    def test_frontier_overlay_ends_at_projected_floor_and_points_toward_unseen_side(self):
        obstacle_map = HabitatObstacleMap(size=21, pixels_per_meter=10)
        navigable = np.ones((21, 21), dtype=np.uint8)
        explored = np.zeros((21, 21), dtype=np.uint8)
        explored[:, :11] = 1
        unseen_direction = obstacle_map._unexplored_direction_xy(
            np.array([10, 10]), navigable, explored
        )
        np.testing.assert_allclose(unseen_direction, (0.0, -1.0), atol=1e-6)

        transform = np.eye(4)
        transform[:2, 0] = unseen_direction
        screen_direction = FrontierImageVLM._frontier_screen_direction(unseen_direction, transform)
        np.testing.assert_allclose(screen_direction, (1.0, 0.0), atol=1e-6)
        transform[:2, 0] = -unseen_direction
        np.testing.assert_allclose(
            FrontierImageVLM._frontier_screen_direction(unseen_direction, transform),
            (-1.0, 0.0),
            atol=1e-6,
        )

        camera = SimpleNamespace(
            rgb=np.zeros((100, 100, 3), dtype=np.uint8),
            intrinsics=np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        )
        camera_transform = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        marker = FrontierImageVLM(SharedState())._frontier_image_point(
            (4.0, 0.0), camera, camera_transform
        )
        np.testing.assert_allclose(marker, (50.0, 75.0), atol=1e-6)

        rendered = FrontierImageVLM._annotate_frontier_overlay(
            np.zeros((100, 100, 3), dtype=np.uint8),
            (50.0, 70.0),
            screen_direction,
        )
        red = (rendered[:, :, 0] > 200) & (rendered[:, :, 1] < 50)
        self.assertGreater(np.count_nonzero(red[:71, 50]), 60)
        self.assertFalse(np.any(red[75:, 50]))
        self.assertTrue(np.any(red[25:50, 38:48]))
        self.assertFalse(np.any(red[:, 20]))
        self.assertFalse(np.any(red[:, 80]))

        for marker_x, direction, arrow_slice in (
            (0.0, (-1.0, 0.0), slice(25, 35)),
            (99.0, (1.0, 0.0), slice(65, 75)),
        ):
            edge_rendered = FrontierImageVLM._annotate_frontier_overlay(
                np.zeros((100, 100, 3), dtype=np.uint8),
                (marker_x, 70.0),
                direction,
            )
            edge_red = (edge_rendered[:, :, 0] > 200) & (edge_rendered[:, :, 1] < 50)
            self.assertTrue(np.any(edge_red[25:45, arrow_slice]))

        prompt = FrontierImageVLM._build_prompt("toilet", [])
        self.assertIn("arrow points from explored space toward unexplored space", prompt)
        self.assertNotIn("occluded", prompt)

    def test_active_frontier_tracks_live_centroid_and_cancels_when_lost(self):
        class Map:
            _combined_blocked = np.zeros((10, 10), dtype=bool)

            @staticmethod
            def xy_to_px(points):
                return np.asarray(points, dtype=int)

            @staticmethod
            def px_to_xy(points):
                return np.asarray(points, dtype=float)

            @staticmethod
            def find_navigable_frontier_goal(centroid):
                centroid = np.asarray(centroid, dtype=float)
                return centroid + (0.2, 0.0), centroid.astype(int)

        state = SharedState()
        strategy = FrontierStrategy(state, FrontierConfig(), DecisionConfig())
        self.assertEqual(strategy._sticky_match_m, 0.5)
        strategy._last_frontier_xy = np.array([0.0, 0.0], dtype=np.float32)
        obstacle_map = Map()

        def refresh(centroid):
            output = SimpleNamespace(
                frontier_clusters=[np.array([[0, 0]])],
                frontier_centroids=np.array([centroid]),
            )
            return strategy.refresh_goal(
                {
                    "frontier_output": output,
                    "obstacle_map": obstacle_map,
                    "robot_pose": (0.0, 0.0, 0.0),
                }
            )

        goal, reason = refresh((0.2, 0.0))
        self.assertEqual(reason, "")
        np.testing.assert_allclose(goal[0], (0.4, 0.0))

        goal, reason = refresh((0.4, 0.0))
        self.assertEqual(reason, "")
        np.testing.assert_allclose(goal[0], (0.6, 0.0))

        goal, reason = refresh((1.0, 0.0))
        self.assertIsNone(goal)
        self.assertIn("nearest live frontier", reason)

        failed = []
        decision = object.__new__(DecisionThread)
        decision._pending_goal = {"xy": (2.4, 0.0)}
        decision.frontier = SimpleNamespace(
            refresh_goal=lambda _context: (None, "no match"),
            on_goal_failed=lambda xy, reason: failed.append((xy, reason)),
        )
        update = NavUpdate()
        changed = decision._refresh_active_frontier(
            SimpleNamespace(context=lambda: {}, obstacle_map=obstacle_map), update
        )
        self.assertTrue(changed)
        self.assertEqual(failed, [((2.4, 0.0), "frontier_vanished")])
        self.assertIsNone(decision._pending_goal)
        self.assertTrue(update.dirty)

        decision = object.__new__(DecisionThread)
        decision.shared_state = state
        decision._pending_goal = {"xy": (2.4, 0.0)}
        decision._last_nav_id = 0
        decision._no_goal_start = None
        decision.frontier = SimpleNamespace(
            refresh_goal=lambda _context: (((2.8, 0.0), (28, 0), {}), "")
        )
        update = NavUpdate()
        changed = decision._refresh_active_frontier(
            SimpleNamespace(context=lambda: {}, obstacle_map=obstacle_map), update
        )
        self.assertTrue(changed)
        self.assertEqual(update.goal_xy, (2.8, 0.0))
        self.assertEqual(update.nav_id, 1)


if __name__ == "__main__":
    unittest.main()
