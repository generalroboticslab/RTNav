#!/usr/bin/env python3
"""ROS input bridge: subscribe to /obs, decode sensors, write HabitatObservation
to shared_state.sensor.habitat_obs, and forward DONE_SENTINEL messages to the
eval node. Geometry is handled downstream by PerceptionThread → MappingThread."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from ros_helpers import DONE_SENTINEL

from rtnav.core.data_types import HabitatObservation

QOS_OBS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class ObservationSubscriberNode(Node):
    """ROS2 node: decode /obs into shared_state.sensor.habitat_obs.
    (robot_api is fed its observations separately by BaseAgentNode.)

    `on_episode_done` is called for every DONE_SENTINEL message so the eval
    node can detect episode end without its own /obs subscription."""

    def __init__(
        self,
        shared_state,
        obs_topic: str = "/obs",
        verbose: bool = False,
        on_episode_done: Callable | None = None,
    ):
        super().__init__("observation_subscriber")
        self.shared_state = shared_state
        self.verbose = verbose
        self._on_episode_done = on_episode_done
        self._paused = False

        from ros_msgs.msg import Observation as ObservationMsg

        self.create_subscription(ObservationMsg, obs_topic, self._on_obs, QOS_OBS)
        self.get_logger().info(f"ObservationSubscriberNode listening on {obs_topic}")

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def _on_obs(self, msg) -> None:
        if int(msg.step_id) == DONE_SENTINEL:
            if self._on_episode_done is not None:
                self._on_episode_done(msg)
            return
        if self._paused:
            return
        rgb = self._decode_rgb(msg.rgb)
        depth = self._decode_depth(msg.depth)
        if rgb is None or depth is None:
            if self.verbose:
                self.get_logger().warn("Malformed RGB-D observation; skipping")
            return
        obs = HabitatObservation(
            step_id=int(msg.step_id),
            rgb=rgb,
            depth=depth,
            gps=(float(msg.gps[0]), float(msg.gps[1])),
            compass=float(msg.compass),
            timestamp=time.time(),
        )
        with self.shared_state.lock:
            self.shared_state.sensor.habitat_obs = obs

    @staticmethod
    def _decode_rgb(msg) -> np.ndarray | None:
        """sensor_msgs/Image (rgb8) → HxWx3 uint8, or None on failure."""
        try:
            if msg.encoding == "rgb8":
                return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            if msg.encoding == "bgr8":
                import cv2

                bgr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            pass  # malformed message → skip frame
        return None

    @staticmethod
    def _decode_depth(msg) -> np.ndarray | None:
        """sensor_msgs/Image (32FC1) → HxW float32 meters, or None on failure."""
        try:
            if msg.encoding == "32FC1":
                return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
            if msg.encoding == "16UC1":
                return (
                    np.frombuffer(msg.data, np.uint16)
                    .reshape(msg.height, msg.width)
                    .astype(np.float32)
                    / 1000.0
                )
        except Exception:
            pass  # malformed message → skip frame
        return None
