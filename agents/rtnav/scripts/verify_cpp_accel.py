#!/usr/bin/env python3
"""Smoke-test cpp_accel imports in a launcher-created RTNav agent container.

  docker exec rtnav_agent_0 \
    python3 /opt/rt_ovn/agents/rtnav/scripts/verify_cpp_accel.py
"""

from __future__ import annotations

from rtnav.modules.perception.cpp_accel import fill_small_holes as _pe
from rtnav.modules.scenegraph.cpp_accel import project_detections_to_3d as _sg

print(f"scenegraph: {_sg is not None}\nperception: {_pe is not None}")
