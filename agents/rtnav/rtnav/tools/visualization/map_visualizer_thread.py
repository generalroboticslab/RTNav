"""Small top-down renderer used by videos, decision traces, and web views."""

from __future__ import annotations

import cv2
import numpy as np

from rtnav.modules.decision.target_utils import label_matches_any
from rtnav.tools.visualization.obstacle_map_renderer import render_region, world_to_view_px


class MapVisualizerBase:
    """Render the map plus the state that directly drives the robot."""

    def __init__(self, shared_state):
        self.shared_state = shared_state

    def _get_state(self):
        acquired = self.shared_state.lock.acquire(timeout=0.05)
        if not acquired:
            return None
        try:
            ss = self.shared_state
            graph = ss.scenegraph.scene_graph
            raw_nodes = getattr(graph, "nodes", []) if graph is not None else []
            nodes = list(raw_nodes.values()) if hasattr(raw_nodes, "values") else list(raw_nodes)
            frontier_output = ss.frontier.frontier_output
            return {
                "map": ss.mapping.obstacle_map,
                "frontiers": frontier_output,
                "vlfm_map_snapshot": getattr(
                    frontier_output, "vlfm_map_snapshot", None
                ),
                "chosen_frontier_xy": ss.frontier.chosen_frontier_xy,
                "nav_goal_xy": ss.nav.goal_xy,
                "nav_goal_source": ss.nav.goal_source,
                "odom": ss.sensor.latest_odom,
                "nodes": [node for node in nodes if getattr(node, "is_confirmed", False)],
                "target": str(ss.task.goal_category or "").strip(),
                "synonyms": tuple(ss.task.synonym_to_canonical),
            }
        finally:
            self.shared_state.lock.release()

    @staticmethod
    def _world_to_view_px(state, xy):
        return world_to_view_px(state["map"], xy, state["view_bounds"])

    @staticmethod
    def _inside(image, pixel):
        x, y = pixel
        return 0 <= x < image.shape[1] and 0 <= y < image.shape[0]

    @staticmethod
    def _label(image, text, pixel, color):
        x, y = pixel
        cv2.putText(
            image,
            str(text),
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(text),
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    def _draw_frontiers(self, image, state):
        output = state["frontiers"]
        obstacle_map = state["map"]
        centroids = getattr(output, "frontier_centroids", None) if output else None
        if centroids is not None and len(centroids):
            try:
                points_xy = obstacle_map.px_to_xy(np.asarray(centroids))
            except Exception:
                points_xy = []
            for index, xy in enumerate(points_xy):
                pixel = self._world_to_view_px(state, xy)
                if not self._inside(image, pixel):
                    continue
                cv2.circle(image, pixel, 6, (0, 150, 220), 2, cv2.LINE_AA)
                self._label(image, f"F{index}", pixel, (0, 120, 190))

        chosen = state["chosen_frontier_xy"]
        if chosen is not None:
            pixel = self._world_to_view_px(state, chosen)
            if self._inside(image, pixel):
                cv2.circle(image, pixel, 12, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(image, pixel, 10, (0, 215, 255), 3, cv2.LINE_AA)

    def _draw_target_nodes(self, image, state):
        names = (state["target"], *state["synonyms"])
        for node in state["nodes"]:
            label = str(
                getattr(node, "chosen_label", None) or getattr(node, "label", "") or ""
            ).strip()
            if names and not label_matches_any(label, names):
                continue
            centroid = getattr(node, "centroid", None)
            if centroid is None:
                continue
            pixel = self._world_to_view_px(state, centroid[:2])
            if not self._inside(image, pixel):
                continue
            cv2.circle(image, pixel, 6, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(image, pixel, 7, (0, 0, 0), 1, cv2.LINE_AA)
            self._label(image, label, pixel, (140, 80, 0))

    def _draw_goal(self, image, state):
        goal = state["nav_goal_xy"]
        if goal is None:
            return
        pixel = self._world_to_view_px(state, goal)
        if not self._inside(image, pixel):
            return
        source = state["nav_goal_source"] or "goal"
        color = (255, 0, 255) if source == "target" else (0, 215, 255)
        cv2.drawMarker(image, pixel, color, cv2.MARKER_DIAMOND, 18, 3, cv2.LINE_AA)
        self._label(image, source.upper(), pixel, color)

    def _draw_robot(self, image, state):
        odom = state["odom"]
        if odom is None:
            return
        pixel = self._world_to_view_px(state, odom[:2])
        if not self._inside(image, pixel):
            return
        x, y = pixel
        tip_xy = (
            float(odom[0]) + 0.35 * np.cos(float(odom[2])),
            float(odom[1]) + 0.35 * np.sin(float(odom[2])),
        )
        tip = self._world_to_view_px(state, tip_xy)
        cv2.circle(image, (x, y), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.line(image, (x, y), tip, (0, 0, 180), 3, cv2.LINE_AA)

    def _render(self, state, view_radius_m):
        if state is None or state["map"] is None:
            return None
        image, state["view_bounds"] = render_region(
            state["map"],
            view_radius_m,
            state.get("vlfm_map_snapshot"),
        )
        image = np.asarray(image, dtype=np.uint8).copy()
        self._draw_frontiers(image, state)
        self._draw_target_nodes(image, state)
        self._draw_goal(image, state)
        self._draw_robot(image, state)
        return image

    def _render_frame(self, state):
        return self._render(state, view_radius_m=15.0)

    def render_decision_frame_clean(self):
        return self._render(self._get_state(), view_radius_m=9.0)


_HTML_PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>RTNav map</title>
<style>
body{margin:0;background:#161820;color:#ddd;font:14px system-ui;display:grid;place-items:center;min-height:100vh}
main{text-align:center}img{max-width:96vw;max-height:90vh;border:1px solid #444}p{margin:.5rem}
</style></head><body><main><img id="frame"><p id="status">Connecting...</p></main>
<script>
const image=document.getElementById('frame'),status=document.getElementById('status');
function connect(){const scheme=location.protocol==='https:'?'wss:':'ws:';const ws=new WebSocket(scheme+'//'+location.host+'/ws');ws.binaryType='arraybuffer';
ws.onopen=()=>status.textContent='Connected';ws.onmessage=e=>{const old=image.src;image.src=URL.createObjectURL(new Blob([e.data],{type:'image/jpeg'}));if(old.startsWith('blob:'))URL.revokeObjectURL(old)};
ws.onclose=()=>{status.textContent='Reconnecting...';setTimeout(connect,2000)};ws.onerror=()=>ws.close()};connect();
</script></body></html>"""
