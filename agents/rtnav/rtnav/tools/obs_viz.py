"""Web visualizers for the rtnav obs / frontier / topdown / heightmap maps,
launched from agent_node.main() after the agent subsystem is built."""

import numpy as np


def _start_vlfm_obs_viz(agent, viz_shutdown, viz_threads):
    """VLFM HabitatObstacleMap MJPEG viz on :8766 (live from VLFMFrontierDetector)."""
    from rtnav.tools.visualization.habitat_obs_map_web_viz_thread import (
        HabitatObsMapWebVizThread,
    )

    def _safe(fn, default):
        # Swallow transient shared-state races rather than killing the viz.
        try:
            return fn()
        except Exception:
            return default

    def _selected_goal_xy():
        # The chosen frontier picked from frontier candidates.
        def _f():
            ss = agent.shared_state
            if ss is None:
                return None
            with ss.lock:
                g = getattr(ss.frontier, "chosen_frontier_xy", None)
                # Fall back to nav.goal_xy if no explicit frontier choice tracked.
                if g is None and getattr(ss.nav, "goal_source", None) == "frontier":
                    g = getattr(ss.nav, "goal_xy", None)
            return tuple(g) if g is not None else None

        return _safe(_f, None)

    def _target_nav_xy():
        # Actual object target during exploitation: track the live SG node
        # centroid (by current_target_node_id), not the stale nav.goal_xy.
        def _f():
            ss = agent.shared_state
            if ss is None:
                return None
            with ss.lock:
                source = (getattr(ss.nav, "goal_source", None) or "").lower()
                if "target" not in source:
                    return None
                node_id = getattr(ss.target, "current_target_node_id", None)
                sg = getattr(ss.scenegraph, "scene_graph", None)
            if node_id is None or sg is None or not getattr(sg, "nodes", None):
                return None
            try:
                target_id = int(node_id)
            except (TypeError, ValueError):
                return None
            for nd in sg.nodes:
                if int(getattr(nd, "node_id", -1)) != target_id:
                    continue
                c = getattr(nd, "centroid", None)
                if c is None:
                    return None
                return (float(c[0]), float(c[1]))
            return None

        return _safe(_f, None)

    def _goal_text():
        def _f():
            ss = agent.shared_state
            if ss is None:
                return None
            with ss.lock:
                t = (getattr(ss.task, "goal_category", "") or "").strip()
            return t or None

        return _safe(_f, None)

    def _found_objects(top_n=12):
        def _f():
            ss = agent.shared_state
            if ss is None:
                return []
            with ss.lock:
                target = (getattr(ss.task, "goal_category", "") or "").strip().lower()
                lookup = dict(getattr(ss.task, "synonym_to_canonical", {}) or {})
                sg = getattr(ss.scenegraph, "scene_graph", None)
                # nodes may be a list or a dict; accept both.
                raw_nodes = getattr(sg, "nodes", None) if sg else None
                if raw_nodes is None:
                    nodes = []
                elif hasattr(raw_nodes, "values"):
                    nodes = list(raw_nodes.values())
                else:
                    nodes = list(raw_nodes)
            # Show the primary target plus every synonym (lookup keys).
            seen, canonicals = set(), []
            ordered = [target] + [str(k or "").strip().lower() for k in lookup.keys()]
            for c in ordered:
                if c and c not in seen:
                    seen.add(c)
                    canonicals.append(c)
            counts: dict[str, int] = {}
            for n in nodes:
                if not getattr(n, "is_confirmed", False):
                    continue
                label = (
                    (getattr(n, "chosen_label", None) or getattr(n, "label", "") or "")
                    .strip()
                    .lower()
                )
                if label:
                    counts[label] = counts.get(label, 0) + int(getattr(n, "view_count", 1))
            return [(c, counts.get(c, 0)) for c in canonicals[:top_n]]

        return _safe(_f, [])

    def _agent_habitat_map():
        a = agent
        ft = getattr(a, "frontier", None)
        det = getattr(ft, "detector", None)
        return getattr(det, "_habitat_map", None)

    def _sg_confirmed_nodes():
        """Return [(x, y, label), ...] for every confirmed SG node whose label
        matches the current search target or one of its synonyms."""

        def _f():
            ss = agent.shared_state
            if ss is None:
                return []
            with ss.lock:
                target = (getattr(ss.task, "goal_category", "") or "").strip().lower()
                lookup = getattr(ss.task, "synonym_to_canonical", {}) or {}
                synonyms = set(str(k).lower() for k in lookup.keys())
                if target:
                    synonyms.add(target)
                sg = getattr(ss.scenegraph, "scene_graph", None)
                nodes = list(getattr(sg, "nodes", []) or [])
            result = []
            for n in nodes:
                if not getattr(n, "is_confirmed", False):
                    continue
                c = getattr(n, "centroid", None)
                if c is None:
                    continue
                lbl = str(getattr(n, "chosen_label", None) or getattr(n, "label", "") or "").strip()
                if synonyms and lbl.lower() not in synonyms:
                    continue
                try:
                    result.append((float(c[0]), float(c[1]), lbl))
                except Exception:
                    continue
            return result

        return _safe(_f, [])

    def _agent_xy():
        def _f():
            ss = agent.shared_state
            if ss is None:
                return None
            with ss.lock:
                det = getattr(getattr(ss, "perception", None), "detection_result", None)
            T = getattr(det, "T_world_base", None) if det is not None else None
            if T is None:
                return None
            return np.array([float(T[0, 3]), float(T[1, 3])], dtype=np.float32)

        return _safe(_f, None)

    viz = HabitatObsMapWebVizThread(
        habitat_map_getter=_agent_habitat_map,
        shutdown_event=viz_shutdown,
        agent_xy_getter=_agent_xy,
        selected_goal_xy_getter=_selected_goal_xy,
        target_nav_xy_getter=_target_nav_xy,
        goal_text_getter=_goal_text,
        found_objects_getter=_found_objects,
        sg_nodes_getter=_sg_confirmed_nodes,
        shared_state_getter=lambda: agent.shared_state,
        port=8766,
        view_size=800,
        fps=4,
    )
    viz.start()
    viz_threads.append(viz)
    print(f"[eval] VLFM HabitatObstacleMap viz — http://<host>:{viz.port}")


def start_persistent_viz(args, agent, viz_shutdown, viz_threads):
    if args.map_viz_web:
        _start_vlfm_obs_viz(agent, viz_shutdown, viz_threads)
