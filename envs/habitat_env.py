import os
import time

import magnum as mn
import numpy as np
import quaternion
from typing import Any, Dict
from omegaconf import DictConfig
from habitat import Env, make_dataset
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector
from habitat_sim.agent import AgentState, SixDOFPose
from habitat_sim.physics import VelocityControl
from habitat_sim._ext.habitat_sim_bindings import RigidState


def episodic_gps_3d(agent_position, start_position, start_rotation) -> np.ndarray:
    """Match Habitat's 3-D EpisodicGPSSensor coordinate transform."""
    return quaternion_rotate_vector(
        quaternion_from_coeff(start_rotation).inverse(),
        np.asarray(agent_position, dtype=np.float32)
        - np.asarray(start_position, dtype=np.float32),
    ).astype(np.float32)


class HabitatEnvWrapper:
    """
    Provides reset/step for discrete actions and velocity_step for async mode.
    """
    STOP, FORWARD, LEFT, RIGHT, LOOK_UP, LOOK_DOWN = 0, 1, 2, 3, 4, 5

    def __init__(self, config: DictConfig):
        habitat_config = config.habitat if "habitat" in config else config
        dataset = make_dataset(
            habitat_config.dataset.type, config=habitat_config.dataset
        )
        scene_dataset = habitat_config.simulator.scene_dataset
        if scene_dataset != "default":
            for episode in dataset.episodes:
                episode.scene_dataset_config = scene_dataset
        self._env = Env(config=config, dataset=dataset)
        self._habitat_env = self._env
        self._step_count = 0
        self._last_obs = None
        self._forward_step_size = float(
            getattr(config.habitat.simulator, "forward_step_size", 0.25)
        )
        self._turn_angle_deg = float(
            getattr(config.habitat.simulator, "turn_angle", 30.0)
        )
        self._turn_angle_rad = np.deg2rad(self._turn_angle_deg)
        self._linear_progress = 0.0
        self._angular_progress = 0.0
        self._has_velocity_control = self._detect_velocity_control_action()
        self._last_sim_location = None

    @property
    def habitat_env(self):
        return self._habitat_env

    @property
    def current_episode(self):
        return self._habitat_env.current_episode

    @property
    def number_of_episodes(self) -> int:
        return len(self._habitat_env.episodes)

    def reset(self) -> dict:
        obs = self._env.reset()
        self._step_count = 0
        self._last_sim_location = self._get_sim_location()
        obs = self._with_sensor_pose(obs, [0.0, 0.0, 0.0])
        self._last_obs = obs
        self._linear_progress = 0.0
        self._angular_progress = 0.0
        return obs

    def step(self, action: int):
        obs = self._env.step(action)
        sensor_pose = self._get_pose_change()
        obs = self._with_sensor_pose(obs, sensor_pose)
        done = self._env.episode_over
        info = self._env.get_metrics()
        reward = self._reward_from_info(info)
        self._step_count += 1
        self._last_obs = obs
        return obs, float(reward), bool(done), info

    def velocity_step(self, linear: float, angular: float, dt: float = 0.0, camera_pitch: float = 0.0):
        """
        Step the environment from a velocity command.

        When the Habitat task exposes native velocity control, forward the
        command directly every sim tick. Otherwise, accumulate commanded motion
        until it reaches one discrete Habitat step or turn.
        """
        linear = float(linear)
        angular = float(angular)

        if self._has_velocity_control:
            action = {
                "action": ("velocity_control", "velocity_stop"),
                "action_args": {
                    "angular_velocity": np.array([angular], dtype=np.float32),
                    "linear_velocity": np.array([linear], dtype=np.float32),
                    "velocity_stop": np.array([-1.0], dtype=np.float32),
                },
            }
            obs = self._env.step(action)
            sensor_pose = self._get_pose_change()
            obs = self._with_sensor_pose(obs, sensor_pose)
            done = self._env.episode_over
            info = self._env.get_metrics()
            reward = self._reward_from_info(info)
            self._step_count += 1
            self._last_obs = obs
            return obs, float(reward), bool(done), info

        if dt > 0.0:
            self._linear_progress += abs(linear) * dt
            self._angular_progress += abs(angular) * dt

        if self._angular_progress + 1e-9 >= self._turn_angle_rad:
            self._angular_progress = max(0.0, self._angular_progress - self._turn_angle_rad)
            return self.step(self.LEFT if angular > 0 else self.RIGHT)
        if self._linear_progress + 1e-9 >= self._forward_step_size:
            self._linear_progress = max(0.0, self._linear_progress - self._forward_step_size)
            return self.step(self.FORWARD)
        return self._last_obs, 0.0, False, {}

    def _detect_velocity_control_action(self) -> bool:
        task = getattr(self._habitat_env, "task", None)
        action_space = getattr(task, "action_space", None)
        if action_space is None:
            return False

        if hasattr(action_space, "spaces"):
            spaces = action_space.spaces
            if "action" in spaces:
                action_selector = spaces["action"]
                selector_spaces = getattr(action_selector, "spaces", None)
                if selector_spaces is not None:
                    return "velocity_control" in selector_spaces
            return "velocity_control" in spaces

        return False

    def _get_sim_location(self):
        """Match OpenFMNav's Habitat simulator pose convention."""
        agent_state = self._env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        euler = quaternion.as_euler_angles(agent_state.rotation)
        axis = euler[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
            o = euler[1]
        else:
            o = 2 * np.pi - euler[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def _get_pose_change(self):
        curr_sim_location = self._get_sim_location()
        if self._last_sim_location is None:
            self._last_sim_location = curr_sim_location
            return [0.0, 0.0, 0.0]

        x1, y1, o1 = self._last_sim_location
        x2, y2, o2 = curr_sim_location
        theta = np.arctan2(y2 - y1, x2 - x1) - o1
        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        self._last_sim_location = curr_sim_location
        return [
            dist * np.cos(theta),
            dist * np.sin(theta),
            np.arctan2(np.sin(o2 - o1), np.cos(o2 - o1)),
        ]

    def _get_camera_pitch(self) -> float:
        """Return RGB camera pitch relative to the agent body, in radians."""
        try:
            agent_state = self._env.sim.get_agent_state(0)
            sensor_states = getattr(agent_state, "sensor_states", {}) or {}
            sensor_state = None
            for name, state in sensor_states.items():
                lname = str(name).lower()
                if "rgb" in lname or "color" in lname:
                    sensor_state = state
                    break
            if sensor_state is None and sensor_states:
                sensor_state = next(iter(sensor_states.values()))
            if sensor_state is None:
                return 0.0

            rel_rot = agent_state.rotation.conjugate() * sensor_state.rotation
            forward = quaternion.rotate_vectors(rel_rot, np.array([0.0, 0.0, -1.0]))
            horizontal = float(np.linalg.norm([forward[0], forward[2]]))
            return float(np.arctan2(float(forward[1]), horizontal))
        except Exception:
            return 0.0

    def _with_sensor_pose(self, obs: dict, sensor_pose) -> dict:
        obs = dict(obs)
        episode = self.current_episode
        obs["gps_3d"] = episodic_gps_3d(
            self._env.sim.get_agent_state(0).position,
            episode.start_position,
            episode.start_rotation,
        )
        obs["sensor_pose"] = np.asarray(sensor_pose, dtype=np.float32)
        obs["camera_pitch"] = np.float32(self._get_camera_pitch())
        return obs

    @staticmethod
    def _reward_from_info(info: Dict[str, Any]) -> float:
        for key in ("sum_reward", "reward"):
            value = info.get(key)
            if isinstance(value, (int, float, np.number)):
                return float(value)
        return 0.0

    def close(self):
        self._env.close()

_VSTEP_PROFILE = os.environ.get("VSTEP_PROFILE", "0") != "0"


class AsyncHabitatEnv(HabitatEnvWrapper):
    """Continuous-velocity wrapper for async mode.

    Inherits :class:`HabitatEnvWrapper` for the action constants, the
    ``habitat_env`` / ``current_episode`` / ``number_of_episodes`` / ``close``
    surface. Overrides ``velocity_step``
    so ``velocity_step`` becomes a real Euler integrator on
    ``habitat_sim.physics.VelocityControl`` with navmesh projection. Look
    commands are integrated directly on the sensor pose.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        # VelocityControl: habitat-sim rigid-body Euler integrator. We feed
        # it (linear, angular) every tick, integrate_transform over dt, then
        # snap the result onto the navmesh.
        self._vel_ctrl = VelocityControl()
        self._vel_ctrl.controlling_lin_vel = True
        self._vel_ctrl.controlling_ang_vel = True
        self._vel_ctrl.lin_vel_is_local = True
        self._vel_ctrl.ang_vel_is_local = True

        sim_cfg = config.habitat.simulator
        self._allow_sliding = bool(getattr(sim_cfg.habitat_sim_v0, "allow_sliding", True))

        # Async stepping tunables.
        self._vstep_substeps = max(1, int(os.environ.get("ASYNC_VSTEP_SUBSTEPS", "1")))
        self._vstep_use_navmesh = os.environ.get("ASYNC_VSTEP_USE_NAVMESH", "1") != "0"
        # OFF by default. step_physics() after set_agent_state() lets physics
        # revert our XY translation (yaw survives) — "robot rotates but never
        # translates". Only enable for scenes with dynamic objects.
        self._vstep_step_physics = os.environ.get("ASYNC_VSTEP_STEP_PHYSICS", "0") != "0"
        self._vstep_debug = os.environ.get("ASYNC_VSTEP_DEBUG", "0") != "0"
        self._vstep_debug_count = 0

    def reset(self) -> dict:
        return super().reset()

    def velocity_step(
        self,
        linear: float,
        angular: float,
        dt: float = 1.0 / 30.0,
        linear_y: float = 0.0,
        camera_pitch: float = 0.0,
    ):
        """One Euler-integrated step over ``dt`` seconds.

        Δposition = (linear, linear_y) · dt (m, body-frame); Δyaw = angular · dt (rad).
        Habitat is nominally a unicycle, but ``VelocityControl`` accepts a
        3D body-frame linear velocity, so non-zero ``linear_y`` strafes.
        """
        hab_env = self._habitat_env
        sim, task = hab_env.sim, hab_env.task

        self._set_velocity_control(linear, linear_y, angular)
        collided = self._integrate_pose(sim, dt)

        if abs(float(camera_pitch)) > 1e-12:
            self._apply_look_velocity(camera_pitch, dt)

        sim._prev_sim_obs["collided"] = collided
        if self._vstep_debug and collided and (self._vstep_debug_count % 30) == 0:
            print(
                f"[vstep] motion clipped by navmesh: cmd=(vx={linear:+.2f}, "
                f"vy={linear_y:+.2f}, wz={angular:+.2f})"
            )
        self._vstep_debug_count += 1

        obs, info, done = self._refresh_obs_and_metrics(sim, task, hab_env)
        sensor_pose = self._get_pose_change()
        obs = self._with_sensor_pose(obs, sensor_pose)
        self._step_count += 1
        self._last_obs = obs
        return obs, 0.0, bool(done), info

    # ── velocity_step helpers ────────────────────────────────────────────

    def _set_velocity_control(self, cmd_lin: float, cmd_lin_y: float, cmd_ang: float) -> None:
        """World-frame cmd → habitat-sim local axes (+X right / +Y up / -Z fwd).

        forward → -Z, strafe-left → -X, yaw-CCW → +Y. Sign verified via
        rtnav_async_velocity_probe: ``vel_ctrl.angular > 0`` → ``compass`` d/dt > 0.
        The agent's ``cmd_wz > 0`` increases ``robot_yaw`` (= compass + offset),
        matching this convention.
        """
        self._vel_ctrl.linear_velocity = np.array([-cmd_lin_y, 0.0, -cmd_lin])
        self._vel_ctrl.angular_velocity = np.array([0.0, cmd_ang, 0.0])

    def _integrate_pose(self, sim, dt: float) -> bool:
        """Advance the agent over ``dt`` via ``VelocityControl``.

        Split into ``ASYNC_VSTEP_SUBSTEPS`` substeps; project each onto the
        navmesh (unless ``ASYNC_VSTEP_USE_NAVMESH=0``). Returns True if any
        substep was clipped (collision).
        """
        substeps = max(1, self._vstep_substeps)
        sub_dt = float(dt) / float(substeps)
        collided = False

        for _ in range(substeps):
            agent_state = sim.get_agent_state()
            rot = agent_state.rotation
            cur_rs = RigidState(mn.Quaternion(rot.imag, rot.real), agent_state.position)

            goal_rs = self._vel_ctrl.integrate_transform(sub_dt, cur_rs)

            if self._vstep_use_navmesh:
                step_fn = (
                    sim.pathfinder.try_step
                    if self._allow_sliding
                    else sim.pathfinder.try_step_no_sliding
                )
                final_position = step_fn(agent_state.position, goal_rs.translation)
            else:
                final_position = goal_rs.translation

            final_rotation = [*goal_rs.rotation.vector, goal_rs.rotation.scalar]

            # Collision: try_step shortened our displacement → navmesh clip.
            requested = (goal_rs.translation - agent_state.position).dot()
            actual = (final_position - agent_state.position).dot()
            collided = collided or ((actual + 1e-5) < requested)

            sim.set_agent_state(final_position, final_rotation, reset_sensors=False)
            if self._vstep_step_physics:
                try:
                    sim.step_physics(sub_dt)
                except Exception:
                    pass

        return collided

    def _apply_look_velocity(self, camera_pitch: float, dt: float) -> None:
        amount_rad = float(camera_pitch) * float(dt)
        if abs(amount_rad) <= 1e-9:
            return

        sim = self._env.sim
        agent = sim.get_agent(0) if hasattr(sim, "get_agent") else sim.agents[0]
        state = agent.get_state()
        sensor_states = getattr(state, "sensor_states", {}) or {}
        if not sensor_states:
            return

        delta = quaternion.from_rotation_vector(np.array([amount_rad, 0.0, 0.0]))
        new_sensor_states = {
            name: SixDOFPose(
                position=np.asarray(sensor_state.position),
                rotation=sensor_state.rotation * delta,
            )
            for name, sensor_state in sensor_states.items()
        }
        agent.set_state(
            AgentState(
                position=np.asarray(state.position),
                rotation=state.rotation,
                sensor_states=new_sensor_states,
            ),
            reset_sensors=False,
            infer_sensor_states=False,
        )

    def _refresh_obs_and_metrics(self, sim, task, hab_env) -> tuple:
        """Re-run sensors + measurements (mirrors habitat-lab's HabitatSim.step)
        so the pipeline sees fresh rgb/depth/gps/compass/info after our pose
        update. ``VSTEP_PROFILE=1`` prints per-stage timing every 10 steps."""
        if _VSTEP_PROFILE:
            t0 = time.time()

        sim_obs = sim.get_sensor_observations()
        sim._prev_sim_obs = sim_obs
        obs = sim._sensor_suite.get_observations(sim_obs)
        if _VSTEP_PROFILE:
            t1 = time.time()

        obs.update(
            task.sensor_suite.get_observations(
                observations=obs, task=task, episode=hab_env.current_episode,
            )
        )
        if _VSTEP_PROFILE:
            t2 = time.time()

        task.measurements.update_measures(
            episode=hab_env.current_episode,
            action={"action": "velocity_control"},
            task=task,
            observations=obs,
        )
        if _VSTEP_PROFILE:
            t3 = time.time()
            if (self._step_count % 10) == 0:
                print(
                    f"[vstep] render+sim_sensors={1000*(t1-t0):5.1f}ms "
                    f"task_sensors={1000*(t2-t1):5.1f}ms "
                    f"measurements={1000*(t3-t2):5.1f}ms"
                )

        info = task.measurements.get_metrics()
        done = task.is_episode_active is False
        return obs, info, done
