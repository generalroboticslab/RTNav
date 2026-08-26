"""Block a worker until the episode task is ready or shutdown is requested."""

import threading


def wait_for_task_ready(
    shared_state,
    tag: str,
    shutdown_event: threading.Event | None = None,
) -> bool:
    if shared_state.task_ready.is_set():
        return True

    print(f"[{tag}] waiting for task...")
    while not shared_state.task_ready.wait(timeout=1.0):
        if shutdown_event is not None and shutdown_event.is_set():
            return False

    print(f"[{tag}] task ready")
    return True
