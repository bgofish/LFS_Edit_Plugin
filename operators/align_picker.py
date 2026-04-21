# SPDX-License-Identifier: GPL-3.0-or-later
"""Hover-based point capture for the Align plugin.

No modal operator needed. The panel polls get_hovered_gaussian_id() each
update tick and stores the result when the user presses the capture button.
"""

import lichtfeld as lf
import lichtfeld.selection as sel

# ── State ─────────────────────────────────────────────────────────────────────
_capture_callback  = None
_capture_point_num = 0
_capture_cancelled = False

# Last known hovered gaussian world position (updated every tick while active)
_hovered_world_pos = None
_hovered_gid       = -1


def set_capture_callback(callback, point_num: int):
    global _capture_callback, _capture_point_num, _capture_cancelled
    _capture_callback  = callback
    _capture_point_num = point_num
    _capture_cancelled = False
    lf.log.info(f"PICK_DBG set_capture_callback: point_num={point_num}")


def clear_capture_callback():
    global _capture_callback, _capture_point_num, _capture_cancelled
    global _hovered_world_pos, _hovered_gid
    lf.log.info("PICK_DBG clear_capture_callback")
    _capture_callback  = None
    _capture_point_num = 0
    _capture_cancelled = True
    _hovered_world_pos = None
    _hovered_gid       = -1


def was_capture_cancelled() -> bool:
    global _capture_cancelled
    if _capture_cancelled:
        _capture_cancelled = False
        return True
    return False


# Max pixel distance to accept a nearest-Gaussian hit
_MAX_PICK_DIST_PX = 50.0

def poll_hover() -> bool:
    """Find the Gaussian nearest the mouse cursor using screen positions.
    Updates _hovered_world_pos. Returns True if within _MAX_PICK_DIST_PX.
    """
    global _hovered_world_pos, _hovered_gid

    if _capture_callback is None:
        return False

    import numpy as np

    # ── Get mouse position ────────────────────────────────────────────────────
    try:
        mouse = lf.ui.get_mouse_screen_pos()
        mx, my = float(mouse[0]), float(mouse[1])
    except Exception as exc:
        lf.log.warning(f"PICK_DBG poll_hover: get_mouse_screen_pos failed: {exc}")
        return False

    # ── Get all Gaussian screen positions ─────────────────────────────────────
    try:
        if not sel.has_screen_positions():
            _hovered_world_pos = None
            return False
        sp = sel.get_screen_positions().cpu().numpy()  # (N, 2) float32
    except Exception as exc:
        lf.log.warning(f"PICK_DBG poll_hover: get_screen_positions failed: {exc}")
        return False

    # ── Find nearest Gaussian ─────────────────────────────────────────────────
    dx    = sp[:, 0] - mx
    dy    = sp[:, 1] - my
    dist2 = dx*dx + dy*dy
    gid   = int(np.argmin(dist2))
    dist  = float(np.sqrt(dist2[gid]))

    if dist > _MAX_PICK_DIST_PX:
        _hovered_gid       = -1
        _hovered_world_pos = None
        return False

    if gid == _hovered_gid:
        return _hovered_world_pos is not None  # cached, no work needed

    # ── Map global index → node + world position ──────────────────────────────
    scene = lf.get_scene()
    if scene is None:
        return False

    offset = 0
    found_node = None
    local_idx  = -1
    for n in scene.get_visible_nodes():
        sd = n.splat_data()
        if sd is None:
            continue
        count = sd.num_points
        if offset <= gid < offset + count:
            found_node = n
            local_idx  = gid - offset
            break
        offset += count

    if found_node is None:
        return False

    try:
        means     = found_node.splat_data().get_means().cpu().numpy()
        local_pos = means[local_idx].astype(np.float64)
        wt = found_node.world_transform
        M  = np.array([[wt[r][c] for c in range(4)] for r in range(4)],
                      dtype=np.float64)
        world_pos = (M[:3, :3] @ local_pos + M[:3, 3]).tolist()
        _hovered_gid       = gid
        _hovered_world_pos = world_pos
        lf.log.info(f"PICK_DBG poll_hover: gid={gid} dist={dist:.1f}px "
                    f"node='{found_node.name}' world={[f'{v:.4f}' for v in world_pos]}")
        return True
    except Exception as exc:
        lf.log.warning(f"PICK_DBG poll_hover: world-pos lookup failed: {exc}")
        return False


def capture_hovered_point() -> bool:
    """Called when the user clicks Capture. Fires the callback with the
    last known hovered world position. Returns True on success."""
    global _hovered_world_pos

    if _capture_callback is None:
        lf.log.info("PICK_DBG capture_hovered_point: no callback set")
        return False

    if _hovered_world_pos is None:
        lf.log.info("PICK_DBG capture_hovered_point: no hovered position available")
        return False

    lf.log.info(f"PICK_DBG capture_hovered_point: capturing "
                f"point_num={_capture_point_num}  pos={_hovered_world_pos}")
    _capture_callback(_hovered_world_pos, _capture_point_num)
    clear_capture_callback()
    return True


def get_hovered_pos():
    """Returns the current hovered world position, or None."""
    return _hovered_world_pos
