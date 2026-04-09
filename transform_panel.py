# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: GPL-3.0-or-later

"""Transform Editor Panel for Lichtfeld Studio."""

from __future__ import annotations
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import lichtfeld as lf


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mat_from_trs(tx, ty, tz, rx, ry, rz, sx, sy, sz):
    rx_r = math.radians(rx); ry_r = math.radians(ry); rz_r = math.radians(rz)
    cx, sx_ = math.cos(rx_r), math.sin(rx_r)
    cy, sy_ = math.cos(ry_r), math.sin(ry_r)
    cz, sz_ = math.cos(rz_r), math.sin(rz_r)
    R = np.array([
        [ cy*cz,  cz*sx_*sy_ - cx*sz_,  cx*cz*sy_ + sx_*sz_],
        [ cy*sz_,  cx*cz + sx_*sy_*sz_, -cz*sx_ + cx*sy_*sz_],
        [-sy_,     cy*sx_,               cx*cy               ],
    ], dtype=np.float64)
    RS = R * np.array([sx, sy, sz])
    return [RS[0,0], RS[1,0], RS[2,0], 0.0,
            RS[0,1], RS[1,1], RS[2,1], 0.0,
            RS[0,2], RS[1,2], RS[2,2], 0.0,
            tx, ty, tz, 1.0]


def _decompose_mat(wt):
    M  = np.array([[wt[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)
    t  = M[:3, 3]
    RS = M[:3, :3]
    sx = np.linalg.norm(RS[:, 0])
    sy = np.linalg.norm(RS[:, 1])
    sz = np.linalg.norm(RS[:, 2])
    R  = RS / np.array([sx, sy, sz])
    if abs(R[2, 0]) < 1.0 - 1e-6:
        ry = math.asin(-R[2, 0])
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        ry = math.pi/2 if R[2, 0] < 0 else -math.pi/2
        rx = math.atan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return (float(t[0]), float(t[1]), float(t[2]),
            math.degrees(rx), math.degrees(ry), math.degrees(rz),
            float(sx), float(sy), float(sz))


def _mat_to_quat(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def _quat_mul_batch(q1, q2):
    w1,x1,y1,z1 = q1[0], q1[1], q1[2], q1[3]
    w2,x2,y2,z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], axis=1)


def _merge_visible(name: str) -> str:
    """Merge all visible splat nodes into a single node called *name*.
    Returns an empty string on success or an error message on failure.
    """
    scene = lf.get_scene()
    if scene is None:
        return "No scene loaded."
    nodes = [n for n in scene.get_visible_nodes() if n.splat_data() is not None]
    if not nodes:
        return "No visible splat nodes to merge."
    name = name.strip() or "merged"
    try:
        group_id = scene.add_group(name)
        for n in nodes:
            scene.reparent(n.id, group_id)
        scene.merge_group(name)
        scene.invalidate_cache()
        scene.notify_changed()
        return ""
    except Exception as e:
        import traceback
        lf.log.error(f"EDIT merge error: {traceback.format_exc()}")
        return str(e)


def _unique_node_name(scene, name: str) -> str:
    """Return *name* if no node with that name exists in *scene*.
    Otherwise append an incrementing two-digit suffix (_01, _02, …).
    """
    if scene.get_node(name) is None:
        return name
    counter = 1
    while True:
        candidate = f"{name}_{counter:02d}"
        if scene.get_node(candidate) is None:
            return candidate
        counter += 1


def _bake(node_name: str) -> str:
    """Permanently write the node transform into its Gaussian data.
    Returns an empty string on success or an error message on failure.
    """
    s    = lf.get_scene()
    node = s.get_node(node_name)
    if node is None:
        return f"Node '{node_name}' not found."
    sd = node.splat_data()
    wt = node.world_transform
    M  = np.array([[wt[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)
    t  = M[:3, 3]
    RS = M[:3, :3]
    sx = np.linalg.norm(RS[:, 0])
    sy = np.linalg.norm(RS[:, 1])
    sz = np.linalg.norm(RS[:, 2])
    R  = RS / np.array([sx, sy, sz])
    try:
        means = sd.get_means().cpu().numpy().astype(np.float64)
        sd.means_raw[:] = lf.Tensor.from_numpy((means @ RS.T + t).astype(np.float32)).cuda()
        if not np.allclose([sx, sy, sz], 1.0, atol=1e-5):
            scales = sd.scaling_raw.cpu().numpy().astype(np.float64)
            sd.scaling_raw[:] = lf.Tensor.from_numpy(
                (scales + np.log([sx, sy, sz])).astype(np.float32)).cuda()
        if not np.allclose(R, np.eye(3), atol=1e-5):
            nq   = _mat_to_quat(R)
            rots = sd.rotation_raw.cpu().numpy().astype(np.float64)
            rb   = _quat_mul_batch(nq, rots).astype(np.float32)
            rb  /= np.linalg.norm(rb, axis=-1, keepdims=True)
            sd.rotation_raw[:] = lf.Tensor.from_numpy(rb).cuda()
        lf.set_node_transform(node_name, [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
        return ""
    except Exception as e:
        import traceback
        lf.log.error(f"EDIT bake error: {traceback.format_exc()}")
        return str(e)


# ── Panel ─────────────────────────────────────────────────────────────────────

_T_STEP = 0.1
_R_STEP = 1.0
_S_STEP = 0.01


class TransformPanel(lf.ui.Panel):
    id                 = "edit.transform_panel"
    label              = "Edit"
    space              = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order              = 290
    template           = str(Path(__file__).resolve().with_name("transform_panel.rml"))
    height_mode        = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 100

    def __init__(self):
        self._handle         = None
        self._node_name      = ""
        self._merge_name     = "merged"
        self._folder_name    = "Group"
        self._move_target    = "Selection"
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self._sx = self._sy = self._sz = 1.0
        self._uniform_scale  = True
        self._live           = True
        self._status         = ""
        self._last_node_name = None   # dirty-detection
        self._last_logged    = None   # dedup: last transform written to session log

        # Slider limits — defaults; overridden by settings.json if present
        self._t_min  = -50.0
        self._t_max  =  50.0
        self._r_min  = -180.0
        self._r_max  =  180.0
        self._s_min  =  0.01
        self._s_max  =  5.0
        self._t_step =  0.1
        self._r_step =  1.0
        self._s_step =  0.01

        self._load_settings()

    @classmethod
    def poll(cls, context) -> bool:
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("transform_panel")

        # Visibility guards
        model.bind_func("no_scene",  lambda: not lf.has_scene())
        model.bind_func("no_node",   lambda: lf.has_scene() and not self._node_name)
        model.bind_func("has_node",  lambda: bool(self._node_name))
        model.bind_func("node_name", lambda: self._node_name)

        # Live checkbox
        model.bind("live",
                   lambda: self._live,
                   self._set_live)

        # Slider limit attributes (read-only — driven from settings.json via _load_settings)
        model.bind_func("t_min",  lambda: str(self._t_min))
        model.bind_func("t_max",  lambda: str(self._t_max))
        model.bind_func("t_step", lambda: str(self._t_step))
        model.bind_func("r_min",  lambda: str(self._r_min))
        model.bind_func("r_max",  lambda: str(self._r_max))
        model.bind_func("r_step", lambda: str(self._r_step))
        model.bind_func("s_min",  lambda: str(self._s_min))
        model.bind_func("s_max",  lambda: str(self._s_max))
        model.bind_func("s_step", lambda: str(self._s_step))

        # Tooltip strings — built from loaded limits so they stay in sync with settings.json
        model.bind_func("tip_live",    lambda: "When enabled, every change is applied immediately. Disable to batch changes and apply with the Apply button.")
        model.bind_func("tip_uniform", lambda: "Lock X/Y/Z scale together so any axis scales all three. Uncheck to scale each axis independently.")
        model.bind_func("tip_tx",      lambda: f"Translate the node along the world X axis.  Range {self._t_min} \u2013 {self._t_max}.")
        model.bind_func("tip_ty",      lambda: f"Translate the node along the world Y axis.  Range {self._t_min} \u2013 {self._t_max}.")
        model.bind_func("tip_tz",      lambda: f"Translate the node along the world Z axis.  Range {self._t_min} \u2013 {self._t_max}.")
        model.bind_func("tip_rx",      lambda: f"Rotate the node around the world X axis (pitch).  Range {self._r_min}\u00b0 \u2013 {self._r_max}\u00b0.")
        model.bind_func("tip_ry",      lambda: f"Rotate the node around the world Y axis (yaw).    Range {self._r_min}\u00b0 \u2013 {self._r_max}\u00b0.")
        model.bind_func("tip_rz",      lambda: f"Rotate the node around the world Z axis (roll).   Range {self._r_min}\u00b0 \u2013 {self._r_max}\u00b0.")
        model.bind_func("tip_sx",      lambda: f"Scale the node along the X axis.  Range {self._s_min} \u2013 {self._s_max}.")
        model.bind_func("tip_sy",      lambda: f"Scale the node along the Y axis.  Range {self._s_min} \u2013 {self._s_max}.")
        model.bind_func("tip_sz",      lambda: f"Scale the node along the Z axis.  Range {self._s_min} \u2013 {self._s_max}.")

        # Translation
        model.bind("tx_str",
                   lambda: f"{self._tx:.3f}",
                   lambda v: self._set_trs("tx", v, self._t_min, self._t_max))
        model.bind("ty_str",
                   lambda: f"{self._ty:.3f}",
                   lambda v: self._set_trs("ty", v, self._t_min, self._t_max))
        model.bind("tz_str",
                   lambda: f"{self._tz:.3f}",
                   lambda v: self._set_trs("tz", v, self._t_min, self._t_max))

        # Rotation
        model.bind("rx_str",
                   lambda: f"{self._rx:.1f}",
                   lambda v: self._set_trs("rx", v, self._r_min, self._r_max))
        model.bind("ry_str",
                   lambda: f"{self._ry:.1f}",
                   lambda v: self._set_trs("ry", v, self._r_min, self._r_max))
        model.bind("rz_str",
                   lambda: f"{self._rz:.1f}",
                   lambda v: self._set_trs("rz", v, self._r_min, self._r_max))

        # Scale
        model.bind("uniform_scale",
                   lambda: self._uniform_scale,
                   self._set_uniform_scale)
        model.bind("sx_str",
                   lambda: f"{self._sx:.3f}",
                   lambda v: self._set_trs("sx", v, self._s_min, self._s_max))
        model.bind("sy_str",
                   lambda: f"{self._sy:.3f}",
                   lambda v: self._set_trs("sy", v, self._s_min, self._s_max))
        model.bind("sz_str",
                   lambda: f"{self._sz:.3f}",
                   lambda v: self._set_trs("sz", v, self._s_min, self._s_max))

        # Text inputs
        model.bind("merge_name",
                   lambda: self._merge_name,
                   lambda v: (setattr(self, "_merge_name", str(v)), self._save_settings()))
        model.bind("folder_name",
                   lambda: self._folder_name,
                   lambda v: (setattr(self, "_folder_name", str(v)), self._save_settings()))
        model.bind("move_target",
                   lambda: self._move_target,
                   lambda v: (setattr(self, "_move_target", str(v)), self._save_settings()))

        # Status
        model.bind_func("status_text",  lambda: self._status)
        model.bind_func("status_class", self._status_class)

        # Events
        model.bind_event("do_refresh",         self._on_refresh)
        model.bind_event("do_grab",            self._on_grab)
        model.bind_event("do_apply",           self._on_apply)
        model.bind_event("do_reset",           self._on_reset)
        model.bind_event("do_bake",            self._on_bake)
        model.bind_event("do_merge",           self._on_merge)
        model.bind_event("do_create_folder",   self._on_create_folder)
        model.bind_event("do_move",            self._on_move)
        model.bind_event("num_step",           self._on_num_step)
        model.bind_event("do_reload_settings", self._on_reload_settings)
        model.bind_event("do_open_log",        self._on_open_log)

        self._handle = model.get_handle()
        self._sync_from_scene()

    def on_update(self, doc):
        current = lf.get_selected_node_name() if lf.has_scene() else ""
        if current != self._last_node_name:
            self._last_node_name = current
            self._sync_from_scene()
            self._dirty_all()
            return True
        return False

    def on_unmount(self, doc):
        doc.remove_data_model("transform_panel")
        self._handle = None

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_refresh(self, handle, event, args):
        self._sync_from_scene()
        self._dirty_all()

    def _on_reload_settings(self, handle, event, args):
        self._load_settings()
        self._status = "Settings reloaded from settings.json."
        self._dirty_all()

    def _on_open_log(self, handle, event, args):
        log_path = self._log_path()
        try:
            # Ensure the file exists before trying to open it
            if not log_path.exists():
                log_path.write_text("[]", encoding="utf-8")
            # Try Notepad++ first; fall back to the system default text editor
            npp_candidates = [
                r"C:\Program Files\Notepad++\notepad++.exe",
                r"C:\Program Files (x86)\Notepad++\notepad++.exe",
            ]
            npp = next((p for p in npp_candidates if Path(p).exists()), None)
            if npp:
                subprocess.Popen([npp, str(log_path)])
                self._status = "Opened session_log.json in Notepad++."
            else:
                subprocess.Popen(["notepad.exe", str(log_path)])
                self._status = "Notepad++ not found — opened in Notepad."
        except Exception as e:
            self._status = f"Could not open log: {e}"
        self._dirty("status_text", "status_class")

    def _on_grab(self, handle, event, args):
        self._sync_from_scene()
        self._log_transform("grab")
        self._dirty_all()

    def _on_apply(self, handle, event, args):
        self._apply_to_scene()
        self._log_transform("apply")
        self._status = "Applied."
        self._dirty("status_text", "status_class")

    def _on_reset(self, handle, event, args):
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self._sx = self._sy = self._sz = 1.0
        self._apply_to_scene()
        self._log_transform("reset")
        self._status = "Reset to identity."
        self._save_settings()
        self._dirty("tx_str", "ty_str", "tz_str",
                    "rx_str", "ry_str", "rz_str",
                    "sx_str", "sy_str", "sz_str",
                    "status_text", "status_class")

    def _on_bake(self, handle, event, args):
        self._apply_to_scene()
        self._log_transform("bake")
        err = _bake(self._node_name)
        if err:
            self._status = f"Bake failed: {err}"
        else:
            self._status = "Baked — transform reset to identity."
            self._sync_from_scene()
        self._dirty_all()

    def _on_merge(self, handle, event, args):
        err = _merge_visible(self._merge_name)
        if err:
            self._status = f"Merge failed: {err}"
        else:
            name = self._merge_name.strip() or "merged"
            self._status = f"Merged visible nodes into '{name}'."
            self._sync_from_scene()
        self._dirty("status_text", "status_class")

    def _on_create_folder(self, handle, event, args):
        name = self._folder_name.strip() or "Group"
        try:
            scene = lf.get_scene()
            if scene is None:
                self._status = "No scene loaded."
            else:
                scene.add_group(name)
                scene.notify_changed()
                self._status = f"Created group '{name}'."
        except Exception as e:
            self._status = f"Create group failed: {e}"
        self._dirty("status_text", "status_class")

    def _on_move(self, handle, event, args):
        target_name = self._move_target.strip()
        if not target_name:
            self._status = "Enter a target node name first."
            self._dirty("status_text", "status_class")
            return
        scene = lf.get_scene()
        if scene is None:
            self._status = "No scene loaded."
            self._dirty("status_text", "status_class")
            return
        if target_name != self._node_name:
            unique_target = _unique_node_name(scene, target_name)
            if unique_target != target_name:
                self._status = (
                    f"'{target_name}' already exists — "
                    f"using '{unique_target}' instead."
                )
                target_name       = unique_target
                self._move_target = target_name
                self._dirty("move_target", "status_text", "status_class")
        err = self._move_selected_splats(self._node_name, target_name)
        if err:
            self._status = f"Move failed: {err}"
        else:
            self._status     = f"Moved selected splats \u2192 '{target_name}'."
            self._move_target = "Selection"
            self._sync_from_scene()
        self._dirty_all()

    def _on_num_step(self, handle, event, args):
        if not args or len(args) < 2:
            return
        field     = str(args[0])
        direction = int(args[1])

        steps  = dict(tx=self._t_step, ty=self._t_step, tz=self._t_step,
                      rx=self._r_step, ry=self._r_step, rz=self._r_step,
                      sx=self._s_step, sy=self._s_step, sz=self._s_step)
        ranges = dict(tx=(self._t_min, self._t_max),
                      ty=(self._t_min, self._t_max),
                      tz=(self._t_min, self._t_max),
                      rx=(self._r_min, self._r_max),
                      ry=(self._r_min, self._r_max),
                      rz=(self._r_min, self._r_max),
                      sx=(self._s_min, self._s_max),
                      sy=(self._s_min, self._s_max),
                      sz=(self._s_min, self._s_max))
        if field not in steps:
            return

        lo, hi  = ranges[field]
        current = getattr(self, f"_{field}")
        new_val = round(max(lo, min(hi, current + direction * steps[field])), 4)
        if abs(new_val - current) < 1e-9:
            return
        setattr(self, f"_{field}", new_val)

        if field == "sx" and self._uniform_scale:
            self._sy = self._sz = new_val
            self._dirty("sy_str", "sz_str")

        if self._live:
            self._apply_to_scene()
        self._dirty(f"{field}_str")
        self._save_settings()

    # ── Setters ───────────────────────────────────────────────────────────────

    def _set_live(self, value):
        if isinstance(value, str):
            self._live = value.lower() not in ("false", "0", "")
        else:
            self._live = bool(value)
        self._save_settings()

    def _set_uniform_scale(self, value):
        if isinstance(value, str):
            self._uniform_scale = value.lower() not in ("false", "0", "")
        else:
            self._uniform_scale = bool(value)
        self._save_settings()

    def _set_trs(self, attr: str, value, lo: float, hi: float):
        try:
            v = max(lo, min(hi, float(value)))
        except (TypeError, ValueError):
            return
        if abs(v - getattr(self, f"_{attr}")) < 1e-9:
            return
        setattr(self, f"_{attr}", v)

        if attr == "sx" and self._uniform_scale:
            self._sy = self._sz = v
            self._dirty("sy_str", "sz_str")

        if self._live:
            self._apply_to_scene()
        self._dirty(f"{attr}_str")
        self._save_settings()

    # ── Scene sync ────────────────────────────────────────────────────────────

    def _sync_from_scene(self):
        try:
            name = lf.get_selected_node_name() if lf.has_scene() else ""
            if not name:
                self._node_name = ""
                return
            node = lf.get_scene().get_node(name)
            if node is None:
                return
            self._node_name = name
            (self._tx, self._ty, self._tz,
             self._rx, self._ry, self._rz,
             self._sx, self._sy, self._sz) = _decompose_mat(node.world_transform)
            lf.log.info(f"EDIT synced: t=({self._tx:.3f},{self._ty:.3f},{self._tz:.3f}) "
                        f"r=({self._rx:.1f},{self._ry:.1f},{self._rz:.1f}) "
                        f"s=({self._sx:.3f},{self._sy:.3f},{self._sz:.3f})")
        except Exception as e:
            self._status = f"Sync error: {e}"

    def _apply_to_scene(self):
        if not self._node_name:
            return
        try:
            mat = _mat_from_trs(self._tx, self._ty, self._tz,
                                 self._rx, self._ry, self._rz,
                                 self._sx, self._sy, self._sz)
            lf.set_node_transform(self._node_name, mat)
        except Exception as e:
            self._status = f"Apply error: {e}"

    def _move_selected_splats(self, source_name: str, target_name: str) -> str:
        """Move splats by manually calculating the node's offset in the global mask."""
        try:
            scene       = lf.get_scene()
            global_mask = scene.selection_mask
            if global_mask is None or not lf.has_selection():
                return "Nothing selected."

            visible_splat_nodes = [n for n in scene.get_visible_nodes()
                                   if n.splat_data() is not None]
            start_idx  = 0
            found_node = None
            for n in visible_splat_nodes:
                if n.name == source_name:
                    found_node = n
                    break
                start_idx += n.splat_data().num_points

            if not found_node:
                return f"Source node '{source_name}' not found (or not visible)."

            num_points        = found_node.splat_data().num_points
            local_mask_tensor = global_mask[start_idx : start_idx + num_points]
            mask_np           = local_mask_tensor.cpu().numpy().astype(bool)
            selected_count    = int(mask_np.sum())

            lf.log.info(f"EDIT move: source='{source_name}' total={num_points} "
                        f"selected={selected_count} start_idx={start_idx}")
            if selected_count == 0:
                return "No splats selected in this node."

            scene.clear_selection()

            source_sd    = found_node.splat_data()
            selected_idx = np.where(mask_np)[0]
            inlier_idx   = np.where(~mask_np)[0]

            def _gather(tensor, idx):
                return lf.Tensor.from_numpy(tensor.cpu().numpy()[idx]).cuda()

            sel_means    = _gather(source_sd.means_raw,    selected_idx)
            sel_sh0      = _gather(source_sd.sh0_raw,      selected_idx)
            sel_shN      = _gather(source_sd.shN_raw,      selected_idx)
            sel_scaling  = _gather(source_sd.scaling_raw,  selected_idx)
            sel_rotation = _gather(source_sd.rotation_raw, selected_idx)
            sel_opacity  = _gather(source_sd.opacity_raw,  selected_idx)

            inlier_means    = _gather(source_sd.means_raw,    inlier_idx)
            inlier_sh0      = _gather(source_sd.sh0_raw,      inlier_idx)
            inlier_shN      = _gather(source_sd.shN_raw,      inlier_idx)
            inlier_scaling  = _gather(source_sd.scaling_raw,  inlier_idx)
            inlier_rotation = _gather(source_sd.rotation_raw, inlier_idx)
            inlier_opacity  = _gather(source_sd.opacity_raw,  inlier_idx)

            active_sh = source_sd.active_sh_degree
            s_scale   = source_sd.scene_scale

            target_node = scene.get_node(target_name)
            if target_node is not None:
                target_sd = target_node.splat_data()
                def _cat(a, b):
                    return lf.Tensor.from_numpy(
                        np.concatenate([a.cpu().numpy(), b.cpu().numpy()], axis=0)
                    ).cuda()
                scene.remove_node(target_name)
                scene.add_splat(
                    target_name,
                    _cat(target_sd.means_raw,    sel_means),
                    _cat(target_sd.sh0_raw,      sel_sh0),
                    _cat(target_sd.shN_raw,      sel_shN),
                    _cat(target_sd.scaling_raw,  sel_scaling),
                    _cat(target_sd.rotation_raw, sel_rotation),
                    _cat(target_sd.opacity_raw,  sel_opacity),
                    active_sh, s_scale,
                )
            else:
                scene.add_splat(
                    target_name,
                    sel_means, sel_sh0, sel_shN,
                    sel_scaling, sel_rotation, sel_opacity,
                    active_sh, s_scale,
                )

            lf.log.info(f"EDIT move: target='{target_name}' built with {len(selected_idx)} splats")

            scene.remove_node(source_name)
            scene.add_splat(
                source_name,
                inlier_means, inlier_sh0, inlier_shN,
                inlier_scaling, inlier_rotation, inlier_opacity,
                active_sh, s_scale,
            )

            lf.log.info(f"EDIT move: source='{source_name}' rebuilt with {len(inlier_idx)} splats")

            scene.invalidate_cache()
            scene.notify_changed()
            return ""

        except Exception as e:
            import traceback
            lf.log.error(f"EDIT move error: {traceback.format_exc()}")
            return str(e)

    # ── Settings persistence ──────────────────────────────────────────────────

    @staticmethod
    def _settings_path() -> Path:
        return Path(__file__).resolve().with_name("settings.json")

    @staticmethod
    def _log_path() -> Path:
        return Path(__file__).resolve().with_name("session_log.json")

    def _log_transform(self, action: str = "apply"):
        """Append one transform entry to session_log.json, skipping exact duplicates."""
        try:
            snapshot = (
                action,
                self._node_name,
                round(self._tx, 4), round(self._ty, 4), round(self._tz, 4),
                round(self._rx, 4), round(self._ry, 4), round(self._rz, 4),
                round(self._sx, 4), round(self._sy, 4), round(self._sz, 4),
                self._uniform_scale, self._live,
            )
            if snapshot == self._last_logged:
                return
            self._last_logged = snapshot

            path = self._log_path()
            try:
                log = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(log, list):
                    log = []
            except Exception:
                log = []

            entry = {
                "time":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "action": action,
                "node":   self._node_name,
                "transform": {
                    "tx": round(self._tx, 4),
                    "ty": round(self._ty, 4),
                    "tz": round(self._tz, 4),
                    "rx": round(self._rx, 4),
                    "ry": round(self._ry, 4),
                    "rz": round(self._rz, 4),
                    "sx": round(self._sx, 4),
                    "sy": round(self._sy, 4),
                    "sz": round(self._sz, 4),
                    "uniform_scale": self._uniform_scale,
                    "live":          self._live,
                },
            }
            log.append(entry)
            path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        except Exception as e:
            lf.log.error(f"EDIT session log error: {e}")

    def _load_settings(self):
        try:
            data = json.loads(self._settings_path().read_text(encoding="utf-8"))
            t = data.get("transform", {})
            self._tx            = float(t.get("tx",            self._tx))
            self._ty            = float(t.get("ty",            self._ty))
            self._tz            = float(t.get("tz",            self._tz))
            self._rx            = float(t.get("rx",            self._rx))
            self._ry            = float(t.get("ry",            self._ry))
            self._rz            = float(t.get("rz",            self._rz))
            self._sx            = float(t.get("sx",            self._sx))
            self._sy            = float(t.get("sy",            self._sy))
            self._sz            = float(t.get("sz",            self._sz))
            # Booleans: use explicit `is True` comparison to correctly read
            # JSON false (Python False) rather than truthy/falsy coercion.
            us = t.get("uniform_scale", self._uniform_scale)
            self._uniform_scale = us if isinstance(us, bool) else bool(us)
            lv = t.get("live", self._live)
            self._live          = lv if isinstance(lv, bool) else bool(lv)
            self._merge_name    = str(t.get("merge_name",      self._merge_name))
            self._folder_name   = str(t.get("folder_name",     self._folder_name))
            self._move_target   = str(t.get("move_target",     self._move_target))

            # Slider limits & steps from settings.json
            lim = data.get("limits", {})
            self._t_min  = float(lim.get("translation_min",  self._t_min))
            self._t_max  = float(lim.get("translation_max",  self._t_max))
            self._r_min  = float(lim.get("rotation_min",     self._r_min))
            self._r_max  = float(lim.get("rotation_max",     self._r_max))
            self._s_min  = float(lim.get("scale_min",        self._s_min))
            self._s_max  = float(lim.get("scale_max",        self._s_max))
            self._t_step = float(lim.get("translation_step", self._t_step))
            self._r_step = float(lim.get("rotation_step",    self._r_step))
            self._s_step = float(lim.get("scale_step",       self._s_step))
        except FileNotFoundError:
            pass  # first run — file will be created on first save
        except Exception as e:
            lf.log.error(f"EDIT settings load error: {e}")

    def _save_settings(self):
        try:
            path = self._settings_path()
            # Preserve any existing top-level keys (e.g. load_on_startup)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            data["transform"] = {
                "tx":            round(self._tx, 4),
                "ty":            round(self._ty, 4),
                "tz":            round(self._tz, 4),
                "rx":            round(self._rx, 4),
                "ry":            round(self._ry, 4),
                "rz":            round(self._rz, 4),
                "sx":            round(self._sx, 4),
                "sy":            round(self._sy, 4),
                "sz":            round(self._sz, 4),
                "uniform_scale": self._uniform_scale,
                "live":          self._live,
                "merge_name":    self._merge_name,
                "folder_name":   self._folder_name,
                "move_target":   self._move_target,
            }
            data["limits"] = {
                "translation_min":  self._t_min,
                "translation_max":  self._t_max,
                "rotation_min":     self._r_min,
                "rotation_max":     self._r_max,
                "scale_min":        self._s_min,
                "scale_max":        self._s_max,
                "translation_step": self._t_step,
                "rotation_step":    self._r_step,
                "scale_step":       self._s_step,
            }
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            lf.log.error(f"EDIT settings save error: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dirty(self, *fields):
        if not self._handle:
            return
        for f in fields:
            self._handle.dirty(f)

    def _dirty_all(self):
        self._dirty("no_scene", "no_node", "has_node", "node_name",
                    "tx_str", "ty_str", "tz_str",
                    "rx_str", "ry_str", "rz_str",
                    "sx_str", "sy_str", "sz_str",
                    "live", "uniform_scale",
                    "t_min", "t_max", "t_step",
                    "r_min", "r_max", "r_step",
                    "s_min", "s_max", "s_step",
                    "tip_live", "tip_uniform",
                    "tip_tx", "tip_ty", "tip_tz",
                    "tip_rx", "tip_ry", "tip_rz",
                    "tip_sx", "tip_sy", "tip_sz",
                    "merge_name", "folder_name", "move_target",
                    "status_text", "status_class")

    def _status_class(self) -> str:
        s = self._status
        if any(w in s for w in ("Moved", "Merged", "Created", "Applied",
                                "Reset", "Baked", "Synced")):
            return "text-accent"
        if s and ("failed" in s.lower() or "error" in s.lower()):
            return "text-muted"
        return "text-default"
