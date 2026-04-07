# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: GPL-3.0-or-later

"""Transform Editor Panel for Lichtfeld Studio."""

from __future__ import annotations
import math
import numpy as np
import lichtfeld as lf


# ── Helpers ───────────────────────────────────────────────────────────────────

def _request_redraw():
    for fn in ("tag_redraw", "request_redraw", "redraw", "invalidate"):
        f = getattr(lf.ui, fn, None)
        if callable(f):
            try: f(); return
            except Exception: pass


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

    Mirrors the logic from SORPanel._merge_visible in the SOR plugin.
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
        scene.notify_changed()
        return ""
    except Exception as e:
        import traceback
        lf.log.error(f"EDIT merge error: {traceback.format_exc()}")
        return str(e)


def _bake(node_name):
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

_T_RANGE = 50.0
_R_RANGE = 180.0
_S_MIN   = 0.01
_S_MAX   = 5.0

# Width of the manual input box on the right of each slider row (px).
_INPUT_W = 60


class TransformPanel(lf.ui.Panel):
    id    = "edit.transform_panel"
    label = "Edit"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 290

    def __init__(self):
        self._status         = ""
        self._node_name      = ""
        self._merge_name     = "Merged"
        self._folder_name    = "Group"
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self._sx = self._sy = self._sz = 1.0
        self._uniform_scale  = True
        self._live           = True
        self._last_mat       = None
        self._gen            = 0  # bumped on sync to flush input_float buffers

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def _sync_from_scene(self):
        try:
            name = lf.get_selected_node_name()
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
            self._last_mat = None
            self._gen += 1
            self._status = f"Synced: {name}"
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
            if mat != self._last_mat:
                lf.set_node_transform(self._node_name, mat)
                self._last_mat = mat
        except Exception as e:
            self._status = f"Apply error: {e}"

    def _row(self, ui, label, uid, value, lo, hi, step):
        """One row: label | drag slider | input box, all on the same line.

        Slider fills all space except _INPUT_W px reserved for the input box.
        Stable widget IDs (with _gen suffix flushed only on sync) let ImGui
        retain keyboard focus across frames but reinit after a grab/sync.
        """
        ui.push_item_width(14)
        ui.label(label)
        ui.pop_item_width()
        ui.same_line()

        # Slider fills all remaining space except _INPUT_W px on the right.
        ui.push_item_width(-_INPUT_W)
        ch_d, v_d = ui.drag_float(f"##{uid}_d", value, step, lo, hi)
        ui.pop_item_width()

        ui.same_line()

        # Input box — ID includes _gen so it reinits after a sync/grab.
        ui.push_item_width(_INPUT_W)
        ch_i, v_i = ui.input_float(f"##{uid}_i{self._gen}", v_d if ch_d else value, step, step * 10)
        ui.pop_item_width()

        if ch_i:
            return True, float(v_i)
        if ch_d:
            return True, float(v_d)
        return False, value

    def draw(self, ui):
        ui.heading("Edit Transform")

        if not lf.has_scene():
            ui.text_disabled("No scene loaded.")
            return

        current_name = lf.get_selected_node_name()
        if current_name != self._node_name:
            self._sync_from_scene()

        if not self._node_name:
            ui.text_disabled("Select a splat node in the scene.")
            if ui.button("Refresh##ref"):
                self._sync_from_scene()
                _request_redraw()
            return

        ui.label(f"Node:  {self._node_name}")
        if ui.button("Grab from viewport##grab"):
            self._sync_from_scene()
            _request_redraw()
        ui.same_line()
        _, self._live = ui.checkbox("Live##live", self._live)

        ui.separator()
        changed_any = False

        # ── Translation ───────────────────────────────────────────────────────
        ui.label("Translation")
        ch, v = self._row(ui, "X", "tx", self._tx, -_T_RANGE, _T_RANGE, 0.001)
        if ch: self._tx = v; changed_any = True
        ch, v = self._row(ui, "Y", "ty", self._ty, -_T_RANGE, _T_RANGE, 0.001)
        if ch: self._ty = v; changed_any = True
        ch, v = self._row(ui, "Z", "tz", self._tz, -_T_RANGE, _T_RANGE, 0.001)
        if ch: self._tz = v; changed_any = True

        ui.separator()

        # ── Rotation ──────────────────────────────────────────────────────────
        ui.label("Rotation (°)")
        ch, v = self._row(ui, "X", "rx", self._rx, -_R_RANGE, _R_RANGE, 0.1)
        if ch: self._rx = v; changed_any = True
        ch, v = self._row(ui, "Y", "ry", self._ry, -_R_RANGE, _R_RANGE, 0.1)
        if ch: self._ry = v; changed_any = True
        ch, v = self._row(ui, "Z", "rz", self._rz, -_R_RANGE, _R_RANGE, 0.1)
        if ch: self._rz = v; changed_any = True

        ui.separator()

        # ── Scale ─────────────────────────────────────────────────────────────
        ui.label("Scale")
        _, self._uniform_scale = ui.checkbox("Uniform##uni", self._uniform_scale)

        ch, v = self._row(ui, "X", "sx", self._sx, _S_MIN, _S_MAX, 0.001)
        if ch:
            self._sx = max(1e-6, v)
            if self._uniform_scale:
                self._sy = self._sz = self._sx
            changed_any = True

        ch, v = self._row(ui, "Y", "sy", self._sy, _S_MIN, _S_MAX, 0.001)
        if ch and not self._uniform_scale:
            self._sy = max(1e-6, v); changed_any = True

        ch, v = self._row(ui, "Z", "sz", self._sz, _S_MIN, _S_MAX, 0.001)
        if ch and not self._uniform_scale:
            self._sz = max(1e-6, v); changed_any = True

        # Apply live; bump _gen so input boxes refresh from new slider value.
        if changed_any and self._live:
            self._apply_to_scene()
            self._gen += 1

        ui.separator()

        if ui.button("Apply##ap"):
            self._apply_to_scene()
            self._status = "Applied."
            _request_redraw()
        ui.same_line()
        if ui.button("Reset##rs"):
            self._tx = self._ty = self._tz = 0.0
            self._rx = self._ry = self._rz = 0.0
            self._sx = self._sy = self._sz = 1.0
            self._apply_to_scene()
            self._status = "Reset to identity."
            _request_redraw()

        ui.separator()

        # ── Bake ──────────────────────────────────────────────────────────────
        ui.label("Bake")
        if ui.button_styled("Bake Transform##bk", "primary"):
            self._apply_to_scene()
            err = _bake(self._node_name)
            if err:
                self._status = f"Bake failed: {err}"
            else:
                self._status = "Baked — transform reset to identity."
                self._sync_from_scene()
            _request_redraw()
        ui.text_disabled("Permanently writes transform into Gaussian data.")
        ui.text_disabled("Save a backup before baking.")

        ui.separator()

        # ── Merge Visible Nodes ───────────────────────────────────────────────
        ui.heading("Merge Visible Nodes")
        ui.push_item_width(120)
        _, self._merge_name = ui.input_text("##merge_name", self._merge_name)
        ui.pop_item_width()
        ui.same_line()
        if ui.button_styled("Merge Visible##mv", "primary"):
            err = _merge_visible(self._merge_name)
            if err:
                self._status = f"Merge failed: {err}"
            else:
                name = self._merge_name.strip() or "merged"
                self._status = f"Merged visible nodes into '{name}'."
                self._sync_from_scene()
            _request_redraw()
        ui.text_disabled("Combines all visible splat nodes into one named node.")

        ui.separator()

        # ── New Group Folder ──────────────────────────────────────────────────
        ui.heading("New Group Folder")
        ui.push_item_width(120)
        _, self._folder_name = ui.input_text("##folder_name", self._folder_name)
        ui.pop_item_width()
        ui.same_line()
        if ui.button("Create##cf"):
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
            _request_redraw()
        ui.text_disabled("Adds an empty folder node to the scene hierarchy.")

        ui.separator()
        if self._status:
            ui.label(self._status)