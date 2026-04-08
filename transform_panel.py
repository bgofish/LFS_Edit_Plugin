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
        scene.invalidate_cache()
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
        self._merge_name     = "merged"
        self._folder_name    = "Group"
        self._move_target    = "Selection"
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self._sx = self._sy = self._sz = 1.0
        self._uniform_scale  = True
        self._live           = True
        self._last_mat       = None
        self._gen            = 0  # bumped on sync to flush input_float buffers
        # Per-field display values for the input boxes — only updated on
        # sync/grab so that typing a decimal is never interrupted by the slider.
        self._input_buf: dict[str, float] = {}

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def _move_selected_splats(self, source_name: str, target_name: str) -> str:
        """Move splats by manually calculating the node's offset in the global mask."""
        try:
            scene = lf.get_scene()
            global_mask = scene.selection_mask
            if global_mask is None or not lf.has_selection():
                return "Nothing selected."

            # 1. CALCULATE TRUE OFFSET
            # The global selection mask is built from VISIBLE splat nodes only,
            # so we must walk the same filtered list to get the correct offset.
            # Using scene.get_nodes() (all nodes, including hidden) was wrong —
            # hidden nodes don't contribute to the mask but were shifting start_idx.
            visible_splat_nodes = [n for n in scene.get_visible_nodes() if n.splat_data() is not None]
            start_idx = 0
            found_node = None

            for n in visible_splat_nodes:
                if n.name == source_name:
                    found_node = n
                    break
                start_idx += n.splat_data().num_points

            if not found_node:
                return f"Source node '{source_name}' not found (or not visible)."

            # 2. CAPTURE LOCAL MASK
            num_points = found_node.splat_data().num_points
            local_mask_tensor = global_mask[start_idx : start_idx + num_points]
            # Cast to bool — ~operator on uint8 gives bitwise NOT (255), not logical NOT
            mask_np = local_mask_tensor.cpu().numpy().astype(bool)

            selected_count = int(mask_np.sum())
            lf.log.info(f"EDIT move: source='{source_name}' total={num_points} selected={selected_count} start_idx={start_idx}")
            if selected_count == 0:
                return "No splats selected in this node."

            # 3. DETACH UI
            scene.clear_selection()

            # 4. PERFORM THE MOVE
            # Snapshot selected splats from source BEFORE any mutation.
            source_sd = found_node.splat_data()
            selected_idx = np.where(mask_np)[0]
            inlier_idx   = np.where(~mask_np)[0]

            def _gather(tensor, idx):
                return lf.Tensor.from_numpy(tensor.cpu().numpy()[idx]).cuda()

            # Selected splats — go to target
            sel_means    = _gather(source_sd.means_raw,    selected_idx)
            sel_sh0      = _gather(source_sd.sh0_raw,      selected_idx)
            sel_shN      = _gather(source_sd.shN_raw,      selected_idx)
            sel_scaling  = _gather(source_sd.scaling_raw,  selected_idx)
            sel_rotation = _gather(source_sd.rotation_raw, selected_idx)
            sel_opacity  = _gather(source_sd.opacity_raw,  selected_idx)

            # Inlier splats — stay in source
            inlier_means    = _gather(source_sd.means_raw,    inlier_idx)
            inlier_sh0      = _gather(source_sd.sh0_raw,      inlier_idx)
            inlier_shN      = _gather(source_sd.shN_raw,      inlier_idx)
            inlier_scaling  = _gather(source_sd.scaling_raw,  inlier_idx)
            inlier_rotation = _gather(source_sd.rotation_raw, inlier_idx)
            inlier_opacity  = _gather(source_sd.opacity_raw,  inlier_idx)

            active_sh = source_sd.active_sh_degree
            s_scale   = source_sd.scene_scale

            # Build / update the target node with selected splats.
            # SOR-style: always remove+add to guarantee a fresh panel entry.
            target_node = scene.get_node(target_name)
            if target_node is not None:
                target_sd = target_node.splat_data()
                # Merge by gathering existing + new splats together
                def _cat(a, b):
                    return lf.Tensor.from_numpy(
                        np.concatenate([a.cpu().numpy(), b.cpu().numpy()], axis=0)
                    ).cuda()
                merged_means    = _cat(target_sd.means_raw,    sel_means)
                merged_sh0      = _cat(target_sd.sh0_raw,      sel_sh0)
                merged_shN      = _cat(target_sd.shN_raw,      sel_shN)
                merged_scaling  = _cat(target_sd.scaling_raw,  sel_scaling)
                merged_rotation = _cat(target_sd.rotation_raw, sel_rotation)
                merged_opacity  = _cat(target_sd.opacity_raw,  sel_opacity)
                scene.remove_node(target_name)
                scene.add_splat(
                    target_name,
                    merged_means, merged_sh0, merged_shN,
                    merged_scaling, merged_rotation, merged_opacity,
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

            # Rebuild source node with only the inlier splats (SOR-style replace).
            scene.remove_node(source_name)
            scene.add_splat(
                source_name,
                inlier_means, inlier_sh0, inlier_shN,
                inlier_scaling, inlier_rotation, inlier_opacity,
                active_sh, s_scale,
            )

            lf.log.info(f"EDIT move: source='{source_name}' rebuilt with {len(inlier_idx)} splats")

            # 5. REFRESH
            scene.invalidate_cache()
            scene.notify_changed()
            return ""  # Success
            
        except Exception as e:
            import traceback
            lf.log.error(f"EDIT move error: {traceback.format_exc()}")
            return str(e)



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
            self._input_buf.clear()
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
        The input box has its own buffer (_input_buf) that is only refreshed on
        sync/grab (when _gen bumps), so typing a decimal is never interrupted
        by slider movement.
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

        # Input box — uses its own buffer so mid-decimal typing isn't clobbered
        # by slider updates.  Buffer is only re-seeded when _gen changes (i.e.
        # after a Grab or Sync), via a fresh widget ID.
        buf_key = f"{uid}_{self._gen}"
        if buf_key not in self._input_buf:
            self._input_buf[buf_key] = value
        ui.push_item_width(_INPUT_W)
        ch_i, v_i = ui.input_float(f"##{uid}_i{self._gen}", self._input_buf[buf_key], 0.0, 0.0)
        ui.pop_item_width()
        if ch_i:
            self._input_buf[buf_key] = float(v_i)

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
            self._input_buf.clear()

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
            self._gen += 1
            self._input_buf.clear()
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
        ui.label("Merge Visible Nodes")
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
        ui.label("New Group Folder")
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

        # ── Move Selected Splats ──────────────────────────────────────────────
        ui.label("Move Selected Splats")
        ui.push_item_width(120)
        _, self._move_target = ui.input_text("##move_target", self._move_target)
        ui.pop_item_width()
        ui.same_line()
        if ui.button_styled("Move##mv_splats", "primary"):
            target_name = self._move_target.strip()
            if not target_name:
                self._status = "Enter a target node name first."
            else:
                # Call the method correctly with both arguments
                err = self._move_selected_splats(self._node_name, target_name)
                if err:
                    self._status = f"Move failed: {err}"
                else:
                    self._status = f"Moved selected splats → '{target_name}'."
                    self._move_target = "Selection" # Reset to default
                    self._sync_from_scene()
            _request_redraw()
        ui.text_disabled("Target node name to move selected splats into.")
        ui.separator()


        if self._status:
            ui.label(self._status)