# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: GPL-3.0-or-later

import lichtfeld as lf
from .transform_panel import TransformPanel

_classes = [TransformPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("EDIT loaded")


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("EDIT unloaded")
