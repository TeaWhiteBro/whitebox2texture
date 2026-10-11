"""Cycles device for the Blender scripts: env CYCLES_DEVICE = OPTIX (default) | CUDA | CPU."""

import os

import bpy


def use_device(scene):
    device = os.environ.get("CYCLES_DEVICE", "OPTIX")
    scene.cycles.device = "CPU" if device == "CPU" else "GPU"
    if device == "CPU":
        return
    prefs = bpy.context.preferences.addons["cycles"].preferences  # device prefs are not stored in the .blend
    prefs.compute_device_type = device
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == device
