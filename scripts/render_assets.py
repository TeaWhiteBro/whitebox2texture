"""Swap several white-box props for their textured meshes at once and render the scene.

Usage: blender -b SCENE_DIR/scene.blend -P render_assets.py -- MODE CAMERA OUT_DIR NAME=MESH_OBJ:ALBEDO [...]
MODE: lit | unlit (emission = albedo, matches the light baked into the video)
CAMERA: original (the white-box camera; the mannequin stays as it is) | orbit (circle over the plaza, character hidden)
NAME is the Blender object name of the white-box prop (Shop, Crates, Barrel).
"""

import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycles_device import use_device  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
mode, camera, out = args[:3]
scene = bpy.context.scene
ORBIT = ((0.0, 11.0, 0.8), 7.5, 4.0, 120)  # target, radius, camera height, frames

for item in args[3:]:
    name, rest = item.split("=", 1)
    mesh_path, albedo_path = rest.rsplit(":", 1)
    bpy.data.objects[name].hide_render = True
    bpy.ops.wm.obj_import(filepath=mesh_path, forward_axis="Y", up_axis="Z")
    obj = bpy.context.selected_objects[0]
    for poly in obj.data.polygons:
        poly.use_smooth = False
    mat = bpy.data.materials.new(f"{name}Textured")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(albedo_path)
    tex.interpolation = "Cubic"
    output = nt.nodes.new("ShaderNodeOutputMaterial")
    if mode == "lit":
        shader = nt.nodes.new("ShaderNodeBsdfPrincipled")
        shader.inputs["Roughness"].default_value = 0.8
        nt.links.new(tex.outputs["Color"], shader.inputs["Base Color"])
        nt.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    else:
        shader = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(tex.outputs["Color"], shader.inputs["Color"])
        nt.links.new(shader.outputs["Emission"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)

if camera == "orbit":
    for obj in bpy.data.objects:
        if obj.pass_index == 9 or obj.name == "Character":
            obj.hide_render = True
    cam = scene.camera
    cam.animation_data_clear()
    centre, radius, height, frames = ORBIT
    target = Vector(centre)
    for f in range(frames):
        theta = 2 * math.pi * f / frames
        cam.location = target + Vector((radius * math.sin(theta), -radius * math.cos(theta), height - centre[2]))
        cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
    scene.frame_start, scene.frame_end = 0, frames - 1

scene.node_tree.nodes.clear() if scene.use_nodes else None
scene.use_nodes = False
scene.frame_step = int(os.environ.get("FRAME_STEP", "1"))
use_device(scene)
scene.cycles.samples = 64
scene.render.filepath = str(Path(out) / "####")
bpy.ops.render.render(animation=True)
