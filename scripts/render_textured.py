"""Put a textured mesh back into the white-box scene and render it with the original (or an orbit) camera.

Usage: [OBJECT=crate] blender -b SCENE_DIR/scene.blend -P render_textured.py -- MESH_OBJ ALBEDO_PNG MODE CAMERA OUT_DIR
The white-box object OBJECT (default house) is replaced by MESH_OBJ; everything else stays white-box.
MODE: lit (Principled, scene sun) | unlit (emission = albedo, matches baked-in video lighting)
CAMERA: original (the 81-frame white-box camera) | orbit (full 360 deg around the object, 96 frames)
"""

import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycles_device import use_device  # noqa: E402

mesh_path, albedo_path, mode, camera, out = sys.argv[sys.argv.index("--") + 1:]
scene = bpy.context.scene
name = os.environ.get("OBJECT", "house")
# orbit target, radius, camera height; the crate orbit runs through the house, so the house is hidden from the camera
ORBITS = {"house": ((0.0, 0.0, 1.8), 9.0, 3.2), "crate": ((2.9, -2.7, 0.5), 2.8, 1.7)}

bpy.data.objects[name.capitalize()].hide_render = True
bpy.ops.wm.obj_import(filepath=mesh_path, forward_axis="Y", up_axis="Z")
obj = bpy.context.selected_objects[0]
for poly in obj.data.polygons:
    poly.use_smooth = False

mat = bpy.data.materials.new("Textured")
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
    cam = scene.camera
    cam.animation_data_clear()
    centre, radius, height = ORBITS[name]
    target = Vector(centre)
    if name != "house":
        bpy.data.objects["House"].visible_camera = False  # still casts shadows
    frames = 96
    for f in range(frames):
        theta = 2 * math.pi * f / frames - math.radians(70)
        pos = Vector((target.x + radius * math.sin(theta), target.y - radius * math.cos(theta), height))
        cam.location = pos
        cam.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
    scene.frame_start, scene.frame_end = 0, frames - 1

scene.node_tree.nodes.clear() if scene.use_nodes else None
scene.use_nodes = False  # no mask/depth outputs needed here
scene.frame_step = int(os.environ.get("FRAME_STEP", "1"))
use_device(scene)
scene.cycles.samples = 64
scene.render.filepath = str(Path(out) / "####")
bpy.ops.render.render(animation=True)
