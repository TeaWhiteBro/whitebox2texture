"""360-degree turntable of one textured prop on its own (unlit, neutral backdrop).

Usage: blender -b --factory-startup -P render_turntable.py -- MESH_OBJ ALBEDO OUT_DIR [FRAMES]
"""

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycles_device import use_device  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
mesh_path, albedo_path, out = args[:3]
frames = int(args[3]) if len(args) > 3 else 96
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.wm.obj_import(filepath=mesh_path, forward_axis="Y", up_axis="Z")
obj = bpy.context.selected_objects[0]
for poly in obj.data.polygons:
    poly.use_smooth = False
mat = bpy.data.materials.new("Albedo")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load(albedo_path)
tex.interpolation = "Cubic"
emit = nt.nodes.new("ShaderNodeEmission")
output = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
nt.links.new(emit.outputs["Emission"], output.inputs["Surface"])
obj.data.materials.clear()  # drop the material the OBJ importer created from the .mtl
obj.data.materials.append(mat)

lo = Vector([min(v.co[i] for v in obj.data.vertices) for i in range(3)])
hi = Vector([max(v.co[i] for v in obj.data.vertices) for i in range(3)])
centre, size = (lo + hi) / 2, (hi - lo).length
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.22, 0.22, 0.24, 1)
cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.lens = 50
for f in range(frames):
    theta = 2 * math.pi * f / frames
    cam.location = centre + size * 1.55 * Vector((math.sin(theta) * 0.94, -math.cos(theta) * 0.94, 0.34))
    cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)
scene.frame_start, scene.frame_end = 0, frames - 1
scene.render.engine = "CYCLES"
use_device(scene)
scene.cycles.samples = 16
scene.view_settings.view_transform = "Standard"
scene.render.resolution_x = scene.render.resolution_y = 480
scene.render.filepath = str(Path(out) / "####")
bpy.ops.render.render(animation=True)
