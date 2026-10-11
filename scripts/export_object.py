"""Export one more white-box object from an existing scene: its mesh (subdivided, smart-UV) and per-frame masks.

Usage: blender -b SCENE_DIR/scene.blend -P export_object.py -- SCENE_DIR OBJECT_NAME
Writes SCENE_DIR/<name>.obj and SCENE_DIR/mask_<name>/####.png (name = OBJECT_NAME lower-cased), same conventions
as house.obj / mask/ from build_scene.py. Only the object-index pass is rendered, so this takes seconds.
"""

import math
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycles_device import use_device  # noqa: E402

scene_dir, name = sys.argv[sys.argv.index("--") + 1:]
scene_dir = Path(scene_dir).resolve()
scene = bpy.context.scene
obj = bpy.data.objects[name]

bpy.ops.object.select_all(action="DESELECT")
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.subdivide(number_cuts=6)  # same density treatment as the house; geometry unchanged
bpy.ops.mesh.quads_convert_to_tris()
bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.02)
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.wm.obj_export(filepath=str(scene_dir / f"{name.lower()}.obj"), export_selected_objects=True, export_uv=True,
                      export_normals=True, export_materials=False, export_triangulated_mesh=True,
                      forward_axis="Y", up_axis="Z")

nt = scene.node_tree
nt.nodes.clear()
rl = nt.nodes.new("CompositorNodeRLayers")
idm = nt.nodes.new("CompositorNodeIDMask")
idm.index = obj.pass_index
idm.use_antialiasing = False
nt.links.new(rl.outputs["IndexOB"], idm.inputs["ID value"])
mask_out = nt.nodes.new("CompositorNodeOutputFile")
mask_out.base_path = str(scene_dir / f"mask_{name.lower()}")
mask_out.format.file_format = "PNG"
mask_out.format.color_mode = "BW"
mask_out.file_slots[0].path = "####"
nt.links.new(idm.outputs["Alpha"], mask_out.inputs[0])

use_device(scene)
scene.cycles.samples = 1
scene.cycles.use_denoising = False
for f in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(f)
    bpy.ops.render.render()  # file-output node writes the mask; the RGB result is not saved
print("exported", name, "pass_index", obj.pass_index, "frames", scene.frame_end - scene.frame_start + 1)
