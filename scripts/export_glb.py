"""Export a textured OBJ (Blender world coords) + albedo as a self-contained GLB for UE / any glTF viewer.

Usage: [OBJECT=crate] blender -b --factory-startup -P export_glb.py -- MESH_OBJ ALBEDO_PNG OUT.glb [ROUGHNESS_IMG METALLIC_IMG]
"""

import os
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
mesh_path, albedo_path, out = args[:3]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=mesh_path, forward_axis="Y", up_axis="Z")
obj = bpy.context.selected_objects[0]
name = os.environ.get("OBJECT", "house").capitalize()
obj.name = name
mat = bpy.data.materials.new(name)
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load(albedo_path)
nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.8
bsdf.inputs["Metallic"].default_value = 0.0
for path, socket in zip(args[3:5], ("Roughness", "Metallic")):
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(path)
    t.image.colorspace_settings.name = "Non-Color"
    nt.links.new(t.outputs["Color"], bsdf.inputs[socket])
obj.data.materials.clear()
obj.data.materials.append(mat)
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=False)
print("exported", out)
