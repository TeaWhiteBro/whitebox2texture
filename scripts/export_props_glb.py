"""Export several textured props (Blender world coords) into one GLB, each with its own albedo material.

Usage: blender -b --factory-startup -P export_props_glb.py -- OUT.glb NAME=MESH_OBJ:ALBEDO [...]
"""

import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
for item in args[1:]:
    name, rest = item.split("=", 1)
    mesh_path, albedo_path = rest.rsplit(":", 1)
    bpy.ops.wm.obj_import(filepath=mesh_path, forward_axis="Y", up_axis="Z")
    obj = bpy.context.selected_objects[0]
    obj.name = name
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(albedo_path)
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    obj.data.materials.clear()
    obj.data.materials.append(mat)
bpy.ops.export_scene.gltf(filepath=args[0], export_format="GLB", use_selection=False)
print("exported", args[0])
