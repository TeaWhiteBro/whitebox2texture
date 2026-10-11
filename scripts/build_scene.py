"""Build a UE-style white-box test scene in Blender and render everything the texture pipeline needs.

Usage: blender -b --factory-startup -P build_scene.py -- OUT_DIR
Writes OUT_DIR/{scene.blend, house.obj, cameras.json, rgb/####.png, mask/####.png, depth/####.exr}.
Blender world is Z-up, metres. The camera sweeps ~110 degrees in front of the house (never sees the back).
"""

import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycles_device import use_device  # noqa: E402

OUT = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
W, H, N, FPS = 832, 480, 81, 15
HFOV = 75.0
HOUSE_INDEX, CRATE_INDEX = 1, 2


def material(name, c1, c2=None, scale=1.0, rough=0.85):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = rough
    if c2 is None:
        bsdf.inputs["Base Color"].default_value = (*c1, 1)
        return mat
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    checker = nt.nodes.new("ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = (*c1, 1)
    checker.inputs["Color2"].default_value = (*c2, 1)
    checker.inputs["Scale"].default_value = 1.0
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    nt.links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def mesh_object(name, verts, faces, mat, index=0):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    obj.pass_index = index
    return obj


def box(x0, x1, y0, y1, z0, z1):
    v = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    f = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return v, f


def merge(parts):
    verts, faces = [], []
    for v, f in parts:
        base = len(verts)
        verts += v
        faces += [tuple(i + base for i in face) for face in f]
    return verts, faces


def gable_roof(x0, x1, y0, y1, z0, z1):
    ym = (y0 + y1) / 2
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, ym, z1), (x1, ym, z1)]
    f = [(0, 3, 2, 1), (0, 1, 5, 4), (2, 3, 4, 5), (0, 4, 3), (1, 2, 5)]
    return v, f


def build_house(mat):
    parts = [
        box(-2.0, 2.0, -1.5, 1.5, 0.0, 2.6),        # body; front door side faces -Y
        gable_roof(-2.3, 2.3, -1.8, 1.8, 2.6, 4.0),  # roof with eave overhang, ridge along X
        box(0.8, 1.3, 0.3, 0.8, 3.0, 4.5),          # chimney on the back slope
        box(-0.7, 0.7, -1.9, -1.5, 0.0, 0.18),      # front step
    ]
    obj = mesh_object("House", *merge(parts), mat, HOUSE_INDEX)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.subdivide(number_cuts=6)  # denser verts for the texturing renderers; geometry unchanged
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
    for poly in obj.data.polygons:
        poly.use_smooth = False
    obj.select_set(False)
    return obj


def build_scene():
    rng = random.Random(7)
    ground = material("Ground", (0.32, 0.32, 0.33), (0.38, 0.38, 0.39), scale=0.5)
    tan = material("BuildingTan", (0.42, 0.34, 0.26), (0.46, 0.37, 0.28), scale=0.5)
    gray = material("BuildingGray", (0.20, 0.20, 0.21), (0.23, 0.23, 0.24), scale=0.5)
    white = material("Whitebox", (0.50, 0.50, 0.50), rough=0.7)
    crate = material("Crate", (0.30, 0.30, 0.30), (0.36, 0.36, 0.36), scale=1.0)

    mesh_object("Ground", [(-60, -60, 0), (60, -60, 0), (60, 60, 0), (-60, 60, 0)], [(0, 1, 2, 3)], ground)
    house = build_house(white)
    mesh_object("Crate", *box(2.4, 3.4, -3.2, -2.2, 0.0, 1.0), crate, CRATE_INDEX)
    # back-drop blocks on three sides, like the street walls in the training scene
    for i in range(14):
        a = math.radians(-40 + 160 * i / 13) + math.pi / 2
        r = rng.uniform(16, 22)
        cx, cy = r * math.cos(a), r * math.sin(a)
        sx, sy, sz = rng.uniform(4, 8), rng.uniform(4, 8), rng.uniform(6, 18)
        mesh_object(f"Block{i}", *box(cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, 0, sz),
                    tan if i % 3 else gray)
    for i in range(8):  # blocks behind the camera so the left/right edges are not empty
        a = math.radians(200 + 140 * i / 7)
        r = rng.uniform(18, 24)
        cx, cy = r * math.cos(a), r * math.sin(a)
        mesh_object(f"BlockB{i}", *box(cx - 3, cx + 3, cy - 3, cy + 3, 0, rng.uniform(5, 14)), gray if i % 2 else tan)
    return house


def setup_world_and_light(scene):
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    sky = nt.nodes.new("ShaderNodeBackground")
    sky.inputs["Color"].default_value = (0.05, 0.25, 0.65, 1)  # saturated blue the camera sees
    sky.inputs["Strength"].default_value = 1.0
    bg.inputs["Color"].default_value = (0.55, 0.62, 0.75, 1)   # softer fill light
    bg.inputs["Strength"].default_value = 0.6
    path = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes["World Output"]
    nt.links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(bg.outputs["Background"], mix.inputs[1])
    nt.links.new(sky.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = 2.6
    sun.angle = math.radians(1.0)
    sun_obj = bpy.data.objects.new("Sun", sun)
    sun_obj.rotation_euler = (math.radians(42), 0, math.radians(-35))
    scene.collection.objects.link(sun_obj)


def setup_camera(scene):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens = 18.0 / math.tan(math.radians(HFOV / 2))
    cam_data.clip_start, cam_data.clip_end = 0.05, 300
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    target = Vector((0.0, 0.0, 1.8))
    for f in range(N):
        t = f / (N - 1)
        s = 0.5 - 0.5 * math.cos(math.pi * t)
        theta = math.radians(-70 + 110 * s)
        radius = 10.0 - 2.0 * s
        pos = Vector((radius * math.sin(theta), -radius * math.cos(theta), 1.7))
        cam.location = pos
        cam.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
    return cam


def setup_render(scene):
    scene.render.engine = "CYCLES"
    use_device(scene)
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = W, H
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 0, N - 1
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(OUT / "rgb" / "####")
    layer = scene.view_layers[0]
    layer.use_pass_z = True
    layer.use_pass_object_index = True
    scene.use_nodes = True
    nt = scene.node_tree
    rl = nt.nodes["Render Layers"]
    idm = nt.nodes.new("CompositorNodeIDMask")
    idm.index = HOUSE_INDEX
    idm.use_antialiasing = False
    nt.links.new(rl.outputs["IndexOB"], idm.inputs["ID value"])
    mask_out = nt.nodes.new("CompositorNodeOutputFile")
    mask_out.base_path = str(OUT / "mask")
    mask_out.format.file_format = "PNG"
    mask_out.format.color_mode = "BW"
    mask_out.file_slots[0].path = "####"
    nt.links.new(idm.outputs["Alpha"], mask_out.inputs[0])
    depth_out = nt.nodes.new("CompositorNodeOutputFile")
    depth_out.base_path = str(OUT / "depth")
    depth_out.format.file_format = "OPEN_EXR"
    depth_out.format.color_depth = "32"
    depth_out.file_slots[0].path = "####"
    nt.links.new(rl.outputs["Depth"], depth_out.inputs[0])


def export(scene, cam, house):
    fx = cam.data.lens / cam.data.sensor_width * W
    frames = []
    for f in range(N):
        scene.frame_set(f)
        frames.append({"frame": f, "c2w_blender": [list(r) for r in cam.matrix_world]})
    meta = {
        "width": W, "height": H, "fps": FPS, "fx": fx, "fy": fx, "cx": W / 2, "cy": H / 2,
        "convention": "Blender camera: looks down -Z, +Y up; world Z-up, metres",
        "house_pass_index": HOUSE_INDEX, "frames": frames,
    }
    (OUT / "cameras.json").write_text(json.dumps(meta, indent=1))
    bpy.ops.object.select_all(action="DESELECT")
    house.select_set(True)
    bpy.ops.wm.obj_export(filepath=str(OUT / "house.obj"), export_selected_objects=True, export_uv=True,
                          export_normals=True, export_materials=False, export_triangulated_mesh=True,
                          forward_axis="Y", up_axis="Z")
    house.select_set(False)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "scene.blend"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    house = build_scene()
    setup_world_and_light(scene)
    cam = setup_camera(scene)
    setup_render(scene)
    export(scene, cam, house)
    scene.frame_step = int(os.environ.get("FRAME_STEP", "1"))  # e.g. 40 renders a 3-frame preview
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
