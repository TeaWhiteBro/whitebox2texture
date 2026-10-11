"""Third-person white-box plaza (like the GameRender training clips): a mannequin runs across a small plaza, the camera
follows behind it and swings from its left to its right, passing a few props that the texture pipeline then paints.

Usage: blender -b --factory-startup -P build_street_scene.py -- OUT_DIR
Writes OUT_DIR/{scene.blend, cameras.json, rgb/, depth/ (planar EXR), mask_<asset>/, mask_character/, <asset>.obj}
for the assets shop, crates, barrel. Blender world is Z-up, metres; the character runs along +Y.
"""

import json
import math
import os
import random
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycles_device import use_device  # noqa: E402

OUT = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
W, H, N, FPS = 832, 480, 81, 15
HFOV = 80.0
SPEED = 1.6          # m/s along +Y (a jog keeps the props in view longer)
STRIDE_FRAMES = 11   # one full running cycle (two steps)
ASSETS = {"Shop": 1, "Crates": 2, "Barrel": 3}
CHARACTER_INDEX = 9


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


def mesh_object(name, verts, faces, mat, index=0, parent=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    obj.pass_index = index
    obj.parent = parent
    return obj


def box(x0, x1, y0, y1, z0, z1):
    v = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    f = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return v, f


def transformed(part, matrix):
    v, f = part
    return [tuple(matrix @ Vector(p)) for p in v], f


def cylinder(cx, cy, r, z0, z1, segments=28):
    v = [(cx + r * math.cos(2 * math.pi * k / segments), cy + r * math.sin(2 * math.pi * k / segments), z)
         for z in (z0, z1) for k in range(segments)]
    v += [(cx, cy, z0), (cx, cy, z1)]
    b, t = 2 * segments, 2 * segments + 1
    f = [(k, (k + 1) % segments, segments + (k + 1) % segments, segments + k) for k in range(segments)]
    f += [((k + 1) % segments, k, b) for k in range(segments)]
    f += [(segments + k, segments + (k + 1) % segments, t) for k in range(segments)]
    return v, f


def merge(parts):
    verts, faces = [], []
    for v, f in parts:
        base = len(verts)
        verts += v
        faces += [tuple(i + base for i in face) for face in f]
    return verts, faces


def prepare_asset(obj):
    """Denser triangles + smart UVs, same treatment as the house; geometry unchanged."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.mesh.subdivide(number_cuts=4)
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
    for poly in obj.data.polygons:
        poly.use_smooth = False
    obj.select_set(False)


def placed(part, x, y, yaw_deg):
    return transformed(part, Matrix.Translation((x, y, 0)) @ Matrix.Rotation(math.radians(yaw_deg), 4, "Z"))


def build_props():
    """Props sit beside the run line and are turned toward the approaching camera, so they stay in view for long."""
    prop = material("PropWhite", (0.55, 0.55, 0.55), rough=0.7)
    shop_parts = [  # local frame: shop front faces +X, centre at the origin
        box(-1.75, 1.75, -2.5, 2.5, 0.0, 3.4),
        transformed(box(-0.05, 1.05, -2.2, 2.2, -0.04, 0.04),                       # awning sloping down to the plaza
                    Matrix.Translation((1.75, 0, 2.8)) @ Matrix.Rotation(math.radians(22), 4, "Y")),
        box(1.70, 1.93, -1.5, 1.5, 2.85, 3.3),                                      # sign board above the awning
        box(1.70, 1.85, -0.6, 0.6, 0.0, 2.2),                                       # door frame (inner faces hidden)
        box(1.70, 1.80, -2.1, -1.1, 0.0, 0.9),                                      # counter
    ]
    shop = mesh_object("Shop", *merge([placed(p, -5.0, 9.5, -40) for p in shop_parts]), prop, ASSETS["Shop"])
    crate_parts = [
        box(-0.5, 0.5, -0.5, 0.5, 0.0, 1.0),
        transformed(box(-0.35, 0.35, -0.35, 0.35, 0.0, 0.7),
                    Matrix.Translation((0.0, 0.0, 1.0)) @ Matrix.Rotation(math.radians(25), 4, "Z")),
        box(-0.4, 0.4, 0.65, 1.45, 0.0, 0.8),
    ]
    crates = mesh_object("Crates", *merge([placed(p, 3.8, 3.8, 20) for p in crate_parts]), prop, ASSETS["Crates"])
    barrel = mesh_object("Barrel", *cylinder(-2.3, 3.2, 0.42, 0.0, 1.15), prop, ASSETS["Barrel"])
    for obj in (shop, crates, barrel):
        prepare_asset(obj)
    return [shop, crates, barrel]


def build_street():
    rng = random.Random(3)
    ground = material("Ground", (0.30, 0.30, 0.31), (0.36, 0.36, 0.37), scale=0.5)
    tan = material("BuildingTan", (0.42, 0.34, 0.26), (0.46, 0.37, 0.28), scale=0.5)
    gray = material("BuildingGray", (0.18, 0.18, 0.19), (0.22, 0.22, 0.23), scale=0.5)
    mesh_object("Ground", [(-80, -80, 0), (80, -80, 0), (80, 80, 0), (-80, 80, 0)], [(0, 1, 2, 3)], ground)
    blocks = [  # (x0, x1, y0, y1, h, material): plaza edges, well away from the run line
        (-20, -10.5, -8, 6, 9, gray), (-18, -10.5, 6.5, 18, 7, tan), (-20, -9, 18.5, 30, 11, gray),
        (9, 17, -8, 4, 7, tan), (8.5, 15, 4.5, 16, 8, gray), (9, 18, 16.5, 30, 10, tan),
        (-9, 9, 31, 34, 8, gray),
    ]
    for i, (x0, x1, y0, y1, h, mat) in enumerate(blocks):
        mesh_object(f"Block{i}", *box(x0, x1, y0, y1, 0, h), mat)
    for i in range(10):  # taller skyline further out, like the training clips
        side = -1 if i % 2 else 1
        x = side * rng.uniform(22, 32)
        y = rng.uniform(-10, 45)
        mesh_object(f"Far{i}", *box(x - 4, x + 4, y - 4, y + 4, 0, rng.uniform(14, 32)), tan if i % 3 else gray)


def segment(name, length, radius, mat, parent, offset, up=False, depth=1.0):
    """Ellipsoid limb whose origin is the joint; extends -Z (or +Z if up) by `length`; depth scales its Y radius."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=1.0)
    sign = 1 if up else -1
    for v in bm.verts:
        v.co = Vector((v.co.x * radius, v.co.y * radius * depth, sign * (v.co.z + 1) * length / 2))
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    obj.pass_index = CHARACTER_INDEX
    obj.parent = parent
    obj.location = offset
    obj.rotation_mode = "XYZ"
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


def joint(name, radius, mat, parent, offset):
    obj = segment(name, 2 * radius, radius, mat, parent, (offset[0], offset[1], offset[2] + radius))
    return obj


def build_character():
    """UE-mannequin-like figure: white segments with dark joints, rigid hierarchy animated with a run cycle."""
    white = material("MannequinWhite", (0.8, 0.8, 0.8), rough=0.35)
    dark = material("MannequinJoint", (0.05, 0.05, 0.05), rough=0.5)
    root = bpy.data.objects.new("Character", None)
    bpy.context.scene.collection.objects.link(root)
    pelvis = segment("Pelvis", 0.22, 0.16, white, root, (0, 0, 0.97), up=False, depth=0.7)
    pelvis.location = (0, 0, 0.97)
    torso = segment("Torso", 0.55, 0.19, white, pelvis, (0, 0, 0.02), up=True, depth=0.62)
    joint("Neck", 0.05, dark, torso, (0, 0, 0.52))
    head = segment("Head", 0.25, 0.115, white, torso, (0, 0, 0.58), up=True)
    parts = {"pelvis": pelvis, "torso": torso, "head": head}
    for side, sx in (("L", -1), ("R", 1)):
        upper = segment(f"UpperArm{side}", 0.30, 0.055, white, torso, (sx * 0.22, 0, 0.47))
        joint(f"Shoulder{side}", 0.06, dark, torso, (sx * 0.22, 0, 0.41))
        fore = segment(f"Forearm{side}", 0.27, 0.045, white, upper, (0, 0, -0.30))
        joint(f"Elbow{side}", 0.045, dark, upper, (0, 0, -0.345))
        segment(f"Hand{side}", 0.10, 0.045, white, fore, (0, 0, -0.27))
        thigh = segment(f"Thigh{side}", 0.45, 0.085, white, pelvis, (sx * 0.10, 0, -0.06))
        shin = segment(f"Shin{side}", 0.44, 0.065, white, thigh, (0, 0, -0.45))
        joint(f"Knee{side}", 0.06, dark, thigh, (0, 0, -0.51))
        foot = bpy.data.objects.new(f"Foot{side}", bpy.data.meshes.new(f"Foot{side}"))
        v, f = box(-0.05, 0.05, -0.06, 0.18, -0.05, 0.03)
        foot.data.from_pydata(v, [], f)
        foot.data.materials.append(dark)
        foot.pass_index = CHARACTER_INDEX
        bpy.context.scene.collection.objects.link(foot)
        foot.parent = shin
        foot.location = (0, 0, -0.44)
        parts[f"upper{side}"], parts[f"fore{side}"] = upper, fore
        parts[f"thigh{side}"], parts[f"shin{side}"] = thigh, shin
    for f in range(N):
        p = 2 * math.pi * f / STRIDE_FRAMES
        root.location = (0, SPEED * f / FPS, 0)
        root.keyframe_insert("location", frame=f)
        pelvis.location = (0, 0, 0.97 + 0.035 * math.cos(2 * p))
        pelvis.rotation_euler = (0, 0, 0.12 * math.sin(p))
        torso.rotation_euler = (-0.18, 0, -0.2 * math.sin(p))
        head.rotation_euler = (0.12, 0, 0.15 * math.sin(p))
        for side, phase in (("L", 0.0), ("R", math.pi)):
            q = p + phase
            parts[f"thigh{side}"].rotation_euler = (0.65 * math.sin(q), 0, 0)
            parts[f"shin{side}"].rotation_euler = (-(0.25 + 0.95 * max(0.0, math.sin(q - 1.6))), 0, 0)
            parts[f"upper{side}"].rotation_euler = (-0.55 * math.sin(q), 0, 0)
            parts[f"fore{side}"].rotation_euler = (1.25, 0, 0)
        for obj in [pelvis, torso, head] + [parts[k] for k in parts if k[:-1] in ("upper", "fore", "thigh", "shin")]:
            obj.keyframe_insert("rotation_euler", frame=f)
        pelvis.keyframe_insert("location", frame=f)
    return root


def setup_world_and_light(scene):
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    sky = nt.nodes.new("ShaderNodeBackground")
    sky.inputs["Color"].default_value = (0.05, 0.25, 0.65, 1)
    bg.inputs["Color"].default_value = (0.55, 0.62, 0.75, 1)
    bg.inputs["Strength"].default_value = 0.6
    path = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(bg.outputs["Background"], mix.inputs[1])
    nt.links.new(sky.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], nt.nodes["World Output"].inputs["Surface"])
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = 2.6
    sun.angle = math.radians(1.0)
    sun_obj = bpy.data.objects.new("Sun", sun)
    sun_obj.rotation_euler = (math.radians(40), 0, math.radians(30))
    scene.collection.objects.link(sun_obj)


def setup_camera(scene):
    """Follow cam 3.6 m behind the runner, swinging from 20 deg left-behind to 20 deg right-behind."""
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens = 18.0 / math.tan(math.radians(HFOV / 2))
    cam_data.clip_start, cam_data.clip_end = 0.05, 300
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    for f in range(N):
        s = 0.5 - 0.5 * math.cos(math.pi * f / (N - 1))
        phi = math.radians(-20 + 40 * s)
        c = Vector((0, SPEED * f / FPS, 0))
        pos = c + Vector((3.6 * math.sin(phi), -3.6 * math.cos(phi), 1.9))
        target = c + Vector((0, 1.2, 1.15))
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
    for name, index in list(ASSETS.items()) + [("Character", CHARACTER_INDEX)]:
        idm = nt.nodes.new("CompositorNodeIDMask")
        idm.index = index
        idm.use_antialiasing = False
        nt.links.new(rl.outputs["IndexOB"], idm.inputs["ID value"])
        out = nt.nodes.new("CompositorNodeOutputFile")
        out.base_path = str(OUT / f"mask_{name.lower()}")
        out.format.file_format = "PNG"
        out.format.color_mode = "BW"
        out.file_slots[0].path = "####"
        nt.links.new(idm.outputs["Alpha"], out.inputs[0])
    depth_out = nt.nodes.new("CompositorNodeOutputFile")
    depth_out.base_path = str(OUT / "depth")
    depth_out.format.file_format = "OPEN_EXR"
    depth_out.format.color_depth = "32"
    depth_out.file_slots[0].path = "####"
    nt.links.new(rl.outputs["Depth"], depth_out.inputs[0])


def export(scene, cam, assets):
    fx = cam.data.lens / cam.data.sensor_width * W
    frames = []
    for f in range(N):
        scene.frame_set(f)
        frames.append({"frame": f, "c2w_blender": [list(r) for r in cam.matrix_world]})
    meta = {
        "width": W, "height": H, "fps": FPS, "fx": fx, "fy": fx, "cx": W / 2, "cy": H / 2,
        "convention": "Blender camera: looks down -Z, +Y up; world Z-up, metres",
        "assets": ASSETS, "character_pass_index": CHARACTER_INDEX, "frames": frames,
    }
    (OUT / "cameras.json").write_text(json.dumps(meta, indent=1))
    for obj in assets:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.wm.obj_export(filepath=str(OUT / f"{obj.name.lower()}.obj"), export_selected_objects=True,
                              export_uv=True, export_normals=True, export_materials=False,
                              export_triangulated_mesh=True, forward_axis="Y", up_axis="Z")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "scene.blend"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    build_street()
    assets = build_props()
    build_character()
    setup_world_and_light(scene)
    cam = setup_camera(scene)
    setup_render(scene)
    export(scene, cam, assets)
    scene.frame_step = int(os.environ.get("FRAME_STEP", "1"))
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
