"""Small OBJ helpers + Blender<->Hunyuan frame conversion shared by the texture scripts."""

import math

import numpy as np


def read_obj(path):
    """Return (V[n,3], VT[m,2] or None, F_v[f,3], F_vt[f,3] or None, header lines like mtllib)."""
    V, VT, FV, FT, header = [], [], [], [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            VT.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            corners = [c.split("/") for c in line.split()[1:]]
            for k in range(1, len(corners) - 1):  # fan-triangulate
                tri = [corners[0], corners[k], corners[k + 1]]
                FV.append([int(c[0]) - 1 for c in tri])
                if len(tri[0]) > 1 and tri[0][1]:
                    FT.append([int(c[1]) - 1 for c in tri])
        elif line.startswith(("mtllib", "usemtl")):
            header.append(line.rstrip("\n"))
    VT = np.array(VT, np.float64) if VT else None
    FT = np.array(FT, np.int64) if FT else None
    return np.array(V, np.float64), VT, np.array(FV, np.int64), FT, header


def write_obj(path, V, F, VT=None, FT=None, header=()):
    with open(path, "w") as f:
        for line in header:
            if line.startswith("mtllib"):
                f.write(line + "\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if VT is not None:
            for t in VT:
                f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        for line in header:
            if line.startswith("usemtl"):
                f.write(line + "\n")
        for i, tri in enumerate(F):
            if VT is not None:
                t = FT[i]
                f.write(f"f {tri[0]+1}/{t[0]+1} {tri[1]+1}/{t[1]+1} {tri[2]+1}/{t[2]+1}\n")
            else:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def blender_to_hunyuan(V, yaw_deg):
    """Blender world (Z-up) -> yaw about Z -> glTF-style Y-up frame that Hunyuan3D-Paint expects."""
    R = rot_z(yaw_deg)
    P = V @ R.T
    return np.stack([P[:, 0], P[:, 2], -P[:, 1]], axis=1)


def hunyuan_to_blender(V, yaw_deg):
    P = np.stack([V[:, 0], -V[:, 2], V[:, 1]], axis=1)
    return P @ rot_z(-yaw_deg).T


def object_name():
    """Which white-box object the texture scripts work on (env OBJECT, default the house)."""
    import os
    return os.environ.get("OBJECT", "house")


def mask_dir(scene_dir):
    name = object_name()
    return scene_dir / ("mask" if name == "house" else f"mask_{name}")
