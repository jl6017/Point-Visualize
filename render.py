import os
import sys
import time
import glob
import subprocess
from pathlib import Path
import numpy as np
from math import radians, cos, sin

## control pose
def apply_rpy(xyz: np.ndarray, roll=0.0, pitch=0.0, yaw=0.0, degrees=True, order="zyx"):
    """
    Intrinsic roll (x), pitch (y), yaw (z) rotation.
    order="zyx" means: first roll (x), then pitch (y), then yaw (z) in object space.
    """
    if degrees:
        roll, pitch, yaw = map(radians, (roll, pitch, yaw))

    Rx = np.array([[1,0,0],
                   [0,cos(roll),-sin(roll)],
                   [0,sin(roll), cos(roll)]], dtype=np.float32)
    Ry = np.array([[ cos(pitch),0,sin(pitch)],
                   [0,1,0],
                   [-sin(pitch),0,cos(pitch)]], dtype=np.float32)
    Rz = np.array([[cos(yaw),-sin(yaw),0],
                   [sin(yaw), cos(yaw),0],
                   [0,0,1]], dtype=np.float32)

    mats = {"x": Rx, "y": Ry, "z": Rz}
    R = np.eye(3, dtype=np.float32)
    for ax in order:     # compose in given order
        R = mats[ax] @ R
    return (xyz @ R.T).astype(np.float32)

def place_on_ground(xyz: np.ndarray, sphere_radius: float):
    # lift so the lowest point just touches z=0 plane (your ground rectangle)
    zmin = float(xyz[:,2].min())
    return xyz - np.array([0,0,zmin - sphere_radius], dtype=np.float32)


## color mismatch issue
def srgb_to_linear(c):
    # c: (..., 3) in [0,1]
    a = 0.055
    return np.where(c <= 0.04045, c/12.92, ((c + a)/(1 + a))**2.4)

def subsample_indices(n, k=None):
    if k is None or k >= n:
        return np.arange(n, dtype=np.int64)
    idx = np.random.choice(n, k, replace=False)
    np.random.shuffle(idx)
    return idx

def standardize_bbox(pcl):
    """Center to bbox center and scale to unit longest side."""
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.0
    scale = np.max(maxs - mins)
    if scale <= 0:
        scale = 1.0
    return ((pcl - center) / scale).astype(np.float32)

def load_ply_xyz_rgb(ply_path: str, points_per_object: int | None = None, assume_srgb: bool = True):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise ValueError(f"No points in {ply_path}")

    xyz = np.asarray(pcd.points, dtype=np.float32)          # (N,3)
    rgb = None
    if pcd.has_colors():
        rgb = np.asarray(pcd.colors, dtype=np.float32)      # Open3D gives [0,1]
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        rgb = np.clip(rgb, 0.0, 1.0)
        if assume_srgb:
            rgb = srgb_to_linear(rgb)                       # Mitsuba expects linear

    # --- subsample with the SAME indices for xyz/rgb ---
    idx = subsample_indices(xyz.shape[0], points_per_object)
    xyz = xyz[idx]
    if rgb is not None:
        rgb = rgb[idx]

    # --- normalize AFTER subsampling so radius is consistent with image scale ---
    xyz = standardize_bbox(xyz)

    # after normalization (and any axis remap you keep)
    xyz = apply_rpy(xyz, roll=0, pitch=90, yaw=90, degrees=True)   # e.g. lay it like glasses on table

    # --- optional axis remap like before ---
    xyz = xyz[:, [2, 0, 1]].copy()
    xyz[:, 0] *= -1.0

    return xyz, rgb

# ---- optional: Mitsuba Python for EXR->PNG conversion ----
_have_mi = False
try:
    import mitsuba as mi  # Mitsuba 3 Python API
    mi.set_variant(os.environ.get("MI_VARIANT", "cuda_ad_rgb"))
    _have_mi = True
except Exception:
    _have_mi = False  # we'll keep the EXRs if PNG conversion isn't possible

# ---- Open3D to read .ply ----
try:
    import open3d as o3d
except ImportError as e:
    print("Open3D is required: pip install open3d", file=sys.stderr)
    raise

# ----------------- geometry helpers -----------------
def standardize_bbox(pcl: np.ndarray, points_per_object: int = None):
    """
    Center to bbox center and scale so the largest bbox side becomes 1.0.
    (Matches your behavior; output in ~[-0.5, 0.5] on largest axis.)
    Optionally subsample to points_per_object.
    """
    assert pcl.ndim == 2 and pcl.shape[1] == 3, "pcl must be (N, 3)"
    if points_per_object is not None and points_per_object < pcl.shape[0]:
        pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
        np.random.shuffle(pt_indices)
        pcl = pcl[pt_indices]

    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.0
    scale = np.max(maxs - mins)
    if scale <= 0:
        scale = 1.0
    pcl = (pcl - center) / scale  # roughly inside [-0.5, 0.5] on the longest axis
    return pcl.astype(np.float32)

# def load_ply_xyz_rgb(ply_path: str):
#     """
#     Returns:
#       xyz: (N, 3) float32
#       rgb: (N, 3) float32 in [0,1] (if no color in file, returns None)
#     """
#     pcd = o3d.io.read_point_cloud(ply_path)
#     if len(pcd.points) == 0:
#         raise ValueError(f"No points found in: {ply_path}")

#     xyz = np.asarray(pcd.points, dtype=np.float32)
#     rgb = None
#     if pcd.has_colors():
#         rgb = np.asarray(pcd.colors, dtype=np.float32)
#         # Open3D colors are already [0,1]; guard NaNs
#         rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
#         rgb = np.clip(rgb, 0.0, 1.0)

#     return xyz, rgb

# ----------------- Mitsuba XML builders (v3) -----------------
def xml_header(width=1600, height=1200, fov=25, spp=256):
    # Mitsuba 3 uses snake_case property names and whitespace-separated vectors.
    return f"""<scene version="3.0.0">
    <integrator type="path">
        <integer name="max_depth" value="-1"/>
    </integrator>

    <sensor type="perspective">
        <float name="near_clip" value="0.1"/>
        <float name="far_clip" value="100"/>
        <transform name="to_world">
            <lookat origin="3 3 3" target="0 0 0" up="0 0 1"/>
        </transform>
        <float name="fov" value="{fov}"/>

        <sampler type="independent">
            <integer name="sample_count" value="{spp}"/>
        </sampler>

        <film type="hdrfilm">
            <integer name="width" value="{width}"/>
            <integer name="height" value="{height}"/>
            <rfilter type="gaussian"/>
            <boolean name="sample_border" value="true"/>
        </film>
    </sensor>

    <!-- a glossy ground material (unused by spheres which carry their own bsdf) -->
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="int_ior" value="1.46"/>
        <rgb name="diffuse_reflectance" value="1 1 1"/>
    </bsdf>
"""

def xml_sphere(x, y, z, r, rgb=None, albedo_scale=0.85):  # <— tweak 0.7–0.95
    if rgb is None: rgb = (1.0, 1.0, 1.0)
    rr, gg, bb = [max(0.0, min(1.0, c * albedo_scale)) for c in rgb]
    return f"""
    <shape type="sphere">
        <float name="radius" value="{r}"/>
        <transform name="to_world">
            <translate x="{x}" y="{y}" z="{z}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{rr} {gg} {bb}"/>
        </bsdf>
    </shape>
"""

def xml_tail():
    # Ground plane and an area light (rectangle with emitter)
    return """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="to_world">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>

    <shape type="rectangle">
        <transform name="to_world">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4 4 20" target="0 0 0" up="0 0 1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6 6 6"/>
        </emitter>
    </shape>
</scene>
"""

def build_scene_xml(xyz: np.ndarray, rgb: np.ndarray | None, sphere_radius=0.025,
                    width=1600, height=1200, fov=25, spp=256):
    parts = [xml_header(width, height, fov, spp)]
    # small z-lift so spheres don't z-fight with ground when normalized close to -0.5
    z_lift = 0.5 * sphere_radius
    for i in range(xyz.shape[0]):
        x, y, z = float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2] + z_lift)
        color = None if rgb is None else (float(rgb[i, 0]), float(rgb[i, 1]), float(rgb[i, 2]))
        parts.append(xml_sphere(x, y, z, sphere_radius, color))
    parts.append(xml_tail())
    return "".join(parts)

# ----------------- rendering pipeline -----------------
def render_xml_with_mitsuba(xml_path: Path, exr_path: Path, mitsuba_exe="mitsuba", variant="cuda_ad_rgb"):
    """
    Renders the given XML to EXR using the Mitsuba CLI.
    """
    cmd = [mitsuba_exe, "-m", variant, "-o", str(exr_path), str(xml_path)]
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def convert_exr_to_png(exr_path: Path, png_path: Path):
    """
    Convert EXR->PNG using Mitsuba Bitmap if available; otherwise skip.
    """
    if not _have_mi:
        print(f"[warn] Mitsuba Python not available, keeping EXR only: {exr_path.name}")
        return
    bmp = mi.Bitmap(str(exr_path))
    out = bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
    out.write(str(png_path))

def render_ply_folder(
    in_folder: str,
    out_root: str = "out",
    points_per_object: int | None = None,
    sphere_radius: float = 0.025,
    img_w: int = 1600,
    img_h: int = 1200,
    fov: float = 25.0,
    spp: int = 256,
    mitsuba_exe: str = "mitsuba",
    mitsuba_variant: str = "cuda_ad_rgb",
):
    """
    Loads every .ply in in_folder, uses PLY colors (if present), normalizes geometry,
    writes a Mitsuba 3 XML, renders EXR, and converts to PNG (if possible).
    Outputs into out/<timestamp>/.
    """
    in_folder = Path(in_folder)
    ply_files = sorted(glob.glob(str(in_folder / "*.ply")))
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in: {in_folder}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Found {len(ply_files)} .ply files.")
    for idx, ply_path in enumerate(ply_files, 1):
        ply_path = Path(ply_path)
        name = ply_path.stem
        print(f"[{idx}/{len(ply_files)}] {name}")

        # Load and normalize
        # xyz, rgb = load_ply_xyz_rgb(str(ply_path))
        xyz, rgb = load_ply_xyz_rgb(
            str(ply_path),
            points_per_object=points_per_object,
            assume_srgb=True,      # set False if your PLY colors are already linear
        )
        xyz = standardize_bbox(xyz, points_per_object=points_per_object)

        # Optional axis remap like your original code (swap to z,x,y; flip x); comment out if not needed
        xyz = xyz[:, [2, 0, 1]].copy()
        xyz[:, 0] *= -1.0

        # If colors exist but count mismatches after subsampling/reindexing, trim
        if rgb is not None and rgb.shape[0] != xyz.shape[0]:
            # If you needed to preserve sampling correspondence, you would sample both together.
            # Here we just trim/pad conservatively:
            m = min(rgb.shape[0], xyz.shape[0])
            rgb = rgb[:m]
            xyz = xyz[:m]

        # Build XML
        xml = build_scene_xml(
            xyz, rgb,
            sphere_radius=sphere_radius,
            width=img_w, height=img_h, fov=fov, spp=spp
        )

        # Paths
        xml_path = out_dir / f"{name}.xml"
        exr_path = out_dir / f"{name}.exr"
        png_path = out_dir / f"{name}.png"

        # Write XML
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml)

        # Render
        try:
            render_xml_with_mitsuba(xml_path, exr_path, mitsuba_exe=mitsuba_exe, variant=mitsuba_variant)
        except subprocess.CalledProcessError as e:
            print(f"[error] Mitsuba render failed for {name}: {e}", file=sys.stderr)
            continue

        # Convert to PNG (optional)
        try:
            convert_exr_to_png(exr_path, png_path)
        except Exception as e:
            print(f"[warn] PNG conversion failed for {name}: {e}", file=sys.stderr)

    print(f"[done] Outputs saved in: {out_dir}")

# ----------------- CLI -----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Render a folder of PLY point clouds with Mitsuba 3")
    parser.add_argument("in_folder", type=str, help="Folder containing .ply files")
    parser.add_argument("--out_root", type=str, default="out", help="Root output folder (a timestamped subfolder will be created)")
    parser.add_argument("--points", type=int, default=None, help="Optional subsample count per object")
    parser.add_argument("--radius", type=float, default=0.025, help="Sphere radius per point")
    parser.add_argument("--w", type=int, default=1600, help="Image width")
    parser.add_argument("--h", type=int, default=1200, help="Image height")
    parser.add_argument("--fov", type=float, default=25.0, help="Perspective fov in degrees")
    parser.add_argument("--spp", type=int, default=256, help="Samples per pixel")
    parser.add_argument("--mitsuba", type=str, default="mitsuba", help="Path to mitsuba executable if not on PATH")
    parser.add_argument("--variant", type=str, default="cuda_ad_rgb", help="Mitsuba variant, e.g., cuda_ad_rgb / llvm_ad_rgb")

    args = parser.parse_args()

    render_ply_folder(
        args.in_folder,
        out_root=args.out_root,
        points_per_object=args.points,
        sphere_radius=args.radius,
        img_w=args.w,
        img_h=args.h,
        fov=args.fov,
        spp=args.spp,
        mitsuba_exe=args.mitsuba,
        mitsuba_variant=args.variant,
    )
