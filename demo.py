import sys
import os
import json
import argparse
import numpy as np
import imageio
import torch
import OpenEXR
import Imath
from PIL import Image
from pytorch3d.transforms import Transform3d

# import inference code
sys.path.append("notebook")
from inference import (
    Inference,
    load_image,
    load_single_mask,
    make_scene,
    ready_gaussian_for_video_rendering,
    render_video,
)
from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera


def get_args():
    parser = argparse.ArgumentParser(description='SAM 3D Objects demo with scale correction')
    # parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing image.png and mask files')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image.png')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to mask.png')
    
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for saving results')
    parser.add_argument('--pointmap_path', type=str, default=None, help='Path to points.exr from MoGe (H,W,3) XYZ in camera space; skips internal MoGe if provided')
    # parser.add_argument('--mask_index', type=int, default=0, help='Index of mask to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for inference')
    return parser.parse_args()


def main():
    args = get_args()
    
    IMAGE_PATH = args.image_path
    MASK_PATH = args.mask_path
    OUTPUT_FOLDER = args.output_dir
    
    # ============ LOAD MODEL ============
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    # ============ LOAD IMAGE & MASK ============
    image = load_image(IMAGE_PATH)
    mask = load_image(MASK_PATH)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., -1]

    # ============ LOAD PRE-COMPUTED POINTMAP (optional) ============
    pointmap_tensor = None
    if args.pointmap_path:
        print(f"Loading pre-computed pointmap: {args.pointmap_path}")
        _f = OpenEXR.InputFile(args.pointmap_path)
        _dw = _f.header()['dataWindow']
        _H = _dw.max.y - _dw.min.y + 1
        _W = _dw.max.x - _dw.min.x + 1
        _pt = Imath.PixelType(Imath.PixelType.FLOAT)
        _r = np.frombuffer(_f.channel('R', _pt), dtype=np.float32).reshape(_H, _W)
        _g = np.frombuffer(_f.channel('G', _pt), dtype=np.float32).reshape(_H, _W)
        _b = np.frombuffer(_f.channel('B', _pt), dtype=np.float32).reshape(_H, _W)
        points = np.stack([_r, _g, _b], axis=-1)  # (H, W, 3) float32, XYZ in camera space
        cam_transform = Transform3d().rotate(camera_to_pytorch3d_camera().rotation)
        pointmap_tensor = cam_transform.transform_points(torch.from_numpy(points))  # (H, W, 3)

    # ============ RUN MODEL ============
    print("Running SAM-3D inference...")
    output = inference(image, mask, seed=args.seed, pointmap=pointmap_tensor)

    # ============ SAVE OUTPUTS ============
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Gaussian Splat (.ply)
    ply_path = f"{OUTPUT_FOLDER}/gaussian.ply"
    output["gs"].save_ply(ply_path)
    print(f"✓ Saved Gaussian Splat: {ply_path}")

    # 2. GLB Mesh (.glb) - original scale
    glb_path = f"{OUTPUT_FOLDER}/mesh.glb"
    if output.get("glb") is not None:
        output["glb"].export(glb_path)
        print(f"✓ Saved GLB Mesh: {glb_path}")
    else:
        print("✗ GLB mesh not available")

    # 3. OBJ Mesh (.obj) - original scale
    obj_path = f"{OUTPUT_FOLDER}/mesh.obj"
    if output.get("glb") is not None:
        output["glb"].export(obj_path)
        print(f"✓ Saved OBJ Mesh: {obj_path}")
    else:
        print("✗ OBJ mesh not available (no GLB to convert)")

    # 4. Scaled Mesh (.obj) - in METERS (using metric scale from pose decoder)
    if output.get("glb") is not None:
        mesh = output["glb"].copy()
        scale = output["scale"].cpu().numpy().squeeze()  # (3,) metric scale from MoGe depth
        mesh.vertices = np.array(mesh.vertices) * scale
        scaled_obj_path = f"{OUTPUT_FOLDER}/mesh_scaled.obj"
        mesh.export(scaled_obj_path)
        print(f"✓ Saved Scaled OBJ Mesh (meters): {scaled_obj_path}")

    # 5. Pose Data (.json)
    pose_path = f"{OUTPUT_FOLDER}/pose.json"
    pose_data = {
        "rotation": output["rotation"].cpu().numpy().tolist() if "rotation" in output else None,
        "translation": output["translation"].cpu().numpy().tolist() if "translation" in output else None,
        "scale": output["scale"].cpu().numpy().tolist() if "scale" in output else None,
    }
    with open(pose_path, "w") as f:
        json.dump(pose_data, f, indent=2)
    print(f"✓ Saved Pose Data: {pose_path}")

    # 6. Pointmap (.npy)
    pointmap_path = f"{OUTPUT_FOLDER}/pointmap.npy"
    if "pointmap" in output:
        np.save(pointmap_path, output["pointmap"].cpu().numpy())
        print(f"✓ Saved Pointmap: {pointmap_path}")

    # 8. Pointmap Colors (.npy)
    pointmap_colors_path = f"{OUTPUT_FOLDER}/pointmap_colors.npy"
    if "pointmap_colors" in output:
        np.save(pointmap_colors_path, output["pointmap_colors"].cpu().numpy())
        print(f"✓ Saved Pointmap Colors: {pointmap_colors_path}")

    # 9. Turntable GIF (.gif)
    gif_path = f"{OUTPUT_FOLDER}/turntable.gif"
    print("Rendering turntable animation...")
    scene_gs = make_scene(output)
    scene_gs = ready_gaussian_for_video_rendering(scene_gs)
    video = render_video(
        scene_gs,
        r=1,
        fov=60,
        pitch_deg=15,
        yaw_start_deg=-45,
        resolution=512,
        num_frames=60,
    )["color"]
    imageio.mimsave(gif_path, video, format="GIF", duration=1000 / 30, loop=0)
    print(f"✓ Saved Turntable GIF: {gif_path}")

    # ============ SUMMARY ============
    print("\n" + "=" * 50)
    print("✅ ALL OUTPUTS SAVED TO:", OUTPUT_FOLDER)
    print("=" * 50)
    print("Files:")
    print(f"  - gaussian.ply      (Gaussian Splat)")
    print(f"  - mesh.glb          (GLB Mesh - original scale)")
    print(f"  - mesh.obj          (OBJ Mesh - original scale)")
    print(f"  - mesh_scaled.obj   (OBJ Mesh - metric scale in meters)")
    print(f"  - pose.json         (Rotation, Translation, Scale)")
    print(f"  - pointmap.npy      (Depth Pointmap)")
    print(f"  - pointmap_colors.npy (Pointmap Colors)")
    print(f"  - turntable.gif     (Visualization)")
    print("=" * 50)


if __name__ == '__main__':
    main()
