import os
import trimesh
import numpy as np
from PIL import Image
from io import BytesIO
import argparse
from src.utils.camera_util import FOV_to_intrinsics
import torch
import torch.nn.functional as F

# 구형 좌표계에 따른 카메라 위치를 생성하는 함수
def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    # 카메라 포즈 계산
    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws

def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 4, 4)
    """
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics

def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics

# CLI 인자 처리
parser = argparse.ArgumentParser(description="3D Object Renderer")
parser.add_argument('--input_folder', type=str, required=True, help="Folder containing OBJ, PNG, MTL files")
parser.add_argument('--output_folder', type=str, required=True, help="Folder to save the rendered images")
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder

# 카메라 설정 값
camera_settings = {
    'radius': 2.5,  # 카메라와 오브젝트 사이의 거리
    'num_views': 5,  # 카메라 시점의 수 (4개 45도 + 1개 80도)
    'fov': 30.0,  # 카메라의 시야각 (FOV)
    'image_size': 512  # 출력 이미지 크기
}

# 폴더 내 파일 검색 및 그룹화
obj_files = [f for f in os.listdir(input_folder) if f.endswith('.obj')]

# 출력 폴더가 없다면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 이미지 생성 함수
def render_and_save_images(mesh, file_name, camera_poses, output_folder, image_size):
    for i, pose in enumerate(camera_poses):
        # Trimesh로 장면 설정
        scene = trimesh.Scene(mesh)
        
        # Trimesh에서 카메라 포즈 적용 (extrinsics 변환)
        scene.camera_transform = pose.numpy()

        # 장면 렌더링 (bytes 반환)
        rendered_image = scene.save_image(resolution=(image_size, image_size))

        # bytes 데이터를 PIL 이미지로 변환
        image = Image.open(BytesIO(rendered_image))

        # 이미지 상하 반전
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # 저장 경로 설정 및 저장
        output_path = os.path.join(output_folder, f"{file_name}_view_{i}.png")
        image.save(output_path)
        print(f"Saved {output_path}")

# 구형 좌표계 카메라 포즈 설정
def setup_camera(camera_settings):
    # 카메라 포즈 생성
    azimuths = np.linspace(0, 360, camera_settings['num_views'], endpoint=False)
    
    # 4개는 40도, 1개는 85도 elevation
    elevations = np.array([40] * 4 + [85])

    camera_poses = spherical_camera_pose(azimuths, elevations, radius=camera_settings['radius'])

    # 카메라의 시야각 설정 (FOV -> Intrinsics 변환)
    intrinsics = FOV_to_intrinsics(camera_settings['fov']).numpy()

    return camera_poses, intrinsics

# 카메라 설정 적용
camera_poses, intrinsics = setup_camera(camera_settings)

# 각 OBJ 파일에 대해 처리
for obj_file in obj_files:
    base_name = os.path.splitext(obj_file)[0]
    
    # OBJ 파일 로드
    obj_path = os.path.join(input_folder, obj_file)
    mesh = trimesh.load(obj_path)
    
    # 이미지 저장
    render_and_save_images(mesh, base_name, camera_poses, output_folder, camera_settings['image_size'])

# 실행 명령어
#python 5view.py --input_folder "C:\Users\minch\Downloads\InstantMesh-main\outputs\instant-mesh-large\meshes" --output_folder "C:\Users\minch\Downloads\InstantMesh-main\outputs\instant-mesh-large\5images"
