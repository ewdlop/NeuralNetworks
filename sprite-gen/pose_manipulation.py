#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
姿勢操作模組
用於精確控制和修改 OpenPose 關鍵點來生成動畫序列
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw

# OpenPose 關鍵點索引定義 (COCO 格式)
POSE_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2, 
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# 關鍵點連接定義 (用於繪製骨架)
POSE_CONNECTIONS = [
    # 頭部
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear'),
    
    # 軀幹
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    
    # 左臂
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    
    # 右臂
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    
    # 左腿
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    
    # 右腿
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
]

@dataclass
class KeyPoint:
    """關鍵點類"""
    x: float
    y: float
    confidence: float = 1.0
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def distance_to(self, other: 'KeyPoint') -> float:
        """計算到另一個關鍵點的距離"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Pose:
    """姿勢類，包含所有關鍵點"""
    keypoints: Dict[str, KeyPoint]
    image_size: Tuple[int, int] = (512, 512)
    
    def get_keypoint(self, name: str) -> Optional[KeyPoint]:
        """獲取指定的關鍵點"""
        return self.keypoints.get(name)
    
    def set_keypoint(self, name: str, point: KeyPoint):
        """設置關鍵點"""
        self.keypoints[name] = point
    
    def get_center(self) -> KeyPoint:
        """獲取姿勢中心點"""
        valid_points = [p for p in self.keypoints.values() if p.confidence > 0]
        if not valid_points:
            return KeyPoint(self.image_size[0]/2, self.image_size[1]/2)
        
        avg_x = sum(p.x for p in valid_points) / len(valid_points)
        avg_y = sum(p.y for p in valid_points) / len(valid_points)
        return KeyPoint(avg_x, avg_y)
    
    def scale(self, factor: float, center: Optional[KeyPoint] = None):
        """縮放姿勢"""
        if center is None:
            center = self.get_center()
        
        for point in self.keypoints.values():
            if point.confidence > 0:
                point.x = center.x + (point.x - center.x) * factor
                point.y = center.y + (point.y - center.y) * factor
    
    def translate(self, dx: float, dy: float):
        """平移姿勢"""
        for point in self.keypoints.values():
            if point.confidence > 0:
                point.x += dx
                point.y += dy
    
    def rotate(self, angle: float, center: Optional[KeyPoint] = None):
        """旋轉姿勢"""
        if center is None:
            center = self.get_center()
        
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        for point in self.keypoints.values():
            if point.confidence > 0:
                # 相對於中心的座標
                rel_x = point.x - center.x
                rel_y = point.y - center.y
                
                # 旋轉
                new_x = rel_x * cos_a - rel_y * sin_a
                new_y = rel_x * sin_a + rel_y * cos_a
                
                # 轉回絕對座標
                point.x = center.x + new_x
                point.y = center.y + new_y
    
    def copy(self) -> 'Pose':
        """複製姿勢"""
        new_keypoints = {}
        for name, point in self.keypoints.items():
            new_keypoints[name] = KeyPoint(point.x, point.y, point.confidence)
        return Pose(new_keypoints, self.image_size)
    
    def to_openpose_format(self) -> np.ndarray:
        """轉換為 OpenPose 格式的數組"""
        pose_array = np.zeros((17, 3))  # 17個關鍵點，每個3個值(x, y, confidence)
        
        for name, idx in POSE_KEYPOINTS.items():
            if name in self.keypoints:
                point = self.keypoints[name]
                pose_array[idx] = [point.x, point.y, point.confidence]
        
        return pose_array

class PoseManipulator:
    """姿勢操作器"""
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.image_size = image_size
    
    def extract_pose_from_openpose(self, openpose_image: np.ndarray) -> Pose:
        """從 OpenPose 圖像中提取姿勢關鍵點"""
        # 這裡需要實現從 OpenPose 圖像中提取關鍵點的邏輯
        # 簡化版本：假設輸入已經是關鍵點數組
        
        if openpose_image.shape[0] == 17 and openpose_image.shape[1] >= 2:
            # 如果輸入是關鍵點數組
            return self._array_to_pose(openpose_image)
        else:
            # 如果輸入是圖像，需要進行關鍵點檢測
            return self._detect_keypoints_from_image(openpose_image)
    
    def _array_to_pose(self, keypoints_array: np.ndarray) -> Pose:
        """將關鍵點數組轉換為 Pose 對象"""
        keypoints = {}
        
        for name, idx in POSE_KEYPOINTS.items():
            if idx < len(keypoints_array):
                x, y = keypoints_array[idx][:2]
                confidence = keypoints_array[idx][2] if keypoints_array.shape[1] > 2 else 1.0
                keypoints[name] = KeyPoint(x, y, confidence)
        
        return Pose(keypoints, self.image_size)
    
    def _detect_keypoints_from_image(self, image: np.ndarray) -> Pose:
        """從圖像中檢測關鍵點（簡化版本）"""
        # 這裡應該實現實際的關鍵點檢測
        # 簡化版本：返回預設姿勢
        return self.create_default_pose()
    
    def create_default_pose(self) -> Pose:
        """創建預設的標準姿勢"""
        w, h = self.image_size
        center_x, center_y = w // 2, h // 2
        
        # 創建一個標準的站立姿勢
        keypoints = {
            'nose': KeyPoint(center_x, center_y - 120),
            'left_eye': KeyPoint(center_x - 15, center_y - 130),
            'right_eye': KeyPoint(center_x + 15, center_y - 130),
            'left_ear': KeyPoint(center_x - 25, center_y - 125),
            'right_ear': KeyPoint(center_x + 25, center_y - 125),
            
            'left_shoulder': KeyPoint(center_x - 40, center_y - 80),
            'right_shoulder': KeyPoint(center_x + 40, center_y - 80),
            
            'left_elbow': KeyPoint(center_x - 60, center_y - 30),
            'right_elbow': KeyPoint(center_x + 60, center_y - 30),
            
            'left_wrist': KeyPoint(center_x - 70, center_y + 10),
            'right_wrist': KeyPoint(center_x + 70, center_y + 10),
            
            'left_hip': KeyPoint(center_x - 25, center_y + 20),
            'right_hip': KeyPoint(center_x + 25, center_y + 20),
            
            'left_knee': KeyPoint(center_x - 30, center_y + 80),
            'right_knee': KeyPoint(center_x + 30, center_y + 80),
            
            'left_ankle': KeyPoint(center_x - 35, center_y + 140),
            'right_ankle': KeyPoint(center_x + 35, center_y + 140),
        }
        
        return Pose(keypoints, self.image_size)
    
    def create_walking_animation(self, base_pose: Pose, steps: int = 8) -> List[Pose]:
        """創建行走動畫序列"""
        poses = []
        
        for i in range(steps):
            # 計算行走週期
            cycle = (i / steps) * 2 * math.pi
            
            # 複製基礎姿勢
            pose = base_pose.copy()
            
            # 行走動畫的關鍵：腿部和臂部的交替運動
            
            # 腿部運動
            left_leg_offset = math.sin(cycle) * 30
            right_leg_offset = math.sin(cycle + math.pi) * 30
            
            # 調整膝蓋位置
            if 'left_knee' in pose.keypoints:
                pose.keypoints['left_knee'].x += left_leg_offset * 0.5
                pose.keypoints['left_knee'].y += abs(left_leg_offset) * 0.3
            
            if 'right_knee' in pose.keypoints:
                pose.keypoints['right_knee'].x += right_leg_offset * 0.5
                pose.keypoints['right_knee'].y += abs(right_leg_offset) * 0.3
            
            # 調整腳踝位置
            if 'left_ankle' in pose.keypoints:
                pose.keypoints['left_ankle'].x += left_leg_offset
            
            if 'right_ankle' in pose.keypoints:
                pose.keypoints['right_ankle'].x += right_leg_offset
            
            # 臂部運動（與腿部相反）
            left_arm_offset = math.sin(cycle + math.pi) * 20
            right_arm_offset = math.sin(cycle) * 20
            
            # 調整手肘位置
            if 'left_elbow' in pose.keypoints:
                pose.keypoints['left_elbow'].x += left_arm_offset * 0.7
                pose.keypoints['left_elbow'].y += abs(left_arm_offset) * 0.3
            
            if 'right_elbow' in pose.keypoints:
                pose.keypoints['right_elbow'].x += right_arm_offset * 0.7
                pose.keypoints['right_elbow'].y += abs(right_arm_offset) * 0.3
            
            # 調整手腕位置
            if 'left_wrist' in pose.keypoints:
                pose.keypoints['left_wrist'].x += left_arm_offset
                pose.keypoints['left_wrist'].y += left_arm_offset * 0.5
            
            if 'right_wrist' in pose.keypoints:
                pose.keypoints['right_wrist'].x += right_arm_offset
                pose.keypoints['right_wrist'].y += right_arm_offset * 0.5
            
            # 輕微的身體搖擺
            body_sway = math.sin(cycle * 2) * 5
            torso_points = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
            for point_name in torso_points:
                if point_name in pose.keypoints:
                    pose.keypoints[point_name].x += body_sway
            
            poses.append(pose)
        
        return poses
    
    def create_jumping_animation(self, base_pose: Pose, steps: int = 6) -> List[Pose]:
        """創建跳躍動畫序列"""
        poses = []
        
        # 跳躍階段：準備(2) -> 空中(2) -> 落地(2)
        for i in range(steps):
            pose = base_pose.copy()
            
            if i < 2:  # 準備階段 - 蹲下
                progress = i / 2.0
                crouch_amount = progress * 30
                
                # 降低身體高度
                for point_name in pose.keypoints:
                    if 'knee' in point_name or 'ankle' in point_name:
                        pose.keypoints[point_name].y += crouch_amount
                    elif point_name in ['left_hip', 'right_hip']:
                        pose.keypoints[point_name].y += crouch_amount * 0.7
                
                # 手臂向後擺動準備
                if 'left_wrist' in pose.keypoints:
                    pose.keypoints['left_wrist'].x -= progress * 20
                    pose.keypoints['left_wrist'].y += progress * 15
                if 'right_wrist' in pose.keypoints:
                    pose.keypoints['right_wrist'].x -= progress * 20
                    pose.keypoints['right_wrist'].y += progress * 15
                    
            elif i < 4:  # 空中階段 - 跳躍
                progress = (i - 2) / 2.0
                jump_height = 40 * (1 - (progress - 0.5)**2 * 4)  # 拋物線
                
                # 整體上升
                for point_name in pose.keypoints:
                    pose.keypoints[point_name].y -= jump_height
                
                # 腿部彎曲
                knee_bend = progress * 40
                if 'left_knee' in pose.keypoints:
                    pose.keypoints['left_knee'].y -= knee_bend
                if 'right_knee' in pose.keypoints:
                    pose.keypoints['right_knee'].y -= knee_bend
                
                # 手臂向上擺動
                arm_raise = progress * 30
                if 'left_wrist' in pose.keypoints:
                    pose.keypoints['left_wrist'].y -= arm_raise
                if 'right_wrist' in pose.keypoints:
                    pose.keypoints['right_wrist'].y -= arm_raise
                    
            else:  # 落地階段
                progress = (i - 4) / 2.0
                land_amount = progress * 20
                
                # 輕微下沉
                for point_name in pose.keypoints:
                    if 'knee' in point_name or 'ankle' in point_name:
                        pose.keypoints[point_name].y += land_amount
                
                # 手臂放下
                if 'left_wrist' in pose.keypoints:
                    pose.keypoints['left_wrist'].y += progress * 20
                if 'right_wrist' in pose.keypoints:
                    pose.keypoints['right_wrist'].y += progress * 20
            
            poses.append(pose)
        
        return poses
    
    def create_attack_animation(self, base_pose: Pose, steps: int = 4) -> List[Pose]:
        """創建攻擊動畫序列"""
        poses = []
        
        for i in range(steps):
            pose = base_pose.copy()
            
            if i < 2:  # 準備和出擊
                progress = i / 2.0
                
                # 右手攻擊動作
                if 'right_shoulder' in pose.keypoints and 'right_elbow' in pose.keypoints:
                    # 肩膀前移
                    pose.keypoints['right_shoulder'].x += progress * 15
                    
                    # 手肘伸展
                    elbow_extend = progress * 50
                    pose.keypoints['right_elbow'].x += elbow_extend
                    pose.keypoints['right_elbow'].y -= progress * 10
                
                if 'right_wrist' in pose.keypoints:
                    # 手腕向前伸出
                    pose.keypoints['right_wrist'].x += progress * 80
                    pose.keypoints['right_wrist'].y -= progress * 20
                
                # 身體前傾
                torso_lean = progress * 10
                torso_points = ['nose', 'left_shoulder', 'right_shoulder']
                for point_name in torso_points:
                    if point_name in pose.keypoints:
                        pose.keypoints[point_name].x += torso_lean
                
                # 左腳向前穩定
                if 'left_ankle' in pose.keypoints:
                    pose.keypoints['left_ankle'].x += progress * 15
                    
            else:  # 收回
                progress = (i - 2) / 2.0
                return_factor = 1 - progress
                
                # 逐漸回到原位
                if 'right_shoulder' in pose.keypoints:
                    pose.keypoints['right_shoulder'].x += return_factor * 15
                
                if 'right_elbow' in pose.keypoints:
                    pose.keypoints['right_elbow'].x += return_factor * 50
                    pose.keypoints['right_elbow'].y -= return_factor * 10
                
                if 'right_wrist' in pose.keypoints:
                    pose.keypoints['right_wrist'].x += return_factor * 80
                    pose.keypoints['right_wrist'].y -= return_factor * 20
                
                # 身體回正
                torso_lean = return_factor * 10
                torso_points = ['nose', 'left_shoulder', 'right_shoulder']
                for point_name in torso_points:
                    if point_name in pose.keypoints:
                        pose.keypoints[point_name].x += torso_lean
            
            poses.append(pose)
        
        return poses
    
    def render_pose_to_image(self, pose: Pose, 
                           line_color: Tuple[int, int, int] = (255, 255, 255),
                           point_color: Tuple[int, int, int] = (0, 255, 0),
                           background_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
        """將姿勢渲染為圖像"""
        
        # 創建畫布
        image = Image.new('RGB', pose.image_size, background_color)
        draw = ImageDraw.Draw(image)
        
        # 繪製連接線
        for connection in POSE_CONNECTIONS:
            point1_name, point2_name = connection
            
            point1 = pose.get_keypoint(point1_name)
            point2 = pose.get_keypoint(point2_name)
            
            if point1 and point2 and point1.confidence > 0 and point2.confidence > 0:
                draw.line([point1.to_tuple(), point2.to_tuple()], 
                         fill=line_color, width=3)
        
        # 繪製關鍵點
        for point in pose.keypoints.values():
            if point.confidence > 0:
                x, y = point.to_tuple()
                radius = 4
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=point_color, outline=line_color)
        
        return image
    
    def save_animation_sequence(self, poses: List[Pose], output_dir: str, prefix: str = "frame"):
        """保存動畫序列"""
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, pose in enumerate(poses):
            image = self.render_pose_to_image(pose)
            image.save(output_path / f"{prefix}_{i:03d}.png")
        
        # 保存元數據
        metadata = {
            "frame_count": len(poses),
            "image_size": poses[0].image_size if poses else self.image_size,
            "keypoint_names": list(POSE_KEYPOINTS.keys())
        }
        
        with open(output_path / f"{prefix}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

def main():
    """測試姿勢操作功能"""
    manipulator = PoseManipulator((512, 512))
    
    # 創建預設姿勢
    base_pose = manipulator.create_default_pose()
    
    # 創建動畫序列
    walk_poses = manipulator.create_walking_animation(base_pose, 8)
    jump_poses = manipulator.create_jumping_animation(base_pose, 6)
    attack_poses = manipulator.create_attack_animation(base_pose, 4)
    
    # 保存動畫序列
    manipulator.save_animation_sequence(walk_poses, "./test_output", "walk")
    manipulator.save_animation_sequence(jump_poses, "./test_output", "jump")
    manipulator.save_animation_sequence(attack_poses, "./test_output", "attack")
    
    print("姿勢動畫序列已保存到 ./test_output")

if __name__ == "__main__":
    main() 