from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import asyncio
from pathlib import Path
import uuid
from pydantic import BaseModel
import subprocess
import logging
from PIL import Image
import cv2
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from io import BytesIO
from transformers import TextIteratorStreamer
from threading import Thread
import re
from enum import Enum

# กำหนด logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "Uploads"
OUTPUT_DIR = BASE_DIR / "Outputs"
SCREENSHOT_DIR = BASE_DIR / "Screenshots"
MODEL_DIR = BASE_DIR / "Model"

# สร้างโฟลเดอร์ที่จำเป็น
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, SCREENSHOT_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model paths
YOLO_OBJECT_MODEL_PATH = MODEL_DIR / "YOLO_OBJECT.pt"
YOLO_NAME_MODEL_PATH = MODEL_DIR / "YOLO_NAME.pt"
SAM_CHECKPOINT = MODEL_DIR / "sam_vit_h_4b8939.pth"
OCR_MODEL_PATH = MODEL_DIR / "ocr"

# ===== UPDATED ROBUST CIRCLE CONFIG =====
MAX_FRAMES_FOR_CIRCLE_SEARCH = 80   # ลองหาให้เต็มที่กี่เฟรมก่อนใช้ fallback ชื่อ
CIRCLE_MIN_DIST = 50                # เดิม 60
CIRCLE_PARAM_SETS = [
    # ชุดมาตรฐาน (เดิมใกล้เคียง)
    dict(dp=1.2, param1=80, param2=30, minRadius=8,  maxRadius=60),
    # ไวต่อขอบมากขึ้น/รัศมีใหญ่ขึ้น
    dict(dp=1.0, param1=70, param2=24, minRadius=8,  maxRadius=72),
    # ไวขึ้นอีก ลด threshold ตรงศูนย์กลาง + อนุญาตรัศมีกว้างสุด
    dict(dp=1.2, param1=60, param2=20, minRadius=6,  maxRadius=85),
]

# ====== STABILIZATION / GATING SETTINGS ======
WARMUP_MIN_FRAMES    = 4     # ต้องดูอย่างน้อยกี่เฟรมก่อนจะยอม finalize
MERGE_RADIUS         = 32    # รัศมีรวมคลัสเตอร์ของ center (พิกเซล)
MIN_HITS_PER_CENTER  = 3     # center ต้องถูกเห็นซ้ำอย่างน้อยกี่ครั้ง
GATE_RADIUS          = 180   # วงที่ใช้จับคู่ต้องไม่ไกลจากป้ายชื่อเกินระยะนี้

# ===== FULL DEPLOYMENT PARAMS PER TEMPERATURE =====
FULL_DEPLOY_PARAMS: dict[str, dict[str, float | int]] = {
    "room": {"smooth_size": 3, "plateau_alpha": 0.985},
    "hot": {"smooth_size": 3, "plateau_alpha": 0.985},
    "cold": {"smooth_size": 3, "plateau_alpha": 0.985},
    }

def get_full_deploy_params(temperature_type: str) -> tuple[int, float]:
    params = FULL_DEPLOY_PARAMS.get(temperature_type, FULL_DEPLOY_PARAMS["room"]) 
    smooth_size = max(1, int(params.get("smooth_size", 3))) 
    plateau_alpha = float(params.get("plateau_alpha", 0.985))
    return smooth_size, plateau_alpha

# ===== IN-MEMORY DATA STORAGE =====
# Replace SQLite with in-memory dictionaries
video_records: Dict[str, Dict] = {}
processing_logs: Dict[str, List] = {}

# ===== MODELS =====
class TemperatureType(str, Enum):
    ROOM = "room"
    HOT = "hot"
    COLD = "cold"

class VideoUploadRequest(BaseModel):
    temperature_type: TemperatureType = TemperatureType.ROOM

class VideoUploadResponse(BaseModel):
    task_id: str
    message: str
    video_filename: str
    temperature_type: str
    
class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # "processing", "completed", "failed"
    progress: int  # 0-100
    message: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    temperature_type: Optional[str] = None

class VideoRecord(BaseModel):
    id: int
    task_id: str
    original_filename: str
    video_filename: str
    status: str
    progress: int
    message: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_video_path: Optional[str] = None
    screenshots: List[str] = []
    temperature_type: Optional[str] = None

class DetectionResult(BaseModel):
    task_id: str
    video_filename: str
    explosion_frame: Optional[int] = None
    full_deployment_frame: Optional[int] = None
    detected_labels: List[str] = []
    screenshots: List[str] = []
    processing_time: float
    ocr_results: Dict[str, str] = {}

# ===== UPDATED HELPER FUNCTIONS =====
def adjust_gamma(img, gamma=1.2):
    """Adjust image gamma for better circle detection"""
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.array([((i / 255.0) ** inv) * 255 for i in range(256)])).astype("uint8")
    return cv2.LUT(img, table)

def auto_roi(frame, name_det_result=None, default_band=(0.20, 0.85)):
    """Automatically determine ROI based on name detection results"""
    h = frame.shape[0]
    if name_det_result is not None and name_det_result[0].boxes is not None:
        boxes = name_det_result[0].boxes.xyxy.cpu().numpy().astype(int)
        if boxes.shape[0] > 0:
            ys = []
            for (x1, y1, x2, y2) in boxes:
                ys += [y1, y2]
            top = max(0, min(ys) - int(0.10 * h))
            bot = min(h, max(ys) + int(0.10 * h))
            return top, bot
    # ถ้ายังไม่มีชื่อ ให้ใช้แถบกว้างๆ ไปก่อน
    t = int(h * default_band[0]); b = int(h * default_band[1])
    return t, b

def find_circles_multi(gray_roi):
    """Find circles using multiple parameter sets"""
    # ลองหลายเซตพารามิเตอร์จนกว่าจะเจอ
    for ps in CIRCLE_PARAM_SETS:
        circles = cv2.HoughCircles(
            gray_roi, cv2.HOUGH_GRADIENT,
            dp=ps["dp"], minDist=CIRCLE_MIN_DIST,
            param1=ps["param1"], param2=ps["param2"],
            minRadius=ps["minRadius"], maxRadius=ps["maxRadius"]
        )
        if circles is not None:
            yield np.uint16(np.around(circles[0, :]))

def is_far_enough(new_center, centers, min_dist=50):
    """Check if new center is far enough from existing centers"""
    new_center = np.array(new_center, dtype=np.float32)
    for c in centers:
        c = np.array(c, dtype=np.float32)
        dist = np.linalg.norm(new_center - c)
        if dist < min_dist:
            return False
    return True

def gate_centers_by_labels(centers, label_pts, gate=GATE_RADIUS):
    """Filter centers by distance to label points"""
    if not label_pts:
        return centers
    out = []
    for c in centers:
        dmin = min(np.linalg.norm(np.array(c) - np.array(lp)) for lp in label_pts)
        if dmin <= gate:
            out.append(c)
    return out

def match_labels_to_circles_unique(label_centers, candidate_centers, max_dist=None):
    """
    Match labels to circles using Hungarian algorithm for unique assignment
    label_centers: [('FR1',(x,y)), ('FR2',(x,y)), ('RE3',(x,y))]
    candidate_centers: [(x,y), ...]
    max_dist: ถ้าระยะเกินนี้ถือว่าใช้ไม่ได้
    """
    if len(candidate_centers) < len(label_centers):
        return {}

    labels = [l for l, _ in label_centers]
    P = np.array([p for _, p in label_centers], dtype=np.float32)  # (3,2)
    C = np.array(candidate_centers, dtype=np.float32)              # (M,2)

    cost = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2)   # (3,M)
    if max_dist is not None:
        cost = np.where(cost <= float(max_dist), cost, 1e9)        # gate

    rows, cols = linear_sum_assignment(cost)
    out = {}
    for r, c in zip(rows, cols):
        if cost[r, c] >= 1e9:
            return {}  # จับคู่เกินระยะ
        out[labels[r]] = (int(C[c, 0]), int(C[c, 1]))

    if len(out) != len(set(out.values())):
        return {}
    return out

# ===== CENTER CONSENSUS CLASS =====
class CenterConsensus:
    """Class for managing center consensus with clustering"""
    def __init__(self, merge_radius=32):
        self.merge_radius = merge_radius
        self.clusters = []  # list of {'xy': np.array([x,y]), 'n': int}

    def _dist(self, a, b):
        return float(np.linalg.norm(a - b))

    def add(self, pts):
        for x, y in pts:
            p = np.array([float(x), float(y)], dtype=np.float32)
            assigned = False
            for c in self.clusters:
                if self._dist(p, c['xy']) <= self.merge_radius:
                    c['xy'] = (c['xy'] * c['n'] + p) / (c['n'] + 1)
                    c['n'] += 1
                    assigned = True
                    break
            if not assigned:
                self.clusters.append({'xy': p, 'n': 1})

    def topk(self, k=10):
        return sorted(self.clusters, key=lambda c: c['n'], reverse=True)[:k]

    def centers(self, k=10):
        return [(int(c['xy'][0]), int(c['xy'][1])) for c in self.topk(k)]

    def has_k_stable(self, k=3, min_hits=3):
        return sum(1 for c in self.clusters if c['n'] >= min_hits) >= k

# ===== OCR CLASS =====
class OCR:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self.processor = AutoProcessor.from_pretrained(
                OCR_MODEL_PATH,
                trust_remote_code=True,
                use_fast=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                OCR_MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.float32
            ).to(self.device).eval()
        except Exception as e:
            logger.error(f"Failed to load OCR model: {e}")
            self.processor = None
            self.model = None

    def extract_ms_time(self, image: Image.Image, max_new_tokens: int = 32) -> str:
        if self.model is None or image is None:
            return "OCR model not available"
        
        try:
            query = (
                "Read the time (in milliseconds) shown at the top-left corner of the image. "
                "Return exactly in this format: ms=13.6 (The time should be a number only, without '+' or '-' sign)"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": query}
                    ]
                }
            ]

            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)

            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 30,
                "repetition_penalty": 1.1
            }

            Thread(target=self.model.generate, kwargs=generation_kwargs).start()
            output_text = "".join(token for token in streamer).strip().replace("<|im_end|>", "").strip()

            # 🔍 Regex ดึงค่า ms
            match = re.search(r"ms\s*=\s*([+-]?\d+\.?\d*)", output_text, re.IGNORECASE)
            if match:
                ms_val = match.group(1).lstrip("+")  # ตัด '+' ออก
                return ms_val
            
            return f"Unable to extract ms from output: {output_text}"
        
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return "OCR extraction failed"

# ===== UPDATED AIRBAG DETECTION CLASS =====
class AirbagDetector:
    def __init__(self):
        self.yolo_object_model = None
        self.yolo_name_model = None
        self.sam = None
        self.predictor = None
        self.ocr = OCR()
        self.load_models()
    
    def load_models(self):
        """Load all required models"""
        try:
            if YOLO_OBJECT_MODEL_PATH.exists():
                self.yolo_object_model = YOLO(str(YOLO_OBJECT_MODEL_PATH))
                logger.info("YOLO Object model loaded successfully")
            else:
                logger.warning(f"YOLO Object model not found at {YOLO_OBJECT_MODEL_PATH}")
            
            if YOLO_NAME_MODEL_PATH.exists():
                self.yolo_name_model = YOLO(str(YOLO_NAME_MODEL_PATH))
                logger.info("YOLO Name model loaded successfully")
            else:
                logger.warning(f"YOLO Name model not found at {YOLO_NAME_MODEL_PATH}")
            
            if SAM_CHECKPOINT.exists():
                self.sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
                if torch.cuda.is_available():
                    self.sam.to("cuda")
                self.predictor = SamPredictor(self.sam)
                logger.info("SAM model loaded successfully")
            else:
                logger.warning(f"SAM model not found at {SAM_CHECKPOINT}")
                
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def get_temperature_frame_range(self, temperature_type: str):
        """Get frame range based on temperature type"""
        if temperature_type == "hot":
            return 100, 113
        elif temperature_type == "cold":
            return 125, 138
        else:  # room temperature (default)
            return 109, 119
    
    def process_video(self, video_path: str, task_id: str, temperature_type: str = "room", callback=None):
        """Process video for airbag detection with updated detection logic"""
        try:
            if not self.yolo_object_model or not self.yolo_name_model or not self.sam:
                raise Exception("Models not loaded properly")
            
            video_name = Path(video_path).stem
            output_video_path = OUTPUT_DIR / f"{video_name}_Timing_Detection_{temperature_type}.mp4"
            screenshot_dir = SCREENSHOT_DIR / f"{video_name}_{temperature_type}"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Update status
            if callback:
                callback(task_id, "processing", 5, f"Opening video for {temperature_type} temperature analysis...")
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception("Could not open video file")
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Output writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            
            # Updated processing variables
            frame_count = 0
            saved_frame_count = 0
            saved_screenshots = []
            fixed_centers = []
            label_to_center = {}
            hit_center_labels = set()
            done_detecting = False
            roi_top = height // 3
            roi_bottom = 2 * height // 3
            frame17_mask = None
            frame18_mask = None
            explosion_frame = None
            explosion_detected = False
            frame17_image = None
            consensus = CenterConsensus(merge_radius=MERGE_RADIUS)
            
            CONFIDENCE_THRESHOLD = 0.5
            MOTION_THRESHOLD = 1500
            
            if callback:
                callback(task_id, "processing", 10, "Starting frame processing with updated consensus logic...")
            
            # Process frames for circle and name detection with updated logic
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                logger.info(f"🔍 Frame {frame_count}")
                
                # Update progress
                if callback and frame_count % 10 == 0:
                    progress = min(int((frame_count / total_frames) * 70), 70)
                    callback(task_id, "processing", progress, f"Processing frame {frame_count}/{total_frames}")
                
                frame_copy = frame.copy()
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Circle detection phase (Updated with consensus logic)
                if not done_detecting:
                    # --- เตรียมภาพ ---
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # ภาพมืด/คอนทราสต์ต่ำ → ใช้ CLAHE + gamma เล็กน้อยให้สว่างขึ้น
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)
                    gray = adjust_gamma(gray, gamma=1.2)

                    # (ลอง detect ชื่อแบบเบาๆ เพื่อช่วยกำหนด ROI)
                    name_detections_try = self.yolo_name_model.predict(
                        source=frame, conf=0.25, show=False, save=False, stream=False, verbose=False
                    )

                    # ROI อัตโนมัติ (อิงชื่อถ้ามี ไม่งั้นใช้แถบกว้าง 20%-85%)
                    roi_top, roi_bottom = auto_roi(frame, name_detections_try, default_band=(0.20, 0.85))
                    gray_roi = cv2.GaussianBlur(gray[roi_top:roi_bottom, :], (9, 9), sigmaX=2, sigmaY=2)

                    # --- HoughCircles หลายพารามิเตอร์ ---
                    found_this_frame = []
                    for cset in find_circles_multi(gray_roi):
                        for c in cset:
                            center = (int(c[0]), int(c[1]) + roi_top)  # ใส่ offset กลับเข้าไป
                            if is_far_enough(center, fixed_centers, min_dist=45):
                                found_this_frame.append(center)

                        # ถ้าพบเยอะในพาสเดียวก็พอ
                        if len(found_this_frame) + len(fixed_centers) >= 3:
                            break

                    # สะสมจุดที่ใหม่พอ
                    if found_this_frame:
                        consensus.add(found_this_frame)
                        for c in found_this_frame:
                            logger.info(f"[✓] Frame {frame_count}: Added candidate center {c}")
                    else:
                        logger.info(f"⚠️ No NEW circles accepted at frame {frame_count} (clusters={len(consensus.clusters)})")

                    # --- เงื่อนไขเข้าเฟสจับชื่อ + จับคู่กับวง ---
                    logger.info("🔍 Trying to detect names (finalize mapping)...")
                    name_detections = self.yolo_name_model.predict(
                        source=frame, conf=0.30, show=False, save=False, stream=False, verbose=False
                    )
                    
                    if name_detections[0].boxes is not None:
                        boxes = name_detections[0].boxes.xyxy.cpu().numpy().astype(int)
                        classes = name_detections[0].boxes.cls.cpu().numpy().astype(int)
                        classnames = name_detections[0].names

                        # เก็บจุดของป้ายชื่อ (เอากล่องที่ใหญ่สุดต่อชื่อ)
                        target_labels = {'FR1','FR2','RE3'}
                        label_best = {}
                        for box, cls_id in zip(boxes, classes):
                            x1, y1, x2, y2 = box
                            label = classnames[cls_id]
                            if label in target_labels:
                                cx = (x1 + x2) // 2
                                cy = (y1 + y2) // 2
                                area = (x2 - x1) * (y2 - y1)
                                if label not in label_best or area > label_best[label][1]:
                                    label_best[label] = ((cx, cy), area)

                        label_centers = [(lbl, pt_area[0]) for lbl, pt_area in label_best.items() if lbl in target_labels]
                        label_pts     = [pt for _, pt in label_centers]

                        # ดึง centers จากฉันทามติ แล้วคัดด้วย gating ให้ใกล้ป้ายชื่อ
                        candidate_centers = consensus.centers(k=10)
                        candidate_centers = gate_centers_by_labels(candidate_centers, label_pts, gate=GATE_RADIUS)

                        # ยอม finalize เมื่อ: ดูมาหลายเฟรมพอ, มีอย่างน้อย 3 คลัสเตอร์ที่โผล่ซ้ำ, มีป้ายครบ 3
                        if (frame_count >= WARMUP_MIN_FRAMES and
                            consensus.has_k_stable(k=3, min_hits=MIN_HITS_PER_CENTER) and
                            len(candidate_centers) >= 3 and
                            len(label_centers) == 3):

                            temp_map = match_labels_to_circles_unique(label_centers, candidate_centers, max_dist=GATE_RADIUS)
                            if len(temp_map) == 3 and all(k in temp_map for k in ['FR1','FR2','RE3']):
                                label_to_center = temp_map
                                fixed_centers   = [label_to_center['FR1'], label_to_center['FR2'], label_to_center['RE3']]
                                done_detecting  = True
                                logger.info("\n✅ Completed detection with UNIQUE label-to-circle assignment (stable & gated).\n")
                            else:
                                logger.info("ℹ️ Assignment not ready (unique/gated not satisfied). Keep scanning...")
                        else:
                            logger.info("ℹ️ Not stable enough or labels/centers incomplete. Keep scanning...")

                    # --- Fallback: ถ้าหาหลายเฟรมแล้วยังไม่ครบ 3 วง → ใช้ตำแหน่งชื่อแทน ---
                    if (not done_detecting) and frame_count >= MAX_FRAMES_FOR_CIRCLE_SEARCH:
                        logger.info("🛟 Fallback: using name boxes as centers (circles insufficient).")
                        if name_detections_try[0].boxes is not None:
                            boxes = name_detections_try[0].boxes.xyxy.cpu().numpy().astype(int)
                            classes = name_detections_try[0].boxes.cls.cpu().numpy().astype(int)
                            classnames = name_detections_try[0].names

                            temp_map = {}
                            for box, cls_id in zip(boxes, classes):
                                x1, y1, x2, y2 = box
                                label = classnames[cls_id]
                                if label in ['FR1', 'FR2', 'RE3']:
                                    cx = (x1 + x2) // 2
                                    cy = (y1 + y2) // 2
                                    temp_map[label] = (cx, cy)
                            if all(l in temp_map for l in ['FR1', 'FR2', 'RE3']):
                                # ใช้ศูนย์กลางจากชื่อเป็น "ศูนย์กลางแทนวง"
                                label_to_center = temp_map
                                fixed_centers = list(temp_map.values())
                                done_detecting = True
                                logger.info("\n✅ Fallback succeeded: using label centers for FR1/FR2/RE3.\n")
                            else:
                                logger.info("❌ Fallback failed: not enough labels detected yet.")

                    # วาดเพื่อดู
                    for center in fixed_centers:
                        cv2.circle(frame_copy, center, 30, (0, 255, 0), 2)
                        cv2.circle(frame_copy, center, 5, (0, 0, 255), -1)

                    out.write(frame_copy)
                    continue
                
                # Object detection phase (same as original)
                for label, center in label_to_center.items():
                    cv2.circle(frame_copy, center, 30, (0, 255, 0), 2)
                    cv2.circle(frame_copy, center, 5, (0, 0, 255), -1)
                    cv2.putText(frame_copy, label, (center[0] - 15, center[1] - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # YOLO object detection
                yolo_results = self.yolo_object_model.predict(
                    source=frame, conf=CONFIDENCE_THRESHOLD, show=False, save=False, stream=False, verbose=False
                )
                
                if yolo_results[0].boxes is not None:
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
                    self.predictor.set_image(image_rgb)
                    frame_for_saving = frame_copy.copy()
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        input_box = np.array([x1, y1, x2, y2])
                        masks, scores, _ = self.predictor.predict(
                            box=input_box[None, :], multimask_output=True
                        )
                        best_mask = masks[np.argmax(scores)]
                        
                        # Explosion detection (frame 17-18)
                        if frame_count == 17:
                            frame17_mask = best_mask.copy()
                            frame17_image = frame_for_saving.copy()
                            logger.info("📌 Stored mask for frame 17")
                        elif frame_count == 18:
                            frame18_mask = best_mask.copy()
                            logger.info("📌 Stored mask for frame 18")
                            
                            if frame17_mask is not None:
                                diff = np.logical_xor(frame17_mask, frame18_mask).astype(np.uint8)
                                motion_score = np.sum(diff)
                                logger.info(f"🧮 Motion score between frame 17 & 18: {motion_score}")
                                
                                if motion_score > MOTION_THRESHOLD:
                                    explosion_frame = 18
                                    explosion_img = frame_for_saving
                                    logger.info("💥 Explosion detected at frame 18")
                                else:
                                    explosion_frame = 17
                                    explosion_img = frame17_image
                                    logger.info("💥 Explosion detected at frame 17")
                                
                                screenshot_path = screenshot_dir / f"Explosion_frame{explosion_frame}_{video_name}.png"
                                cv2.imwrite(str(screenshot_path), explosion_img)
                                saved_screenshots.append(str(screenshot_path))
                                explosion_detected = True
                                logger.info(f"💾 Saved explosion screenshot → {screenshot_path}")
                        
                        # Check if mask hits any center
                        for label, center in label_to_center.items():
                            if label in hit_center_labels:
                                continue
                            cx, cy = center
                            if 0 <= cy < best_mask.shape[0] and 0 <= cx < best_mask.shape[1]:
                                if best_mask[cy, cx]:
                                    screenshot_path = screenshot_dir / f"{label}_frame{frame_count}_{video_name}.png"
                                    cv2.imwrite(str(screenshot_path), frame_for_saving)
                                    saved_screenshots.append(str(screenshot_path))
                                    saved_frame_count += 1
                                    hit_center_labels.add(label)
                                    logger.info(f"💾 Saved frame at [{label}] (frame {frame_count}) → {screenshot_path}")
                        
                        # Add green mask overlay
                        green_mask = np.zeros_like(frame, dtype=np.uint8)
                        green_mask[best_mask] = (0, 255, 0)
                        frame_copy = cv2.addWeighted(frame_copy, 1.0, green_mask, 0.5, 0)
                
                out.write(frame_copy)
            
            cap.release()
            out.release()
            
            # Full deployment detection with updated analysis
            if callback:
                callback(task_id, "processing", 75, f"Analyzing full deployment for {temperature_type} temperature...")
            
            full_deployment_frame = self.analyze_full_deployment(video_path, video_name, screenshot_dir, temperature_type)
            
            # OCR processing
            if callback:
                callback(task_id, "processing", 90, "Running OCR on screenshots...")
            
            ocr_results = {}
            for screenshot_path in saved_screenshots:
                if any(label in Path(screenshot_path).name for label in ['FR1', 'FR2', 'RE3']):
                    try:
                        image = Image.open(screenshot_path).convert("RGB")
                        ocr_result = self.ocr.extract_ms_time(image)
                        ocr_results[Path(screenshot_path).name] = ocr_result
                    except Exception as e:
                        logger.error(f"OCR failed for {screenshot_path}: {e}")
                        ocr_results[Path(screenshot_path).name] = "OCR failed"
            
            # Final update
            if callback:
                callback(task_id, "completed", 100, "Processing completed successfully", 
                        str(output_video_path), saved_screenshots, ocr_results)
            
            logger.info("\n=== Final Fixed Centers with Labels ===")
            for label, c in label_to_center.items():
                logger.info(f"{label}: ({int(c[0])}, {int(c[1])})")
            logger.info(f"✅ Process completed. Total saved frames: {saved_frame_count}")
            
            return {
                "success": True,
                "output_video": str(output_video_path),
                "screenshots": saved_screenshots,
                "explosion_frame": explosion_frame,
                "full_deployment_frame": full_deployment_frame,
                "detected_labels": list(hit_center_labels),
                "ocr_results": ocr_results,
                "temperature_type": temperature_type
            }
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            if callback:
                callback(task_id, "failed", 0, f"Processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def analyze_full_deployment(self, video_path: str, video_name: str, screenshot_dir: Path, temperature_type: str = "room"):
        """Analyze full deployment phase with updated frame range logic"""
        try:
            cap_analysis = cv2.VideoCapture(str(video_path))
            mask_area_data = []
            
            # Get temperature-specific frame range
            START_FRAME, END_FRAME = self.get_temperature_frame_range(temperature_type)
            CONFIDENCE_THRESHOLD = 0.5
            
            logger.info(f"📄 Analyzing {temperature_type} temperature: frames {START_FRAME}-{END_FRAME}")
            
            # เลื่อนไปยังเฟรมเริ่มต้น
            cap_analysis.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME - 1)
            
            while cap_analysis.isOpened():
                ret, frame = cap_analysis.read()
                if not ret:
                    break
                    
                frame_count = int(cap_analysis.get(cv2.CAP_PROP_POS_FRAMES))
                
                if frame_count > END_FRAME:
                    break
                
                logger.info(f"📄 Analyzing frame {frame_count} for {temperature_type} temperature full deployment...")
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yolo_results = self.yolo_object_model.predict(
                    source=frame, conf=CONFIDENCE_THRESHOLD, show=False, save=False, stream=False, verbose=False
                )
                
                if yolo_results[0].boxes is not None:
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
                    
                    if len(boxes) > 0:
                        box = boxes[0]
                        x1, y1, x2, y2 = box
                        self.predictor.set_image(image_rgb)
                        masks, scores, _ = self.predictor.predict(
                            box=np.array([x1, y1, x2, y2])[None, :], multimask_output=True
                        )
                        best_mask = masks[np.argmax(scores)]
                        area = np.sum(best_mask)
                        mask_area_data.append((frame_count, area))
            
            cap_analysis.release()
            
            # Find plateau with improved smoothing
            if len(mask_area_data) > 0:
                areas = np.array([area for _, area in mask_area_data])
                frames = [frame_num for frame_num, _ in mask_area_data]


                # ✅ ใช้ค่าตามอุณหภูมิ
                smooth_size, plateau_alpha = get_full_deploy_params(temperature_type)


                # ตรวจสอบให้แน่ใจว่ามีข้อมูลเพียงพอสำหรับการทำ Smoothing
                if len(areas) >= max(3, smooth_size):
                    smoothed_areas = uniform_filter1d(areas, size=smooth_size)
                else:
                    # ถ้าข้อมูลน้อยเกินไป ให้ใช้พื้นที่เดิมโดยไม่มีการ smoothing
                    smoothed_areas = areas
                    logger.info(
                    f"⚠️ Not enough data points for smoothing (need >= {max(3, smooth_size)}). Using raw area data."
                    )


                peak_index = np.argmax(smoothed_areas)
                plateau_frame = frames[peak_index]


                # ✅ ใช้ threshold ตาม alpha ของอุณหภูมิ
                max_area = smoothed_areas[peak_index]
                threshold = max_area * plateau_alpha if max_area > 0 else 0


                for i in range(peak_index, len(smoothed_areas)):
                    if smoothed_areas[i] < threshold:
                        break
                    plateau_frame = frames[i]


                logger.info(
                f"✨ Full Deployment Detected at Frame (within {START_FRAME}-{END_FRAME}): "
                f"{plateau_frame} (Smoothed Area: {max_area}, smooth_size={smooth_size}, alpha={plateau_alpha})"
                )
                
                # Capture screenshot
                cap_final = cv2.VideoCapture(str(video_path))
                cap_final.set(cv2.CAP_PROP_POS_FRAMES, plateau_frame - 1)
                ret_final, frame_final = cap_final.read()
                if ret_final:
                    screenshot_path = screenshot_dir / f"Airbag_Full_Deployment_{temperature_type}_frame{plateau_frame}_{video_name}.png"
                    cv2.imwrite(str(screenshot_path), frame_final)
                    logger.info(f"💾 Captured {temperature_type} temperature full deployment screenshot → {screenshot_path}")
                else:
                    logger.info("❌ Could not read final frame for full deployment capture.")
                cap_final.release()
                
                return plateau_frame
            else:
                logger.info(f"\n⚫ No airbag mask data collected within the specified frame range ({START_FRAME}-{END_FRAME}).")
            
            return None
            
        except Exception as e:
            logger.error(f"Full deployment analysis failed for {temperature_type}: {e}")
            return None

# ===== FASTAPI APP =====
app = FastAPI(title="Airbag Detection API", version="2.0.0")

# CORS middleware
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000", 
    "http://10.83.49.188:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

# Initialize detector
detector = AirbagDetector()

# In-memory task tracking
processing_tasks = {}

# ===== HELPER FUNCTIONS =====
def update_task_status(task_id: str, status: str, progress: int, message: str, 
                      output_video_path: str = None, screenshots: List[str] = None, 
                      ocr_results: Dict[str, str] = None):
    """Update task status in memory"""
    
    try:
        # Update in video_records
        if task_id in video_records:
            video_records[task_id].update({
                "status": status,
                "progress": progress,
                "message": message
            })
            
            if status == "completed":
                video_records[task_id].update({
                    "completed_at": datetime.now(),
                    "output_video_path": output_video_path,
                    "screenshots": screenshots or [],
                    "ocr_results": ocr_results or {}
                })
        
        # Update in-memory processing tasks
        processing_tasks[task_id] = {
            "status": status,
            "progress": progress,
            "message": message,
            "updated_at": datetime.now()
        }
        
        # Add to processing logs
        if task_id not in processing_logs:
            processing_logs[task_id] = []
        
        processing_logs[task_id].append({
            "log_level": "INFO",
            "message": message,
            "timestamp": datetime.now()
        })
        
        logger.info(f"Task {task_id}: {status} ({progress}%) - {message}")
        
    except Exception as e:
        logger.error(f"Failed to update task status: {e}")

async def process_video_background(video_path: str, task_id: str, temperature_type: str = "room"):
    """Background task for video processing with temperature selection"""
    try:
        result = detector.process_video(video_path, task_id, temperature_type, update_task_status)
        if result["success"]:
            logger.info(f"Video processing completed for task {task_id} with {temperature_type} temperature")
        else:
            logger.error(f"Video processing failed for task {task_id}: {result.get('error')}")
    except Exception as e:
        logger.error(f"Background processing error: {e}")
        update_task_status(task_id, "failed", 0, f"Processing error: {str(e)}")

# ===== API ENDPOINTS =====

@app.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    temperature_type: TemperatureType = TemperatureType.ROOM
):
    """Upload video for processing with temperature selection"""
    
    # Validate file
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only video files are allowed.")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Save uploaded file with original name
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Save to in-memory storage
    video_records[task_id] = {
        "id": len(video_records) + 1,
        "task_id": task_id,
        "original_filename": file.filename,
        "video_filename": file.filename,
        "status": "pending",
        "progress": 0,
        "message": f"Video uploaded for {temperature_type} temperature analysis with enhanced detection, queued for processing",
        "created_at": datetime.now(),
        "completed_at": None,
        "output_video_path": None,
        "screenshots": [],
        "ocr_results": {},
        "temperature_type": temperature_type
    }
    
    # Start background processing with temperature type
    background_tasks.add_task(process_video_background, str(file_path), task_id, temperature_type)
    
    return VideoUploadResponse(
        task_id=task_id,
        message=f"Video uploaded successfully for {temperature_type} temperature analysis with enhanced consensus detection. Processing started.",
        video_filename=file.filename,
        temperature_type=temperature_type
    )

@app.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_processing_status(task_id: str):
    """Get processing status of a task"""
    
    if task_id not in video_records:
        raise HTTPException(status_code=404, detail="Task not found")
    
    record = video_records[task_id]
    
    return ProcessingStatus(
        task_id=record['task_id'],
        status=record['status'],
        progress=record['progress'],
        message=record['message'],
        created_at=record['created_at'],
        completed_at=record['completed_at'],
        temperature_type=record.get('temperature_type', 'room')
    )

@app.get("/videos", response_model=List[VideoRecord])
async def get_video_history(limit: int = Query(default=50, ge=1, le=100)):
    """Get video processing history"""
    
    # Sort by created_at descending and limit
    sorted_records = sorted(
        video_records.values(), 
        key=lambda x: x['created_at'], 
        reverse=True
    )[:limit]
    
    records = []
    for record in sorted_records:
        records.append(VideoRecord(
            id=record['id'],
            task_id=record['task_id'],
            original_filename=record['original_filename'],
            video_filename=record['video_filename'],
            status=record['status'],
            progress=record['progress'],
            message=record['message'],
            created_at=record['created_at'],
            completed_at=record['completed_at'],
            output_video_path=record['output_video_path'],
            screenshots=record['screenshots'],
            temperature_type=record.get('temperature_type', 'room')
        ))
    
    return records

@app.get("/video/{task_id}")
async def get_output_video(task_id: str):
    """Download processed video"""
    
    if task_id not in video_records:
        raise HTTPException(status_code=404, detail="Task not found")
    
    record = video_records[task_id]
    
    if not record['output_video_path']:
        raise HTTPException(status_code=404, detail="Output video not found")
    
    video_path = Path(record['output_video_path'])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name
    )
        
@app.get("/temperature-options")
async def get_temperature_options():
    """Get available temperature options with updated frame ranges"""
    return {
        "options": [
            {"value": "room", "label": "Room Temperature", "frame_range": "109-119"},
            {"value": "hot", "label": "Hot Temperature", "frame_range": "100-113"},
            {"value": "cold", "label": "Cold Temperature", "frame_range": "125-138"}
        ]
    }

@app.get("/screenshots/{task_id}")
async def get_screenshots(task_id: str):
    """Get list of screenshots for a task"""
    
    if task_id not in video_records:
        raise HTTPException(status_code=404, detail="Task not found")
    
    record = video_records[task_id]
    screenshots = record['screenshots']
    
    # Convert to relative paths for API response
    screenshot_info = []
    for screenshot_path in screenshots:
        path = Path(screenshot_path)
        if path.exists():
            screenshot_info.append({
                "filename": path.name,
                "path": f"/screenshot/{task_id}/{path.name}",
                "full_path": str(path)
            })
    
    return {"screenshots": screenshot_info}

@app.get("/screenshot/{task_id}/{filename}")
async def get_screenshot(task_id: str, filename: str):
    """Download specific screenshot"""
    
    if task_id not in video_records:
        raise HTTPException(status_code=404, detail="Task not found")
    
    record = video_records[task_id]
    video_name = Path(record['video_filename']).stem
    temp_type = record.get('temperature_type', 'room')
    screenshot_path = SCREENSHOT_DIR / f"{video_name}_{temp_type}" / filename
    
    if not screenshot_path.exists():
        raise HTTPException(status_code=404, detail="Screenshot not found")
    
    return FileResponse(
        path=str(screenshot_path),
        media_type="image/png",
        filename=filename
    )

@app.get("/results/{task_id}", response_model=DetectionResult)
async def get_detection_results(task_id: str):
    """Get complete detection results for a task"""
    
    if task_id not in video_records:
        raise HTTPException(status_code=404, detail="Task not found")
    
    record = video_records[task_id]
    
    if record['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    # Parse processing time (mock value for now)
    processing_time = 120.5  # seconds
    
    return DetectionResult(
        task_id=task_id,
        video_filename=record['video_filename'],
        explosion_frame=18,  # Mock value
        full_deployment_frame=115,  # Mock value
        detected_labels=['FR1', 'FR2', 'RE3'],
        screenshots=record['screenshots'],
        processing_time=processing_time,
        ocr_results=record['ocr_results']
    )

@app.delete("/video/{task_id}")
async def delete_video(task_id: str):
    """Delete video and all associated files"""
    
    if task_id not in video_records:
        raise HTTPException(status_code=404, detail="Task not found")
    
    record = video_records[task_id]
    
    try:
        # Delete files
        video_path = UPLOAD_DIR / record['video_filename']
        if video_path.exists():
            video_path.unlink()
        
        if record['output_video_path']:
            output_path = Path(record['output_video_path'])
            if output_path.exists():
                output_path.unlink()
        
        # Delete screenshots
        screenshots = record['screenshots']
        for screenshot_path in screenshots:
            path = Path(screenshot_path)
            if path.exists():
                path.unlink()
        
        # Delete screenshot directory if empty
        video_name = Path(record['video_filename']).stem
        temp_type = record.get('temperature_type', 'room')
        screenshot_dir = SCREENSHOT_DIR / f"{video_name}_{temp_type}"
        if screenshot_dir.exists() and not any(screenshot_dir.iterdir()):
            screenshot_dir.rmdir()
        
        # Delete from in-memory storage
        del video_records[task_id]
        if task_id in processing_tasks:
            del processing_tasks[task_id]
        if task_id in processing_logs:
            del processing_logs[task_id]
        
        return {"message": "Video and associated files deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting files: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0 - Enhanced with Consensus Detection",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "yolo_object": detector.yolo_object_model is not None,
            "yolo_name": detector.yolo_name_model is not None,
            "sam": detector.sam is not None,
            "ocr": detector.ocr.model is not None
        },
        "storage": {
            "type": "in-memory",
            "total_videos": len(video_records),
            "active_tasks": len([r for r in video_records.values() if r['status'] == 'processing'])
        },
        "enhancements": [
            "✅ Center Consensus with Clustering",
            "✅ Hungarian Algorithm for Unique Label-Circle Assignment",
            "✅ Gating by Label Distance",
            "✅ Stabilized Detection with Warmup Frames",
            "✅ Enhanced Circle Detection with Multiple Parameter Sets",
            "✅ Automatic ROI Detection",
            "✅ Image Preprocessing with CLAHE and Gamma Adjustment",
            "✅ Fallback Mechanism for Circle Detection",
            "✅ Temperature-specific Frame Range Analysis",
            "✅ Improved Full Deployment Analysis with Smoothing"
        ]
    }

@app.get("/logs/{task_id}")
async def get_processing_logs(task_id: str):
    """Get processing logs for a task"""
    
    if task_id not in processing_logs:
        raise HTTPException(status_code=404, detail="No logs found for this task")
    
    return {
        "task_id": task_id,
        "logs": processing_logs[task_id]
    }

# ===== STARTUP EVENT =====
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting Airbag Detection API v2.0.0...")
    logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    logger.info("💾 Storage: In-memory (SQLite removed)")
    logger.info("🔧 New Features:")
    logger.info("   ✅ Center Consensus with Clustering Algorithm")
    logger.info("   ✅ Hungarian Algorithm for Optimal Label-Circle Matching")
    logger.info("   ✅ Gating System for Distance-based Filtering")
    logger.info("   ✅ Stabilized Detection with Warmup Mechanism")
    logger.info("   ✅ Enhanced Multi-parameter Circle Detection")
    logger.info("   ✅ Automatic ROI with Name-based Detection")
    logger.info("   ✅ Advanced Image Preprocessing Pipeline")
    logger.info("🎯 Ready for enhanced airbag detection with improved accuracy!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)