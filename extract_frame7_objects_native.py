"""Frame-7 object extraction from ORIGINAL native-resolution videos.

Row order per split is reconstructed exactly like the tensor cache build
(sorted glob over source videos filtered by split subjects), so outputs align
row-for-row with {split}_float16.npy. The 7th of 13 uniformly sampled frames
is decoded from each mp4 at native resolution and detected at --imgsz
(default 1280). Output dict format matches extract_frame7_objects.py.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_boxes  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args():
    p = argparse.ArgumentParser(description="Frame-7 detection on native-resolution source videos.")
    p.add_argument("--x3d-metadata", default="/workspace/X3D/data/clipgcn_tensor_cs_70_10_20/metadata.json")
    p.add_argument("--split", required=True, choices=("train", "val", "test"))
    p.add_argument("--weights", default=str(ROOT / "runs/train/coco_custom50_fixed2/weights/best.pt"))
    p.add_argument("--output", required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="1")
    p.add_argument("--conf-thres", type=float, default=0.25)
    p.add_argument("--iou-thres", type=float, default=0.45)
    p.add_argument("--num-frames", type=int, default=13)
    p.add_argument("--frame-number", type=int, default=7, help="1-based index within the sampled frames.")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def action_id(path):
    return re.match(r"(A\d+)_", Path(path).name).group(1)


def split_videos(metadata, split):
    src = Path(metadata["source_root"])
    subjects = set(metadata["split_subjects"][split])
    excluded = set(metadata.get("excluded_actions") or [])
    videos = sorted(
        p for p in src.glob("P*/**/*.mp4")
        if action_id(p) not in excluded and p.relative_to(src).parts[0] in subjects
    )
    labels = [metadata["class_to_idx"][action_id(p)] for p in videos]
    rel = [str(p.relative_to(src)) for p in videos]
    return videos, rel, labels


def read_middle_frame(video_path, num_frames, frame_number):
    """Sequential decode up to the (frame_number-1)-th of num_frames uniform samples; returns RGB or None."""
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    target = int(np.round(np.linspace(0, total - 1, num_frames))[frame_number - 1])
    frame = None
    idx = 0
    while idx <= target:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
    cap.release()
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()
    metadata = json.load(open(args.x3d_metadata))
    videos, rel, labels = split_videos(metadata, args.split)
    if args.limit:
        videos, rel, labels = videos[: args.limit], rel[: args.limit], labels[: args.limit]
    total = len(videos)
    print(f"{args.split}: {total} videos (native-resolution frame-7 extraction @ imgsz {args.imgsz})")

    device = select_device(args.device)
    model = DetectMultiBackend(args.weights, device=device, dnn=False, fp16=device.type != "cpu")
    imgsz = check_img_size(args.imgsz, s=model.stride)
    names = model.names
    nc = len(names)
    class_ids = np.arange(nc, dtype=np.int64) if nc != 80 else np.array([0, *range(31, 80)], dtype=np.int64)
    class_to_slot = {int(c): i for i, c in enumerate(class_ids.tolist())}

    presence = np.zeros((total, len(class_ids)), dtype=np.uint8)
    center_xyz = np.zeros((total, len(class_ids), 3), dtype=np.float32)
    confidence = np.zeros((total, len(class_ids)), dtype=np.float32)
    failures = []

    def run_batch(batch_frames, batch_indices):
        ims, shapes = [], []
        for f in batch_frames:
            im = letterbox(f, imgsz, stride=model.stride, auto=False)[0]
            ims.append(np.ascontiguousarray(im.transpose(2, 0, 1)))
            shapes.append(f.shape)
        im = torch.from_numpy(np.stack(ims)).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255.0
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres,
                                   classes=class_ids.tolist(), max_det=300)
        for k, det in enumerate(pred):
            if det is None or len(det) == 0:
                continue
            si = batch_indices[k]
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shapes[k]).round()
            h, w = shapes[k][:2]
            for *xyxy, conf, cls in det.tolist():
                slot = class_to_slot.get(int(cls))
                if slot is None or conf <= confidence[si, slot]:
                    continue
                presence[si, slot] = 1
                center_xyz[si, slot] = (((xyxy[0] + xyxy[2]) * 0.5) / w,
                                        ((xyxy[1] + xyxy[3]) * 0.5) / h, 0.0)
                confidence[si, slot] = conf

    buf_frames, buf_idx = [], []
    iterator = range(total)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"native frame7 {args.split}")
    t0 = time.time()
    for i in iterator:
        frame = read_middle_frame(videos[i], args.num_frames, args.frame_number)
        if frame is None:
            failures.append(rel[i])
            continue
        buf_frames.append(frame)
        buf_idx.append(i)
        if len(buf_frames) >= args.batch_size:
            run_batch(buf_frames, buf_idx)
            buf_frames, buf_idx = [], []
    if buf_frames:
        run_batch(buf_frames, buf_idx)

    manifest = np.array([[r, int(l)] for r, l in zip(rel, labels)], dtype=object)
    payload = {
        "class_ids": class_ids,
        "class_names": np.array([names[int(c)] for c in class_ids], dtype=object),
        "presence": presence,
        "center_xyz": center_xyz,
        "confidence": confidence,
        "manifest": manifest,
        "split": args.split,
        "frame_number": args.frame_number,
        "frame_index": args.frame_number - 1,
        "coordinate_format": "normalized_x_y_z0",
        "note": f"native-resolution source frames, detection imgsz={imgsz}; rows follow sorted-glob tensor order",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, payload, allow_pickle=True)
    meta = {
        "output": str(out), "split": args.split, "weights": str(Path(args.weights).resolve()),
        "imgsz": imgsz, "conf_thres": args.conf_thres, "iou_thres": args.iou_thres,
        "num_videos": total, "failures": failures,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}; failures={len(failures)}; elapsed {meta['elapsed_sec']}s")
    det_rate = presence.any(axis=1).mean()
    print(f"样本级检出率: {det_rate:.3f}; 出现过的类别数: {int((presence.sum(axis=0) > 0).sum())}")


if __name__ == "__main__":
    main()
