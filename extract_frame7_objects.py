import argparse
import json
import sys
from pathlib import Path

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
except ImportError:  # pragma: no cover
    tqdm = None


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLOv5m on the 7th frame of the CLIPGCN train tensor cache and "
            "save per-class object presence plus center coordinates."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="/workspace/X3D/data/clipgcn_tensor_cs_70_10_20",
        help="Directory containing train_float16.npy and train_manifest.txt.",
    )
    parser.add_argument(
        "--weights",
        default=str(ROOT / "weights/yolov5m.pt"),
        help="YOLOv5 weights path.",
    )
    parser.add_argument(
        "--output",
        default="/workspace/X3D/data/clipgcn_tensor_cs_70_10_20/train_frame7_yolov5m_objects.npy",
        help="Output .npy path.",
    )
    parser.add_argument("--split", default="train", choices=("train", "val", "test", "test_seen"))
    parser.add_argument(
        "--frame-number",
        type=int,
        default=7,
        help="1-based frame number inside the 13-frame tensor clip.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0, 1, 2, or cpu.")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of samples for quick smoke tests.",
    )
    return parser.parse_args()


def iter_range(total, batch_size):
    ranges = range(0, total, batch_size)
    if tqdm is None:
        return ranges
    return tqdm(ranges, desc="YOLOv5m frame-7 inference")


def load_manifest(path, limit=None):
    if not path.is_file():
        return None
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            video_path, label = line.rsplit(" ", 1)
            rows.append((video_path, int(label)))
            if limit is not None and len(rows) >= limit:
                break
    return np.array(rows, dtype=object)


def tensor_frames_to_uint8(batch):
    """Convert [B, C, H, W] ImageNet-normalized RGB tensors to uint8 RGB."""
    batch = batch.astype(np.float32, copy=False)
    batch = (batch * IMAGENET_STD.reshape(1, 3, 1, 1)) + IMAGENET_MEAN.reshape(1, 3, 1, 1)
    batch = np.clip(batch, 0.0, 1.0)
    batch = np.transpose(batch, (0, 2, 3, 1))
    return np.rint(batch * 255.0).astype(np.uint8)


def prepare_yolo_batch(frames, imgsz, stride, pt):
    images = []
    for frame in frames:
        image = letterbox(frame, imgsz, stride=stride, auto=pt)[0]
        image = image.transpose((2, 0, 1))  # RGB HWC to RGB CHW for YOLOv5.
        images.append(np.ascontiguousarray(image))
    return np.stack(images, axis=0)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_path = data_dir / f"{args.split}_float16.npy"
    manifest_path = data_dir / f"{args.split}_manifest.txt"
    output_path = Path(args.output)

    data = np.load(data_path, mmap_mode="r")
    if data.ndim != 5:
        raise ValueError(f"Expected [N, C, T, H, W], got {data.shape} from {data_path}")

    frame_index = args.frame_number - 1
    if frame_index < 0 or frame_index >= data.shape[2]:
        raise ValueError(f"frame-number must be within 1..{data.shape[2]}, got {args.frame_number}")

    total = data.shape[0] if args.limit is None else min(args.limit, data.shape[0])

    device = select_device(args.device)
    model = DetectMultiBackend(
        args.weights,
        device=device,
        dnn=False,
        data=str(ROOT / "data/coco128.yaml"),
        fp16=device.type != "cpu",
    )
    imgsz = check_img_size(args.imgsz, s=model.stride)
    names = model.names

    nc = len(names)
    if nc == 80:
        # Legacy stock-COCO weights: person (0) plus household-object range 31..79.
        class_ids = np.array([0] + list(range(31, 80)), dtype=np.int64)
    else:
        # Custom model (e.g. coco_custom50): its classes are already the wanted set.
        class_ids = np.arange(nc, dtype=np.int64)
    class_to_slot = {int(coco_id): idx for idx, coco_id in enumerate(class_ids.tolist())}

    presence = np.zeros((total, len(class_ids)), dtype=np.uint8)
    center_xyz = np.zeros((total, len(class_ids), 3), dtype=np.float32)
    confidence = np.zeros((total, len(class_ids)), dtype=np.float32)
    class_names = np.array([names[int(coco_id)] for coco_id in class_ids], dtype=object)

    for start in iter_range(total, args.batch_size):
        end = min(start + args.batch_size, total)
        frames = tensor_frames_to_uint8(data[start:end, :, frame_index, :, :])
        yolo_batch = prepare_yolo_batch(frames, imgsz, model.stride, model.pt)
        im = torch.from_numpy(yolo_batch).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255.0

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            classes=class_ids.tolist(),
            max_det=300,
        )

        for local_idx, det in enumerate(pred):
            if det is None or len(det) == 0:
                continue
            sample_idx = start + local_idx
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frames[local_idx].shape).round()
            height, width = frames[local_idx].shape[:2]
            for *xyxy, conf, cls in det.tolist():
                cls = int(cls)
                slot = class_to_slot.get(cls)
                if slot is None or conf <= confidence[sample_idx, slot]:
                    continue
                x1, y1, x2, y2 = xyxy
                center_x = ((x1 + x2) * 0.5) / width
                center_y = ((y1 + y2) * 0.5) / height
                presence[sample_idx, slot] = 1
                # The source cache is RGB only, so depth is not observable here.
                # Keep z as 0.0 and store confidence separately.
                center_xyz[sample_idx, slot] = (center_x, center_y, 0.0)
                confidence[sample_idx, slot] = conf

    manifest = load_manifest(manifest_path, limit=total)
    payload = {
        "class_ids": class_ids,
        "class_names": class_names,
        "presence": presence,
        "center_xyz": center_xyz,
        "confidence": confidence,
        "manifest": manifest,
        "split": args.split,
        "frame_number": args.frame_number,
        "frame_index": frame_index,
        "coordinate_format": "normalized_x_y_z0",
        "note": "z is 0.0 because clipgcn_tensor_cs_70_10_20 stores RGB frames, not depth.",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, payload, allow_pickle=True)

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = {
        "output": str(output_path),
        "split": args.split,
        "input_data": str(data_path),
        "weights": str(Path(args.weights).resolve()),
        "shape": {
            "presence": list(presence.shape),
            "center_xyz": list(center_xyz.shape),
            "confidence": list(confidence.shape),
        },
        "frame_number": args.frame_number,
        "frame_index": frame_index,
        "allowed_coco_class_ids": class_ids.tolist(),
        "allowed_coco_class_names": class_names.tolist(),
        "coordinate_format": "normalized_x_y_z0",
        "conf_thres": args.conf_thres,
        "iou_thres": args.iou_thres,
        "imgsz": imgsz,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {output_path}")
    print(f"saved {metadata_path}")
    print(f"presence shape: {presence.shape}")
    print(f"center_xyz shape: {center_xyz.shape}")


if __name__ == "__main__":
    main()
