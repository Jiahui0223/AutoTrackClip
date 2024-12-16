import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from easydict import EasyDict

# Arguments
args = EasyDict({
    "mixsort_txt_path": "C:\\Users\\hyeongjin\\MixSort\\YOLOX_outputs\\sportsmot_experiment\\track_results\\clip2.txt",
    "video_path": "C:\\Users\\hyeongjin\\MixSort\\datasets\\SportsMOT\\val\\clip2\\img1",
    "seq_length": 16,
    "vid_stride": 8,
    "num_classes": 10,
    "pretrained": True,
    "model_path": "model_checkpoints/r2plus1d_augmented-2/",
    "base_model_name": "r2plus1d_multiclass",
    "start_epoch": 19,
    "lr": 0.0001,
    "output_path": "C:\\Users\\hyeongjin\\MixSort\\output_results\\clip2.mp4",
    "labels": {
        "0": "block",
        "1": "pass",
        "2": "run",
        "3": "dribble",
        "4": "shoot",
        "5": "ball in hand",
        "6": "defense",
        "7": "pick",
        "8": "no_action",
        "9": "walk",
    },
    "target_size": (176, 128),
})

def parse_mixsort_txt(txt_path, target_id=None):
    """
    Parse the MixSort TXT file and filter by target player ID.
    """
    frame_boxes = {}
    with open(txt_path, "r") as file:
        for line in file:
            frame_id, obj_id, x, y, w, h, score, *_ = map(float, line.strip().split(","))
            frame_id = int(frame_id)
            obj_id = int(obj_id)
            
            # Only include bounding boxes for the specified player ID
            if target_id is not None and obj_id != target_id:
                continue
            
            if frame_id not in frame_boxes:
                frame_boxes[frame_id] = []
            frame_boxes[frame_id].append([x, y, w, h, obj_id, score])
    return frame_boxes
# Load video frames
def load_video_frames(video_path):
    frame_paths = sorted(os.listdir(video_path))
    video_frames = [cv2.imread(os.path.join(video_path, frame)) for frame in frame_paths]
    return video_frames

# Crop and resize a frame based on bounding box
def crop_and_resize_fixed(frame, bbox, target_size=(176, 128)):
    x, y, w, h, *_ = map(int, bbox)
    cropped = frame[max(0, y):y + h, max(0, x):x + w]
    
    if cropped.size == 0:
        # 바운딩 박스가 잘못된 경우 빈 이미지 반환
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # 비율 유지하면서 resize
    aspect_ratio = w / h
    target_w, target_h = target_size
    pad_w, pad_h = 0, 0

    if aspect_ratio > target_w / target_h:
        new_w = target_w
        new_h = int(h * target_w / w)
        pad_h = (target_h - new_h) // 2
    else:
        new_h = target_h
        new_w = int(w * target_h / h)
        pad_w = (target_w - new_w) // 2

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, pad_h, target_h - new_h - pad_h, pad_w, target_w - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded

# Prepare input tensor for R(2+1)D model
def prepare_input_tensor(frames):
    input_tensor = torch.FloatTensor(frames).permute(0, 3, 1, 2).unsqueeze(0)
    return input_tensor

# Predict player actions
def predict_player_actions(video_frames, frame_boxes, seq_length, model, device):
    player_predictions = {}
    player_frames = {}

    # Initialize storage for each player
    for frame_id, bboxes in frame_boxes.items():
        for bbox in bboxes:
            obj_id = int(bbox[4])
            if obj_id not in player_frames:
                player_frames[obj_id] = []

    # Process each frame
    for frame_id, bboxes in frame_boxes.items():
        for bbox in bboxes:
            obj_id = int(bbox[4])
            cropped_frame = crop_and_resize_fixed(video_frames[frame_id - 1], bbox,(176, 128))
            player_frames[obj_id].append(cropped_frame)

            # If enough frames are accumulated, make a prediction
            if len(player_frames[obj_id]) == seq_length:
                clip_tensor = torch.FloatTensor(player_frames[obj_id]).unsqueeze(0).permute(0, 4, 1, 2, 3).to(device)

                with torch.no_grad():
                    output = model(clip_tensor)
                    pred = output.argmax(dim=1).item()
                    
                #print(f"Player {obj_id}, Frame {frame_id}:")
                #print(f"Raw Output: {output.cpu().numpy()}")
                #print(f"Predicted Label Index: {pred}, Label: {args.labels[str(pred)]}")
                if obj_id not in player_predictions:
                    player_predictions[obj_id] = []
                player_predictions[obj_id].append(pred)

                # Clear frames for this player
                player_frames[obj_id] = []
                if pred == 4:  # Label "4" corresponds to "shoot"
                    print(f"Frame {frame_id}, Player {obj_id}: Predicted Label - shoot")
                    #print(f"Raw Output: {output.cpu().numpy()}")
    return player_predictions

# Write output video
def write_video(output_path, video_frames, frame_boxes, predictions):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_path, fourcc, 10, (width, height))

    for frame_id, frame in enumerate(video_frames):
        if frame_id + 1 in frame_boxes:
            for bbox in frame_boxes[frame_id + 1]:
                x, y, w, h, obj_id, _ = map(int, bbox)
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if obj_id in predictions and len(predictions[obj_id]) > 0:
                    label = args.labels[str(predictions[obj_id][0])]
                    cv2.putText(
                        frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
        video_writer.write(frame)
    video_writer.release()

# Main function
def main():
    frame_boxes = parse_mixsort_txt(args.mixsort_txt_path)
    print(f"Parsed {len(frame_boxes)} frames with bounding boxes.")
    video_frames = load_video_frames(args.video_path)
    print(f"Loaded {len(video_frames)} video frames.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.video.r2plus1d_18(pretrained=args.pretrained, progress=True)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)

    model_file = f"{args['model_path']}/{args['base_model_name']}_{args['start_epoch']}_{args['lr']}.pt"
    checkpoint = torch.load(model_file)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    predictions = predict_player_actions(video_frames, frame_boxes, args.seq_length, model, device)
    write_video(args.output_path, video_frames, frame_boxes, predictions)
    print(f"Video saved to {args.output_path}")

if __name__ == "__main__":
    main()
