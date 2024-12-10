from src.pipeline import process_and_visualize_video
from src.pipeline import process_video
from src.pipeline import process_video_with_model
from src.pipeline import process_video_with_debug_model
from src.pipeline import process_video_for_highlight_detection

import cv2
import numpy as np
from src.modules.clipper.basic_clipper import BasicClipper

# python run.py --input .\assets\videos\data_short.mov

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Auto Track Clip Tool")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    args = parser.parse_args()

    # process_and_visualize_video(input_path=args.input)
    # process_video(args.input)
    # process_video_with_model(args.input)
    # process_video_with_debug_model(args.input)
    highlight_scores_np, highlights, highlight_threshold = process_video_for_highlight_detection(args.input)
    clipper = BasicClipper()
    # print(f"Highlight scores shape _np: {highlight_scores_np.shape}")
    # print(f"Highlight scores shape _s: {highlights.shape}")

    output_path = 'assets/highlight_output.mp4'
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 打印 highlight_scores_np 的信息
    # print("Highlight Scores NP:",  highlight_scores_np[0])
    # print("Highlight Scores NP Shape:", np.array(highlight_scores_np).shape)
    # print("Highlight Scores NP Type:", type(highlight_scores_np))
    print(f"highlits = {highlights[1]}")
    highlight_scores_indices = highlight_scores_np[0][highlights[1]]
    print(f"temp = {highlight_scores_indices}")
    # highlight_scores_indices = np.nonzero(highlight_scores_np[0])[0].tolist()
    # highlights_time = clipper.convert_frame_indices_to_time_ranges(highlight_scores_indices , fps)

    # print(f"Highlight scores shape =  : {highlights_time.shape}")
    # print(f"Highlight scores = :  {highlights_time}")
    video_duration = clipper.get_video_duration(args.input)
    final_highlights = clipper.convert_highlight_scores_to_intervals(highlight_scores_np, highlight_threshold, video_duration )
    final_video_path = clipper.clip(args.input, final_highlights, output_path)

    print(f"Final highlighted video saved at {final_video_path}")
