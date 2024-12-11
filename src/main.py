from src.pipeline import process_and_visualize_video
from src.pipeline import process_video
from src.pipeline import process_video_with_model
from src.pipeline import process_video_with_debug_model
from src.pipeline import process_video_for_highlight_detection

import cv2
import numpy as np
from src.modules.clipper.basic_clipper import BasicClipper
import argparse

# python run.py --input .\assets\videos\NCAA_Short.mp4 --mode 5

def main():
    parser = argparse.ArgumentParser(description="Auto Track Clip Tool")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--mode",type=int,required=True, choices=[1,2,3,4,5],
                        help="Processing mode:"
                                "1= visualize, 2=basic processing, 3=with model, 4 = with debug model, 5=highlight detection")
    args = parser.parse_args()

    input_path = args.input
    output_path = 'assets/highlight_output_test.mp4'

    clipper = BasicClipper()
    try:
        if( args.mode == 1):
            print("Mode 1: Visualizing video...")
            process_and_visualize_video(input_path=input_path)
        elif( args.mode == 2):
            print("Mode 2: Basic processing...")
            process_video(input_path)
        elif(args.mode == 3):
            print("Mode 3: Processing video with model...")
            process_video_with_model(input_path)
        elif(args.mode == 4):
            print("Mode 4: Processing video with debug model...")
            process_video_with_debug_model(input_path)
        elif(args.mode == 5):
            print("Mode 5: Highlight detection and video clipping...")
            highlight_scores_np, highlights, highlight_threshold = process_video_for_highlight_detection(input_path)
            print(f"Highlight Scores Shape: {highlight_scores_np.shape}")
            print(f"Highlights Indices: {highlights}")
            print(f"Highlight Threshold: {highlight_threshold}")

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Error: Unable to open video file {input_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            video_duration = clipper.get_video_duration(input_path)
            final_highlights = clipper.convert_highlight_scores_to_intervals(
                highlight_scores_np, highlight_threshold, video_duration
            )
            print(f"Final Highlight Intervals: {final_highlights}")

            final_video_path = clipper.clip(input_path, final_highlights, output_path)
            print(f"Final highlighted video saved at {final_video_path}")

    except Exception as e:
        print(f"An error occured:{e}")


### JIAHUI DRAFT

    # highlight_scores_np, highlights, highlight_threshold = process_video_for_highlight_detection(args.input)
    # print(f"Highlight scores shape _np: {highlight_scores_np.shape}")
    # print(f"Highlight scores shape _s: {highlights.shape}")
    # cap = cv2.VideoCapture(args.input)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # if not cap.isOpened():
    #     raise ValueError(f"Error: Unable to open video file {args.input}")

    # 打印 highlight_scores_np 的信息
    # print("Highlight Scores NP:",  highlight_scores_np[0])
    # print("Highlight Scores NP Shape:", np.array(highlight_scores_np).shape)
    # print("Highlight Scores NP Type:", type(highlight_scores_np))
    # print(f"highlits = {highlights[1]}")
    # highlight_scores_indices = highlight_scores_np[0][highlights[1]]
    # print(f"temp = {highlight_scores_indices}")
    # highlight_scores_indices = np.nonzero(highlight_scores_np[0])[0].tolist()
    # highlights_time = clipper.convert_frame_indices_to_time_ranges(highlight_scores_indices , fps)

    # print(f"Highlight scores shape =  : {highlights_time.shape}")
    # print(f"Highlight scores = :  {highlights_time}")
    # video_duration = clipper.get_video_duration(args.input)
    # final_highlights = clipper.convert_highlight_scores_to_intervals(highlight_scores_np, highlight_threshold, video_duration )
    # final_video_path = clipper.clip(args.input, final_highlights, output_path)

    # print(f"Final highlighted video saved at {final_video_path}")
    # cap.release()

