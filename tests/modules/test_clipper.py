import unittest
import os
import numpy as np
from moviepy.editor import VideoFileClip
from basic_clipper import BasicClipper


class TestBasicClipper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up shared resources for all tests."""
        cls.clipper = BasicClipper()
        cls.test_video_path = "test_video.mp4"
        cls.output_video_path = "output_video.mp4"

        # Create a test video for testing
        if not os.path.exists(cls.test_video_path):
            from moviepy.editor import ColorClip
            clip = ColorClip(size=(640, 480), color=(255, 0, 0), duration=10)  # 10-second red video
            clip.write_videofile(cls.test_video_path, codec="libx264")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests."""
        if os.path.exists(cls.test_video_path):
            os.remove(cls.test_video_path)

    def tearDown(self):
        """Remove output files after each test."""
        if os.path.exists(self.output_video_path):
            os.remove(self.output_video_path)

    def test_get_video_duration(self):
        """Test getting the duration of the video."""
        duration = self.clipper.get_video_duration(self.test_video_path)
        self.assertAlmostEqual(duration, 10, places=1, msg="Duration should be approximately 10 seconds.")

    def test_convert_highlight_scores_to_intervals(self):
        """Test converting highlight scores to time intervals."""
        highlight_scores = np.array([[0.1, 0.8, 0.9, 0.3, 0.85, 0.4, 0.95]])
        threshold = 0.8
        video_duration = 7  # Total duration in seconds
        intervals = self.clipper.convert_highlight_scores_to_intervals(highlight_scores, threshold, video_duration)
        expected_intervals = [(1.0, 2.0), (4.0, 5.0), (6.0, 7.0)]
        self.assertEqual(intervals, expected_intervals, msg="Intervals do not match the expected results.")

    def test_convert_frame_indices_to_time_ranges(self):
        """Test converting frame indices to time intervals."""
        frame_indices = [0, 1, 2, 5, 6, 10]
        fps = 30  # Frames per second
        intervals = self.clipper.convert_frame_indices_to_time_ranges(frame_indices, fps)
        expected_intervals = [(0.0, 0.06666666666666667), (0.16666666666666666, 0.2), (0.3333333333333333, 0.3333333333333333)]
        self.assertEqual(intervals, expected_intervals, msg="Frame-to-time conversion failed.")

    def test_clip_video(self):
        """Test clipping a video with highlights."""
        highlights = [(1, 3), (5, 7)]  # Highlight times in seconds
        result_path = self.clipper.clip(self.test_video_path, highlights, self.output_video_path, transition_duration=1.0)
        self.assertTrue(os.path.exists(result_path), "Output video file should exist.")
        output_video_duration = self.clipper.get_video_duration(result_path)
        expected_duration = 4.0  # Total duration including transitions
        self.assertAlmostEqual(output_video_duration, expected_duration, places=1, msg="Output video duration mismatch.")

    def test_empty_highlights(self):
        """Test behavior when no highlights are provided."""
        highlights = []  # No highlights
        with self.assertRaises(ValueError, msg="An empty highlight list should raise a ValueError."):
            self.clipper.clip(self.test_video_path, highlights, self.output_video_path)

    def test_invalid_video_path(self):
        """Test behavior when an invalid video path is provided."""
        invalid_path = "non_existent_video.mp4"
        highlights = [(1, 3)]
        with self.assertRaises(ValueError, msg="An invalid video path should raise a ValueError."):
            self.clipper.clip(invalid_path, highlights, self.output_video_path)


if __name__ == "__main__":
    unittest.main()
