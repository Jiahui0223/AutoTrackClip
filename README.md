# Auto Track Clip

Automatically track people in fast-paced video content using 3D recognition and tracking technology, ensuring focus remains on them even if they’re temporarily obscured or off-screen.

## Features

- **Automatic object tracking**: Based on deep learning models, high-precision object tracking is achieved.
- **Smart editing**：Automatically generate clips based on tracking results.

## Installation and Running

### Environment Dependency

Please make sure the following software and libraries are installed in your system:

- Python >= 3.8
- OpenCV >= 4.5
- PyTorch >= 1.10
- ffmpeg

### Installation steps

1. Clone the project code to local:

    ```bash
    git clone https://github.com/your-repo/AutoTrackClip.git
    cd AutoTrackClip

2. Pose Feature
   1) Multi-player tracking
      Please visit https://github.com/MCG-NJU/SportsMOT for download SportsMOT dataset to train MixSort.
      Use our example datasets to predict tracking of players.
      Save output .txt file and input img files.
      
   2) Multi-player action recognition
      We trained R(2+1)D model with spacejam. You can move on to our model check-points.
      Yolov3 could not track the players in occlusion or fast player movement or camera view switching.
      We used MixSort output so that we can predict action recognition on MixSort based cropped bounding boxes. You can run mix.py with setting correct paths.
      Working on maintaining quality of prediction. 

### Thanks
1)Simone Francia for the basketball action dataset and paper on action classification with 3D-CNNs.
2)hkair(https://github.com/hkair) for introducing a way for augumenting datasets.
3)Authors of SportsMOT for releasing datasets and MixSort
