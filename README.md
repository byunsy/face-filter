# face-filter

## Descriptions

Face-filter is a simple program that can detect human faces in a webcam feed and apply up to ten (or more) different filters. It uses the SSD (Single Shot MultiBox Detector) based neural network, more specifically, 'res10_300x300_ssd_iter_140000_fp16.caffemodel' and 'deploy.prototxt' all of which can be found [here](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector). Once the program is executed, it will detect human faces within the frame, and users will be able to change filters using a Space bar.

The main objective of the face-filter project was to build a more advanced face-recognition program than a typical Haar Cascade face-detection program. While I did not, in any way, constructed the neural network, I learned a lot from applying a preexisting SSD network through OpenCV. I also wished to go beyond just detecting a face within a frame and worked on overlaying various transparent PNG filters on top of the detected faces. The face filters are adaptive in that they calculate the size of the faces so that they can remain mostly proportional and natural when a user moves closer or farther from the camera.

There are still room for improvements. For example, I hope to make the face filters adapt to different types of geometric/perspective transformations like rotating along with the face. I also wish to explore ways for the program to recognize individual features of a detected face (eyes, nose, mouth, etc) so that it can apply filters that perfectly fit a person's face.

## Installation

I used the OpenCV package for python (version 4.1.0.25 or above) with Python 3.7.2

```bash
pip install opencv-python==4.1.0.25
```

## Usage

Clone the lane-detection repository in your directory.

```bash
git clone https://github.com/byunsy/face-filter.git
```

## Demonstrations

The program can accurately detect a human face and apply different filters. The face filter remains proportional to the face regardless of how far or close a user is to the camera.
![](images/face_filter1.gif)

The program can also detect multiple human faces within a frame and apply filters to each of those faces.
![](images/face_filter2.gif)
