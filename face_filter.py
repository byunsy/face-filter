"""============================================================================
TITLE       : face_filter.py
BY          : Sang Yoon Byun
DESCRIPTION : A program that can detect human faces and apply a face filter 
============================================================================"""
import sys
import numpy as np
import cv2

"""============================================================================
PROCEDURE:
    overlay
PARAMETERS:
    img, a source image to overlay the filter on
    flower, a filter
    pos, x and y coordinate tuple to position to filter
PURPOSE:
    overlays a four-channel iamge on a three-channel source image
PRODUCES:
    None - a void function
============================================================================"""
def overlay(img, flower, pos):

    # Compute the start and end xy coordinates to overlay 
    # - sx, sy : start x and y
    # - ex, ey : end x and y
    sx = pos[0]
    ex = pos[0] + flower.shape[1]
    sy = pos[1]
    ey = pos[1] + flower.shape[0]

    # If filter goes outside the source image boundaries, then stop overlay
    if sx < 0 or sy < 0 or ex > img.shape[1] or ey > img.shape[0]:
        return

    # img1: portion of source image
    # img2: filter iamge
    img1 = img[sy:ey, sx:ex]               # shape=(h, w, 3)
    img2 = flower[:, :, 0:3]               # shape=(h, w, 3)
    alpha = 1. - (flower[:, :, 3] / 255.)  # shape=(h, w)

    # Compute weighted sum of img1 and img2 per BGR channel
    img1[..., 0] = (img1[..., 0] * alpha + img2[..., 0] * (1. - alpha)).astype(np.uint8)
    img1[..., 1] = (img1[..., 1] * alpha + img2[..., 1] * (1. - alpha)).astype(np.uint8)
    img1[..., 2] = (img1[..., 2] * alpha + img2[..., 2] * (1. - alpha)).astype(np.uint8)


"""============================================================================
                                     MAIN
============================================================================"""
def main():
        
    # Set up model and configuration for face detection
    model = 'opencv_face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    config = 'opencv_face_detector/deploy.prototxt'

    # Open Camera to capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Check camera input
    if not cap.isOpened():
        print('Camera open failed!')
        sys.exit()

    # Attain the neural network described above
    net = cv2.dnn.readNet(model, config)

    # Check net status
    if net.empty():
        print('Net open failed!')
        sys.exit()

    # Open filter image
    flowers = cv2.imread('./filter/filter1.png', cv2.IMREAD_UNCHANGED)

    # Check image input
    if flowers is None:
        print('Error: Failed to open PNG image.')
        sys.exit()

    # Attain flower filter dimensions and midpoint
    flowers_h, flowers_w = flowers.shape[:2]
    flowers_mid = flowers_w // 2

    while True:
        # Read each frame
        ret, frame = cap.read()
        (h, w) = frame.shape[:2]

        # Check frame input
        if not ret:
            break

        # Prepare blob object to pass into the neural net
        blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
        net.setInput(blob)
        out = net.forward()

        # Get the detected objects: the ignore the first two dimensions 
        detect = out[0, 0, :, :]

        for i in range(detect.shape[0]):
            confidence = detect[i, 2]

            # Only when confidence level is above 0.5, draw the border box
            if confidence < 0.5:
                break

            # x and y coordinates for the border box
            x1 = int(detect[i, 3] * w)
            y1 = int(detect[i, 4] * h)
            x2 = int(detect[i, 5] * w)
            y2 = int(detect[i, 6] * h)

            # Calculate border box width and midpoint 
            border_box_w = x2 - x1
            border_box_mid = border_box_w // 2

            # FOR REFERENCE -- Draw border box around face
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

            # FOR REFERENCE -- Write label at the top of the border box
            # label = f'Face: {confidence:4.2f}'
            # cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

            # Scaling factor for filter
            fx = (border_box_w / flowers_w) * 1.5
            flowers_resized = cv2.resize(flowers, (0, 0), fx=fx, fy=fx, 
                                         interpolation=cv2.INTER_AREA)

            # Calculate the xy coordinates to overlay the resized filter
            # - the top left corner coordinate
            pos = ( int(x1 + border_box_mid - flowers_mid * fx), 
                    int(y1 - 200 * fx))

            # FOR REFERENCE -- Draw border around the flower filter
            # pos2 = ( int(x1 + border_box_mid - flowers_mid*fx) + int(flowers_w*fx), int(y1 - 200 * fx) + int(flowers_h*fx))
            # cv2.rectangle(frame, pos, pos2, (0, 255, 0))

            # Overlay the filter on a frame
            overlay(frame, flowers_resized, pos)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

"""============================================================================
REFERENCE
Transparent filter image from:
https://freepngimg.com/png/16981-snapchat-filters-png-image

============================================================================"""