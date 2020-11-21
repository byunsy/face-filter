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
    fil, a filter
    pos, x and y coordinate tuple to position to filter
PURPOSE:
    overlays a four-channel iamge on a three-channel source image
PRODUCES:
    None - a void function
============================================================================"""
def overlay(img, fil, pos):

    # Compute the start and end xy coordinates to overlay 
    # - sx, sy : start x and y
    # - ex, ey : end x and y
    sx = pos[0]
    ex = pos[0] + fil.shape[1]
    sy = pos[1]
    ey = pos[1] + fil.shape[0]

    # If filter goes outside the source image boundaries, then stop overlay
    if sx < 0 or sy < 0 or ex > img.shape[1] or ey > img.shape[0]:
        return

    # img1: portion of source image
    # img2: filter iamge
    img1 = img[sy:ey, sx:ex]            # shape=(h, w, 3)
    img2 = fil[:, :, 0:3]               # shape=(h, w, 3)
    alpha = 1. - (fil[:, :, 3] / 255.)  # shape=(h, w)

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

    # Open filter images
    flowers = cv2.imread('./filter/filter1.png', cv2.IMREAD_UNCHANGED)
    dog     = cv2.imread('./filter/filter2.png', cv2.IMREAD_UNCHANGED)
    love    = cv2.imread('./filter/filter3.png', cv2.IMREAD_UNCHANGED)
    cat     = cv2.imread('./filter/filter4.png', cv2.IMREAD_UNCHANGED)
    bunny   = cv2.imread('./filter/filter5.png', cv2.IMREAD_UNCHANGED)
    wig     = cv2.imread('./filter/filter6.png', cv2.IMREAD_UNCHANGED)
    fox     = cv2.imread('./filter/filter7.png', cv2.IMREAD_UNCHANGED)
    dog2    = cv2.imread('./filter/filter8.png', cv2.IMREAD_UNCHANGED)
    crown   = cv2.imread('./filter/filter9.png', cv2.IMREAD_UNCHANGED)
    rat     = cv2.imread('./filter/filter10.png', cv2.IMREAD_UNCHANGED)

    # List to store filters
    # - filt_y_list determines the y position of the filter 
    # - need to finetune through trial and error for each filter
    filter_list = [flowers, dog, love, cat, bunny, wig, fox, dog2, crown, rat]
    fil_y_list = [200, 80, 200, 100, 300, 320, 120, 200, 300, 60]    

    # Set current filter and its y position
    counter = 0
    cur_fil = filter_list[counter]
    cur_fil_y = fil_y_list[counter]

    # Check image input
    for fil in filter_list:
        if fil is None:
            print('Error: Failed to open PNG image.')
            sys.exit()

    # Attain current filter dimensions and midpoint
    cur_fil_h, cur_fil_w = cur_fil.shape[:2]
    cur_fil_mid = cur_fil_w // 2

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
            # cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 
            #             0.8, (0, 255, 0), 1, cv2.LINE_AA)

            # Scaling factors for each filter
            fx1 = (border_box_w / cur_fil_w) * 1.5
            fx2 = (border_box_w / cur_fil_w) * 1.3
            fx3 = (border_box_w / cur_fil_w) * 1.5
            fx4 = (border_box_w / cur_fil_w) * 1.5
            fx5 = (border_box_w / cur_fil_w) * 2
            fx6 = (border_box_w / cur_fil_w) * 2.3
            fx7 = (border_box_w / cur_fil_w) * 1.2
            fx8 = (border_box_w / cur_fil_w) * 1.2
            fx9 = (border_box_w / cur_fil_w) * 1.2
            fx10 = (border_box_w / cur_fil_w) * 1.2
            fx_list = [fx1, fx2, fx3, fx4, fx5, fx6, fx7, fx8, fx9, fx10]

            # Resize filter using scaling factors
            fx = fx_list[counter]
            cur_fil_resized = cv2.resize(cur_fil, (0, 0), fx=fx, fy=fx, 
                                         interpolation=cv2.INTER_AREA)

            # Calculate the xy coordinates to overlay the resized filter
            # - the top left corner coordinate
            pos = ( int(x1 + border_box_mid - cur_fil_mid * fx), 
                    int(y1 - cur_fil_y * fx))

            # FOR REFERENCE -- Draw border around the  filter
            # pos2 = (int(x1 + border_box_mid - cur_fil_mid*fx) + int(cur_fil_w*fx), 
            #         int(y1 - cur_fil_y * fx) + int(cur_fil_h*fx))
            # cv2.rectangle(frame, pos, pos2, (0, 255, 0))

            # Overlay the filter on a frame
            overlay(frame, cur_fil_resized, pos)

        cv2.imshow('Face Filter', frame)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

        # Press space to change filters
        if key == 32:  # Space
            if counter < len(filter_list) - 1:
                # Change to next filter
                counter += 1
                cur_fil = filter_list[counter]
                cur_fil_y = fil_y_list[counter]

                # Attain the new dimensions 
                cur_fil_h, cur_fil_w = cur_fil.shape[:2]
                cur_fil_mid = cur_fil_w // 2
            else:
                # Change to first filter
                counter = 0
                cur_fil = filter_list[counter]
                cur_fil_y = fil_y_list[counter]

                # Attain the new dimensions 
                cur_fil_h, cur_fil_w = cur_fil.shape[:2]
                cur_fil_mid = cur_fil_w // 2

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


""" REFERENCES
Transparent filter image from:
1.  https://freepngimg.com/png/16981-snapchat-filters-png-image
2.  https://freepngimg.com/png/16984-snapchat-filters-png-clipart
3.  https://freepngimg.com/png/16980-snapchat-filters-png-picture
4.  https://freepngimg.com/png/16977-snapchat-filters-png
5.  https://freepngimg.com/png/16989-snapchat-filters-free-png-image
6.  https://www.stickpng.com/img/clothes/wigs/wig-yellow-bob
7.  https://www.pinclipart.com/pindetail/ThmJim_snapchat-filters-clipart-pink-flower-animal-filter-png/
8.  https://www.pinclipart.com/pindetail/iTwRobw_dogears-snapchat-snapchatfilter-glassesfilter-glasses-dog-ears-with/
9.  https://www.pinclipart.com/pindetail/iTwRoii_snapchat-filters-clipart-aesthetic-transparent-background-aesthetic-heart/
10. https://www.pinclipart.com/pindetail/ibToTbb_snapchat-filters-clipart-snow-snapchat-filtros-de-hallowen/
"""