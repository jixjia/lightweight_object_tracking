import numpy as np
import pandas as pd
import cv2
import logging
import argparse


def registerNewObject(cx, cy):
    global totalObjects
    global df

    objId = "Id"+str(totalObjects)
    df[objId] = ""
    df.at[int(frameNum), objId] = [cx, cy]
    totalObjects += 1
    print("New object {0}, coords:{1}".format(objId, [cx, cy]))
    print("Current total objects:", totalObjects)


if __name__ == '__main__':

    # Arg parser
    parser = argparse.ArgumentParser(description='Specify path to video')
    parser.add_argument('-p', '--video_path', type=str,
                        default='videos/test_animation.mp4', help='Enter path to video file')
    args = parser.parse_args()

    # Load video
    cap = cv2.VideoCapture(args.video_path)
    frames_count, fps, width, height = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                                        int(cap.get(cv2.CAP_PROP_FPS)),
                                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Total Frames: {}, FPS: {}, W: {}, H:{}".format(
        frames_count, fps, width, height))

    # Running parameter
    totalObjects = 0
    frameNum = 0
    minArea = 1000
    maxArea = 50000
    scaleRatio = 0.8
    upperLinePos = int(height*scaleRatio*0.3)
    upperLineColor = (0, 255, 0)  # green
    lowerLinePos = int(height*scaleRatio*0.59)
    lowerLineColor = (0, 255, 255)  # yellow
    lineWidth = 3

    # Create a tracking dataframe, each video frame corresponds to a single row
    df = pd.DataFrame(index=range(int(frames_count)))
    df.index.name = "Frame Number"

    # OpenCV Gaussian BG Subtractor module
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    while True:
        ret, frame = cap.read()

        ##### Process Image #####
        if ret:
            # 1.1 FGBG Mask preparation
            # Downsize input frame to sepcified ratio
            image = cv2.resize(frame, None, fx=scaleRatio, fy=scaleRatio)

            # Change color space
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Fgbg mask
            fgmask = fgbg.apply(gray)

            # 1.2 Apply 2D convolutions to enhance contrast of objects (Morphological Transformation)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            _, bins = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            dilation = cv2.dilate(bins, kernel)

            # 1.3 Get contour of objects
            contours, _ = cv2.findContours(
                dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # 1.4 Apply Convex hull to create polygon around contours
            hull = [cv2.convexHull(c) for c in contours]
            cv2.drawContours(image, hull, -1, (0, 255, 0), 1)

            # 1.5 Draw two lines: Upper Line and Lower Line.
            # Only objects bound by the two lines will be processed
            # Upper Line
            cv2.line(
                image,
                (0, lowerLinePos),
                (width, lowerLinePos),
                lowerLineColor,
                lineWidth
            )
            # Lower Line
            cv2.line(
                image,
                (0, upperLinePos),
                (width, upperLinePos),
                upperLineColor,
                lineWidth
            )

            # 1.6 Process contours
            # Create two numpy lists to store contour's centroid coordiantes (x and y)
            contoursX = np.zeros(len(contours))
            contoursY = np.zeros(len(contours))

            for i in range(len(contours)):
                # Compute area size of a contor
                area = cv2.contourArea(contours[i])

                # If area size exceeds our threshold then process it
                if minArea < area < maxArea:
                    cnt = contours[i]
                    # Calculate contor's centroid using Moments
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # If the contor's centroid falls between the upper and lower line
                    if upperLinePos < cy < lowerLinePos:
                        # Draw bounding box around contour
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(
                            image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        # Draw contour's centroid
                        cv2.drawMarker(
                            image,
                            (cx, cy),
                            (0, 0, 255),
                            cv2.MARKER_STAR,
                            markerSize=10,
                            thickness=1,
                            line_type=cv2.LINE_AA
                        )

                        # Add to centroid list
                        contoursX[i] = cx
                        contoursY[i] = cy

            # Discard elements did not satisfy criteria
            contoursX = contoursX[contoursX != 0]
            contoursY = contoursY[contoursY != 0]

            ##### Tracking Program #####

            # Check if there are contours bound between two lines
            if len(contoursX):
                # Check if tracking dataframe is empty (0 objects tracked)
                if frameNum == 0 or df.shape[1] == 0:
                    for i in range(len(contoursX)):
                        # Register current contour object for tracking
                        registerNewObject(contoursX[i], contoursY[i])
                 # If there is object being tracked, look for it's coordinate from previous frame
                else:
                    sum = 0
                    for i in range(totalObjects):
                        sum += len(df.iloc[int(frameNum-1), i])

                    if sum > 0:
                        for i in range(totalObjects):
                            if (len(contoursX)):
                                dist = np.zeros((totalObjects, len(contoursX)))
                                objId = "Id"+str(i)
                                prevCoords = df.iloc[int(frameNum-1)][objId]

                                if not prevCoords:
                                    continue
                                else:
                                    for j in range(len(contoursX)):
                                        dist[i, j] = np.abs(
                                            contoursX[j] - prevCoords[0]) + np.abs(contoursY[j] - prevCoords[1])

                                    minIndex = int(np.argmin(dist[i, :]))

                                    # Assign closest coords to the object
                                    df.at[int(frameNum), objId] = [
                                        contoursX[minIndex], contoursY[minIndex]]

                                    logging.info("Coords {0} allocated to {1} {2} ".format(
                                        [contoursX[minIndex], contoursY[minIndex]], objId, prevCoords))

                                    # Get velocity of the object
                                    velocity = np.abs(
                                        contoursX[minIndex] - prevCoords[0]) + np.abs(contoursY[minIndex] - prevCoords[1]) * fps

                                    cv2.putText(image, "(Speed: "+str(int(velocity))+" km/h)", (int(contoursX[minIndex]) + 10, int(
                                        contoursY[minIndex]) + 10), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)

                                    # Get rid of allocated coords from contoursX/contoursY list
                                    contoursX = np.delete(contoursX, minIndex)
                                    contoursY = np.delete(contoursY, minIndex)
                            else:
                                break

                         # For residual coords register them as new objects
                        if(len(contoursX)):
                            for i in range(len(contoursX)):
                                registerNewObject(contoursX[i], contoursY[i])

                    else:
                        print(
                            "Registering objects for tracking")
                        for i in range(len(contoursX)):
                            registerNewObject(contoursX[i], contoursY[i])

            # 2.1 Display total object counts
            cv2.rectangle(image, (0, 0), (160, 70), (162, 115, 0), -1)
            cv2.putText(image, "COUNTS: "+str(totalObjects), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)

            # 3.1 Display image stream
            fgmask_window = cv2.resize(fgmask, None, fx=.5, fy=.5)
            dilation_window = cv2.resize(dilation, None, fx=.5, fy=.5)
            bins_window = cv2.resize(bins, None, fx=.5, fy=.5)

            cv2.imshow("Traffic Monitor", image)
            cv2.moveWindow("Traffic Monitor", 0, 0)

            cv2.imshow("fgmask", fgmask_window)
            cv2.moveWindow("fgmask", int(width * scaleRatio), 0)

            cv2.imshow("Dilation", dilation_window)
            cv2.moveWindow("Dilation", int(width * scaleRatio),
                           int(height * scaleRatio * 0.5))

            frameNum += 1

            k = cv2.waitKey(int(1000/fps)) & 0xff  # press ESC to quit
            if k == 27:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save dataframe to csv file for analysis
    df.to_csv('tracking.csv', sep=',')
    print("Program Finished! Saved tracking result to CSV")
