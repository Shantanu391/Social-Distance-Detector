import  socialdistanceconfig as config
import detectpeople
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", type=int, default=1,help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
print(LABELS)

weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

print("[INFO] accessing video stream...")

vs = cv2.VideoCapture("vid2.mp4")

#begin processing frames and determining if people are maintaining safe social distance

# loop over the frames from the video stream
while True:
    # read the next frame from the file
    grabbed, frame = vs.read()

    # if the frame was not grabbed, then we have reached the end
	# of the stream
    if not grabbed:
        break

    # resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=500)
    personIdx = LABELS.index('person')

    #Using our detect_people function implemented earlier, 
    #we grab results of YOLO object detection
    results = detectpeople.detect_people(frame, net, ln,personIdx)

    # initialize the set of indexes that violate the minimum social
	# distance
    violate = set()
    
    # ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
    if len(results) >= 2:
        
        # extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                
                # check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels
                if D[i, j] < config.MIN_DISTANCE:

                    # update our violation set with the indexes of
					# the centroid pairs
                    violate.add(i)
                    violate.add(j)
                    
#let’s annotate our frame with rectangles, circles, and text:                    
     
    # loop over the results               
    for (i, (prob, bbox, centroid)) in enumerate(results):

        # extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation to green   
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # if the index pair exists within the violation set, then
		# update the color to red
        if i in violate:
            color = (0, 0, 255)

        # draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    # display the total number of social distancing violations on the
	# output frame
    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    # check to see if the output frame should be displayed to our
	# screen
    if args["display"] > 0:

      # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

      # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

vs.release()
cv2.destroyAllWindows()