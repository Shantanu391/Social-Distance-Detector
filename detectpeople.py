from socialdistanceconfig import NMS_THRESH
from socialdistanceconfig import MIN_CONF
import numpy as np
import cv2

'''detect_people; the function accepts four parameters:
frame: The frame from our video file 
net: The pre-initialized and pre-trained YOLO object detection model
ln: The YOLO CNN output layer names
personIdx: The YOLO model can detect many types of objects; 
this index is specifically for the person class, as we won’t be considering other objects'''
def detect_people(frame, net, ln, personIdx=0):

    (H, W) = frame.shape[:2]  #grabs the frame dimensions for scaling purposes
    
    '''We then initialize our results list, which the function ultimately returns. 
    The results consist of (1) the person prediction probability, 
    (2) bounding box coordinates for the detection, and 
    (3) the centroid of the object.'''
    results=[]

    # construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, centroids, and
	# confidences, respectively
    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:     # loop over each of the layer outputs
            for detection in output:  # loop over each of the detections
              
              # extract the class ID and confidence (i.e., probability)
			                              # of the current object detection
                scores = detection[5:] 
                classID = np.argmax(scores)
                confidence = scores[classID]

            # filter detections by (1) ensuring that the object
			# detected was a person and (2) that the minimum
			# confidence is met
                if classID == personIdx and confidence > MIN_CONF:
                  
                  # scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
				# and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                
                # update our list of bounding box coordinates,
				# centroids, and confidences
                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
    if len(idxs) > 0:
	    for i in idxs.flatten():
		    (x, y) = (boxes[i][0], boxes[i][1])
		    (w, h) = (boxes[i][2], boxes[i][3])

		    r = (confidences[i], (x, y, x + w, y + h), centroids[i])
		    results.append(r)

    return results