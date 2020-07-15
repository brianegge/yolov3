
def bb_intersection_over_union(boxA, boxB):
    if isinstance(boxA,dict):
	    # determine the (x, y)-coordinates of the intersection rectangle in x1,y1,x2,y2 format
	    xA = max(boxA['left'], boxB['left'])
	    yA = max(boxA['top'], boxB['top'])
	    xB = min(boxA['left'] + boxA['width'], boxB['left'] + boxB['width'])
	    yB = min(boxA['top'] + boxA['height'], boxB['top'] + boxB['height'])
	    # compute the area of intersection rectangle
	    interArea = max(0, xB - xA) * max(0, yB - yA)
	    # compute the area of both the prediction and ground-truth
	    # rectangles
	    boxAArea = (boxA['width']) * (boxA['height'])
	    boxBArea = (boxB['width']) * (boxB['height'])
	    # compute the intersection over union by taking the intersection
	    # area and dividing it by the sum of prediction + ground-truth
	    # areas - the interesection area
	    iou = interArea / float(boxAArea + boxBArea - interArea)
	    # return the intersection over union value
	    return iou
    else:
	    # determine the (x, y)-coordinates of the intersection rectangle in x1,y1,x2,y2 format
	    xA = max(boxA[0], boxB[0])
	    yA = max(boxA[1], boxB[1])
	    xB = min(boxA[2], boxB[2])
	    yB = min(boxA[3], boxB[3])
	    # compute the area of intersection rectangle
	    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	    # compute the area of both the prediction and ground-truth
	    # rectangles
	    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	    # compute the intersection over union by taking the intersection
	    # area and dividing it by the sum of prediction + ground-truth
	    # areas - the interesection area
	    iou = interArea / float(boxAArea + boxBArea - interArea)
	    # return the intersection over union value
	    return iou
