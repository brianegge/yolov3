from PIL import ImageDraw,ImageFont,ImageColor
from pprint import pprint

def draw_bbox(image, p, color, label=None, width=4):
    w,h = image.size
    # {'probability': 0.60014141, 'tagId': 1, 'tagName': 'deer', 'boundingBox': {'left': 0.94383056, 'top': 0.82897264, 'width': 0.05527838, 'height': 0.18486874}}
    bbox = p['boundingBox']
    rect_start = (w * bbox['left'], h * bbox['top'])
    rect_end = (w * (bbox['left'] + bbox['width']), h * (bbox['top'] + bbox['height']))
    draw = ImageDraw.Draw(image, 'RGBA')
    outline = ImageColor.getrgb(color) + (128, )
    draw.rectangle((rect_start, rect_end), outline = outline, width=width)
    if label:
        font = ImageFont.truetype("arial.ttf", size=52)
        draw.text( (w * bbox['left'], h * bbox['top'] - 52), text=label, fill=color, font=font)
    del draw

def draw_road(image, points):
    w,h = image.size
    xy = list(map(lambda t: (t[0]*w, t[1] * h), points))
    draw = ImageDraw.Draw(image, 'RGBA')
    fill = ImageColor.getrgb("yellow") + (128, )
    draw.line(xy, fill=fill, width=4)
    del draw

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
