import cv2
import numpy as np
import scripts.Image_Process as ip

human_cascade = cv2.CascadeClassifier()
if not human_cascade.load("data/lbpcascade_frontalface_improved.xml"):
    print("Error loading face cascade")
    exit(0)


def human_detectAndDisplay(img, cmdtype=""):
    frame = img
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = human_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:

        if cmdtype == "mosaic":
            ip.Get_Mosaic(frame, x, y, x + 2 * w, y + 2 * h, 10)
        elif cmdtype == "black":
            for i in range(x, x + w):
                for j in range(y, y + h):
                    frame[j, i] = (0, 0, 0)
        elif cmdtype == "ellipse":
            center = (x + w // 2, y + h // 2)
            frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        else:
            center_x = x + w // 2
            center_y = y + h // 2
            return center_x, center_y


def Face_Change(img,usecamera):
    frame = img
    clipart = cv2.imread("data/mask_cat.jpg")
    sh = clipart.shape
    h = sh[0]
    half_h=int(h/2)
    w = sh[1]
    half_w=int(w/2)
    try:
        center_x, center_y = human_detectAndDisplay(frame)
    except:
        return frame
    B, G, R = cv2.split(clipart)
    mask = ~((B == 255) & (G == 255) & (R == 255))
    src = mask.astype(np.uint8)
    structingElement = np.ones((3, 3))
    src = cv2.erode(src, structingElement)
    src = cv2.dilate(src, structingElement)
    mask=src.astype(bool)
    sh=frame.shape
    try:
        for i in range(h):
            for j in range(w):
                if mask[i,j]:
                    frame[(center_y - half_h+i), (center_x - half_w+j)]=clipart[i,j]
        if usecamera:
            cv2.imshow("Mask", frame)
    except:
        cv2.imshow("Mask", frame)