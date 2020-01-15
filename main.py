import cv2
import numpy as np
import scripts.Image_Process as ip
import scripts.Face_Process as fp
from imutils.video import VideoStream
import time

# main
drawing = False  # true if mouse is pressed
useCamera = False
mode = "face_change"
ix, iy = -1, -1
vertex = (-1, -1)
img = cv2.imread('data/test1.jpg')
OriImg = img.copy()
PreImg = img.copy()
sp = img.shape
vertex = (sp[0], sp[1])
ip.Get_Size(sp[0], sp[1])
cv2.namedWindow('image')


def Get_img():
    return img


def process_mode():
    global PreImg
    PreImg = img.copy()
    if mode == "gray":
        ip.Get_GrayImg(img, ix, iy, vertex[0], vertex[1])
    elif mode == "neg":
        ip.Get_Negative(img, ix, iy, vertex[0], vertex[1])
    elif mode == "blur":
        ip.Get_Blur(img, ix, iy, vertex[0], vertex[1])
    elif mode == "mosaic":
        ip.Get_Mosaic(img, ix, iy, vertex[0], vertex[1])
    elif mode == "edge_overlay":
        ip.Get_EdgeImg(img, ix, iy, vertex[0], vertex[1], img)
    elif mode == "edge":
        ip.Get_EdgeImg(img, ix, iy, vertex[0], vertex[1], OriImg)
    elif mode == "face_mosaic":
        fp.human_detectAndDisplay(img, "mosaic")
    elif mode == "face_black":
        fp.human_detectAndDisplay(img, "black")
    elif mode == "face_ellipse":
        fp.human_detectAndDisplay(img, "ellipse")
    elif mode == "face_mask":
        fp.Face_Change(img,useCamera)
    cv2.imshow('image', img)


def draw_rectangle(event, x, y, flags, param):  # select the region which going to change
    global ix, iy, drawing, vertex
    # Click
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y  # click position
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 255), 0)
        vertex = (x, y)
        process_mode()


cv2.setMouseCallback('image', draw_rectangle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if useCamera:
        print("use camera")
        cam = VideoStream(src=0).start()
        time.sleep(1.0)
        mode="gray"
        while True:
            img = cam.read()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                useCamera = False
                break
            elif key == ord("m"):
                cv2.destroyWindow("Mask")
                mode = "face_mosaic"
            elif key == ord("b"):
                cv2.destroyWindow("Mask")
                mode = "face_black"
            elif key == ord("a"):
                mode = "face_mask"
            process_mode()
            cv2.imshow('image', img)

    if k == ord("n"):  # negative img
        mode = "neg"
        print("mode = ", mode)
    elif k == ord("g"):  # gray scale
        mode = "gray"
        print("mode = ", mode)
    elif k == ord("b"):  # gray scale
        mode = "blur"
        print("mode = ", mode)
    elif k == ord("m"):
        mode = "mosaic"
        print("mode = ", mode)
    elif k == ord("e"):  # gray scale
        mode = "edge"
        print("mode = ", mode)
    elif k == ord("o"):  # gray scale
        mode = "edge_overlay"
        print("mode = ", mode)
    elif k== ord("f"):
        mode="face_mosaic"
        img = OriImg.copy()
        print("mode = ", mode)
        process_mode()
    elif k== ord("k"):
        mode="face_black"
        img=OriImg.copy()
        print("mode = ", mode)
        process_mode()
    elif k== ord ("a"):
        mode = "face_mask"
        img = OriImg.copy()
        print("mode = ", mode)
        process_mode()
    elif k == ord("z"):  # cancel previous step
        img = PreImg.copy()
    elif k == ord("c"):  # clear
        img = OriImg.copy()
    elif k == ord("v"):
        useCamera = True
    elif k == ord("q"):
        break
cv2.destroyAllWindows()
