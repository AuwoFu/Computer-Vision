import cv2
import pyautogui

imgSize_x = 0
imgSize_y = 0


def Get_Size(y, x):
    global imgSize_x, imgSize_y
    imgSize_x = x
    imgSize_y = y


def Bound_Process(x1, y1, x2, y2):
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > imgSize_x:
        x2 = imgSize_x
    if y2 > imgSize_y:
        y2 = imgSize_y
    return x1, y1, x2, y2


def Get_GrayImg(img, x1, y1, x2, y2):
    x1, y1, x2, y2 = Bound_Process(x1, y1, x2, y2)
    trans_img = img
    clip = img[y1: y2, x1: x2]
    gray_img = cv2.cvtColor(clip, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    trans_img[y1: y2, x1: x2] = gray_img


def Get_Negative(img, x1, y1, x2, y2):
    x1, y1, x2, y2 = Bound_Process(x1, y1, x2, y2)
    trans_img = img
    gray_img = cv2.cvtColor(img[y1: y2, x1: x2], cv2.COLOR_BGR2GRAY)
    for j in range(x2 - x1):
        for i in range(y2 - y1):
            gray_img[i, j] = 255 - gray_img[i, j]
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    trans_img[y1: y2, x1: x2] = gray_img


def Get_Blur(img, x1, y1, x2, y2):
    x1, y1, x2, y2 = Bound_Process(x1, y1, x2, y2)
    trans_img = img
    kernel_size = (5, 5)
    sigma = 1.5
    clip = img[y1:y2, x1:x2]
    blur_img = cv2.GaussianBlur(clip, kernel_size, sigma)
    trans_img[y1:y2, x1:x2] = blur_img


def Get_Mosaic(img, x1, y1, x2, y2, scale=5):
    x1, y1, x2, y2 = Bound_Process(x1, y1, x2, y2)
    trans_img = img
    clip = img[y1:y2, x1:x2]
    mas_img = clip.copy()
    for j in range(0, x2 - x1, scale):
        for i in range(0, y2 - y1, scale):
            for l in range(scale):
                for k in range(scale):
                    if (i + k) >= (y2 - y1) or (j + l) >= (x2 - x1):
                        break
                    mas_img[i + k, j + l] = clip[i, j]

    trans_img[y1:y2, x1:x2] = mas_img


def Get_EdgeImg(img, x1, y1, x2, y2, src):
    x1, y1, x2, y2 = Bound_Process(x1, y1, x2, y2)
    trans_img = img
    clip = src[y1:y2, x1:x2]
    gray_img = cv2.cvtColor(clip, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(gray_img, 100, 200)
    edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
    trans_img[y1:y2, x1:x2] = edge_img


