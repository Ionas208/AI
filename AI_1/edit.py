import cv2

homer = cv2.resize(cv2.imread('homer.jpg'),(400,350))

edges = cv2.Canny(homer, 100, 200)

cv2.imshow('edges', edges)

cv2.waitKey()