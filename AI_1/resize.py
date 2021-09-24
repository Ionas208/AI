import cv2

homer = cv2.imread('homer.jpg')


scale_factor = 0.1
dim = (int(homer.shape[1]*scale_factor), int(homer.shape[0]*scale_factor)) 

linear = cv2.resize(homer, dim, interpolation=cv2.INTER_LINEAR)
cubic = cv2.resize(homer, dim, interpolation=cv2.INTER_CUBIC)
nearest = cv2.resize(homer, dim, interpolation=cv2.INTER_NEAREST)
area = cv2.resize(homer, dim, interpolation=cv2.INTER_AREA)

cv2.imshow('linear', linear)
cv2.imshow('cubic', cubic)
cv2.imshow('nearest', nearest)
cv2.imshow('area', area)

cv2.waitKey()