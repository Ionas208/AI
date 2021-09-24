import cv2

bit1 = cv2.imread('bit-1.png')
bit2 = cv2.imread('bit-2.png')

anded = cv2.bitwise_and(bit1, bit2, mask=None)
ord = cv2.bitwise_or(bit1, bit2, mask=None)
xord = cv2.bitwise_xor(bit1, bit2, mask=None)
notted = cv2.bitwise_not(bit1, bit2, mask=None)
cv2.imshow('AND', anded)
cv2.imshow('OR', ord)
cv2.imshow('XOR', xord)
cv2.imshow('NOT', notted)

cv2.waitKey()