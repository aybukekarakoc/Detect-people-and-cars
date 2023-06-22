import cv2
import numpy as np

# resim oku
img = cv2.imread("formula.jpg")

# resim gri yap
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gürültü azalt
blurred = cv2.GaussianBlur(gray, (3,3), 0)

# arka plan maskeleme uygula
mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)[1]

# nesnelere canny edge uygula
edges = cv2.Canny(mask, 100, 200)

# nesneleri belirleyen contourlar
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# contourları orjinal resme çiz
cv2.drawContours(img, contours, -1, (0,255,0), 3)

# son resmi göster
cv2.imshow("Contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


