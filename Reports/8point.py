import cv2
import numpy as np

from matplotlib import pyplot as plt

def drawlines(img1, img2, lines, pts1, pts2, user_input):

    if user_input == '1':
        r, c = img1.shape
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    elif user_input == '2':
        r, c = img1.shape

        img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)

    color = ((0, 0, 0),                                     #선의 색깔 8개를 결정한다.
             (255, 0, 0), (0, 255, 0), (0, 0, 255),
             (255, 255, 0), (0, 255, 255), (255, 0, 255),
             (255, 255, 255))

    for r, pt1, pt2, number in zip(lines, pts1, pts2, range(8)):
        x0,y0 = map(int, [0, -r[2]/r[1] ])  #(0, y) line 위의 적당한 점이다.
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ]) #(c, y), line 위의 적당한 점이다.
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color[number], 1)
        #img1 = cv2.circle(img1, tuple(pt1), 5, color[number], -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color[number], -1)
    return img1, img2

path_image = '8point_image/'

user_input = input("1 또는 2를 입력하시오 : ")
if user_input == '1':
    img1 = cv2.imread(path_image + 'left_image.jpg', 0)
    img2 = cv2.imread(path_image + 'right_image.jpg', 0)
    pt1 = [[51, 146], [49, 226], [48, 306], [112, 173], [111, 226], [111, 282], [202, 202], [203, 282]]
    pt2 = [[112, 147], [110, 221], [111, 295], [162, 168], [161, 219], [162, 273], [246, 190], [245, 272]]
elif user_input == '2':
    img1 = cv2.imread(path_image + '8point-1.jpg', 1)
    img2 = cv2.imread(path_image + '8point-2.jpg', 1)
    pt1 = [[184, 85], [172, 116], [445, 39], [454, 73], [176, 213], [265, 207], [487, 130], [477, 98]]
    pt2 = [[84, 207], [72, 239], [202, 82], [196, 116], [73, 326], [109, 309], [365, 175], [350, 136]]
else:
    print("wrong input")
    exit()
# pt1 = [[79, 57], [75, 214], [74, 374], [181, 111], [180, 216], [179, 322], [334, 16], [334, 166]]
# pt2 = [[175, 82], [173, 227], [174, 374], [258, 124], [258, 225], [257, 327], [396, 10], [397, 166]]
pt1 = np.int32(pt1)
pt2 = np.int32(pt2)
# print("pt", pt1.shape, pt2.shape)

# pt1 = cv2.convertPointsToHomogeneous(pt1, pt1)
# pt2 = cv2.convertPointsToHomogeneous(pt2, pt2)
# print("homo", pt1.shape, pt2.shape, pt1[0])
#
# A = np.array([])
#
# for i in range(8):
#     list = []
#     for j in range(3):
#         for k in range(3):
#             value = pt1[i][0][j] * pt2[i][0][k]
#             list.append(value)
#     A = np.concatenate((A, list[0:9]))
# A = A.reshape(-1, 9)
# print(A.shape, A[0])
#
#
#
# #
#
# #
# #
# [U, D, V] = np.linalg.svd(A)
# print("SVD", U.shape, D.shape, V.shape)
# F_ = np.array(V[:, 8])
# F_ = F_.reshape(3, 3)
# print("F\'", F_)
#
# [U, D, V] = np.linalg.svd(F_)
# print("SVD", U.shape, D.shape, V.shape)
# D[2] = 0
# print(D)
# F = U * D * V
# print(F.shape, F)

#print("ex", np.dot(pt1[0][0], F))
#pt2 = pt2.reshape(-1, 3, 1)
# print(pt2[0])
Fmatrix, _ = cv2.findFundamentalMat(pt1, pt2, 8)

pt1_h = cv2.convertPointsToHomogeneous(pt1, pt1)    #homo형태로 변환
pt2_h = cv2.convertPointsToHomogeneous(pt2, pt2)
pt2_h = pt2_h.reshape(-1, 3, 1)                     #계산을 위해 3,1 행렬로 변환

print(pt1_h.shape, pt2_h.shape)
equation = np.dot(pt1_h[0], Fmatrix)
equation = np.dot(equation, pt2_h[0])
print(equation)


#left line draw
line1 = cv2.computeCorrespondEpilines(pt2, 2, Fmatrix)
# print(line1.shape)
# print(line1[0])

line1 = line1.reshape(-1, 3)
img_l1, img_l2 = drawlines(img1, img2, line1, pt1, pt2, user_input)



#right line draw
line2 = cv2.computeCorrespondEpilines(pt1, 1, Fmatrix)

line2 = line2.reshape(-1, 3)
img_r1, img_r2 = drawlines(img2, img1, line2, pt2, pt1, user_input)

plt.subplot(122), plt.imshow(img_r1)
plt.subplot(121), plt.imshow(img_r2)
plt.show()