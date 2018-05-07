import cv2
import numpy as np

from matplotlib import pyplot as plt

def drawlines(img1, img2, lines, pts1, pts2, user_input):
    if user_input == '1':
        r, c = img1.shape
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    elif user_input == '2':
        r, c, _ = img1.shape
        img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)

    color = ((0, 0, 0),                                     #선의 색깔 8개를 결정한다.
             (255, 0, 0), (0, 255, 0), (0, 0, 255),
             (255, 255, 0), (0, 255, 255), (255, 0, 255),
             (255, 255, 255))

    for r, pt1, pt2, number in zip(lines, pts1, pts2, range(8)):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
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

pt1 = np.int32(pt1)
pt2 = np.int32(pt2)

A = np.array([])

pt1_h = cv2.convertPointsToHomogeneous(pt1, pt1)
pt2_h = cv2.convertPointsToHomogeneous(pt2, pt2)
print("homo", pt1.shape, pt2.shape, pt1_h[0])


for i in range(8):
    list = []
    for j in range(3):
        for k in range(3):
            #print(j, k)
            value = pt1_h[i][0][j] * pt2_h[i][0][k]
            list.append(value)
    A = np.concatenate((A, list))
A = A.reshape(-1, 9)            #8쌍의 점으로 만듬 => 8*9행렬 마지막 열은 1(homo를 써서 그럼)

U, D, V_t = np.linalg.svd(A)
print(U.shape, D.shape, V_t.shape)
#axis_min = np.argmin(D)

F_ = np.array(V_t[len(D)])      #F' = V'의 마지막 행
F_ = F_.reshape(3, 3)

U, D, V_t = np.linalg.svd(F_)

axis_min = np.argmin(D)
D[axis_min] = 0                 #가장 작은 값을 0으로

D_ = np.zeros((len(U), len(V_t)))
D_[:len(V_t), :len(V_t)] = np.diag(D)
print(D_)
F = np.dot(np.dot(U, D_), V_t)      #F = U * D * V'
print(F[0], F_[0])
pt2_h = pt2_h.reshape(-1, 3, 1)

equation = np.dot(np.dot(pt1_h[0], F), pt2_h[0])
print(equation)

line1 = cv2.computeCorrespondEpilines(pt2, 2, F)
# print(line1.shape)
# print(line1[0])

line1 = line1.reshape(-1, 3)
img_l1, img_l2 = drawlines(img1, img2, line1, pt1, pt2, user_input)

line2 = cv2.computeCorrespondEpilines(pt1, 1, F)

line2 = line2.reshape(-1, 3)
img_r1, img_r2 = drawlines(img2, img1, line2, pt2, pt1, user_input)



plt.subplot(122), plt.imshow(img_r1)
plt.subplot(121), plt.imshow(img_r2)
plt.show()