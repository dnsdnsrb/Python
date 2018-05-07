import cv2
import numpy as np
import math
from moviepy.editor import VideoFileClip

class LaneDetector:
    def __init__(self):
        self.image = []
        self.video_path = "Video/"

        self.left_average = (0, 0, 0, 0)
        self.right_average = (0, 0, 0 ,0)

    def gaussian_blur(self, image, kernel_size):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def perp(self, a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    def seg_intersect(self, x1, x2, y1, y2):
        dx = x2 - x1
        dy = y2 - y1
        dp = x1 - y1
        dx_p = self.perp(dx)
        denormallize = np.dot(dx_p, dy)
        num = np.dot(dx_p, dp)

        return (num / denormallize.astype(float)) * dy + y1

    def movingAverage(self, average, sample, N=20):
        if average == 0:
            return sample
        average = average - average / N #1개 지우고
        average = average + sample / N  #1개 새로 추가
        return average

    def canny(self, image, low=40, high=50):
        return cv2.Canny(image, low, high)

    def region_of_interest(self, image, vertices):
        mask = np.zeros_like(image)

        if len(image.shape) >= 3: #3 = rgb, others = maybe grey
            channel = image.shape[2]
            maskBackground = (255,) * channel
        else:
            maskBackground = 255

        cv2.fillPoly(mask, vertices, maskBackground)

        image_masked = cv2.bitwise_and(image, mask)
        return image_masked

    def hough(self, image, rho, theta,threshold, minLineLength , maxLineGap):
        image_line = cv2.HoughLinesP(image, rho, theta,
                                      threshold, np.array([]), minLineLength=minLineLength, maxLineGap=maxLineGap)

        image = np.zeros((*image.shape, 3), dtype=np.uint8)
        self.drawline(image, image_line)
        return image

    def line(self, image, line_most, point_top1, point_top2, point_bottom1, point_bottom2, side):
        p3 = np.array( [line_most.coor1[0], line_most.coor1[1]])
        p4 = np.array( [line_most.coor2[0], line_most.coor2[1]])
        point_top = self.seg_intersect(point_top1, point_top2, p3, p4)  #위 반쪽 영역(좌우 중 하나)
        point_bottom = self.seg_intersect(point_bottom1, point_bottom2, p3, p4) #아래 반쪽 영역(좌우 중 하나)
        # print("test", point_top[0])

        # None
        if math.isnan(point_top[0]) or math.isnan(point_bottom[0]):
            x1_average, y1_average, x2_average, y2_average = self.left_average
            cv2.line(image,
                     (int(x1_average), int(y1_average)),
                     (int(x2_average), int(y2_average)),
                     [255, 255, 255], 12)
            x1_average, y1_average, x2_average, y2_average = self.right_average
            cv2.line(image,
                     (int(x1_average), int(y1_average)),
                     (int(x2_average), int(y2_average)),
                     [255, 255, 255], 12)
            return

        cv2.line(image,
                 (int(point_top[0]), int(point_top[1])),
                 (int(point_bottom[0]), int(point_bottom[1])),
                 [0, 0, 255], 8)

        # Average
        if side=='left':
            average = self.left_average
        else:
            average = self.right_average

        x1_average, y1_average, x2_average, y2_average = average
        average = (self.movingAverage(x1_average, point_top[0]),
                   self.movingAverage(y1_average, point_top[1]),
                   self.movingAverage(x2_average, point_bottom[0]),
                   self.movingAverage(y2_average, point_bottom[1]))

        x1_average, y1_average, x2_average, y2_average = average

        if side=='left':
            self.left_average = average
        else:
            self.right_average = average

        # print("white", self.left_average, self.right_average)
        cv2.line(image,
                 (int(x1_average), int(y1_average)),
                 (int(x2_average), int(y2_average)),
                 [255, 255, 255], 12)

    def drawline(self, image, lines, color=[255, 0, 0], thickness=2):
        class Line:
            pass
        leftmost = Line()
        leftmost.size = 0
        leftmost.coor1 = (0, 0)
        leftmost.coor2 = (0, 0)

        rightmost = Line()
        rightmost.size = 0
        rightmost.coor1 = (0, 0)
        rightmost.coor2 = (0, 0)

        #
        if lines is None:
            x1_average, y1_average, x2_average, y2_average = self.left_average
            cv2.line(image, (int(x1_average), int(y1_average)), (int(x2_average), int(y2_average)), [255, 255, 255], 12)
            x1_average, y1_average, x2_average, y2_average = self.right_average
            cv2.line(image, (int(x1_average), int(y1_average)), (int(x2_average), int(y2_average)), [255, 255, 255], 12)
            return

        for line in lines:
            for x1, y1, x2, y2 in line:
                size = math.hypot(x2 - x1, y2 - y1) #hypot(3, 4) => sqrt(3^2 + 4^2) = 5
                slope = (y2 - y1) / (x2 - x1)

                if slope > 0.5:
                    if size > rightmost.size:
                        rightmost.coor1 = (x1, y1)
                        rightmost.coor2 = (x2, y2)
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
                elif slope < -0.5:
                    if size > leftmost.size:
                        leftmost.coor1 = (x1, y1)
                        leftmost.coor2 = (x2, y2)
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)

        # Top & Bottom
        image_height, image_width = image.shape[0], image.shape[1]
        p = int(image_height - image_height / 3)
        point_top1 = np.array([0, p])
        point_top2 = np.array([int(image_width), p])
        point_bottom1 = np.array([0, int(image_height)])
        point_bottom2 = np.array([int(image_width), int(image_height)])

        # Side
        self.line(image, leftmost, point_top1, point_top2, point_bottom1, point_bottom2, 'left')
        self.line(image, rightmost, point_top1, point_top2, point_bottom1, point_bottom2, 'right')


        # # Left
        # p3 = np.array( [leftmost.coor1[0], leftmost.coor1[1]])
        # p4 = np.array( [leftmost.coor2[0], leftmost.coor2[1]])
        # point_topLeft = self.seg_intersect(point_top1, point_top2, p3, p4)
        # point_bottomLeft = self.seg_intersect(point_bottom1, point_bottom2, p3, p4)
        #
        #
        #     # None
        # if math.isnan(point_topLeft[0]) or math.isnan(point_bottomLeft[0]):
        #     x1_average, y1_average, x2_average, y2_average = self.left_average
        #     cv2.line(image,
        #              (int(x1_average), int(y1_average)),
        #              (int(x2_average), int(y2_average)),
        #              [255, 255, 255], 12)
        #     x1_average, y1_average, x2_average, y2_average = self.right_average
        #     cv2.line(image,
        #              (int(x1_average), int(y1_average)),
        #              (int(x2_average), int(y2_average)),
        #              [255, 255, 255], 12)
        #     return
        #
        # cv2.line(image,
        #          (int(point_topLeft[0]), int(point_topLeft[1])),
        #          (int(point_bottomLeft[0]), int(point_bottomLeft[1])),
        #          [0, 0, 255], 8)
        #
        #     # Average Left
        # x1_average, y1_average, x2_average, y2_average = self.left_average
        # self.left_average = (self.movingAverage(x1_average, point_topLeft[0]),
        #                      self.movingAverage(y1_average, point_topLeft[1]),
        #                      self.movingAverage(x2_average, point_bottomLeft[0]),
        #                      self.movingAverage(y2_average, point_bottomLeft[1]))
        #
        # x1_average, y1_average, x2_average, y2_average = self.left_average
        #
        # cv2.line(image,
        #          (int(x1_average), int(y1_average)),
        #          (int(x2_average), int(y2_average)),
        #          [255, 255, 255], 12)
        # #
        #
        # # Right
        # p5 = np.array( [rightmost.coor1[0], rightmost.coor1[1]])
        # p6 = np.array([rightmost.coor2[0], rightmost.coor2[1]])
        # point_topRight = self.seg_intersect(point_top1, point_top2, p5, p6)
        # point_bottomRight = self.seg_intersect(point_bottom1, point_bottom2, p5, p6)
        #
        # if(math.is)

    def weighted_img(self, image, initial_img, a=0.8, b=1.0, rambda=0.0):
        return cv2.addWeighted(initial_img, a, image, b, rambda)

    def process(self, image_origin):
        image = self.gaussian_blur(image_origin, 7)
        image = self.canny(image, 40, 50)

        # 탐색 범위 정하는데 사용
        height = image.shape[0]
        width = image.shape[1]
        vertices = np.array([[[4 * width / 4, 3 * height / 5], [width / 4, 3 * height / 5],
                             [40, height], [width - 40, height]]],
                            dtype=np.int32)
        #

        image = self.region_of_interest(image, vertices)

        image = self.hough(image, 1, np.pi/180, 40, 30, 200)

        image = self.weighted_img(image, image_origin)
        return image

    def test(self, name):


        video = VideoFileClip(name)
        frames = video.fl_image(self.process)

        frames.write_videofile('result.avi', codec='mpeg4', audio=False)



video = input("입력 파일 : ")

detector = LaneDetector()
detector.test(video)


