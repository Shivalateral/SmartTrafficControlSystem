import cv2
import numpy as np

class CannyEdgeDetector:
    def __init__(self, imgs, sigma=1.0, kernel_size=5, weak_pixel=75, strong_pixel=255,
                 lowthreshold=0.05, highthreshold=0.15):
        self.imgs = imgs
        self.imgs_final = []
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold

    def gaussian_blur(self, img):
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)

    def sobel_filters(self, img):
        Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        G = np.hypot(Ix, Iy)
        G = (G / G.max()) * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)

    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.uint8)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                q = 255
                r = 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0
        return Z

    def threshold(self, img):
        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold
        M, N = img.shape
        res = np.zeros((M, N), dtype=np.uint8)

        strong = np.uint8(self.strong_pixel)
        weak = np.uint8(self.weak_pixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        return res

    def hysteresis(self, img):
        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if img[i, j] == weak:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
        return img

    def detect(self):
        imgs_final = []
        for img in self.imgs:
            blurred = self.gaussian_blur(img)
            gradientMat, thetaMat = self.sobel_filters(blurred)
            nonMaxImg = self.non_max_suppression(gradientMat, thetaMat)
            thresholdImg = self.threshold(nonMaxImg)
            img_final = self.hysteresis(thresholdImg)
            imgs_final.append(img_final)
        return imgs_final
