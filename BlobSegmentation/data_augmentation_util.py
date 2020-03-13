# coding: utf-8
# all images are supposed to be BGR
import cv2
import numpy
import random


def random_noise(image, shake_interval):
    random_shake = numpy.random.randint(shake_interval[0], shake_interval[1], image.shape)
    image = image + random_shake # do not use += !!!!!!
    image[image > 255] = 255
    image[image < 0] = 0
    return image.astype(dtype=numpy.uint8)


def darkening(image, dark_value):
    image = image.astype(dtype=numpy.int32) - dark_value
    image[image < 0] = 0
    return image.astype(dtype=numpy.uint8)


def brightening(image, bright_value):
    image = image.astype(dtype=numpy.int32) + bright_value
    image[image > 255] = 255
    return image.astype(dtype=numpy.uint8)


def horizon_flip(image):
    return cv2.flip(image, 1)


def vertical_flip(image):
    return cv2.flip(image, 0)

def get_rotation_center(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = float(64)/max(img_gray.shape)
    threshold = 100
    img_gray = cv2.resize(img_gray, (0,0),fx=scale, fy=scale)

    sum_y = numpy.sum(img_gray, axis=1)
    sum_x = numpy.sum(img_gray, axis=0)

    for i in xrange(0, sum_y.shape[0], 1):
        if sum_y[i]>threshold and sum_y[i+1]>threshold and sum_y[i+2]>threshold:

            top=i
            break
    for i in xrange(sum_y.shape[0]-1, 0, -1):
        if sum_y[i]>threshold and sum_y[i-1]>threshold and sum_y[i-2]>threshold:

            bottom = i
            break
    for i in xrange(0, sum_x.shape[0], 1):
        if sum_x[i]>threshold and sum_x[i+1]>threshold and sum_x[i+2]>threshold:

            left = i
            break
    for i in xrange(sum_x.shape[0]-1, 0, -1):
        if sum_x[i]>threshold  and sum_x[i-1]>threshold and sum_x[i-2]>threshold:

            right = i
            break
    return (int((left/scale+right/scale)/2), int((top/scale+bottom/scale)/2))

# positive angle means anticlockwise
def rotation_image(image, angle, center):
    height = image.shape[0]
    width = image.shape[1]

    rotateMat = cv2.getRotationMatrix2D(center, angle, 1)
    rotateImg = cv2.warpAffine(image, rotateMat, (width, height), borderMode=cv2.BORDER_CONSTANT)

    return rotateImg

def get_rotation_mat(angle, center):
    return cv2.getRotationMatrix2D(center, angle, 1)


def blurring_by_resize(image, resize_factor):
    # randomly choose a interpolation method
    interpolation_method = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA]#, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
    height = image.shape[0]
    width = image.shape[1]

    temp_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor,interpolation=random.choice(interpolation_method))
    result_image = cv2.resize(temp_image, (width, height), interpolation=random.choice(interpolation_method))

    return result_image

def blurring_by_compression(image, jpg_compression_quality):
    if image.ndim != 3:
        print 'function [blurring_by_compression] needs RGB images as input.'
        return image
    ret, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, jpg_compression_quality])
    if not ret:
        print 'encode in [blurring_by_compression] is failed.'
        return image

    result_image = cv2.imdecode(buf, cv2.CV_LOAD_IMAGE_COLOR)
    return result_image



if __name__ == '__main__':
    image = cv2.imread('./test_image/test1.jpg', cv2.IMREAD_COLOR)

    for loop in range(100):
        quality = random.randint(10, 100)
        print quality
        image_blur = blurring_by_compression(image, quality)
        cv2.imshow('image', image)
        cv2.imshow('image_change', image_blur)
        cv2.waitKey()
    cv2.destroyAllWindows()
