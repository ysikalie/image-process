# *_*coding:utf-8 *_*

from PIL import Image
from numpy import average, dot, linalg
import os
import cv2

# 对图片进行统一化处理
def get_thum(image, size=(64, 64), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image


# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数？？
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res

IMAGE1_DIR = './annotation'
COMPETITION_DIR = './competition/zuoyebang'
TARGET_DIR = './TARGET_DIR'

# for img1 in os.listdir(IMAGE1_DIR):
#     img1_path = os.path.join(IMAGE1_DIR, img1)
#     for img2 in os.listdir(COMPETITION_DIR):
#         img2_path = os.path.join(COMPETITION_DIR, img2)
#         image1 = Image.open(img1_path)
#         image2 = Image.open(img2_path)
#         cosin = image_similarity_vectors_via_numpy(image1, image2)
#         if cosin > 0.9:
#             print(img1, img2)
#         break
#     break


from skimage.measure import compare_ssim
from imageio import imread
import numpy as np


from PIL import Image


# 将图片转化为RGB
def make_regalur_image(img, size=(64, 64)):
    gray_image = img.resize(size).convert('RGB')
    return gray_image


# 计算直方图
def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    hist = sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)
    return hist


# 计算相似度
def calc_similar(li, ri):
    calc_sim = hist_similar(li.histogram(), ri.histogram())
    return calc_sim

image1 = '/1.png'
image2 = '/Screenshot_2020-04-17-10-59-58-951_ai.zuoye.app.png'






def main():
    img1_list, img2_list = get_image_path_list()
    for img1 in img1_list:
        for img2 in img2_list:
            img1_path = os.path.join(IMAGE1_DIR, img1)
            img2_path = os.path.join(COMPETITION_DIR, img2)
            image1 = Image.open(img1_path)
            image1 = make_regalur_image(image1)

            image2 = Image.open(img2_path)
            image2 = make_regalur_image(image2)

            val = calc_similar(image1, image2)
            post_process(img1_path, img2_path, val)


def post_process(img1_path, img2_path, val):
    '''
    Save image with similarity more than 0.5.Save coresponding img2_path image to TARGET_DIR.

    :param img1_path: str,
    :param img2_path: str,
    :param val: int, similarity
    :return:
    '''

    img1 = img1_path.split('\\')[-1][:-4]
    count = 0

    if val > 0.5:
        while True:
            # .../0_i.jpg
            temp_img_pth = os.path.join(TARGET_DIR, img1+'_'+str(count)+'.jpg')
            if os.path.exists(temp_img_pth):
                count += 1
            else:
                temp_image = cv2.imread(img2_path)
                cv2.imwrite(temp_img_pth, temp_image)
                print(temp_img_pth)
                break






def get_image_path_list():
    '''
    Get image list.And image list sample ['1.jpg', '2.jpg'.....'100.jpg']
    :return:
    '''

    img1_pth = os.listdir(IMAGE1_DIR)
    img1_pth.sort(key=lambda x: int(x[:-4]))

    img2_pth = os.listdir(COMPETITION_DIR)
    img2_pth.sort(key=lambda x: int(x[:-4]))
    return img1_pth, img2_pth


    # for i in sorted(result)[-5:]:
    #     print(result.index(i))

    # img1_path = os.path.join(IMAGE1_DIR, img1)
    # image1 = Image.open(img1_path)
    # image1 = make_regalur_image(image1)
    # image2 = Image.open(img2_path)
    # image2 = make_regalur_image(image2)
    #
    # print(calc_similar(image1, image2))

if __name__ == '__main__':
    main()
