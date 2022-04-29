# Correctness cannot be guaranteed
import math
import sys

import numpy as np
import numpy.fft as fft
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from tqdm import tqdm


class ImageProcessing:
    @staticmethod
    def get_grayscale_image(image):
        output_image = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
        return output_image

    @staticmethod
    def minmax_scale(image):
        mmax = np.max(image)
        mmin = np.min(image)
        if mmax == mmin:
            return np.zeros_like(image)
        return (image - mmin) / (mmax - mmin)

    @staticmethod
    def minmax_scale_byte(image):
        mmax = np.max(image)
        mmin = np.min(image)
        if mmax - mmin < 1e-9:
            return np.zeros_like(image).astype("int")
        return ((image - mmin) / (mmax - mmin) * 255).astype("int")

    @staticmethod
    def clip_channel(image):
        image = np.maximum(0, image)
        image = np.minimum(image, 1)
        return image


class ImageFiltering:
    @staticmethod
    def get_log_filter(kernel_size, sigma) -> np.ndarray:
        kernel = np.zeros((kernel_size, kernel_size)).astype("float64")
        kernel_pos_x = np.zeros_like(kernel)
        kernel_pos_y = np.zeros_like(kernel)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel_pos_x[i, j] = (i - center)
                kernel_pos_y[i, j] = (j - center)
        kernel_pos_x = kernel_pos_x.astype("float64")
        kernel_pos_y = kernel_pos_y.astype("float64")
        gaussian_kernel = np.exp(-(kernel_pos_x * kernel_pos_x + kernel_pos_y * kernel_pos_y) / 2.0 / (sigma ** 2))
        kernel_sum = np.sum(gaussian_kernel)
        gaussian_kernel /= kernel_sum
        kernel = gaussian_kernel * 1.0
        kernel *= ((kernel_pos_y ** 2.0 + kernel_pos_x ** 2.0)  - (2 * (sigma ** 2))) / (sigma ** 2)
        # np.save(str(kernel_size)+"_"+str(sigma)+".npy", kernel)
        return kernel

    @staticmethod
    def get_gaussian_filter(kernel_size, sigma) -> np.ndarray:
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        kernel_sum = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2) / 2.0 / (sigma ** 2))
                kernel_sum += kernel[i, j]
        kernel = kernel / kernel_sum
        return kernel

    @staticmethod
    def get_average_filter3d(kernel_size) -> np.ndarray:
        kernel = np.zeros((kernel_size, kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                for k in range(kernel_size):
                    kernel[i, j, k] = 1.0 / (kernel_size ** 3)
        return kernel

    @staticmethod
    def get_sum_filter3d(kernel_size) -> np.ndarray:
        kernel = np.zeros((kernel_size, kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                for k in range(kernel_size):
                    kernel[i, j, k] = 1.0
        return kernel

    @staticmethod
    def __image_conv_operator(image_clip, conv_filter):
        conv_result = 0
        for i in range(image_clip.shape[0]):
            for j in range(image_clip.shape[1]):
                conv_result += image_clip[i, j] * conv_filter[i, j]
        return conv_result

    @staticmethod
    def image_convolution(image, conv_filter, filter_size) -> np.ndarray:
        pad_length = filter_size // 2
        padded_image = np.pad(image, ((pad_length, pad_length), (pad_length, pad_length)), mode="edge")
        padded_shape = padded_image.shape
        proc_image = np.zeros_like(image)
        with tqdm(total=image.shape[0] * image.shape[1], file=sys.stdout, desc="Conv") as t:
            for i in range(pad_length, padded_shape[0] - pad_length):
                for j in range(pad_length, padded_shape[1] - pad_length):
                    px, py = i - pad_length, j - pad_length
                    rx, ry = i + pad_length, j + pad_length
                    t.update(1)
                    proc_image[px, py] = ImageFiltering.__image_conv_operator(padded_image[px:rx, py:ry], conv_filter)
        return proc_image

    @staticmethod
    def image_convolution_fft(image, conv_filter, filter_size) -> np.ndarray:
        pad_length = filter_size // 2
        padded_image = np.pad(image, ((pad_length, pad_length), (pad_length, pad_length)), mode="edge")
        padded_kernel = np.zeros((*padded_image.shape,))
        padded_kernel[:conv_filter.shape[0], :conv_filter.shape[1]] = conv_filter
        kv = int(np.floor(conv_filter.shape[0] / 2))
        kh = int(np.floor(conv_filter.shape[1] / 2))
        padded_kernel = np.roll(padded_kernel, -kv, axis=0)
        padded_kernel = np.roll(padded_kernel, -kh, axis=1)
        proc_image = fft.ifft2(fft.fft2(padded_image) * fft.fft2(padded_kernel))
        proc_image = np.real(proc_image)
        proc_image = proc_image[pad_length:pad_length + image.shape[0], pad_length:pad_length + image.shape[1]]
        return proc_image

    @staticmethod
    def image_convolution_fft3(image, conv_filter, filter_size) -> np.ndarray:
        pad_length = filter_size // 2
        padded_image = np.pad(image, ((pad_length, pad_length), (pad_length, pad_length), (pad_length, pad_length)),
                              mode="edge")
        padded_kernel = np.zeros((*padded_image.shape,))
        padded_kernel[:conv_filter.shape[0], :conv_filter.shape[1], ::conv_filter.shape[2]] = conv_filter
        kv = int(np.floor(conv_filter.shape[0] / 2))
        kh = int(np.floor(conv_filter.shape[1] / 2))
        ks = int(np.floor(conv_filter.shape[2] / 2))
        padded_kernel = np.roll(padded_kernel, -kv, axis=0)
        padded_kernel = np.roll(padded_kernel, -kh, axis=1)
        padded_kernel = np.roll(padded_kernel, -ks, axis=2)
        proc_image = fft.ifftn(fft.fftn(padded_image) * fft.fftn(padded_kernel))
        proc_image = np.real(proc_image)
        proc_image = proc_image[pad_length:pad_length + image.shape[0],
                     pad_length:pad_length + image.shape[1],
                     pad_length:pad_length + image.shape[2]]
        return proc_image

    @staticmethod
    def gaussian_blur(image, sigma):
        filter_size = 2 * math.ceil(3 * sigma) + 1
        gaussian_kernel = ImageFiltering.get_gaussian_filter(filter_size, sigma)
        conv = ImageFiltering.image_convolution_fft(image, gaussian_kernel, filter_size)
        # conv = cv2.filter2D(image, -1, gaussian_kernel)
        return ImageProcessing.clip_channel(conv)


class ScaleInvariantFeatureTransform:
    @staticmethod
    def apply_log_operator(image, kernel_size, sigma):
        conv_filter = ImageFiltering.get_log_filter(kernel_size, sigma)
        processed_image = ImageFiltering.image_convolution_fft(image, conv_filter, kernel_size)
        # processed_image = cv2.filter2D(image,-1,conv_filter)
        print(np.max(processed_image)*255, np.min(processed_image)*255)
        # processed_image = ImageProcessing.clip_channel(processed_image)
        return processed_image

    @staticmethod
    def create_spatial_space(image, sigma_modifier=2, base_sigma=1.6, samples=5):
        octave = np.zeros((*image.shape, samples))
        octave_minmax = np.zeros((*image.shape, samples))
        sigma_ret = []
        for i in range(samples):
            sigma = base_sigma * (sigma_modifier ** i)
            sigma_ret.append(sigma)
            kernel_size = 2 * int(math.ceil(sigma * 3)) + 1
            octave[:, :, i] = ScaleInvariantFeatureTransform.apply_log_operator(image, kernel_size, sigma)
            octave_minmax[:, :, i] = ImageProcessing.clip_channel(octave[:, :, i])
        return octave_minmax, octave, sigma_ret

    @staticmethod
    def find_local_max_v1(octave, neighbor_width=3, sup_thresh=0.005):
        octave_d = octave
        # Maximum filter aims to figure out values larger than or equal to the neighbour
        # The operation trades of accuracy for the speed
        proc_octave = maximum_filter(octave_d, neighbor_width)
        proc_octave = maximum_filter(octave_d, neighbor_width)
        proc_octave_nm = np.maximum(octave_d - proc_octave + 1e-9, 0)
        sum_kernel = ImageFiltering.get_sum_filter3d(3)
        sproc_octave = ImageFiltering.image_convolution_fft3(proc_octave_nm, sum_kernel, 3)
        sproc_cond = sproc_octave >= 1e-9
        octave_d[sproc_cond] = 0

        # Non-maximum suppression
        thresh_cond = proc_octave < sup_thresh
        octave_d[thresh_cond] = 0
        local_max_mat = (proc_octave == octave_d).astype("int")
        proc_octave2 = octave_d * local_max_mat
        return local_max_mat, proc_octave

    @staticmethod
    def find_local_max(octave, neighbor_width=3, sup_thresh=0.3):
        octave_d = octave
        octave_r = np.zeros_like(octave_d)
        layers = octave.shape[2]
        s1, s2 = octave.shape[0], octave.shape[1]
        for i in range(1, layers-1):
            for j in range(1, octave.shape[0] - 1):
                for k in range(1, octave.shape[1] - 1):
                    cpm = octave_d[max(j - 1, 0):min(j + 2, s1), max(k - 1, 0):min(k + 2, s2),
                          max(i - 1, 0):min(layers, i + 2)]
                    imx = np.max(cpm)
                    if imx == octave_d[j, k, i]:
                        pos, _, __ = np.where(cpm == octave_d[j, k, i])
                        if pos.shape[0] == 1:
                            octave_r[j, k, i] = 1.0
                    # imx = np.min(cpm)
                    # if imx == octave_d[j, k, i]:
                    #    pos, _, __ = np.where(cpm == octave_d[j, k, i])
                    #    if pos.shape[0] == 1:
                    #        octave_r[j, k, i] = 1.0
        return octave_r, octave_r

    @staticmethod
    def after_processing(octave, src_image, base_sigma=1.6, s_ratio=2, s_modifier=1.41 * 2):
        ret = []
        src_image_copy = src_image
        total_rc = 0
        for i in range(octave.shape[2]):
            r, c = np.where(octave[:, :, i])
            total_rc += len(r)
            octave_e = np.expand_dims(octave[:, :, i], axis=-1)
            octave_e = ImageProcessing.minmax_scale_byte(octave_e)
            octave_d = np.concatenate((octave_e, octave_e, octave_e), axis=-1)
            print("Octave",i," Features:",len(r))
            for j in range(len(r)):
                cv2.circle(octave_d, (c[j], r[j]), int(base_sigma * s_modifier * (s_ratio ** (i - 1))), (255, 0, 0),
                           thickness=1)
                cv2.circle(src_image_copy, (c[j], r[j]), int(base_sigma * s_modifier * (s_ratio ** (i - 1))),
                           (0, 0, 255),
                           thickness=1)
            ret.append(octave_d)
        print("Total Features:", total_rc)
        return ret, src_image_copy


if __name__ == "__main__":
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.5, hspace=None)
    src_img_org = cv2.imread("origin.jpeg")
    src_img = cv2.cvtColor(src_img_org, cv2.COLOR_BGR2GRAY)
    src_img_scaled = ImageProcessing.minmax_scale(src_img)
    print("Performing LoG calculation...")
    octaves_mm, octaves, sret = ScaleInvariantFeatureTransform.create_spatial_space(src_img_scaled, samples=5)
    imw = 5
    for i in range(octaves.shape[2]):
        plt.subplot(math.ceil(octaves.shape[2] / imw) * 2, imw, i + 1)
        plt.title("LoG Filtered \n sigma=" + str(sret[i]), fontsize=8)
        plt.imshow((octaves_mm[:, :, i]).astype("int"), "gray")
    print("Calculating local maxima...")
    octaves, _octaves = ScaleInvariantFeatureTransform.find_local_max(octaves)
    print("After processing...")
    ret, ret_final = ScaleInvariantFeatureTransform.after_processing(octaves, src_img_org)
    octp = octaves[:, :, 0]
    for i in range(octaves.shape[2]):
        plt.subplot(math.ceil(octaves.shape[2] / imw) * 2, imw, i + 1 * math.ceil(octaves.shape[2] / imw) * imw + 1)
        plt.title("Features \n sigma=" + str(sret[i]), fontsize=8)
        plt.imshow(ret[i])
    print("Close this window to show the final result.")
    plt.show()
    plt.figure()
    ret_final = cv2.cvtColor(ret_final, cv2.COLOR_BGR2RGB)
    plt.imshow(ret_final)
    plt.show()
