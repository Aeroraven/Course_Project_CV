import random
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt

FLOAT_EPS = 1e-10
RAND_SEED = 42


class ImageProcessing:
    @staticmethod
    def get_grayscale_image(image):
        output_image = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
        return output_image

    @staticmethod
    def minmax_scale(image):
        mmax = np.max(image)
        mmin = np.min(image)
        return (image - mmin) / (mmax - mmin)

    @staticmethod
    def minmax_scale_byte(image):
        mmax = np.max(image)
        mmin = np.min(image)
        return ((image - mmin) / (mmax - mmin) * 255).astype("int")

    @staticmethod
    def relu(x):
        return max(0.,x)


class HomographyProcessing:
    @staticmethod
    def least_square_approx(mat_a: np.ndarray):
        """
        Solve the approximately optimal and nonzero solution for Ax=0.

        :param mat_a: Matrix A
        :return: Optimal eigenvalue (Approx to 0), Optimal eigenvector
        """
        mat_at = np.transpose(mat_a, (1, 0))
        imp_mat = np.matmul(mat_at, mat_a)
        eig_val, eig_vec = np.linalg.eig(imp_mat)
        opt_idx = 0
        for i in range(eig_val.shape[0]):
            if eig_val[i] < eig_val[opt_idx]:
                opt_idx = i
        if eig_val[opt_idx] >= 1e-6:
            warnings.warn("Estimation precision lost!", UserWarning)
        return eig_val[opt_idx], eig_vec[:, opt_idx]

    @staticmethod
    def to_homogeneous(point):
        res = np.zeros((3, 1))
        res[0, 0] = point[0]
        res[1, 0] = point[1]
        res[2, 0] = 1
        return res

    @staticmethod
    def to_cartesian(point):
        return [point[0, 0] / point[2, 0], point[1, 0] / point[2, 0]]

    @staticmethod
    def cartesian_sq_dist(point_a, point_b):
        return (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2

    @staticmethod
    def homography_estimate(src_points: list[np.ndarray], dst_points: list[np.ndarray]):
        """
        Estimate or determine the homography mapping matrix

        :param src_points: Points on the source plane
        :param dst_points: Points on the destination plane
        :return: Homography matrix
        """
        sl, dl = len(src_points), len(dst_points)
        if sl != dl:
            raise Exception("Indexes mismatch")
        if sl < 4:
            raise Exception("Length of array should be larger than 4")
        length = sl
        coef_mat = np.zeros((length * 2, 9))
        for i in range(length):
            coef_mat[2 * i, :] = np.array([-src_points[i][0], -src_points[i][1], -1, 0, 0, 0,
                                           dst_points[i][0] * src_points[i][0], dst_points[i][0] * src_points[i][1],
                                           dst_points[i][0]])
            coef_mat[2 * i + 1, :] = np.array([0, 0, 0, -src_points[i][0], -src_points[i][1], -1,
                                               dst_points[i][1] * src_points[i][0], dst_points[i][1] * src_points[i][1],
                                               dst_points[i][1]])
        homo_eig, homo_vec = HomographyProcessing.least_square_approx(coef_mat)
        homo_mat = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                homo_mat[i, j] = homo_vec[i * 3 + j]
        homo_mat = homo_mat / homo_mat[2, 2]
        return homo_mat

    @staticmethod
    def ransac_estimate(src_points: list[np.ndarray],
                        dst_points: list[np.ndarray],
                        max_iters: int = 2500,
                        inlier_threshold=15,
                        replace_threshold=150):
        max_inliers = 0
        max_inliers_sol = None
        sl, dl = len(src_points), len(dst_points)
        if sl != dl:
            raise Exception("Indexes mismatch")
        if sl < 4:
            raise Exception("Length of array should be larger than 4")
        index_list = [i for i in range(sl)]
        for i in range(max_iters):
            chosen_idx = random.choices(index_list, k=4)
            src_chosen = [src_points[i] for i in chosen_idx]
            dst_chosen = [dst_points[i] for i in chosen_idx]
            homo_mat = HomographyProcessing.homography_estimate(src_chosen, dst_chosen)
            inliers = 0
            selected_idx = []
            errors = 0
            for j in range(len(index_list)):
                dst_pred_h = np.matmul(homo_mat, HomographyProcessing.to_homogeneous(src_points[j]))
                dst_pred_c = HomographyProcessing.to_cartesian(dst_pred_h)
                dist = HomographyProcessing.cartesian_sq_dist(dst_pred_c, dst_points[j])
                errors += dist
                if dist < inlier_threshold ** 2:
                    inliers += 1
                    selected_idx.append(j)
                if len(selected_idx) > replace_threshold:
                    index_list = selected_idx
            if inliers > max_inliers:
                max_inliers = inliers
                max_inliers_sol = homo_mat
                print("Iter", i, " Error=", errors, " Inliers=", inliers, "/", len(index_list), "-Update")
            else:
                print("Iter", i, " Error=", errors, " Inliers=", inliers, "/", len(index_list))
        if max_inliers_sol is None:
            raise Exception("No valid homography matrices")
        return max_inliers_sol


class PanoramaStitching:
    @staticmethod
    def sift_detect(org: np.ndarray, image: np.ndarray):
        sift = cv2.SIFT_create()
        keypoint, descriptor = sift.detectAndCompute(image, None)
        return keypoint, descriptor

    @staticmethod
    def descriptor_match(keypoint1, keypoint2, desc1, desc2, img1, img2):
        bf = cv2.BFMatcher()
        match = bf.knnMatch(desc1, desc2, k=2)
        ret = []
        for m, n in match:
            if m.distance < 0.75 * n.distance:
                ret.append([m])
        img3 = cv2.drawMatchesKnn(img1, keypoint1, img2, keypoint2, ret[:], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        print(ret)
        return img3, ret[:]

    @staticmethod
    def ransac_homography_estimate(keypoints_a: list,
                                   keypoints_b: list,
                                   matches: list[list[cv2.DMatch]]):
        src_points = []
        dst_points = []
        for i in range(len(keypoints_a)):
            src_points.append(keypoints_a[i].pt)
        for i in range(len(keypoints_b)):
            dst_points.append(keypoints_b[i].pt)
        src_passing = []
        dst_passing = []
        for i in range(len(matches)):
            src_passing.append(np.array(src_points[matches[i][0].queryIdx]))
            dst_passing.append(np.array(dst_points[matches[i][0].trainIdx]))
        return HomographyProcessing.ransac_estimate(src_passing, dst_passing)

    @staticmethod
    def image_stitch(homography_matrix:np.ndarray, image_src:np.ndarray, image_dst:np.ndarray):
        # Border Estimation
        src_border_keypoints = [[0, 0], [0, image_src.shape[0] - 1], [image_src.shape[1] - 1, 0],
                                [image_src.shape[1] - 1, image_src.shape[0] - 1]]
        src_border_keypoints_mapped = []
        for i in src_border_keypoints:
            mp = np.matmul(homography_matrix, HomographyProcessing.to_homogeneous(i))
            mp = HomographyProcessing.to_cartesian(mp)
            src_border_keypoints_mapped.append(mp)

        src_border_keypoints_mapped = np.array(src_border_keypoints_mapped)
        src_bk_xl, src_bk_xr = np.min(src_border_keypoints_mapped[:, 0]), np.max(src_border_keypoints_mapped[:, 0])
        src_bk_yl, src_bk_yr = np.min(src_border_keypoints_mapped[:, 1]), np.max(src_border_keypoints_mapped[:, 1])

        # Translation
        translation_matrix = np.array([[1, 0, ImageProcessing.relu(-src_bk_xl)],
                                       [0, 1, ImageProcessing.relu(-src_bk_yl)],
                                       [0, 0, 1]])
        mixed_homo_mat = np.matmul(translation_matrix, homography_matrix)
        src_warp_shape = np.array((src_bk_xr + ImageProcessing.relu(- src_bk_xl),
                                   src_bk_yr + ImageProcessing.relu(- src_bk_yl))).astype("int")
        dst_warp_shape = np.array((image_dst.shape[1] + ImageProcessing.relu(- src_bk_xl),
                                   image_dst.shape[0] + ImageProcessing.relu(- src_bk_yl))).astype("int")
        pad_x = max(src_warp_shape[0], dst_warp_shape[0])
        pad_y = max(src_warp_shape[1], dst_warp_shape[1])
        warp_shape = np.array((pad_x, pad_y)).astype("int")

        # Warping & Mask Processing
        src_mask = (np.ones_like(image_src)).astype("uint8")
        dst_mask = (np.ones_like(image_dst)).astype("uint8")
        src_warp = cv2.warpPerspective(image_src, mixed_homo_mat, warp_shape)
        src_mask_warp = cv2.warpPerspective(src_mask, mixed_homo_mat, warp_shape) >= 0.5
        dst_warp = cv2.warpPerspective(image_dst, translation_matrix, warp_shape)
        dst_mask_warp = cv2.warpPerspective(dst_mask, translation_matrix, warp_shape) >= 0.5

        # Intersection Masking
        int_mask_warp = (dst_mask_warp * src_mask_warp) >= 0.5
        int_warp = (src_warp.astype("int16") + dst_warp.astype("int16")) / 2.0
        int_warp = int_warp.astype("uint8")
        int_mask_warp = int_mask_warp.astype("uint8")
        int_warp_after = int_warp * int_mask_warp

        # Target Masking
        dst_mask_warp_backup = dst_mask_warp
        dst_mask_warp = (dst_mask_warp.astype("int") - src_mask_warp.astype("int")) >= 0.5
        dst_mask_warp = dst_mask_warp.astype("uint8")
        dst_warp_after = dst_warp * dst_mask_warp

        # Source Masking
        src_mask_warp = (src_mask_warp.astype("int") - dst_mask_warp_backup.astype("int")) >= 0.5
        src_mask_warp = src_mask_warp.astype("uint8")
        src_warp_after = src_warp * src_mask_warp

        stitched_result = dst_warp_after + src_warp_after + int_warp_after

        # Visualization
        plt.subplot(2, 4, 1)
        plt.imshow(image_src)
        plt.title("Source Image \n (Input)")
        plt.subplot(2, 4, 2)
        plt.imshow(image_dst)
        plt.title("Target Image \n (Input)")
        plt.subplot(2, 4, 3)
        plt.imshow(stitched_result)
        plt.title("Stitched Image \n (Output)")
        plt.subplot(2, 4, 4)
        plt.imshow(src_warp)
        plt.title("Transformed Source Image \n (SIFT Homography + Translation)")
        plt.subplot(2, 4, 5)
        plt.imshow(dst_warp)
        plt.title("Transformed Target Image \n (Translation)")
        plt.subplot(2, 4, 6)
        plt.imshow(src_mask_warp * 255)
        plt.title("Transformed Source Mask \n (SIFT Homography + Translation)")
        plt.subplot(2, 4, 7)
        plt.imshow(dst_mask_warp * 255)
        plt.title("Transformed Target Mask \n (Translation)")
        plt.subplot(2, 4, 8)
        plt.imshow(int_mask_warp * 255)
        plt.title("Transformed Intersection Mask")
        plt.show()

        # Visualization
        plt.figure()
        plt.imshow(stitched_result)
        plt.title("Stitched Image \n (Output)")
        plt.show()

    @staticmethod
    def panorama_stitch(path_src:str,path_dst:str):
        src_im_org = cv2.imread(path_src)
        dst_im_org = cv2.imread(path_dst)
        src_im_org = cv2.cvtColor(src_im_org, cv2.COLOR_BGR2RGB)
        dst_im_org = cv2.cvtColor(dst_im_org, cv2.COLOR_BGR2RGB)
        src_im = cv2.cvtColor(src_im_org, cv2.COLOR_BGR2GRAY)
        dst_im = cv2.cvtColor(dst_im_org, cv2.COLOR_BGR2GRAY)
        src_kp, src_desc = PanoramaStitching.sift_detect(src_im_org, src_im)
        dst_kp, dst_desc = PanoramaStitching.sift_detect(dst_im_org, dst_im)
        mat, matches = PanoramaStitching.descriptor_match(src_kp, dst_kp, src_desc, dst_desc, src_im, dst_im)
        plt.imshow(mat)
        plt.title("Keypoint Matches (SIFT Features)\n(Close this window to show the final result)")
        print("!!!!Close this Visualization Window to show next output!!!!")
        plt.show()
        print("Total Sift Matches", len(matches))
        homo_mat = PanoramaStitching.ransac_homography_estimate(src_kp, dst_kp, matches)
        PanoramaStitching.image_stitch(homo_mat, src_im_org, dst_im_org)


if __name__ == "__main__":
    PanoramaStitching.panorama_stitch("Exp1.JPG","Exp2.JPG")
