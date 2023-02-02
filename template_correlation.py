import cv2 as cv
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt


template = cv.imread('data/others/oasis/oasis_cross-sectional_disc1/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/template2_150.png',0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
# methods = ['cv.TM_CCOEFF'  , 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
#              'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

methods = ['cv.TM_CCOEFF_NORMED']
# ]
# cv.TM_CCORR   152
# TM_CCOEFF_NORMED  169
# TM_CCORR 152
# TM_CCORR_NORMED  169

glob_path = 'data/others/oasis/oasis_cross-sectional_disc1/disc1/OAS1_0002_MR1/PROCESSED/MPRAGE/SUBJ_111'

path = f'{glob_path}/OAS1_0002_MR1_mpr_n4_anon_sbj_111.nii'

big = 'data/others/oasis/oasis_cross-sectional_disc1/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/big4_150.png'
# img = template.copy()
img = cv.imread(big, 0)
img2 = img.copy()
# #
# for meth in methods:
#     img = img2.copy()
#     method = eval(meth)
#     # Apply template Matching
#     res = cv.matchTemplate(img, template, method)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv.rectangle(img,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()


Nifti_img = nib.load(path)
nii_data = Nifti_img.get_fdata()
nii_aff = Nifti_img.affine
nii_hdr = Nifti_img.header
print(nii_aff, '\n', nii_hdr)
print(nii_data.shape)
dict_ = {'path': [],
        'res': []}
if (len(nii_data.shape) == 4):
    for frame in range(nii_data.shape[3]):
        for meth in methods:
            dict_ = {'path': [],
                     'res': []}
            for slice_Number in range(100, nii_data.shape[1]):
            # plt.imsave(f'{glob_path}/shape1_{frame}_{slice_Number}.png', nii_data[:, slice_Number, :, frame])
            # plt.imshow(nii_data[:, :, slice_Number, frame])
            # plt.show()
                img:np.ndarray = cv.imread(f'{glob_path}/shape1_{frame}_{slice_Number}.png', 0)
                big_img = np.zeros((4*img.shape[0], 4*img.shape[1]))
                big_img[big_img.shape[0]//2:big_img.shape[0]//2+img.shape[0],
                                        big_img.shape[1]//2:big_img.shape[1]//2+img.shape[1]] = img
                # img = img.shape[0]//2 i
                # img = template.copy()
                img2 = img.copy()


                img = img2.copy()
                method = eval(meth)
                # Apply template Matching
                res = cv.matchTemplate(big_img.astype('uint8'), template, method)
                dict_['path'].append(f'{glob_path}/shape1_{frame}_{slice_Number}.png')

                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc

                dict_['res'].append(max_val)
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv.rectangle(img,top_left, bottom_right, 255, 2)
            # print(dict_)
            max_ind = dict_['res'].index(max(dict_['res']))
            print(max_ind)
            print(dict_['path'][max_ind])

                # plt.subplot(121),plt.imshow(res,cmap = 'gray')
                # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                # plt.subplot(122),plt.imshow(img,cmap = 'gray')
                # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                # plt.suptitle(meth)
                # plt.show()
# print(dict_)


