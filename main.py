import nibabel as nib
import nibabel as nb
import matplotlib.pyplot as plt

# fname = 'data/others/oasis/oasis_cross-sectional_disc1/disc1/OAS1_0002_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0002_MR1_mpr_n4_anon_sbj_111.img'
# img = nb.load(fname)
# nb.save(img, fname.replace('.img', '.nii'))

glob_path = 'data/others/oasis/oasis_cross-sectional_disc1/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111'

path = f'{glob_path}/OAS1_0001_MR1_mpr_n4_anon_sbj_111.nii'

Nifti_img = nib.load(path)
nii_data = Nifti_img.get_fdata()
nii_aff = Nifti_img.affine
nii_hdr = Nifti_img.header
print(nii_aff, '\n', nii_hdr)
print(nii_data.shape)
if len(nii_data.shape) == 3:
    for slice_Number in range(nii_data.shape[2]):
        # plt.imshow(nii_data[:,:,slice_Number ])
        # plt.show()
        j = []
        j.split()
        # plt.imsave(f'{glob_path}/1_orig.png', test_image_orig_rot)
        #
        # test_image_orig_rot = plt.imread(f'{TRAIN_DATASET_PATH}/{PATIENT}_orig.png')

        plt.imsave(f'{glob_path}/1_{slice_Number}.png', nii_data[:, :, slice_Number])

if (len(nii_data.shape) == 4):
    for frame in range(nii_data.shape[3]):
        for slice_Number in range(nii_data.shape[1]):
            # plt.imsave(f'{glob_path}/shape1_{frame}_{slice_Number}.png', nii_data[:, slice_Number, :, frame])
            plt.imshow(nii_data[:, :, slice_Number, frame])
            plt.show()
            
