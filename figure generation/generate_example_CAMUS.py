import nibabel as nib
import matplotlib.pyplot as plt

# Load the NIFTI images
image1 = nib.load("/Users/davidsaldubehere/Documents/model_research/CAMUS_public/database_nifti/patient0003/patient0003_2CH_ED.nii").get_fdata()
image2 = nib.load("/Users/davidsaldubehere/Documents/model_research/CAMUS_public/database_nifti/patient0003/patient0003_2CH_ED_gt.nii").get_fdata()
image3 = nib.load("/Users/davidsaldubehere/Documents/model_research/CAMUS_public/database_nifti/patient0003/patient0003_2CH_ES.nii").get_fdata()
image4 = nib.load("/Users/davidsaldubehere/Documents/model_research/CAMUS_public/database_nifti/patient0003/patient0003_2CH_ES_gt.nii").get_fdata()

# For 2D or 3D images (taking first slice if 3D)
# Make sure to handle the case where images might be 3D
if len(image1.shape) >= 2:
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Get proper 2D slices if images are 3D
    img1 = image1[:,:,0] if len(image1.shape) > 2 else image1
    img2 = image2[:,:,0] if len(image2.shape) > 2 else image2
    img3 = image3[:,:,0] if len(image3.shape) > 2 else image3
    img4 = image4[:,:,0] if len(image4.shape) > 2 else image4
    
    # Display first image (top-left)
    im1 = axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title('Example ED Input Image')
    axes[0, 0].axis('off')
    
    # Display second image (top-right)
    im2 = axes[0, 1].imshow(img2, cmap='gray')
    axes[0, 1].set_title('Corresponding GT Mask')
    axes[0, 1].axis('off')
    
    # Display third image (bottom-left)
    im3 = axes[1, 0].imshow(img3, cmap='gray')
    axes[1, 0].set_title('Example ES Input Image')
    axes[1, 0].axis('off')
    
    # Display fourth image (bottom-right)
    im4 = axes[1, 1].imshow(img4, cmap='gray')
    axes[1, 1].set_title('Corresponding GT Mask')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()