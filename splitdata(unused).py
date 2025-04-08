import os
import shutil
import random
from pathlib import Path
import argparse
import re

def create_directory(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_patient_id(filename):
    """Extract patient ID from filename."""
    match = re.search(r'patient(\d+)', filename)
    return match.group(1) if match else None

def split_dataset(image_dir, mask_dir, train_dir, test_dir, split_ratio=0.8, seed=42):
    """
    Split dataset into train and test sets.
    
    Args:
        image_dir (str): Directory containing input images
        mask_dir (str): Directory containing corresponding masks
        train_dir (str): Output directory for training data
        test_dir (str): Output directory for test data
        split_ratio (float): Ratio for train split (default: 0.8)
        seed (int): Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)
    
    # Create output directories
    create_directory(os.path.join(train_dir, 'images'))
    create_directory(os.path.join(train_dir, 'masks'))
    create_directory(os.path.join(test_dir, 'images'))
    create_directory(os.path.join(test_dir, 'masks'))
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii')]
    
    # Group files by patient ID to keep all images from the same patient together
    patient_files = {}
    for filename in image_files:
        patient_id = get_patient_id(filename)
        if patient_id:
            if patient_id not in patient_files:
                patient_files[patient_id] = []
            patient_files[patient_id].append(filename)
    
    # Get list of unique patient IDs and shuffle
    patient_ids = list(patient_files.keys())
    random.shuffle(patient_ids)
    
    # Calculate split index for patients
    split_idx = int(len(patient_ids) * split_ratio)
    
    # Split patients into train and test sets
    train_patients = patient_ids[:split_idx]
    test_patients = patient_ids[split_idx:]
    
    # Function to copy files for a set of patients
    def copy_patient_files(patient_ids, output_base_dir):
        copied_files = []
        for patient_id in patient_ids:
            for filename in patient_files[patient_id]:
                # Copy image
                src_image = os.path.join(image_dir, filename)
                dst_image = os.path.join(output_base_dir, 'images', filename)
                shutil.copy2(src_image, dst_image)
                copied_files.append(filename)
                
                # Copy corresponding mask
                src_mask = os.path.join(mask_dir, filename.replace('.nii', '') + "_gt.nii")
                print(src_mask)
                if os.path.exists(src_mask):
                    dst_mask = os.path.join(output_base_dir, 'masks', filename)
                    shutil.copy2(src_mask, dst_mask)
        return copied_files
    
    # Copy files to respective directories
    train_files = copy_patient_files(train_patients, train_dir)
    test_files = copy_patient_files(test_patients, test_dir)
    
    # Print summary
    print(f"\nDataset split complete:")
    print(f"Total patients: {len(patient_ids)}")
    print(f"Training set: {len(train_patients)} patients ({len(train_files)} files)")
    print(f"Test set: {len(test_patients)} patients ({len(test_files)} files)")
    
    # Print patient IDs in each set
    print(f"\nTrain set patients: {', '.join(sorted(train_patients))}")
    print(f"Test set patients: {', '.join(sorted(test_patients))}")

def main():
    parser = argparse.ArgumentParser(description='Split medical image dataset into train and test sets')
    parser.add_argument('--image-dir', default = '/scratch/das6859/UNetVGNNCAMUS/dataset/full/images', help='Directory containing input images')
    parser.add_argument('--mask-dir', default = '/scratch/das6859/UNetVGNNCAMUS/dataset/full/masks', help='Directory containing input masks')
    parser.add_argument('--train-dir', default = '/scratch/das6859/UNetVGNNCAMUS/dataset/train', help='Output directory for training data')
    parser.add_argument('--test-dir', default = '/scratch/das6859/UNetVGNNCAMUS/dataset/test', help='Output directory for test data')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='Ratio for train split (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    split_dataset(
        args.image_dir,
        args.mask_dir,
        args.train_dir,
        args.test_dir,
        args.split_ratio,
        args.seed
    )

if __name__ == '__main__':
    main()