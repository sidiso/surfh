import numpy as np

import scipy.optimize as opt
from scipy.ndimage import shift
from skimage.transform import rotate as ski_rotate


class AlignmentCorrection:
    """
    Classe pour effectuer des corrections d'alignement (rotation et translation)
    entre deux images.
    """

    @staticmethod
    def chi2(toBeAligned_img, ref_img, angle, y_shift, x_shift):
        """
        Perform chi-squared calculation between a reference image and a shifted + rotated image.

        Args:
            toBeAligned_img (np.ndarray): Image to be aligned.
            ref_img (np.ndarray): Reference image.
            angle (float): Rotation angle in degrees.
            y_shift (float): Shift in the y direction.
            x_shift (float): Shift in the x direction.
        Returns:
            float: Chi-squared value.
        """
        img_rot = ski_rotate(toBeAligned_img, angle=angle, resize=False, order=3, preserve_range=True)  
        img_rot_shift = shift(img_rot, shift=(y_shift, x_shift), order=3)  
        return np.sum((img_rot_shift - ref_img)**2)

    @staticmethod
    def apply_img_shift_rotation(img, angle, y_shift, x_shift):
        """
        Apply rotation and shift to an image.
        
        Args:
            img (np.ndarray): Image to be rotated and shifted.
            angle (float): Rotation angle in degrees.
            y_shift (float): Shift in the y direction.
            x_shift (float): Shift in the x direction.
        Returns:
            np.ndarray: Rotated and shifted image.
        """
        img_rot = ski_rotate(img, angle=angle, resize=False, order=3, preserve_range=True)
        img_rot_shift = shift(img_rot, shift=(y_shift, x_shift), order=3)
        return img_rot_shift

    @staticmethod
    def apply_cube_shift_rotation(cube, angle, y_shift, x_shift):
        """
        Apply rotation and shift to each image (slice) in a cube.
        
        Args:
            cube (np.ndarray): 3D array where each slice along the first axis is an image.
            angle (float): Rotation angle in degrees.
            y_shift (float): Shift in the y direction.
            x_shift (float): Shift in the x direction.
        Returns:
            np.ndarray: Cube with each image rotated and shifted.
        """
        rotated_shifted_cube = []
        for img in cube:
            img_rot_shift = AlignmentCorrection.apply_img_shift_rotation(img, angle, y_shift, x_shift)
            rotated_shifted_cube.append(img_rot_shift)
        return np.array(rotated_shifted_cube)
        

    @staticmethod
    def alignement_correction(minimization_fct, ref_img, toBeAligned_img, initial_guess=(0,0,0)):
        """
        Perform alignment correction using optimization to minimize the chi-squared function.

        Args:
            minimization_fct (callable): Function to minimize, should take (toBeAligned_img, ref_img, angle, y_shift, x_shift).
            ref_img (np.ndarray): Reference image.
            toBeAligned_img (np.ndarray): Image to be aligned.
            initial_guess (tuple): Initial guess for (angle, y_shift, x_shift).
        Returns:
            tuple: Corrected image and optimization result object.
        """
        fct_chi2 = lambda x : minimization_fct(toBeAligned_img, ref_img, angle=x[0], y_shift=x[1], x_shift=x[2])
        result = opt.minimize(
            fct_chi2, 
            initial_guess, 
            method='Nelder-Mead', 
            options={'disp': True, 'maxiter': 2000, 'maxfev': 5000, 'xatol':1e-3, 'fatol':1e-3},
        )
        corrected_img = AlignmentCorrection.apply_img_shift_rotation(toBeAligned_img, *result.x)
        return corrected_img, result