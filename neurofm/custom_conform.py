import nibabel as nib
import numpy as np
from loguru import logger
from nibabel.orientations import apply_orientation, axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform


def pad_volume_to_shape(volume, target_shape=(256,256,256)):
    """
    Given a volume, adds necessary constant padding to yield desired shape.
    Does NOT handle if the volume is *larger* than the desired shape. Only smaller!
    """
    # padding = [(0, max(target_dim - current_dim, 0)) for target_dim, current_dim in zip(target_shape, volume.shape)]

    padding = []
    for current_dim, target_dim in zip(volume.shape, target_shape):
        total_padding = target_dim - current_dim
        # Split padding evenly between 'before' and 'after', with extra padding added to 'after' if odd
        before_padding = total_padding // 2
        after_padding = total_padding // 2 + total_padding % 2  # Add the extra padding to 'after' if odd
        padding.append((before_padding, after_padding))

    padded_vol = np.pad(volume, padding, mode='constant', constant_values=0)
    return padded_vol, padding

def pad_orient_conform(
    from_img,
    out_shape=(256, 256, 256),
    orientation='RAS',
    pad=False
):
    if out_shape is None:
        out_shape = from_img.shape

    # Only support 3D images. This can be made more general in the future, once tests
    # are written.
    required_ndim = 3
    if from_img.ndim != required_ndim:
        raise ValueError('Only 3D images are supported.')
    elif len(out_shape) != required_ndim:
        raise ValueError(f'`out_shape` must have {required_ndim} values')
        
    start_ornt = io_orientation(from_img.affine)
    end_ornt = axcodes2ornt(orientation) if orientation is not None else start_ornt
        
    transform = ornt_transform(start_ornt, end_ornt)

    # Reorient first to ensure shape matches expectations
    reoriented, padding = pad_and_orient(from_img, transform, pad, out_shape, orientation)

    return reoriented, padding, start_ornt, end_ornt

def pad_and_orient(img, ornt, pad, out_shape, orientation):
    X_data = np.asanyarray(img.dataobj)
    padding = None
    if out_shape is not None and tuple(X_data.shape) != tuple(out_shape):
        if not pad:
            logger.error(f'ERROR: volume has shape {X_data.shape} != {out_shape}: exiting...')
            assert tuple(X_data.shape) == tuple(out_shape) # Padding is probably needed. Remember to turn on padding! :)
        else:
            X_data, padding = pad_volume_to_shape(X_data, target_shape=out_shape)
    
    # Adjust affine if padding was applied
    new_affine = img.affine.copy()

    if padding is not None: # Adjust affine transform matrix using padding
        new_transform = np.eye(4)
        new_transform[0:3, -1] = [j[0] for j in padding]
        new_affine = new_affine @ np.linalg.inv(new_transform)
        
    if np.array_equal(ornt, [[0, 1], [1, 1], [2, 1]]) or \
                    orientation is None or \
                    nib.aff2axcodes(img.affine) == tuple(orientation):
        return nib.Nifti1Image(X_data, new_affine, header=img.header), padding
            
    t_arr = apply_orientation(X_data, ornt)

    # Apply the inverse orientation affine to update the image orientation
    new_aff = new_affine.dot(inv_ornt_aff(ornt, t_arr.shape))

    return nib.Nifti1Image(t_arr, new_aff, header=img.header), padding

