import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import sobel

from src.preprocess import crop_center_roi_grayscale


def extract_glcm(gray_img, config):
    gray_uint8 = (gray_img * 255).clip(0, 255).astype(np.uint8)
    levels = config['features']['glcm_levels']
    if gray_uint8.max() >= levels:
        gray_uint8 = (gray_uint8 / 255 * (levels - 1)).astype(np.uint8)

    distances = config['features']['glcm_distances']
    angles = np.deg2rad(config['features']['glcm_angles'])

    glcm = graycomatrix(gray_uint8, distances, angles, levels=levels,
                        symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'correlation']
    feats = np.hstack([graycoprops(glcm, prop).ravel() for prop in props])
    return feats.flatten()


def extract_lbp(gray_img, config):
    radius = config['features']['lbp_radius']
    n_points = config['features']['lbp_points']
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), density=True)
    return hist


def extract_glrlm_proxy(gray_img):
    edges = sobel(gray_img)
    hist = np.histogram(edges.ravel(), bins=32, density=True)[0]
    return hist


def extract_all_features(img_rgb, config):
    gray = crop_center_roi_grayscale(img_rgb)
    glcm_features = extract_glcm(gray, config)
    glrlm_features = extract_glrlm_proxy(gray)
    lbp_features = extract_lbp(gray, config)
    feats = np.hstack([
        glcm_features,
        glrlm_features,
        lbp_features
    ])

    feats = feats / (np.linalg.norm(feats) + 1e-12)
    return feats

