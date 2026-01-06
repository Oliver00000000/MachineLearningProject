from skimage.color import rgb2gray


def crop_center_roi_grayscale(img_rgb, roi_fraction=0.5):
    gray_image = rgb2gray(img_rgb)
    h, w = gray_image.shape[:2]
    roi_h, roi_w = int(h * roi_fraction), int(w * roi_fraction)

    y1, x1 = (h - roi_h) // 2, (w - roi_w) // 2
    roi = gray_image[y1:y1 + roi_h, x1:x1 + roi_w]

    return roi



