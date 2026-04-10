import numpy as np
from PIL import Image
from skimage import measure, feature
from skimage.color import rgb2gray
from pathlib import Path

def extract_morphology(mask):
    props = measure.regionprops(mask.astype(int))
    if not props:
        return np.zeros(6)
    p = props[0]
    area = p.area
    perimeter = p.perimeter
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
    eccentricity = p.eccentricity
    solidity = p.solidity
    extent = p.extent
    return np.array([area, perimeter, circularity, eccentricity, solidity, extent])

def extract_glcm(gray_img):
    from skimage.feature import graycomatrix, graycoprops
    gray = (gray_img * 255).astype(np.uint8) if gray_img.max() <= 1 else gray_img.astype(np.uint8)
    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    return np.array([contrast, correlation, energy, homogeneity])

def extract_color(img_array):
    r, g, b =img_array[:,:,0].mean(), img_array[:,:,1].mean(), img_array[:,:,2].mean()
    gray = rgb2gray(img_array)
    center = gray[gray.shape[0]//4:3*gray.shape[0]//4, gray.shape[1]//4:3*gray.shape[1]//4]
    pallor=(center < center.mean()).sum() / (center.size + 1e-8)
    return np.array([r,g,b, pallor])

def get_mask_from_image(img_array):
    """Simple Otsu threshold to get RBC mask."""
    from skimage.filters import threshold_otsu
    gray = rgb2gray(img_array)
    thresh = threshold_otsu(gray)
    mask = gray < thresh
    return mask

def extract_all_features(img_path):
    """Extract all handcrafted features from image path."""
    img = np.array(Image.open(img_path).convert('RGB'))
    gray = rgb2gray(img)
    mask = get_mask_from_image(img)
   
    morph = extract_morphology(mask)
    texture = extract_glcm(gray)
    color = extract_color(img)
   
    return np.concatenate([morph, texture, color])

def extract_features_batch(image_paths):
    """Extract features for list of image paths."""
    features = []
    for path in image_paths:
        try:
            f = extract_all_features(path)
            features.append(f)
        except Exception as e:
            features.append(np.zeros(14))
    return np.array(features, dtype=np.float32)

def extract_all_features_from_array(img_array):
    """
    Extract all handcrafted features from image array.
    img_array: numpy (H, W, 3) - from PIL or uploaded image
    Usage (for inference / UI when user uploads image):
        img_array = np.array(pil_image)
        hand_raw = extract_all_features_from_array(img_array)
        hand_feat = scaler.transform(hand_raw.reshape(1, -1))
        hand_tensor = torch.tensor(hand_feat, dtype=torch.float32).to(device)
    """
    gray = rgb2gray(img_array)
    mask = get_mask_from_image(img_array)
    morph = extract_morphology(mask)
    texture = extract_glcm(gray)
    color = extract_color(img_array)
    return np.concatenate([morph, texture, color])
