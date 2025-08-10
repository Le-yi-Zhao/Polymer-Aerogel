
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
import xgboost as xgb
import shap
import os
import warnings
import logging
import time
from pathlib import Path
import cv2
from skimage import io, measure
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from skimage.measure import regionprops
from skimage.feature import canny
from skimage import filters
from scipy import ndimage as ndi
from skimage import img_as_ubyte

logging.basicConfig(
    filename="results/aerogel_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)
logger = logging.getLogger()

sns.set(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = ['SimHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

# 图像处理：提取孔隙特征
def extract_porosity_features(img_path: str):
    # 读取并预处理图像
    img = imread_gray_8bit(img_path)
    img = enhance_contrast(img)
    img = denoise(img)
    binary_img = binarize_pores(img)

    # 计算孔隙率（Porosity）
    porosity = np.sum(binary_img == 255) / binary_img.size

    # 使用标签函数检测不同区域
    labeled_img, num_labels = measure.label(binary_img, connectivity=2, return_num=True)
    region_props = measure.regionprops(labeled_img)

    # 孔径：计算每个区域的平均、标准差和中位孔径
    pore_sizes = [region.area for region in region_props if region.area > 50]
    avg_pore_size = np.mean(pore_sizes)
    median_pore_size = np.median(pore_sizes)
    std_pore_size = np.std(pore_sizes)

    # 孔隙形态学特征：圆度、长宽比、实心度
    circularity = []
    aspect_ratios = []
    solidity = []

    for region in region_props:
        circularity.append(region.eccentricity)  # 孔隙圆度
        aspect_ratios.append(region.major_axis_length / region.minor_axis_length)  # 长宽比
        solidity.append(region.solidity)  # 实心度

    # 骨架分析：计算骨架长度密度、分叉点、端点
    skeleton = skeletonize(binary_img)
    skeleton_dilated = ndi.binary_dilation(skeleton)

    # 计算骨架长度密度
    skeleton_length_density = np.sum(skeleton) / binary_img.size

    # 计算分叉点数量和端点数量
    skeleton_endpoints = ndi.binary_erosion(skeleton) & skeleton
    skeleton_branchpoints = ndi.convolve(skeleton, np.ones((3, 3)), mode='constant') == 2

    num_endpoints = np.sum(skeleton_endpoints)
    num_branchpoints = np.sum(skeleton_branchpoints)

    # 计算孔隙结构的分形维数
    box_counting_dim = calculate_fractal_dimension(skeleton)

    return {
        'porosity': porosity,
        'avg_pore_size': avg_pore_size,
        'median_pore_size': median_pore_size,
        'std_pore_size': std_pore_size,
        'circularity': np.mean(circularity),
        'aspect_ratio': np.mean(aspect_ratios),
        'solidity': np.mean(solidity),
        'skeleton_length_density': skeleton_length_density,
        'num_endpoints': num_endpoints,
        'num_branchpoints': num_branchpoints,
        'fractal_dimension': box_counting_dim
    }

# 计算分形维数的函数（基于盒计数法）
def calculate_fractal_dimension(image):
    # 盒计数法
    threshold_values = np.arange(1, min(image.shape) // 2)
    box_counts = []

    for size in threshold_values:
        boxes = np.zeros_like(image)
        for i in range(0, image.shape[0], size):
            for j in range(0, image.shape[1], size):
                boxes[i:i+size, j:j+size] = 1
        box_counts.append(np.sum(image[boxes == 1]))

    # 使用对数回归计算分形维数
    log_box_counts = np.log(box_counts)
    log_threshold_values = np.log(threshold_values)
    coeffs = np.polyfit(log_threshold_values, log_box_counts, 1)
    fractal_dimension = -coeffs[0]

    return fractal_dimension

def validate_data(pva_df):
    try:
        pva_cols = ['concentration', 'density', 'volume_shrinkage', 'modulus', 'modulus_std', 'transmittance']
        if not all(col in pva_df.columns for col in pva_cols):
            logger.error(f"PVA气凝胶数据缺少必要列: {set(pva_cols) - set(pva_df.columns)}")
            return False
        if not pva_df['concentration'].dtype in [np.float64, np.int64]:
            logger.error(f"PVA气凝胶数据的浓度列必须为数值类型")
            return False
        if not (pva_df['concentration'] > 0).all():
            logger.error(f"PVA气凝胶数据包含非正浓度值")
            return False
        missing = pva_df.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"PVA气凝胶数据包含{missing}个缺失值")
        logger.info("数据验证通过")
        return True
    except Exception as e:
        logger.error(f"数据验证失败: {e}")
        return False

def load_and_preprocess_data(image_paths, expand=True, target_samples=200):
    start_time = time.time()
    cache_file = Path("results/cached_dataset.csv")
    if cache_file.exists():
        logger.info("加载缓存数据集")
        try:
            final_df = pd.read_csv(cache_file)
            logger.info(f"从缓存加载数据集，耗时{time.time() - start_time:.2f}秒")
            return final_df
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}，重新处理数据")
    try:
        pva_df = pd.read_excel("PVA Aerogel Data.xlsx", sheet_name="Sheet2")
        pva_df.columns = ['concentration', 'density', 'volume_shrinkage', 'modulus', 'modulus_std', 'transmittance']
        if not validate_data(pva_df):
            return None
        # 提取图像特征并将其与浓度相匹配
        porosity_features = []
        for img_path in image_paths:
            features = extract_porosity_features(img_path)
            features['concentration'] = float(img_path.split('_')[-1].split('.')[0])  # 假设图片文件名包含浓度信息
            porosity_features.append(features)

        # 将图像特征整合到PVA DataFrame中
        porosity_df = pd.DataFrame(porosity_features)
        pva_df = pd.merge(pva_df, porosity_df, on='concentration', how='left')

        imputer = KNNImputer(n_neighbors=3)
        pva_df[['modulus', 'modulus_std']] = imputer.fit_transform(pva_df[['modulus', 'modulus_std']])

        # 扩展浓度范围
        if expand:
            logger.info(f"原始样本数: {len(pva_df)}")
            pva_df = expand_concentration_range(pva_df, target_samples)
            logger.info(f"扩展后样本数: {len(pva_df)}")

        pva_df.to_csv(cache_file, index=False)
        logger.info(f"数据预处理完成，缓存保存至{cache_file}，耗时{time.time() - start_time:.2f}秒")
        return pva_df
    except FileNotFoundError as e:
        logger.error(f"数据文件未找到: {e}")
        return None
    except Exception as e:
        logger.error(f"数据加载或预处理失败: {e}")
        return None

# 执行数据分析：可以在此进行进一步的建模或其他分析
def analyze_pva_features(pva_df):
    logger.info("PVA气凝胶数据的特征分析：
" + str(pva_df.describe()))
    return pva_df
