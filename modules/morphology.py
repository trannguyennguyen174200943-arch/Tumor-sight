"""
Module 3: Phân tích Hình thái & Radiomics Nâu cao
Trích xuất các đặc trưng shape, texture, và radiomics tinh vi
Features: GLCM, GLRLM, Wavelet, Shape descriptors, Entropy
"""
import numpy as np  # Xử lý mảng
from typing import Dict, Tuple  # Type hints
from scipy import stats, ndimage  # Các hàm toán học


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Trả về một shell boolean quanh đối tượng nhị phân"""
    gradient = ndimage.morphological_gradient(mask.astype(np.uint8), size=(3, 3, 3))  # Tính gradient hình thái
    return gradient > 0  # Trả về boundary


class AdvancedMorphologyAnalyzer:
    """Trích xuất các đặc trưng hình thái học và radiomics nâu cao"""
    
    def __init__(self, voxel_spacing: tuple = (1.0, 1.0, 1.0)):
        self.voxel_spacing = voxel_spacing  # Kích thước voxel trong từng chiều
    
    # =====================================================================
    # SHAPE FEATURES (1st Order)
    # =====================================================================
    
    def extract_shape_features(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Trích xuất các đặc trưng shape nâu cao
        
        Args:
            mask: Binary segmentation mask (mặt nạ phân đoạn nhị phân)
        
        Returns:
            features: Dict của các đặc trưng shape
        """
        coords = np.argwhere(mask > 0)  # Lấy tọa độ của các voxel u
        
        if len(coords) < 10:  # Nếu quá ít voxel
            return {}  # Trả về rỗng
        
        # Tâm khối lượng
        centroid = coords.mean(axis=0)  # Trung bình các tọa độ
        
        # Ma trận hiệp phương sai và PCA
        coords_centered = coords - centroid  # Tâm tại gốc
        cov = np.cov(coords_centered.T)  # Ma trận hiệp phương sai
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # Eigenvalue decomposition
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sắp xếp giảm dần
        
        # Độ dài trục
        l1, l2, l3 = np.sqrt(eigenvalues)  # Căn bậc 2 của eigenvalue
        
        # Thể tích
        volume = len(coords)  # Số lượng voxel
        surface_area = self._estimate_surface_area(mask)  # Ước tính diện tích bề mặt
        
        # Các mô tả shape nâu cao
        elongation = l1 / (l2 + 1e-8)  # Độ kéo dài (lớn = kéo dài)
        flatness = l2 / (l3 + 1e-8)  # Độ phẳng (lớn = phẳng)
        compactness = (4 * np.pi * volume ** 2) / (3 * surface_area ** 3 + 1e-8)  # Độ nhỏ gọn
        
        # Sphericity (0-1, 1 = hình cầu)
        sphericity = ((np.pi ** (1/3)) * (6 * volume) ** (2/3)) / (surface_area + 1e-8)  # Độ hình cầu
        
        # Tỉ lệ bề mặt trên thể tích
        surface_volume_ratio = surface_area / (volume + 1e-8)  # Tỉ lệ SA/V
        
        # Asphericity (0 = hình cầu, 1 = kéo dài)
        asphericity = (l1 - l3) / (l1 + 1e-8)  # Mức độ không là hình cầu
        
        # Extent (tỉ lệ volume so với bounding box)
        extent = volume / (l1 * l2 * l3 + 1e-8)  # Mức độ điền đầy hộp
        
        # Solidity (tỉ lệ volume so với convex hull) - ước tính
        solidity = 1.0 - (surface_area - volume) / (surface_area + 1e-8)  # Độ rắn chắc
        
        return {
            "centroid_x": float(centroid[0]),  # Tọa độ X tâm
            "centroid_y": float(centroid[1]),  # Tọa độ Y tâm
            "centroid_z": float(centroid[2]),
            "volume_voxels": float(volume),
            "surface_area_voxels": float(surface_area),
            "elongation": float(elongation),
            "flatness": float(flatness),
            "compactness": float(compactness),
            "sphericity": float(np.clip(sphericity, 0, 1)),
            "surface_volume_ratio": float(surface_volume_ratio),
            "asphericity": float(asphericity),
            "extent": float(extent),
            "solidity": float(solidity),
            "max_axis_length": float(l1),
            "mid_axis_length": float(l2),
            "min_axis_length": float(l3),
        }
    
    # =====================================================================
    # GLCM FEATURES (Gray Level Co-occurrence Matrix)
    # =====================================================================
    
    def compute_glcm_features(self, volume: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Extract GLCM-based texture features
        
        Args:
            volume: Raw 3D volume
            mask: Binary mask
        
        Returns:
            features: GLCM features
        """
        # Get tumor voxels
        tumor_intensities = volume[mask > 0]
        
        if len(tumor_intensities) < 10:
            return {}
        
        # Quantize intensities to 32 levels for GLCM
        tumor_min, tumor_max = tumor_intensities.min(), tumor_intensities.max()
        quantized = np.clip(
            ((tumor_intensities - tumor_min) / (tumor_max - tumor_min + 1e-8) * 31).astype(int),
            0, 31
        )
        
        # Build GLCM for 3D (simplified: use 2D slices)
        glcm = np.zeros((32, 32), dtype=np.float32)
        
        # Extract 2D slices and compute GLCM
        for z in range(mask.shape[2]):
            slice_mask = mask[:, :, z] > 0
            slice_volume = volume[:, :, z]
            
            if slice_mask.sum() > 0:
                slice_intensities = slice_volume[slice_mask]
                slice_quantized = np.clip(
                    ((slice_intensities - tumor_min) / (tumor_max - tumor_min + 1e-8) * 31).astype(int),
                    0, 31
                )
                
                # Build co-occurrence for this slice (horizontal neighbors)
                for i in range(len(slice_quantized) - 1):
                    glcm[slice_quantized[i], slice_quantized[i+1]] += 1
                    glcm[slice_quantized[i+1], slice_quantized[i]] += 1
        
        # Normalize GLCM
        glcm = glcm / (glcm.sum() + 1e-8)
        
        # Compute GLCM features
        features = {}
        
        # Mean and variance
        px = glcm.sum(axis=1)
        py = glcm.sum(axis=0)
        mean_x = np.sum(np.arange(32) * px)
        mean_y = np.sum(np.arange(32) * py)
        std_x = np.sqrt(np.sum((np.arange(32) - mean_x) ** 2 * px))
        std_y = np.sqrt(np.sum((np.arange(32) - mean_y) ** 2 * py))
        
        # Contrast
        contrast = 0
        for i in range(32):
            for j in range(32):
                contrast += glcm[i, j] * (i - j) ** 2
        features["glcm_contrast"] = float(contrast)
        
        # Dissimilarity
        dissimilarity = 0
        for i in range(32):
            for j in range(32):
                dissimilarity += glcm[i, j] * np.abs(i - j)
        features["glcm_dissimilarity"] = float(dissimilarity)
        
        # Homogeneity
        homogeneity = 0
        for i in range(32):
            for j in range(32):
                homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
        features["glcm_homogeneity"] = float(homogeneity)
        
        # ASM (Angular Second Moment)
        asm = np.sum(glcm ** 2)
        features["glcm_asm"] = float(asm)
        
        # Energy
        features["glcm_energy"] = float(np.sqrt(asm))
        
        # Correlation
        correlation = 0
        if std_x > 0 and std_y > 0:
            for i in range(32):
                for j in range(32):
                    correlation += glcm[i, j] * (i - mean_x) * (j - mean_y) / (std_x * std_y)
        features["glcm_correlation"] = float(correlation)
        
        return features
    
    # =====================================================================
    # HISTOGRAM & INTENSITY FEATURES
    # =====================================================================
    
    def extract_intensity_features(self, volume: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Extract advanced intensity-based features
        """
        tumor_intensities = volume[mask > 0]
        
        if len(tumor_intensities) < 10:
            return {}
        
        # Basic statistics
        mean_int = float(np.mean(tumor_intensities))
        std_int = float(np.std(tumor_intensities))
        min_int = float(np.min(tumor_intensities))
        max_int = float(np.max(tumor_intensities))
        median_int = float(np.median(tumor_intensities))
        
        # Higher order statistics
        skewness = float(stats.skew(tumor_intensities))
        kurtosis = float(stats.kurtosis(tumor_intensities))
        
        # Entropy & Energy
        hist, _ = np.histogram(tumor_intensities, bins=64)
        hist = hist / (hist.sum() + 1e-8)
        entropy = float(-np.sum(hist * np.log(hist + 1e-8)))
        energy = float(np.sum(hist ** 2))
        
        # Quantile features
        q10 = float(np.percentile(tumor_intensities, 10))
        q25 = float(np.percentile(tumor_intensities, 25))
        q75 = float(np.percentile(tumor_intensities, 75))
        q90 = float(np.percentile(tumor_intensities, 90))
        
        # Range and IQR
        range_val = max_int - min_int
        iqr = q75 - q25
        
        # Robust statistics
        mad = float(np.mean(np.abs(tumor_intensities - median_int)))
        
        return {
            "intensity_mean": mean_int,
            "intensity_std": std_int,
            "intensity_min": min_int,
            "intensity_max": max_int,
            "intensity_median": median_int,
            "intensity_skewness": skewness,
            "intensity_kurtosis": kurtosis,
            "intensity_entropy": entropy,
            "intensity_energy": energy,
            "intensity_q10": q10,
            "intensity_q25": q25,
            "intensity_q75": q75,
            "intensity_q90": q90,
            "intensity_range": float(range_val),
            "intensity_iqr": float(iqr),
            "intensity_mad": mad,
        }
    
    # =====================================================================
    # GRADIENT & EDGE FEATURES (Radiomics)
    # =====================================================================
    
    def extract_gradient_features(self, volume: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Extract gradient and edge-based radiomics features
        """
        # Compute gradients
        grad_x = ndimage.sobel(volume.astype(float), axis=0)
        grad_y = ndimage.sobel(volume.astype(float), axis=1)
        grad_z = ndimage.sobel(volume.astype(float), axis=2)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Get gradient values in tumor region
        grad_tumor = grad_mag[mask > 0]
        
        if len(grad_tumor) < 10:
            return {}
        
        return {
            "gradient_mean": float(np.mean(grad_tumor)),
            "gradient_std": float(np.std(grad_tumor)),
            "gradient_max": float(np.max(grad_tumor)),
            "gradient_min": float(np.min(grad_tumor)),
            "gradient_median": float(np.median(grad_tumor)),
            "gradient_energy": float(np.sum(grad_tumor ** 2)),
            "gradient_entropy": float(-np.sum((grad_tumor / (grad_tumor.sum() + 1e-8)) * 
                                             np.log(grad_tumor / (grad_tumor.sum() + 1e-8) + 1e-8))),
        }
    
    # =====================================================================
    # FRACTAL DIMENSION
    # =====================================================================
    
    def compute_fractal_dimension(self, mask: np.ndarray) -> float:
        """
        Estimate fractal dimension of tumor boundary
        Uses box-counting method
        """
        boundary = _binary_boundary(mask)
        
        if np.sum(boundary) < 10:
            return 2.0
        
        scales = []
        counts = []
        
        for scale in range(2, min(32, min(mask.shape) // 2), 2):
            scaled = ndimage.zoom(boundary.astype(float), 1/scale, order=0) > 0.5
            if scaled.sum() > 0:
                scales.append(np.log(scale))
                counts.append(np.log(scaled.sum()))
        
        if len(scales) < 2:
            return 2.0
        
        # Linear regression for fractal dimension
        scales = np.array(scales)
        counts = np.array(counts)
        fractal_dim = -np.polyfit(scales, counts, 1)[0]
        
        return float(np.clip(fractal_dim, 1.0, 3.0))
    
    # =====================================================================
    # HELPER FUNCTIONS
    # =====================================================================
    
    def _estimate_surface_area(self, mask: np.ndarray) -> float:
        """Estimate surface area from binary mask"""
        boundary = _binary_boundary(mask)
        return float(np.sum(boundary))
    
    # =====================================================================
    # MAIN ANALYSIS FUNCTION
    # =====================================================================
    
    def analyze(self, volume: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Complete advanced morphology analysis
        
        Args:
            volume: Raw 3D volume
            mask: Binary segmentation mask
        
        Returns:
            analysis: Dict with all morphology features (50+ features)
        """
        shape_features = self.extract_shape_features(mask)
        intensity_features = self.extract_intensity_features(volume, mask)
        glcm_features = self.compute_glcm_features(volume, mask)
        gradient_features = self.extract_gradient_features(volume, mask)
        
        # Fractal dimension
        fractal_dim = self.compute_fractal_dimension(mask)
        
        # Combine all features
        analysis = {
            **shape_features,
            **intensity_features,
            **glcm_features,
            **gradient_features,
            "fractal_dimension": fractal_dim,
        }

        intensity_std = float(intensity_features.get("intensity_std", 0.0))
        gradient_energy = float(gradient_features.get("gradient_energy", 0.0))
        solidity = float(shape_features.get("solidity", 1.0))
        analysis["surface_irregularity"] = float(np.clip(1.0 - solidity, 0.0, 1.0))
        analysis["surface_irregularity_index"] = analysis["surface_irregularity"]
        analysis["std_intensity"] = intensity_std
        analysis["grad_energy"] = gradient_energy
        analysis["gradient_distribution"] = float(gradient_features.get("gradient_std", 0.0))
        analysis["intensity_heterogeneity_index"] = float(
            intensity_std / (abs(float(intensity_features.get("intensity_mean", 0.0))) + 1e-8)
        )
        
        return analysis


# Malignancy scoring based on advanced radiomics
ADVANCED_MALIGNANCY_INDICATORS = {
    "high_contrast": lambda m: m.get("glcm_contrast", 0) > 50,
    "low_homogeneity": lambda m: m.get("glcm_homogeneity", 1) < 0.3,
    "high_entropy": lambda m: m.get("intensity_entropy", 0) > 4.0,
    "irregular_surface": lambda m: m.get("surface_irregularity", 0) > 0.4,
    "elongated_shape": lambda m: m.get("elongation", 1) > 1.8,
    "high_gradient": lambda m: m.get("gradient_std", 0) > 10,
    "fractal_complexity": lambda m: m.get("fractal_dimension", 2) > 2.4,
}


if __name__ == "__main__":
    analyzer = AdvancedMorphologyAnalyzer()

    mask = np.zeros((64, 64, 64), dtype=np.uint8)
    volume = np.random.randn(64, 64, 64) * 20 + 100

    for i in range(64):
        for j in range(64):
            for k in range(64):
                dist = np.sqrt((i - 32) ** 2 + (j - 32) ** 2 + (k - 32) ** 2)
                if dist < 15:
                    mask[i, j, k] = 1
                    volume[i, j, k] += 30

    analysis = analyzer.analyze(volume, mask)
    print("Advanced Morphology Analysis:")
    print(f"Total features extracted: {len(analysis)}")
    for key, value in list(analysis.items())[:20]:
        print(f"  {key}: {value:.3f}")
