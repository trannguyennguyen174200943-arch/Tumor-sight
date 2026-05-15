"""
Module 2: Tái dựng 3D & Trực quan hóa
Chuyển đổi mặt nạ phân đoạn thành mesh 3D
Sử dụng thuật toán Marching Cubes
Tính toán: Thể tích, Diện tích bề mặt, Sphericity, v.v.
"""
import numpy as np  # Xử lý mảng
from typing import Tuple, List, Dict, Optional  # Type hints
from scipy import ndimage  # Xử lý ảnh
from skimage.measure import marching_cubes  # Marching cubes algorithm
import json  # JSON processing


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Trả về một shell boolean quanh đối tượng nhị phân"""
    gradient = ndimage.morphological_gradient(mask.astype(np.uint8), size=(3, 3, 3))  # Tính gradient hình thái
    return gradient > 0  # Trả về boundary


class Reconstruction3D:
    """Xử lý tái dựng 3D từ mặt nạ phân đoạn"""
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Khởi tạo reconstructor
        
        Args:
            voxel_spacing: Khoảng cách vật lý của voxel trong mm (x, y, z)
        """
        self.voxel_spacing = voxel_spacing  # Kích thước voxel

    def _crop_bounds(self, mask: np.ndarray, margin: int = 3) -> Tuple[slice, slice, slice]:
        """Tìm bounding box của mask với lề"""
        coords = np.argwhere(mask > 0)  # Tìm tất cả voxel u
        if len(coords) == 0:  # Nếu không có voxel
            return tuple(slice(0, int(dim)) for dim in mask.shape)  # type: ignore[return-value]  # Trả về full volume

        lower = np.maximum(coords.min(axis=0) - margin, 0)  # Cận dưới với lề
        upper = np.minimum(coords.max(axis=0) + margin + 1, np.asarray(mask.shape))  # Cận trên với lề
        return (
            slice(int(lower[0]), int(upper[0])),  # Slice chiều 0
            slice(int(lower[1]), int(upper[1])),  # Slice chiều 1
            slice(int(lower[2]), int(upper[2])),  # Slice chiều 2
        )

    def _offset_vertices(self, vertices: np.ndarray, crop_slices: Tuple[slice, slice, slice]) -> np.ndarray:
        """Dịch các đỉnh từ tọa độ crop sang tọa độ toàn bộ volume"""
        if vertices.size == 0:  # Nếu không có đỉnh
            return vertices.astype(np.float32)  # Trả về rỗng
        offset = np.asarray(
            [
                float(crop_slices[0].start or 0) * float(self.voxel_spacing[0]),  # Dịch X
                float(crop_slices[1].start or 0) * float(self.voxel_spacing[1]),
                float(crop_slices[2].start or 0) * float(self.voxel_spacing[2]),
            ],
            dtype=np.float32,
        )
        return (vertices.astype(np.float32) + offset[None, :]).astype(np.float32)

    def _smoothed_field(
        self,
        mask: np.ndarray,
        probability: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        spacing = np.asarray(self.voxel_spacing, dtype=np.float32)
        sigma = np.clip(0.75 / np.maximum(spacing, 1e-3), 0.45, 1.15)
        field = mask.astype(np.float32)

        if probability is not None and probability.shape == mask.shape:
            clipped_probability = np.clip(probability.astype(np.float32), 0.0, 1.0)
            field = np.maximum(field, clipped_probability)

        return ndimage.gaussian_filter(field.astype(np.float32), sigma=tuple(float(item) for item in sigma))

    def _mesh_quality(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        if len(vertices) == 0 or len(faces) == 0:
            return 0.0

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        edge_lengths = np.concatenate(
            [
                np.linalg.norm(v1 - v0, axis=1),
                np.linalg.norm(v2 - v1, axis=1),
                np.linalg.norm(v0 - v2, axis=1),
            ]
        )
        if edge_lengths.size == 0:
            return 0.0

        mean_edge = float(edge_lengths.mean())
        std_edge = float(edge_lengths.std())
        regularity = 1.0 - min(std_edge / max(mean_edge, 1e-6), 1.0)
        density_penalty = min(len(faces) / 48000.0, 1.0) * 0.18
        return float(np.clip((regularity * 0.82) + 0.18 - density_penalty, 0.0, 1.0))
    
    def marching_cubes(
        self,
        mask: np.ndarray,
        probability: Optional[np.ndarray] = None,
        level: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D mesh from segmentation mask using Marching Cubes
        
        Args:
            mask: Binary segmentation mask (H, W, D)
        
        Returns:
            vertices: Mesh vertices
            faces: Mesh faces (triangles)
        """
        if np.count_nonzero(mask) < 8:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

        crop_slices = self._crop_bounds(mask.astype(np.uint8))
        cropped_mask = mask[crop_slices[0], crop_slices[1], crop_slices[2]].astype(np.uint8)
        cropped_probability = None
        if probability is not None and probability.shape == mask.shape:
            cropped_probability = probability[crop_slices[0], crop_slices[1], crop_slices[2]].astype(np.float32)

        scalar_field = self._smoothed_field(cropped_mask, cropped_probability)
        marching_level = float(level if level is not None else 0.5)
        marching_level = float(np.clip(marching_level, 0.12, 0.88))

        try:
            vertices, faces, normals, values = marching_cubes(
                scalar_field,
                level=marching_level,
                spacing=self.voxel_spacing,
            )
        except ValueError:
            vertices, faces, normals, values = marching_cubes(
                cropped_mask.astype(np.float32),
                level=0.5,
                spacing=self.voxel_spacing,
            )

        vertices = self._offset_vertices(vertices, crop_slices)
        return vertices.astype(np.float32), faces.astype(np.int32)
    
    def compute_volume(self, mask: np.ndarray) -> float:
        """
        Compute tumor volume
        
        Args:
            mask: Binary segmentation mask
        
        Returns:
            volume: Volume in cm³
        """
        voxel_volume = np.prod(self.voxel_spacing)  # mm³
        num_voxels = np.sum(mask)
        volume_mm3 = num_voxels * voxel_volume
        volume_cm3 = volume_mm3 / 1000  # Convert to cm³
        return float(volume_cm3)
    
    def compute_surface_area(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """
        Compute surface area from mesh
        
        Args:
            vertices: Mesh vertices
            faces: Mesh faces (triangles)
        
        Returns:
            area: Surface area in cm²
        """
        if len(faces) == 0:
            return 0.0
        
        # Calculate area of each triangle
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # Cross product
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Total area in mm²
        total_area_mm2 = np.sum(areas)
        # Convert to cm²
        total_area_cm2 = total_area_mm2 / 100
        
        return float(total_area_cm2)
    
    def compute_sphericity(self, volume: float, surface_area: float) -> float:
        """
        Compute sphericity (0 = elongated, 1 = sphere)
        
        Sphericity = (π^(1/3) * (6*V)^(2/3)) / A
        
        Args:
            volume: Volume in cm³
            surface_area: Surface area in cm²
        
        Returns:
            sphericity: Value between 0 and 1
        """
        if surface_area == 0:
            return 0.0
        
        numerator = (np.pi ** (1/3)) * (6 * volume) ** (2/3)
        sphericity = numerator / surface_area
        return min(float(sphericity), 1.0)
    
    def compute_surface_irregularity(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """
        Compute surface irregularity (roughness) index
        Higher = more irregular surface
        
        Args:
            vertices: Mesh vertices
            faces: Mesh faces
        
        Returns:
            irregularity: Irregularity index
        """
        if len(faces) < 4:
            return 0.0
        
        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        normals = np.cross(v1 - v0, v2 - v0)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        # Compute variance of normal directions
        mean_normal = np.mean(normals, axis=0)
        variance = np.mean(np.linalg.norm(normals - mean_normal, axis=1))
        
        return float(variance)
    
    def compute_fractal_dimension(self, mask: np.ndarray) -> float:
        """
        Estimate fractal dimension of tumor boundary
        Uses box-counting method
        
        Args:
            mask: Binary segmentation mask
        
        Returns:
            fractal_dim: Estimated fractal dimension (1-3)
        """
        # Get surface voxels (boundary)
        boundary = _binary_boundary(mask)
        
        if np.sum(boundary) < 10:
            return 2.0  # Default for small objects
        
        # Box counting with multiple scales
        scales = []
        counts = []
        
        for scale in range(2, min(32, min(mask.shape) // 2), 2):
            # Resample with given scale
            scaled = ndimage.zoom(boundary.astype(float), 1/scale, order=0) > 0.5
            if scaled.sum() > 0:
                scales.append(np.log(scale))
                counts.append(np.log(scaled.sum()))
        
        if len(scales) < 2:
            return 2.0
        
        # Linear regression
        scales = np.array(scales)
        counts = np.array(counts)
        fractal_dim = -np.polyfit(scales, counts, 1)[0]
        
        return float(np.clip(fractal_dim, 1.0, 3.0))

    def compute_spatial_descriptors(self, mask: np.ndarray) -> Dict[str, List[float]]:
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return {
                "centroid_mm": [0.0, 0.0, 0.0],
                "bounds_min_mm": [0.0, 0.0, 0.0],
                "bounds_max_mm": [0.0, 0.0, 0.0],
                "dimensions_mm": [0.0, 0.0, 0.0],
                "principal_axes": [],
            }

        spacing = np.asarray(self.voxel_spacing, dtype=np.float32)
        coords_mm = coords.astype(np.float32) * spacing
        centroid = coords_mm.mean(axis=0)
        bounds_min = coords_mm.min(axis=0)
        bounds_max = coords_mm.max(axis=0)
        dimensions = bounds_max - bounds_min + spacing

        centered = coords_mm - centroid
        principal_axes: List[List[float]] = []
        if len(coords_mm) >= 3:
            covariance = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            for axis_index in range(min(3, eigenvectors.shape[1])):
                axis_vector = eigenvectors[:, axis_index]
                axis_extent = float(np.sqrt(max(eigenvalues[axis_index], 0.0)) * 2.0)
                principal_axes.append(
                    [
                        float(axis_vector[0]),
                        float(axis_vector[1]),
                        float(axis_vector[2]),
                        axis_extent,
                    ]
                )

        return {
            "centroid_mm": centroid.tolist(),
            "bounds_min_mm": bounds_min.tolist(),
            "bounds_max_mm": bounds_max.tolist(),
            "dimensions_mm": dimensions.tolist(),
            "principal_axes": principal_axes,
        }
    
    def analyze(
        self,
        mask: np.ndarray,
        probability: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        Complete 3D analysis of segmentation mask
        
        Args:
            mask: Binary segmentation mask (H, W, D)
        
        Returns:
            analysis: Dict with all computed metrics
        """
        # Generate mesh
        try:
            vertices, faces = self.marching_cubes(mask, probability=probability, level=threshold)
        except ValueError:
            vertices = np.empty((0, 3), dtype=np.float32)
            faces = np.empty((0, 3), dtype=np.int32)

        shell_vertices = np.empty((0, 3), dtype=np.float32)
        shell_faces = np.empty((0, 3), dtype=np.int32)
        if probability is not None and probability.shape == mask.shape and np.count_nonzero(mask) >= 8:
            shell_level = max(0.18, float((threshold if threshold is not None else 0.5) - 0.09))
            support_mask = (probability >= shell_level).astype(np.uint8)
            if np.count_nonzero(support_mask) >= np.count_nonzero(mask):
                try:
                    shell_vertices, shell_faces = self.marching_cubes(
                        support_mask,
                        probability=probability,
                        level=shell_level,
                    )
                except ValueError:
                    shell_vertices = np.empty((0, 3), dtype=np.float32)
                    shell_faces = np.empty((0, 3), dtype=np.int32)
        
        # Compute metrics
        volume = self.compute_volume(mask)
        surface_area = self.compute_surface_area(vertices, faces)
        sphericity = self.compute_sphericity(volume, surface_area)
        surface_irregularity = self.compute_surface_irregularity(vertices, faces)
        fractal_dimension = self.compute_fractal_dimension(mask)
        spatial = self.compute_spatial_descriptors(mask)
        
        return {
            "volume_cm3": volume,
            "surface_area_cm2": surface_area,
            "sphericity": sphericity,
            "surface_irregularity": surface_irregularity,
            "fractal_dimension": fractal_dimension,
            "mesh_quality_score": self._mesh_quality(vertices, faces),
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "voxel_spacing_mm": list(self.voxel_spacing),
            **spatial,
            "mesh": {
                "vertices": vertices.tolist(),
                "faces": faces.tolist(),
                "shell_vertices": shell_vertices.tolist(),
                "shell_faces": shell_faces.tolist(),
            },
        }


if __name__ == "__main__":
    # Test
    recon = Reconstruction3D()
    
    # Create synthetic mask
    mask = np.zeros((64, 64, 64), dtype=np.uint8)
    # Add sphere
    for i in range(64):
        for j in range(64):
            for k in range(64):
                dist = np.sqrt((i-32)**2 + (j-32)**2 + (k-32)**2)
                if dist < 15:
                    mask[i, j, k] = 1
    
    analysis = recon.analyze(mask)
    print(json.dumps({k: v for k, v in analysis.items() if k != 'mesh'}, indent=2))
