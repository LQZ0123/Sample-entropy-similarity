import torch
import scipy

class FPSearch:
    def cosine_similarity(self, a, b):
        dot_product = torch.sum(a * b)
        norm_a = torch.sqrt(torch.sum(a** 2))
        norm_b = torch.sqrt(torch.sum(b** 2))
        
        if norm_a == 0 or norm_b == 0:
            return torch.tensor(0.0)
        return dot_product / (norm_a * norm_b)
    
    def sample_entropy_between(self, a: torch.Tensor, b: torch.Tensor) -> float:
        entropy_a, a = self.get_entropy_and_weighted_intensity(a)
        entropy_b, b = self.get_entropy_and_weighted_intensity(b)

        entropy_merged = scipy.stats.entropy(a + b)
        return 1 - (2 * entropy_merged - entropy_a - entropy_b) / torch.log(torch.tensor(4.0)).item()

    def get_entropy_and_weighted_intensity(self, intensity: torch.Tensor) -> tuple[float, torch.Tensor]:

        spectral_entropy = scipy.stats.entropy(intensity)
        if spectral_entropy < 3:
            weight = 0.25 + 0.25 * spectral_entropy
            weighted_intensity = torch.pow(intensity, weight)
            intensity_sum = torch.sum(weighted_intensity)
            weighted_intensity /= intensity_sum
            spectral_entropy = scipy.stats.entropy(weighted_intensity)
            return spectral_entropy, weighted_intensity
        else:
            return spectral_entropy, intensity
