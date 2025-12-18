import torch
import random
import torchaudio

class TicsAugmentation:
    def __init__(self, mode='baseline', sample_rate=16000):
        self.mode = mode
        self.sample_rate = sample_rate

    def __call__(self, waveform: torch.Tensor, is_view2: bool) -> torch.Tensor:
        # 所有模式共用的基础增强：增益 (Gain)
        aug_wav = waveform.clone()
        gain = random.uniform(0.8, 1.2)
        aug_wav = aug_wav * gain

        # 根据不同实验组执行特定策略
        if self.mode == 'baseline' or self.mode=='shuffle':
            return self._apply_baseline(aug_wav, is_view2)
        elif self.mode == 'flip_comparison':
            return self._apply_flip(aug_wav, is_view2)
        else:
            return torch.clamp(aug_wav, -1.0, 1.0)

    def _apply_baseline(self, wav, is_view2):
        """基准组：仅加噪"""
        if is_view2:
            noise = torch.randn_like(wav) * 0.005
            wav = wav + noise
        return torch.clamp(wav, -1.0, 1.0)

    def _apply_flip(self, wav, is_view2):
        """对比组：物理反转"""
        if is_view2:
            wav = torch.flip(wav, dims=[-1])
        return torch.clamp(wav, -1.0, 1.0)