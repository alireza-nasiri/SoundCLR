import numpy as np

import torch
import torchvision as tv
import librosa

import random



def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value


class ToTensor1D(tv.transforms.ToTensor):

    def __call__(self, tensor: np.ndarray):
        tensor_2d = super(ToTensor1D, self).__call__(tensor[..., np.newaxis])
        
        return tensor_2d.squeeze_(0)

class RandomNoise():
    def __init__(self, min_noise=0.0, max_noise=0.05): #0.002, 0.01
        super(RandomNoise, self).__init__()
        
        self.min_noise = min_noise
        self.max_noise = max_noise
        
    def addNoise(self, wave):
        noise_val = random.uniform(self.min_noise, self.max_noise)
        noise = torch.from_numpy(np.random.normal(0, noise_val, wave.shape[0]))
        noisy_wave = wave + noise
        
        return noisy_wave
    
    def __call__(self, x):
        return self.addNoise(x)



class RandomScale():

    def __init__(self, max_scale: float = 1.25):
        super(RandomScale, self).__init__()

        self.max_scale = max_scale

    @staticmethod
    def random_scale(max_scale: float, signal: torch.Tensor) -> torch.Tensor:
        scaling = np.power(max_scale, np.random.uniform(-1, 1)) #between 1.25**(-1) and 1.25**(1)
        output_size = int(signal.shape[-1] * scaling)
        ref = torch.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)
        
        # ref1 is of size output_size
        ref1 = ref.clone().type(torch.int64)
        ref2 = torch.min(ref1 + 1, torch.full_like(ref1, signal.shape[-1] - 1, dtype=torch.int64))
        
        r = ref - ref1.type(ref.type())
        
        scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r
        
        
        return scaled_signal

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_scale(self.max_scale, x)

    
    

class RandomCrop():

    def __init__(self, out_len: int = 44100, train: bool = True):
        super(RandomCrop, self).__init__()

        self.out_len = out_len
        self.train = train

    def random_crop(self, signal: torch.Tensor) -> torch.Tensor:
        if self.train:
            left = np.random.randint(0, signal.shape[-1] - self.out_len)
        else:
            left = int(round(0.5 * (signal.shape[-1] - self.out_len)))

        orig_std = signal.float().std() * 0.5
        output = signal[..., left:left + self.out_len]

        out_std = output.float().std()
        if out_std < orig_std:
            output = signal[..., :self.out_len]

        new_out_std = output.float().std()
        if orig_std > new_out_std > out_std:
            output = signal[..., -self.out_len:]

        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_crop(x) if x.shape[-1] > self.out_len else x


class RandomPadding():

    def __init__(self, out_len: int = 88200, train: bool = True):
        super(RandomPadding, self).__init__()

        self.out_len = out_len
        self.train = train

    def random_pad(self, signal: torch.Tensor) -> torch.Tensor:
        
        if self.train:
            left = np.random.randint(0, self.out_len - signal.shape[-1])
        else:
            left = int(round(0.5 * (self.out_len - signal.shape[-1])))

        right = self.out_len - (left + signal.shape[-1])

        pad_value_left = signal[..., 0].float().mean().to(signal.dtype)
        pad_value_right = signal[..., -1].float().mean().to(signal.dtype)
        output = torch.cat((
            torch.zeros(signal.shape[:-1] + (left,), dtype=signal.dtype, device=signal.device).fill_(pad_value_left),
            signal,
            torch.zeros(signal.shape[:-1] + (right,), dtype=signal.dtype, device=signal.device).fill_(pad_value_right)
        ), dim=-1)

        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_pad(x) if x.shape[-1] < self.out_len else x
    
            
    
    
class FrequencyMask():
    def __init__(self, max_width, numbers): 
        super(FrequencyMask, self).__init__()
        
        self.max_width = max_width
        self.numbers = numbers
    
    def addFreqMask(self, wave):
        #print(wave.shape)
        for _ in range(self.numbers):
            #choose the length of mask
            mask_len = random.randint(0, self.max_width)
            start = random.randint(0, wave.shape[1] - mask_len) #start of the mask
            end = start + mask_len
            wave[:, start:end, : ] = 0
            
        return wave
    
    def __call__(self, wave):
        return self.addFreqMask(wave)
    
        

class TimeMask():
    def __init__(self, max_width, numbers): 
        super(TimeMask, self).__init__()
        
        self.max_width = max_width
        self.numbers = numbers
    
    
    def addTimeMask(self, wave):
        
        for _ in range(self.numbers):
            #choose the length of mask
            mask_len = random.randint(0, self.max_width)
            start = random.randint(0, wave.shape[2] - mask_len) #start of the mask
            end = start + mask_len
            wave[ : , : , start:end] = 0
            
        return wave
    
    def __call__(self, wave):
        return self.addTimeMask(wave)
        
        
        
        
        
