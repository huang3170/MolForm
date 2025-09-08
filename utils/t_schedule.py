import numpy as np
from scipy import stats
import torch


def sample_mix_up02_beta(num_samples):
    p1, p2 = 1.9, 1.0
    shape = (num_samples,)
    dist = torch.distributions.beta.Beta(p1, p2)
    samples_beta = dist.sample(shape)
    samples_uniform = torch.rand(shape)
    u = torch.rand(shape)
    return torch.where(u < 0.02, samples_uniform, samples_beta)


class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        C = a / (np.exp(a) - 1)
        return C * np.exp(a * x)
    
def sample_ushape_t(num_samples, a=4.0):
    exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')
    t = exponential_distribution.rvs(size=num_samples, a=a)
    t = torch.from_numpy(t).float()
    t = torch.cat([t, 1 - t], dim=0)
    t = t[torch.randperm(t.shape[0])]
    t = t[:num_samples]

    t_min = 1e-5
    t_max = 1-1e-5
    t = t * (t_max - t_min) + t_min
    
    return t