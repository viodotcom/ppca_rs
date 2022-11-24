import numpy as np

from ppca import PPCAModel, Dataset

real_model = PPCAModel(
    transform=np.matrix([[1, 1, 0], [1, 0, 1]], dtype="float32").T,
    isotropic_noise=0.1,
    mean=np.array([0, 1, 0], dtype="float32"),
)
sample = real_model.sample_masked(100, mask_prob=0.2)
model = PPCAModel.init(2, sample)

for it in range(100):
    print(f"At iteration {it + 1} PPCA llk is {model.llk_masked(sample)}")
    model: PPCAModel = model.iterate_masked(sample)

print(model)
