import numpy as np

from ppca_rs import PPCAModel, Prior

def dbg(x): print(x); return x

real_model = PPCAModel(
    transform=np.matrix([[1, 1, 0], [1, 0, 1]], dtype="float64").T,
    isotropic_noise=0.1,
    mean=np.array([[0, 1, 0]], dtype="float64").T,
)
sample = real_model.sample(100, mask_prob=0.2)
model = PPCAModel.init(2, sample)
prior = (
    Prior()
    .with_isotropic_noise_prior(100.0, 100.0)
    .with_mean_prior(np.array([1.0, 0.0, 1.0], dtype="float64"), 0.0001 * np.eye(3, dtype="float64").T)
)

for it in range(100):
    print(f"At iteration {it + 1} PPCA llk is {model.llk(sample)}")
    model: PPCAModel = model.iterate_with_prior(sample, prior)

model = model.to_canonical()

print(model)
print(model.isotropic_noise)
