import numpy as np

from ppca_rs import PPCAModel

real_model = PPCAModel(
    transform=np.array([[1, 1], [0, 1], [0, 1]], dtype="float64"),
    isotropic_noise=0.1,
    mean=np.array([[0], [1], [0]], dtype="float64"),
)
sample = real_model.sample(100, mask_prob=0.2)
model = PPCAModel.init(2, sample)

for it in range(100):
    print(f"At iteration {it + 1} PPCA llk is {model.llk(sample)}")
    model: PPCAModel = model.iterate(sample)

model = model.to_canonical()

print(model)
print(model.singular_values)

inferred = model.infer(sample)
print(inferred.smoothed_covariances_diagonal(model).numpy() ** 0.5)
