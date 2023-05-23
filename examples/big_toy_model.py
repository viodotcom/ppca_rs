import numpy as np
from ppca_rs import PPCAModel

print("Generating model")

transform = np.random.binomial(1.0, 0.1, size=(200, 16))
real_model = PPCAModel(
    transform=np.matrix(transform, dtype="float64"),
    isotropic_noise=0.1,
    mean=np.zeros((200, 1), dtype="float64"),
)

print("Generating synthetic sample")
sample = real_model.sample(100_000, 0.2)

print("Initializing model")
model = PPCAModel.init(16, sample)

print("Starting iterations...")

for it in range(24):
    print(f"At iteration {it + 1} PPCA llk is {model.llk(sample) / len(sample)}")
    model = model.iterate(sample)
    print("Done creating iterated model")

print("Model trained")
