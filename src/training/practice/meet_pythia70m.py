# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="EleutherAI/pythia-70m-deduped")

while True:
    print(pipe(input()))