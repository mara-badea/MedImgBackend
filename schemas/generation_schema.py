from pydantic import BaseModel
from enum import Enum

class GenerationModel(str, Enum):
    gan = "gan"
    diffusion = "diffusion"

class GenerationRequest(BaseModel):
    model_type: GenerationModel  # "gan" or "diffusion"
    disease: str                 # e.g. "pneumonia"
    organ: str              # e.g. "xray"
    seeds: str = "0"           # optional (GAN only)
