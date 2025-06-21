import argparse
import uuid
from typing import Tuple

import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import login
from PIL import Image
import os

login(token="xxx")

DISEASE_PROMPTS = {
    # Chest X-ray
    ("chest", "atelectasis"): "a chest x-ray with atelectasis",
    ("chest", "cardiomegaly"): "a chest x-ray with cardiomegaly",
    ("chest", "consolidation"): "a chest x-ray with consolidation",
    ("chest", "edema"): "a chest x-ray with pulmonary edema",
    ("chest", "effusion"): "a chest x-ray with pleural effusion",
    ("chest", "emphysema"): "a chest x-ray with emphysema",
    ("chest", "fibrosis"): "a chest x-ray with pulmonary fibrosis",
    ("chest", "hernia"): "a chest x-ray with diaphragmatic hernia",
    ("chest", "infiltration"): "a chest x-ray with infiltration",
    ("chest", "mass"): "a chest x-ray showing a lung mass",
    ("chest", "nodule"): "a chest x-ray with a lung nodule",
    ("chest", "pneumonia"): "a chest x-ray showing pneumonia",
    ("chest", "pleural thickening"): "a chest x-ray with pleural thickening",
    ("chest", "pneumothorax"): "a chest x-ray with pneumothorax",
    ("chest", "healthy"): "a normal chest x-ray",

    # Brain MRI
    ("brain", "glioma"): "a brain mri with glioma tumor",
    ("brain", "meningioma"): "a brain mri showing meningioma near the skull",
    ("brain", "pituitary tumor"): "a brain mri with a pituitary tumor",
    ("brain", "healthy"): "a normal brain mri with no tumor",
}


def get_prompt(organ: str, disease: str) -> str:
    key = (organ.lower().strip(), disease.lower().strip())
    if key not in DISEASE_PROMPTS:
        raise ValueError(f"Unsupported combination: {key}")
    return DISEASE_PROMPTS[key]


def generate_sd_image(organ: str, disease: str, out_dir: str = "out") -> Tuple[str, str]:
    prompt = get_prompt(organ, disease)
    print(f"[INFO] Prompt: {prompt}")

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float32
    ).to("cuda")

    # Load correct LoRA weights based on organ
    if organ.lower() == "brain":
        pipe.load_lora_weights(
            "mara2606/stable-diffusion-2-1-brain-mri",
            weight_name="checkpoint-12000/pytorch_lora_weights.safetensors"
        )
    elif organ.lower() == "chest":
        pipe.load_lora_weights(
            "mara2606/stable-diffusion-2-1-chest-xray",
            weight_name="checkpoint-10000/pytorch_lora_weights.safetensors"
        )
    else:
        raise ValueError(f"No LoRA available for organ: {organ}")

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    image = pipe(
        prompt=prompt,
        width=256,
        height=256,
        guidance_scale=1.5
    ).images[0]

    os.makedirs(out_dir, exist_ok=True)

    filename = f"{organ.lower()}_{disease.lower().replace(' ', '_')}_sd_{uuid.uuid4().hex}.png"
    out_path = os.path.join(out_dir, filename)
    image.save(out_path)
    print(f"[INFO] Image saved to: {out_path}")
    return (out_path, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image with Stable Diffusion")
    parser.add_argument("--organ", type=str, required=True, help="Organ name, e.g., brain or chest")
    parser.add_argument("--disease", type=str, required=True, help="Disease name, e.g., glioma, pneumonia")

    args = parser.parse_args()
    generate_sd_image(args.organ, args.disease)

