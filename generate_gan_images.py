import os
import uuid
import argparse
import subprocess

GAN_CLASS_LABELS = {
    ("brain", "no tumor"): 0,
    ("brain", "glioma"): 100,
    ("brain", "meningioma"): 200,
    ("brain", "pituitary tumor"): 300,

    ("chest", "atelectasis"): 0,
    ("chest", "cardiomegaly"): 1,
    ("chest", "consolidation"): 2,
    ("chest", "edema"): 3,
    ("chest", "effusion"): 4,
    ("chest", "emphysema"): 5,
    ("chest", "fibrosis"): 6,
    ("chest", "hernia"): 7,
    ("chest", "infiltration"): 8,
    ("chest", "mass"): 9,
    ("chest", "no finding"): 10,
    ("chest", "nodule"): 11,
    ("chest", "pleural thickening"): 12,
    ("chest", "pneumonia"): 13,
    ("chest", "pneumothorax"): 14,
}

def get_gan_class_label(organ: str, disease: str) -> int:
    key = (organ.lower().strip(), disease.lower().strip())
    if key not in GAN_CLASS_LABELS:
        raise ValueError(f"No GAN class label for: {key}")
    return GAN_CLASS_LABELS[key]

def get_model_path(organ: str) -> str:
    if organ.lower() == "brain":
        return "stylegan2/brain-mri-models/brain-mri-564kimg.pkl"
    elif organ.lower() == "chest":
        return "stylegan2/clahe-chest-xray-models/clahe-chest-xray-1168kimg.pkl"
    else:
        raise ValueError(f"No model available for organ: {organ}")

def generate_gan_image(organ: str, disease: str, out_dir: str = "generated_imgs_poc") -> str:
    class_label = get_gan_class_label(organ, disease)
    seed = uuid.uuid4().int % (2**32)
    filename = f"{organ.lower()}_{disease.lower().replace(' ', '_')}_gan_{uuid.uuid4().hex}.png"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    network_path = get_model_path(organ)

    command = [
        "python3", "stylegan2/stylegan2-ada-pytorch/generate.py",
        "--outdir", out_dir,
        "--seeds", str(seed),
        "--class", str(class_label),
        "--network", network_path
    ]

    print(f"[INFO] Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

    # GAN salvează imaginea ca <seed>.png (ex: 012345.png)
    generated_name = f"seed{seed:06d}.png"
    generated_path = os.path.join(out_dir, generated_name)
    os.rename(generated_path, out_path)

    print(f"[INFO] Image saved as: {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image with StyleGAN2-ADA")
    parser.add_argument("--organ", type=str, required=True)
    parser.add_argument("--disease", type=str, required=True)

    args = parser.parse_args()
    out_path = generate_gan_image(args.organ, args.disease)
    print(out_path)
