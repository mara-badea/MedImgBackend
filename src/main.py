import traceback

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import uuid
import logging

from services.ssh_handler import SshHandler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG for full output
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("app.log")  # Optional: log to a file
    ]
)


# === CONFIG ===
lockfile_path = "/export/home/acs/stud/h/horia_andrei.moraru/tmp/sd_image_gen.lock"
REMOTE_OUTPUT_DIR = "/export/home/acs/stud/h/horia_andrei.moraru/generated_imgs_poc"
LOCAL_OUTPUT_DIR = "/home/mbadea/generated_images"
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

REMOTE_SCRIPT_SD = "/export/home/acs/stud/h/horia_andrei.moraru/generate_diffusion_images.py"
REMOTE_SCRIPT_GAN = "/export/home/acs/stud/h/horia_andrei.moraru/generate_gan_images.py"

SD_SANDBOX = "/export/home/acs/stud/h/horia_andrei.moraru/dpdm/stable-diffusion2-1/sd-lora-sandbox"
GAN_SANDBOX = "/export/home/acs/stud/h/horia_andrei.moraru/stylegan2/stylegan2-cuda-toolkit-sandbox"

SSH_HOST = "fep.grid.pub.ro"
SSH_USER = "horia_andrei.moraru"

# === REQUEST BODY ===
class ImageGenerationRequest(BaseModel):
    organ: str
    disease: str
    model_type: str  # either "gan" or "sd"

@app.post("/generate-image")
def generate_image(req: ImageGenerationRequest):
    organ = req.organ.strip()
    disease = req.disease.strip()
    model_type = req.model_type.lower().strip()

    if model_type not in {"gan", "sd"}:
        raise HTTPException(status_code=400, detail="model_type must be 'gan' or 'sd'")

    script_path = REMOTE_SCRIPT_GAN if model_type == "gan" else REMOTE_SCRIPT_SD
    sandbox_path = GAN_SANDBOX if model_type == "gan" else SD_SANDBOX

    with SshHandler(host=SSH_HOST, user=SSH_USER) as ssh:
        job_id = uuid.uuid4().hex[:6]
        wrapped_command = (
            f"flock -n 9 || exit 1; "
            f"srun -p dgxa100 "
            f"-A student "
            f"--gres=gpu:1 "
            f"--cpus-per-task=5 "
            f"--mem-per-cpu=16G "
            f"--time=00:30:00 "
            f"--job-name=gen-{job_id} "
            f"--export=ALL,GPU_FORCE_MODE_CLOCKS=0 "
            f"apptainer exec --nv {sandbox_path} "
            f"python3 {script_path} --organ \"{organ}\" --disease \"{disease}\""
        )

        command = f"bash -c '({wrapped_command}) 9>{lockfile_path}'"

        output, _ = ssh.run_command(command)
        ls_output, _ = ssh.run_command(f"ls -t {REMOTE_OUTPUT_DIR}/*.png | head -n 1")

        remote_path = ls_output.strip()
        logging.info(f"[BUGFIX] Extracted latest image via ls: {remote_path}")
        logging.info(f"Remote path: {remote_path}")
        logging.info(f"Remote path returned: '{remote_path}'")
        filename = os.path.basename(remote_path)
        local_path = os.path.join(LOCAL_OUTPUT_DIR, filename)

        ssh.download_file(remote_path, local_path)

    return {"download_url": f"/download-image?file={filename}"}

@app.get("/download-image")
def download_image(file: str = Query(...)):
    local_path = os.path.join(LOCAL_OUTPUT_DIR, file)
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(local_path, filename=file, media_type="image/png")
