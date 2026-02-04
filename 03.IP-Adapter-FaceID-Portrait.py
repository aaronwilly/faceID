import os

# -------------------------------------------------------------------
# Set cache dirs FIRST (before any import that uses Hugging Face / disk).
# -------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_CACHE = os.path.join(PROJECT_DIR, "models_cache")
os.makedirs(MODELS_CACHE, exist_ok=True)

os.environ["HF_HOME"] = MODELS_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODELS_CACHE
os.environ["HF_HUB_CACHE"] = MODELS_CACHE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(MODELS_CACHE, "transformers")
os.environ["INSIGHTFACE_HOME"] = os.path.join(MODELS_CACHE, "insightface")

from datetime import datetime

import cv2
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from insightface.app import FaceAnalysis
from PIL import Image

from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID

app = FaceAnalysis(
    name="buffalo_l",
    root=os.environ["INSIGHTFACE_HOME"],
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
)
app.prepare(ctx_id=0, det_size=(640, 640))


images = [
    # "input/dilmurat_1.png", 
    "input/dilmurat_2.png", 
    "input/dilmurat_3.png", 
    "input/dilmurat_4.png", 
    "input/dilmurat_5.png"
]

faceid_embeds = []
for image in images:
    image = cv2.imread(image)
    faces = app.get(image)
    faceid_embeds.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))
faceid_embeds = torch.cat(faceid_embeds, dim=1)


base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_ckpt = os.path.join(MODELS_CACHE, "ip-adapter-faceid-portrait_sd15.bin")
if not os.path.isfile(ip_ckpt):
    ip_ckpt = os.path.join(PROJECT_DIR, "models", "ip-adapter-faceid-portrait_sd15.bin")
if not os.path.isfile(ip_ckpt):
    raise FileNotFoundError("IP-Adapter Portrait checkpoint not found. Put ip-adapter-faceid-portrait_sd15.bin in models_cache/ or models/")
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(
    vae_model_path,
    cache_dir=MODELS_CACHE,
    local_files_only=True,
).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None,
    cache_dir=MODELS_CACHE,
    local_files_only=True,
)


# load ip-adapter
ip_model = IPAdapterFaceID(pipe, ip_ckpt, device, num_tokens=16, n_cond=5)

# generate image
prompt = "photo of a woman in red dress in a garden"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

images = ip_model.generate(
    prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=4, width=512, height=512, num_inference_steps=30, seed=2023
)

# Save generated images to results folder
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for i, img in enumerate(images):
    path = os.path.join(results_dir, f"faceid_{timestamp}_{i}.png")
    img.save(path)
    print(f"Saved {path}")
