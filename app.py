import os

# -------------------------------------------------------------------
# Set cache dirs FIRST (before any import that uses Hugging Face / disk).
# Uses existing models_cache so nothing is downloaded to C:\
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
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
# IP-Adapter: prefer models_cache, fallback to models/ (where your .bin lives)
ip_ckpt = os.path.join(MODELS_CACHE, "ip-adapter-faceid_sd15.bin")
if not os.path.isfile(ip_ckpt):
    ip_ckpt = os.path.join(PROJECT_DIR, "models", "ip-adapter-faceid_sd15.bin")
if not os.path.isfile(ip_ckpt):
    raise FileNotFoundError("IP-Adapter checkpoint not found. Put ip-adapter-faceid_sd15.bin in models_cache/ or models/")
device = "cuda"

app = FaceAnalysis(
    name="buffalo_l",
    root=os.environ["INSIGHTFACE_HOME"],
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
)
app.prepare(ctx_id=0, det_size=(640, 640))

image = cv2.imread("tailor.png")

if image is None:
    raise ValueError("Image not found or failed to load")

faces = app.get(image)

if len(faces) == 0:
    raise ValueError("No face detected in image")

faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

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
).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None,
    cache_dir=MODELS_CACHE,
)

# load ip-adapter
ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)

negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

# Generate 7 images (one per prompt)
prompts = [
    "portrait of a young woman standing outdoors in a garden, soft natural light, green foliage background, shallow depth of field, summer casual outfit, white blouse, realistic skin texture, 85mm lens, photorealistic, high detail, soft shadows",
    "portrait of a young woman wearing a modern hoodie, urban background, soft sunset lighting, natural makeup, candid photography style, 50mm lens, realistic skin detail, cinematic color grading",
    "epic female knight in golden armor, dramatic sunset lighting, medieval city background, cinematic fantasy style, ultra detailed armor reflections, sharp focus, volumetric light, epic concept art",
    "young woman wearing traditional japanese kimono, temple background, wooden architecture, soft warm sunlight, detailed fabric patterns, peaceful atmosphere, photorealistic, shallow depth of field",
    "portrait painting of a young woman, classical oil painting style, brush strokes visible, renaissance lighting, rich color tones, fine art museum quality",
    "artistic watercolor portrait of a young woman, paint splashes, abstract background, soft pastel tones, mixed media illustration, expressive brush strokes, high detail artistic rendering",
    "black and white portrait photography, studio lighting, minimal background, high contrast, sharp focus, classic monochrome film look",
]

# Save generated images to results folder
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for i, prompt in enumerate(prompts):
    images = ip_model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        faceid_embeds=faceid_embeds,
        num_samples=1,
        width=512,
        height=768,
        num_inference_steps=30,
        seed=2023 + i,
    )
    img = images[0]
    path = os.path.join(results_dir, f"faceid_{timestamp}_{i:02d}.png")
    img.save(path)
    print(f"Saved {path}")