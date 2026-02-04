import os

# -------------------------------------------------------------------
# Set cache dirs FIRST (before any import that uses Hugging Face / disk).
# Use existing models_cache so nothing is downloaded to C:\
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
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

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

base_model_path = "SG161222/RealVisXL_V3.0"
ip_ckpt = os.path.join(MODELS_CACHE, "ip-adapter-faceid_sdxl.bin")
if not os.path.isfile(ip_ckpt):
    ip_ckpt = os.path.join(PROJECT_DIR, "models", "ip-adapter-faceid_sdxl.bin")
if not os.path.isfile(ip_ckpt):
    raise FileNotFoundError("IP-Adapter SDXL checkpoint not found. Put ip-adapter-faceid_sdxl.bin in models_cache/ or models/")
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
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    add_watermarker=False,
    cache_dir=MODELS_CACHE,
    local_files_only=True,
)

# load ip-adapter
ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

# Generate 9 images (one per prompt)
negative_prompt = "blurry, low resolution, bad anatomy, distorted face, extra fingers, deformed hands, cross eye, oversharpened, jpeg artifacts, duplicate face, worst quality, low quality"

prompts = [
    "A closeup shot of a beautiful Asian teenage girl in a white dress wearing small silver earrings in the garden, under the soft morning light",
    "professional editorial portrait of a woman, natural window lighting, wearing a soft striped blouse, minimal makeup, studio gray background, 85mm lens photography, ultra detailed skin texture, sharp focus, fashion magazine quality",
    "soft watercolor portrait of a woman, pastel color wash background, gentle brush strokes, light flowing fabric, dreamy atmosphere, fine art watercolor painting, subtle color blending, minimalist composition",
    "high fashion portrait of a woman, patterned blouse, confident pose, studio softbox lighting, sharp editorial photography, vogue magazine style, rich color grading, 50mm lens, high detail",
    "artistic watercolor fashion portrait, colorful abstract background, fluid fabric, soft paint diffusion, expressive brush texture, modern fashion illustration, elegant pastel palette",
    "dramatic beauty portrait of a woman, bold red lipstick, structured blouse, cinematic lighting, strong contrast shadows, studio backdrop, high resolution fashion photography, crisp focus, detailed skin",
    "watercolor fine art portrait, deep color accents, soft paint splashes, moody artistic atmosphere, elegant posture, flowing garments, gallery-quality watercolor rendering",
    "minimal editorial portrait of a woman, soft plaid shirt, natural soft lighting, clean neutral background, modern fashion photography, delicate color tones, realistic skin texture, shallow depth of field",
    "soft pastel watercolor portrait, delicate fabric folds, gentle color gradients, light abstract background, fine brush detailing, elegant minimal aesthetic, fine art painting style",
]

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for i, prompt in enumerate(prompts):
    images = ip_model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        faceid_embeds=faceid_embeds,
        num_samples=1,
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=2023 + i,
    )
    path = os.path.join(results_dir, f"faceid_{timestamp}_{i:02d}.png")
    images[0].save(path)
    print(f"Saved {path}")
