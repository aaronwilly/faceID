# FaceID: InsightFace + IP-Adapter FaceID

Extract face embeddings with InsightFace and generate images conditioned on that face using IP-Adapter FaceID (Stable Diffusion 1.5).

## Setup

### 1. Create environment and install dependencies

```bash
cd faceID
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
pip install git+https://github.com/tencent-ailab/IP-Adapter.git
```

- **InsightFace** downloads its `buffalo_l` model on first run (from Hugging Face).
- **InsightFace + GPU**: The app auto-detects CUDA. If you see an error about `cudnn64_9.dll` missing, InsightFace will fall back to CPU. To use GPU, install [cuDNN 9](https://developer.nvidia.com/cudnn) and add its `bin` (or `lib`) folder to your system PATH so `cudnn64_9.dll` is found. Alternatively use CPU (check **Use CPU** in the app or `--cpu` in CLI).

### 2. Download IP-Adapter FaceID weights

Create a `models` folder and download the SD1.5 checkpoints from [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID):

- **Full adapter (single file):**  
  `ip-adapter-faceid_sd15.bin` → put in `models/`

- **Separate IP-Adapter + LoRA:**  
  `ip-adapter-faceid_sd15.bin` and `ip-adapter-faceid_sd15_lora.safetensors` → put in `models/`

Example with Hugging Face CLI:

```bash
mkdir models
huggingface-cli download h94/IP-Adapter-FaceID ip-adapter-faceid_sd15.bin --local-dir models
huggingface-cli download h94/IP-Adapter-FaceID ip-adapter-faceid_sd15_lora.safetensors --local-dir models
```

Or download the files manually from the repo and place them in `models/`.

Optional: set a custom models path:

```bash
set FACEID_MODELS=D:\path\to\models
```

### 3. Base model and VAE

The scripts use:

- Base: `SG161222/Realistic_Vision_V4.0_noVAE` (downloaded automatically by diffusers)
- VAE: `stabilityai/sd-vae-ft-mse` (downloaded automatically)

No extra download needed if you have Hugging Face cache / login.

## Gradio UI

Run the web interface:

```bash
python app.py
```

Then open the URL shown in the terminal (e.g. http://127.0.0.1:7860). Upload a face image, set your prompt and options (number of images, size, seed, etc.), optionally enable **Use IP-Adapter + LoRA**, and click **Generate**. The pipeline loads on first use (full adapter or LoRA variant). Use **Use CPU** if you don’t have a GPU.

## Usage (CLI)

### Step 1: Extract face embedding

From a photo of a face (e.g. `person.jpg`):

```bash
python extract_face_embedding.py person.jpg -o person_faceid.pt
```

- Output: `person_faceid.pt` (or path given by `-o`).
- Use `--cpu` if you don’t have a CUDA-capable GPU for InsightFace.

### Step 2: Generate images

**Option A – Full IP-Adapter (single .bin):**

```bash
python generate_faceid.py person_faceid.pt -p "photo of a woman in red dress in a garden" -o outputs
```

**Option B – Separate IP-Adapter + LoRA:**

```bash
python generate_faceid_lora.py person_faceid.pt -p "photo of a woman in red dress in a garden" -o outputs
```

Common options for both:

- `-p / --prompt` – text prompt  
- `-n / --negative` – negative prompt  
- `--num-samples` – number of images (default 4)  
- `--width`, `--height` – resolution (default 512x768)  
- `--steps` – inference steps (default 30)  
- `--seed` – random seed  
- `-o / --output-dir` – where to save PNGs  
- `--cpu` – use CPU for diffusion (slow)

## Scripts overview

| Script | Purpose |
|--------|--------|
| `app.py` | **Gradio UI**: upload face, set prompt, generate images in the browser |
| `extract_face_embedding.py` | InsightFace: image → face embedding `.pt` |
| `generate_faceid.py` | Generate with full IP-Adapter FaceID (`ip_adapter_faceid`) |
| `generate_faceid_lora.py` | Generate with IP-Adapter + LoRA (`ip_adapter_faceid_separate`) |

## Requirements

- Python 3.10+
- ~8GB+ VRAM recommended for generation (or use `--cpu` for CPU)
- Disk space for base model, VAE, and IP-Adapter weights

## License / use

InsightFace and IP-Adapter-FaceID are for **non-commercial / research** use. See [IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) and InsightFace terms.
