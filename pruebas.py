
import torch
import torchvision.transforms as transforms
from PIL import Image
from models_2 import DiT_models          # Tu script de modelos
from diffusion import create_diffusion   # Tu módulo de difusión existente
from diffusers.models import AutoencoderKL

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Recursos básicos ---
    model = DiT_models["DiT-S/4"](input_size=32).to(device)
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    model.eval()

    # --- Simula un lote de imágenes x ---
    B = 2
    H, W, C = 256, 256, 3
    x_rgb = torch.randn(B, C, H, W, device=device)

    with torch.no_grad():
        latents = vae.encode(x_rgb).latent_dist.sample() * 0.18215

    # --- Simula el batch de imágenes condicionales ---
    cond_rgb = torch.randn(B, C, H, W, device=device)

    # --- Simula timesteps ---
    t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)

    # --- Forward ---
    try:
        out = model(latents, t, cond_img=cond_rgb)
        print("✅ Forward OK — output shape:", out.shape)
    except Exception as e:
        print("❌ Error en forward:", e)

if __name__ == "__main__":
    main()
