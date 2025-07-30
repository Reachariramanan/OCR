# app.py

import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

# ---------------------------
#  Model & Helper Definitions
# ---------------------------

class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(2, embed_dim, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        y = torch.linspace(-1.0, 1.0, H, device=device)
        xg = torch.linspace(-1.0, 1.0, W, device=device)
        yg = y.view(1,1,H,1).expand(B,1,H,W)
        xg = xg.view(1,1,1,W).expand(B,1,H,W)
        coords = torch.cat([xg, yg], dim=1)
        return x + self.proj(coords)

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.double_conv(x)

class UpConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
    def forward(self, x): return self.up(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g:int, F_l:int, F_int:int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        mask = self.psi(psi)
        return x * mask

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, enc_channels=(128,64,32,16)):
        super().__init__()
        assert len(enc_channels)==4
        # encoder
        self.in_conv     = ConvBlock(in_channels, enc_channels[0])
        self.pos0        = PositionalEncoding2D(enc_channels[0])
        self.pool        = nn.MaxPool2d(2)
        self.enc1        = ConvBlock(enc_channels[0], enc_channels[1])
        self.pos1        = PositionalEncoding2D(enc_channels[1])
        self.enc2        = ConvBlock(enc_channels[1], enc_channels[2])
        self.pos2        = PositionalEncoding2D(enc_channels[2])
        self.enc3        = ConvBlock(enc_channels[2], enc_channels[3])
        self.pos3        = PositionalEncoding2D(enc_channels[3])
        # decoder + attention
        self.up_stage3   = UpConv(enc_channels[3], enc_channels[2])
        self.att_stage3  = AttentionBlock(enc_channels[2], enc_channels[2], enc_channels[2]//2)
        self.conv_stage3 = ConvBlock(enc_channels[2]*2, enc_channels[2])
        self.pos_d3      = PositionalEncoding2D(enc_channels[2])

        self.up_stage2   = UpConv(enc_channels[2], enc_channels[1])
        self.att_stage2  = AttentionBlock(enc_channels[1], enc_channels[1], enc_channels[1]//2)
        self.conv_stage2 = ConvBlock(enc_channels[1]*2, enc_channels[1])
        self.pos_d2      = PositionalEncoding2D(enc_channels[1])

        self.up_stage1   = UpConv(enc_channels[1], enc_channels[0])
        self.att_stage1  = AttentionBlock(enc_channels[0], enc_channels[0], enc_channels[0]//2)
        self.conv_stage1 = ConvBlock(enc_channels[0]*2, enc_channels[0])
        self.pos_d1      = PositionalEncoding2D(enc_channels[0])

        self.out_conv    = nn.Sequential(nn.Conv2d(enc_channels[0], out_channels, 1), nn.Sigmoid())

    def forward(self, x):
        x0 = self.pos0(self.in_conv(x))
        x1 = self.pos1(self.enc1(self.pool(x0)))
        x2 = self.pos2(self.enc2(self.pool(x1)))
        x3 = self.pos3(self.enc3(self.pool(x2)))

        d3 = self.up_stage3(x3)
        x2a = self.att_stage3(x2, d3)
        d3 = self.pos_d3(self.conv_stage3(torch.cat([d3, x2a],dim=1)))

        d2 = self.up_stage2(d3)
        x1a = self.att_stage2(x1, d2)
        d2 = self.pos_d2(self.conv_stage2(torch.cat([d2, x1a],dim=1)))

        d1 = self.up_stage1(d2)
        x0a = self.att_stage1(x0, d1)
        d1 = self.pos_d1(self.conv_stage1(torch.cat([d1, x0a],dim=1)))

        return self.out_conv(d1)

# ---------------------------------------------------
#  Sliding-window inference
# ---------------------------------------------------

def predict_full_image_sliding_window(model, img, patch_size=(512,512), overlap=96, device="cpu"):
    transform = transforms.ToTensor()
    w, h = img.size
    pw, ph = patch_size
    sw, sh = pw-overlap, ph-overlap
    nx = max(1, math.ceil((w - pw + sw)/sw)) if w>pw else 1
    ny = max(1, math.ceil((h - ph + sh)/sh)) if h>ph else 1

    out = torch.zeros(3, h, w, device=device)
    cnt = torch.zeros(1, h, w, device=device)
    model.eval()
    with torch.no_grad():
        for iy in range(ny):
            for ix in range(nx):
                x0 = min(ix*sw, w-pw)
                y0 = min(iy*sh, h-ph)
                patch = img.crop((x0,y0, x0+pw, y0+ph))
                t = transform(patch).unsqueeze(0).to(device)
                pred = model(t)[0]
                out[:, y0:y0+ph, x0:x0+pw] += pred
                cnt[:,y0:y0+ph, x0:x0+pw] += 1

    out = (out/cnt.clamp(min=1)).clamp(0,1).cpu()
    return transforms.ToPILImage()(out)

# -----------------------
#  Load model & checkpoint
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionUNet(in_channels=3, out_channels=3)

# strip off DataParallel 'module.' prefixes
ckpt = torch.load("denoising_unet_patch_checkpoint.pth", map_location=device)
sd  = ckpt.get("model_state_dict", ckpt)
new_sd = OrderedDict()
for k,v in sd.items():
    nk = k[7:] if k.startswith("module.") else k
    new_sd[nk] = v

model.load_state_dict(new_sd)
model.to(device)

# -----------------------
#  Gradio inference fn
# -----------------------

def denoise_image(img: Image.Image) -> Image.Image:
    return predict_full_image_sliding_window(
        model, img,
        patch_size=(512,512),
        overlap=96,
        device=device
    )

# -----------------------
#  Launch Gradio app
# -----------------------

if __name__ == "__main__":
    gr.Interface(
        fn=denoise_image,
        inputs=gr.Image(type="pil", label="Noisy Input Image"),
        outputs=gr.Image(type="pil", label="Denoised Output Image"),
        title="Document Denoising with Attention U‑Net",
        description=(
            "Upload a noisy document image and get a denoised result.\n"
            "– Sliding window: 512×512 patches, 96px overlap.\n"
            "– Make sure your checkpoint is named "
            "`denoising_unet_patch_checkpoint.pth` in this folder."
        ),
        allow_flagging="never"
    ).launch()
