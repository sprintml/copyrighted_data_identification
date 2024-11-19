import matplotlib.pyplot as plt
import seaborn as sns

set_plt = lambda: plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 15,  # Set font size to 11pt
        "axes.labelsize": 15,  # -> axis labels
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
        "text.usetex": False,
        "pgf.rcfonts": False,
    }
)

OURS = "CDI (Ours)"

ATTACKS = [
    "denoising_loss",
    "secmi_stat",
    "pia",
    "pian",
    "gradient_masking",
    "multiple_loss",
    "noise_optim",
    "cdi",
]

ATTACKS_NAME_MAPPING = {
    "denoising_loss": "Denoising Loss",
    "secmi_stat": "SecMI$_{stat}$",
    "pia": "PIA",
    "pian": "PIAN",
    "gradient_masking": "Gradient Masking",
    "multiple_loss": "Multiple Loss",
    "noise_optim": "Noise Optimization",
    "cdi": OURS,
}

ATTACKS_COLORS = {
    attack: color
    for attack, color in zip(ATTACKS_NAME_MAPPING.values(), sns.color_palette("tab10"))
}

MIAS_CITATIONS = {
    "Denoising Loss": "~\citep{carlini2022membership}",
    "SecMI$_{stat}$": "~\citep{duan23bSecMI}",
    "PIA": "~\citep{kong2024an}",
    "PIAN": "~\citep{kong2024an}",
}

MODELS = [
    "ldm",
    "uvit",
    "dit",
    "uvit_512",
    "dit_512",
    "uvit_uncond",
    "uvit_t2i",
    "uvit_t2i_deep",
]

MODELS_NAME_MAPPING = {
    "ldm": "LDM256",
    "uvit": "U-ViT256",
    "dit": "DiT256",
    "uvit_512": "U-ViT512",
    "dit_512": "DiT512",
    "uvit_uncond": "U-ViT256-Uncond",
    "uvit_t2i": "U-ViT256-T2I",
    "uvit_t2i_deep": "U-ViT256-T2I-Deep",
}

MODELS_COLORS = {
    model: color
    for model, color in zip(MODELS_NAME_MAPPING.values(), sns.color_palette("tab10"))
}

MODELS_MARKERS = {
    model: marker
    for model, marker in zip(MODELS_NAME_MAPPING.values(), ["o"] * 5 + ["X"] * 3)
}

MODELS_ORDER = [value for value in MODELS_NAME_MAPPING.values()]

RUN_ID = "25k"
RESAMPLING_CNT = 1000
