"""Every fixed choice used in the paper, in one place.

Nothing in this file is tuned per setting or per backbone.
"""

# ---------------------------------------------------------------- label spaces
LABELS_17 = ["airplane", "bare-soil", "buildings", "cars", "chaparral", "court", "dock",
             "field", "grass", "mobile-home", "pavement", "sand", "sea", "ship", "tanks",
             "trees", "water"]

LABELS_DFC15 = ["impervious", "water", "clutter", "vegetation", "building", "tree",
                "boat", "car"]

# The six categories shared by AID, UC Merced and DFC15 (order matters: it is the
# column order of every shared6 feature file).
SHARED_6 = ["water", "grass", "buildings", "trees", "ship", "cars"]

# DFC15 names two of them differently.
SHARED_6_TO_DFC15 = {"water": "water", "grass": "vegetation", "buildings": "building",
                     "trees": "tree", "ship": "boat", "cars": "car"}

# ---------------------------------------------------------------- prompts
# Each anchor is the renormalized mean of the three normalized template embeddings.
# Category names are used verbatim, with hyphens replaced by spaces.
PROMPTS = ["an aerial photograph containing {}",
           "a satellite image of {}",
           "a remote sensing image showing {}"]

# ---------------------------------------------------------------- transfers
# name -> (source dataset, target dataset, label space)
SETTINGS = {"AID->UCM":   ("aid", "ucm", "native"),
            "UCM->AID":   ("ucm", "aid", "native"),
            "AID->DFC15": ("aid", "dfc15", "shared6"),
            "UCM->DFC15": ("ucm", "dfc15", "shared6")}
ORDER = list(SETTINGS)

# ---------------------------------------------------------------- backbones
# Feature-file prefix -> how the frozen encoder is obtained (see extraction/backbones.py).
BACKBONES = {
    "clip_vitb32_control": dict(kind="open_clip", model="ViT-B-32", pretrained="openai",
                                note="CLIP ViT-B/32, original OpenAI weights (headline)"),
    "remoteclip_vitb32": dict(kind="open_clip_hf", model="ViT-B-32",
                              repo="chendelong/RemoteCLIP",
                              filename="RemoteCLIP-ViT-B-32.pt",
                              note="RemoteCLIP ViT-B/32 (diagnostic; UC Merced contamination)"),
}
HEADLINE = "clip_vitb32_control"
DIAGNOSTIC = "remoteclip_vitb32"

# ---------------------------------------------------------------- protocol
SEEDS = list(range(20))          # split seeds 0..19
SOURCE_FRAC = 0.8                # head fitted on the first 80% of the source permutation
TARGET_FRAC = 0.8                # evaluation on the last 20% of the target permutation

# ---------------------------------------------------------------- CLAB
ALPHA = 0.30                     # blending weight
EPOCHS = 50                      # full-batch gradient-descent steps for the source head
LR = 0.1
WEIGHT_DECAY = 1e-4
INIT_SD = 0.01                   # W ~ N(0, 0.01^2), drawn from RandomState(split seed)
