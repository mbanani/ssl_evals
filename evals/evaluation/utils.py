from pathlib import Path

import clip
import torch
import torch.utils.data
import torchvision
from segment_anything import sam_model_registry
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_state_dict(state_dict, remove_prefix):
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith(remove_prefix):
            state_dict[k[len(remove_prefix) :]] = state_dict[k]
        del state_dict[k]
    return state_dict


def get_model(model_name, ckpt_name):
    if model_name in ["dino", "dinov2"]:
        model = torch.hub.load(f"facebookresearch/{model_name}", ckpt_name)
    elif model_name == "clip":
        clip_models = {
            "resnet50": "RN50",
            "vit_b32": "ViT-B/32",
            "vit_b16": "ViT-B/16",
            "vit_l14": "ViT-L/14",
        }
        model, _ = clip.load(clip_models[ckpt_name], device="cpu")
        model = model.visual
    elif model_name == "SAM_lowres":
        ckpt_path = f"evals/evaluation/baseline_weights/sam_{ckpt_name}.pth"
        sam = sam_model_registry[ckpt_name](checkpoint=ckpt_path)
        model = SAM_ViT(sam, low_res=True)
    elif model_name == "SAM":
        ckpt_path = f"evals/evaluation/baseline_weights/sam_{ckpt_name}.pth"
        sam = sam_model_registry[ckpt_name](checkpoint=ckpt_path)
        model = SAM_ViT(sam)
    elif model_name == "lgssl_checkpoints":
        ckpt_path = Path(__file__) / f"../../../data/checkpoints/{ckpt_name}.ckpt"
        ckpt_path = str(ckpt_path.resolve())

        state_dict = torch.load(ckpt_path, map_location="cpu")
        model = torchvision.models.__dict__["resnet50"]()
        _ = model.load_state_dict(state_dict, strict=False)
        model.fc = torch.nn.Identity()
    else:
        raise ValueError()

    model = model.eval().to(device)
    return model


def extract_features(
    model, loader, normalize=False, norm_stats=None, return_stats=False
):
    """
    Extract global average pooled visual features for linear probe evaluation.
    Args:
        model: Trained model with ``visual`` module for feature extraction.
        dataset laoder: Dataset loader to serve ``(image, label)`` tuples.
    """
    feature_label_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    for images, labels in tqdm(loader, desc="Extracting feats"):
        with torch.inference_mode():
            images = images.to(device)
            features = model(images)
            if type(features) is list:
                assert len(features) == 1
                features = features[0]
            feature_label_pairs.append((features.cpu(), labels))

    all_features = torch.cat([p[0] for p in feature_label_pairs], dim=0)
    all_labels = torch.cat([p[1] for p in feature_label_pairs], dim=0)

    if normalize:
        if norm_stats is None:
            feature_mean = all_features.mean(dim=0, keepdim=True)
            feature_std = all_features.std(dim=0, keepdim=True)
        else:
            feature_mean, feature_std = norm_stats

        all_features = (all_features - feature_mean) / feature_std

    if return_stats:
        return all_features, all_labels, (feature_mean, feature_std)
    else:
        return all_features, all_labels


class SAM_ViT(torch.nn.Module):
    """
    Sequential module of (Linear-BN-ReLU) layers as projection MLP on top of the
    visual backbone. BatchNorm in final layer is not followed by ReLU.
    """

    def __init__(self, sam, low_res=False):
        super().__init__()
        self.print_shapes = True

        self.low_res = low_res
        self.sam = sam.image_encoder

        if self.low_res:
            pos_embed = sam.image_encoder.pos_embed.data.permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(
                pos_embed, size=(14, 14), mode="bicubic"
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1)

            self.sam.pos_embed.data = pos_embed

    @torch.inference_mode()
    def forward(self, image):

        if self.low_res:
            feats = self.sam(image)
        else:
            image = torch.nn.functional.interpolate(
                image, size=(1024, 1024), mode="bilinear"
            )
            feats = [self.sam(image[i, None]) for i in range(image.shape[0])]
            feats = torch.cat(feats, dim=0)

        # out = feats.mean(dim=(2,3)
        out = torch.nn.functional.normalize(feats, dim=1).view(feats.shape[0], -1)

        if self.print_shapes:
            print(f"image: {image.shape} | feats: {feats.shape} | out: {out.shape}")
            self.print_shapes = False

        return out
