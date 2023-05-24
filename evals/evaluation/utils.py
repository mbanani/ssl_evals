from pathlib import Path

import clip
import torch
import torch.utils.data
import torchvision
from segment_anything import sam_model_registry
from segment_anything.modeling import ImageEncoderViT
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
    elif ckpt_name in ["random", "IMAGENET1K_V1", "IMAGENET1K_V2"]:
        ckpt_name = None if ckpt_name == "random" else ckpt_name
        model = torchvision.models.__dict__[model_name](weights=ckpt_name)
        model.fc = torch.nn.Identity()
    elif model_name == "clip":
        clip_models = {
            "resnet50": "RN50",
            "vit_b32": "ViT-B/32",
            "vit_b16": "ViT-B/16",
            "vit_l14": "ViT-L/14",
        }
        model, _ = clip.load(clip_models[ckpt_name], device="cpu")
        model = model.visual
    elif model_name == "dino_gap":
        model = DINOGAP("dino", ckpt_name)
    elif model_name == "dinov2_gap":
        model = DINOGAP("dinov2", ckpt_name)
    elif model_name == "clip_gap":
        clip_models = {
            "vit_b32": "ViT-B/32",
            "vit_b16": "ViT-B/16",
            "vit_l14": "ViT-L/14",
        }
        model, _ = clip.load(clip_models[ckpt_name], device="cpu")
        model = CLIPViT_GAP(model.visual)
    elif model_name == "mae_gap":
        mae_models = {
            "vit_b": "facebook/vit-mae-base",
            "vit_l": "facebook/vit-mae-large",
        }
        model = MAEViT(mae_models[ckpt_name], output="gap")
    elif model_name == "deit_gap":
        model = DeITViT(output="gap")
    elif model_name == "SAM_rand":
        model = SAM_ViT(None, low_res=True, random=True)
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

    def __init__(self, sam, low_res=False, random=False):
        super().__init__()
        self.print_shapes = True
        self.low_res = low_res

        if random:
            self.sam = ImageEncoderViT()
        else:
            self.sam = sam.image_encoder

        if self.low_res:
            pos_embed = self.sam.pos_embed.data.permute(0, 3, 1, 2)
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

        out = feats.mean(dim=(2, 3))
        # out = torch.nn.functional.normalize(feats, dim=1).view(feats.shape[0], -1)

        if self.print_shapes:
            print(f"image: {image.shape} | feats: {feats.shape} | out: {out.shape}")
            self.print_shapes = False

        return out


class CLIPViT_GAP(torch.nn.Module):
    def __init__(self, clip_vit):
        super().__init__()
        self.input_resolution = clip_vit.input_resolution
        self.output_dim = clip_vit.output_dim
        self.conv1 = clip_vit.conv1

        self.class_embedding = clip_vit.class_embedding
        self.positional_embedding = clip_vit.positional_embedding
        self.ln_pre = clip_vit.ln_pre
        self.ln_post = clip_vit.ln_post
        self.proj = clip_vit.proj

        self.transformer = clip_vit.transformer

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x_z = torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + x_z, x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # only changed line -- changed from x[:, 0] to x[:, 1:].mean(dim=1)
        # x = self.ln_post(x[:, 1:, :]).mean(dim=1)
        x = self.ln_post(x[:, 1:, :].mean(dim=1))

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIPViT_NoCLS(torch.nn.Module):
    def __init__(self, clip_vit):
        super().__init__()
        self.input_resolution = clip_vit.input_resolution
        self.output_dim = clip_vit.output_dim
        self.conv1 = clip_vit.conv1

        self.class_embedding = clip_vit.class_embedding
        self.positional_embedding = clip_vit.positional_embedding
        self.ln_pre = clip_vit.ln_pre
        self.ln_post = clip_vit.ln_post
        self.proj = clip_vit.proj

        self.transformer = clip_vit.transformer

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = x + self.positional_embedding.to(x.dtype)[None, 1:, :]

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # only changed line -- changed from x[:, 0] to x[:, 1:].mean(dim=1)
        # x = self.ln_post(x[:, 1:, :].mean(dim=1))
        x = self.ln_post(x[:, :, :].mean(dim=1))

        if self.proj is not None:
            x = x @ self.proj

        return x


class MAEViT(torch.nn.Module):
    def __init__(self, model_name, output):
        super().__init__()
        assert output in ["gap", "cls"]
        self.output = output

        from transformers import ViTMAEForPreTraining

        vit_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        self.vit = vit_model.vit
        self.vit.embeddings.config.mask_ratio = 0.0

    def forward(self, x: torch.Tensor):
        out = self.vit(pixel_values=x).last_hidden_state

        if self.output == "cls":
            out = out[:, 0]
        elif self.output == "gap":
            out = out[:, 1:].mean(dim=1)
        else:
            raise ValueError()

        return out


class DeITViT(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        assert output in ["gap", "cls"]
        self.output = output

        from transformers import ViTForImageClassification

        self.vit = ViTForImageClassification.from_pretrained(
            "facebook/deit-base-patch16-224"
        ).vit

    def forward(self, x: torch.Tensor):
        out = self.vit(pixel_values=x).last_hidden_state

        if self.output == "cls":
            out = out[:, 0]
        elif self.output == "gap":
            out = out[:, 1:].mean(dim=1)
        else:
            raise ValueError()

        return out


class DINOGAP(torch.nn.Module):
    def __init__(self, model="dino", ckpt_name="vitb16"):
        super().__init__()
        assert ckpt_name in ["vitb16", "vitb14"]
        self.vit = torch.hub.load(f"facebookresearch/{model}", f"{model}_{ckpt_name}")

        self.forward = self.forward_v1 if model == "dino" else self.forward_v2

    def forward_v1(self, x: torch.Tensor):
        x = self.vit.prepare_tokens(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        return x[:, 1:].mean(dim=1)

    def forward_v2(self, x: torch.Tensor):
        ret = self.vit.forward_features(x)
        return ret["x_norm_patchtokens"].mean(dim=1)
