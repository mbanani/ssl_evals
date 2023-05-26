from __future__ import annotations

import einops as E
import torch
from torch import nn
from transformers import ViTForImageClassification, ViTMAEForPreTraining
from transformers.models.vit_mae.modeling_vit_mae import get_2d_sincos_pos_embed


class FrozenDINOViT(nn.Module):
    """
    Extension of OpenAI CLIP-ViT models that handles color normalization of
    input images, along with two customizations:

    1. Handle large image resolutions by adaptively resizing position embeddings.
    2. Controllable ``num_blocks`` to break the forward pass at a certain block
       (to train task-specific heads for e.g. segmentation).

    This module will always be frozen and no parts can be fine-tuned.
    """

    def __init__(self, model_name: str = "vitb16", input_resolution: int = 1024):
        super().__init__()
        self.input_resolution = input_resolution

        # create dino model
        checkpoint_name = f"dino_{model_name}"
        dino_vit = torch.hub.load("facebookresearch/dino", checkpoint_name)
        dino_vit = dino_vit.eval().to(torch.float32)
        self.patch_size = dino_vit.patch_embed.proj.kernel_size[0]
        self.embed_dim = dino_vit.embed_dim

        # set to vit and resize pos embed
        self.vit = dino_vit

        pos_embed = self.resize_pos_embed((64, 64))[None, :, :]
        self.vit.pos_embed = torch.nn.Parameter(pos_embed)

        # Extract pixel mean and std from `preprocess`. They will be RGB values
        # in `[0, 1]`. Convert them to `[0, 255]`.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        _mean = [val * 255 for val in mean]
        _std = [val * 255 for val in std]

        self.register_buffer("pixel_mean", torch.tensor(_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(_std).view(-1, 1, 1), False)

        # Freeze all parameters.
        for param in self.vit.parameters():
            param.requires_grad = False

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    def resize_pos_embed(self, new_hw: tuple[int, int]) -> torch.Tensor:
        """
        Resize CLIP's positional embedding for input resolution through bicubic
        interpolation.

        Args:
            pos_embed: CLIP's original positional embedding, a tensor of shape
                ``(1 + num_patches ** 2, embed_dim)``.
            new_hw: Target height and width of the tensor after interpolation.
                Pass height/width as ``(current_image_dim // patch_size)``

        Returns:
            Tensor of shape ``(1 + new_h * new_w, embed_dim)`` of resized embedding.
        """

        # Get original position embedding and extract ``[cls]`` token.
        pos_embed = self.vit.pos_embed[0]
        cls_embed, pos_embed = pos_embed[[0]], pos_embed[1:]
        orig_dim = int(pos_embed.shape[0] ** 0.5)

        # Resize (add fake batch dimension as per functional API).
        pos_embed = E.rearrange(pos_embed, "(h w) embed -> embed h w", h=orig_dim)
        pos_embed = nn.functional.interpolate(
            pos_embed[None, ...], size=new_hw, mode="bicubic", align_corners=False
        )
        pos_embed = E.rearrange(pos_embed[0], "embed h w -> (h w) embed")

        # Add embedding of ``[cls]`` token back after resizing.
        pos_embed = torch.cat([cls_embed, pos_embed], dim=0)
        return pos_embed

    def forward(self, batched_inputs: list[dict]) -> dict[str, torch.Tensor]:
        """
        Encode images through first few transformer blocks until ``num_blocks``.

        Args:
            batched_inputs: Batch of instances as a list of dictionaries. This
                module requires a key ``image`` in every dictionary that gives
                a tensor of shape ``(channels, height, width)``. Each image may
                have different height and width.

        Returns:
            Dictionary with two keys:
            1. cls_token: Tensor of shape ``(batch_size, embed_dim)`` giving
               ``[cls]`` token embeddings for images in batch.
            2. spatial: Tensor of shape ``(batch_size, embed_dim, spatial_dim,
               spatial_dim)`` giving spatial (patch) features, where spatial
               dimension is ``input_resolution // patch_size``.
        """

        image_key = "clip_image" if "clip_image" in batched_inputs[0] else "image"
        images = [inst[image_key].to(self.device) for inst in batched_inputs]

        # Collect features for every image in these lists.
        output_dict = {"cls_token": [], "spatial": []}

        # Process one image at a time to reduce peak GPU memory usage.
        for image in images:
            # Add fake batch dimension and normalize.
            image = (image[None, ...] - self.pixel_mean) / self.pixel_std

            # Pad image after normalization (this is equivalent to the original
            # image being padded with mean color pixels).
            padh = self.input_resolution - image.shape[-2]
            padw = self.input_resolution - image.shape[-1]
            image = nn.functional.pad(image, (0, padw, 0, padh))

            # pass it through
            x = self.vit.prepare_tokens(image)
            for blk in self.vit.blocks:
                x = blk(x)
            x = self.vit.norm(x)

            cls_token = x[:, 0]
            spatial = E.rearrange(x[:, 1:], "b (h w) c -> b c h w", h=64)
            spatial = spatial.contiguous()

            output_dict["cls_token"].append(cls_token)
            output_dict["spatial"].append(spatial)

        # Combine features along batch dimension.
        # shape: (batch_size, embed_dim, ...)
        output_dict["cls_token"] = torch.cat(output_dict["cls_token"], dim=0)
        output_dict["spatial"] = torch.cat(output_dict["spatial"], dim=0)
        return output_dict


class FrozenMAEViT(nn.Module):
    def __init__(self, input_resolution: int = 1024):
        super().__init__()
        self.input_resolution = input_resolution

        vit = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").vit

        # update the config
        vit.embeddings.config.mask_ratio = 0.0

        vit = vit.eval()
        vit.embeddings.image_size = (1024, 1024)
        vit.embeddings.patch_embeddings.image_size = (1024, 1024)

        self.patch_size = vit.config.patch_size
        self.embed_dim = vit.config.hidden_size

        # resize pos embedding
        # initialize (and freeze) position embeddings by sin-cos embedding
        vit.embeddings.position_embeddings = nn.Parameter(
            torch.zeros(1, 4096 + 1, self.embed_dim), requires_grad=False
        )

        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, 64, add_cls_token=True)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
        vit.embeddings.position_embeddings.data.copy_(pos_embed)

        # set to vit and resize pos embed
        self.vit = vit

        # Extract pixel mean and std from `preprocess`. They will be RGB values
        # in `[0, 1]`. Convert them to `[0, 255]`.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        _mean = [val * 255 for val in mean]
        _std = [val * 255 for val in std]

        self.register_buffer("pixel_mean", torch.tensor(_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(_std).view(-1, 1, 1), False)

        # Freeze all parameters.
        for param in self.vit.parameters():
            param.requires_grad = False

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    def embed_forward(self, embedder, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = embedder.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + embedder.position_embeddings[:, 1:, :]

        # append cls token
        cls_token = embedder.cls_token + embedder.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings

    def forward_mae(self, pixel_values):
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x
        # seq_length x seq_length]
        head_mask = self.vit.get_head_mask(None, self.vit.config.num_hidden_layers)

        embedding_output = self.embed_forward(self.vit.embeddings, pixel_values)

        encoder_outputs = self.vit.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=self.vit.config.output_attentions,
            output_hidden_states=self.vit.config.output_hidden_states,
            return_dict=self.vit.config.return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.vit.layernorm(sequence_output)

        return sequence_output

    def forward(self, batched_inputs: list[dict]) -> dict[str, torch.Tensor]:
        """
        Encode images through first few transformer blocks until ``num_blocks``.

        Args:
            batched_inputs: Batch of instances as a list of dictionaries. This
                module requires a key ``image`` in every dictionary that gives
                a tensor of shape ``(channels, height, width)``. Each image may
                have different height and width.

        Returns:
            Dictionary with two keys:
            1. cls_token: Tensor of shape ``(batch_size, embed_dim)`` giving
               ``[cls]`` token embeddings for images in batch.
            2. spatial: Tensor of shape ``(batch_size, embed_dim, spatial_dim,
               spatial_dim)`` giving spatial (patch) features, where spatial
               dimension is ``input_resolution // patch_size``.
        """

        image_key = "clip_image" if "clip_image" in batched_inputs[0] else "image"
        images = [inst[image_key].to(self.device) for inst in batched_inputs]

        # Collect features for every image in these lists.
        output_dict = {"cls_token": [], "spatial": []}

        # Process one image at a time to reduce peak GPU memory usage.
        with torch.inference_mode():
            for image in images:
                # Add fake batch dimension and normalize.
                image = (image[None, ...] - self.pixel_mean) / self.pixel_std

                # Pad image after normalization (this is equivalent to the original
                # image being padded with mean color pixels).
                padh = self.input_resolution - image.shape[-2]
                padw = self.input_resolution - image.shape[-1]
                image = nn.functional.pad(image, (0, padw, 0, padh))

                # pass it through
                hidden = self.forward_mae(image)

                cls_token = hidden[:, 0]

                # reorganize according to ids_restore -- kinda silly, but whatever
                # ids_restore = ids_restore.unsqueeze(2).repeat(1, 1, self.embed_dim)
                spatial = hidden[:, 1:]
                # spatial = torch.gather(spatial, 1, ids_restore)

                spatial = E.rearrange(spatial, "b (h w) c -> b c h w", h=64)
                spatial = spatial.contiguous()

                output_dict["cls_token"].append(cls_token)
                output_dict["spatial"].append(spatial)

        # Combine features along batch dimension.
        # shape: (batch_size, embed_dim, ...)
        output_dict["cls_token"] = torch.cat(output_dict["cls_token"], dim=0)
        output_dict["spatial"] = torch.cat(output_dict["spatial"], dim=0)
        return output_dict


class FrozenDeITViT(nn.Module):
    def __init__(self, input_resolution: int = 1024):
        super().__init__()
        self.input_resolution = input_resolution

        vit = ViTForImageClassification.from_pretrained(
            "facebook/deit-base-patch16-224"
        ).vit

        self.vit = vit.eval()

        self.patch_size = vit.config.patch_size
        self.embed_dim = vit.config.hidden_size

        # Extract pixel mean and std from `preprocess`. They will be RGB values
        # in `[0, 1]`. Convert them to `[0, 255]`.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        _mean = [val * 255 for val in mean]
        _std = [val * 255 for val in std]

        self.register_buffer("pixel_mean", torch.tensor(_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(_std).view(-1, 1, 1), False)

        # Freeze all parameters.
        for param in self.vit.parameters():
            param.requires_grad = False

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    def forward(self, batched_inputs: list[dict]) -> dict[str, torch.Tensor]:
        """
        Encode images through first few transformer blocks until ``num_blocks``.

        Args:
            batched_inputs: Batch of instances as a list of dictionaries. This
                module requires a key ``image`` in every dictionary that gives
                a tensor of shape ``(channels, height, width)``. Each image may
                have different height and width.

        Returns:
            Dictionary with two keys:
            1. cls_token: Tensor of shape ``(batch_size, embed_dim)`` giving
               ``[cls]`` token embeddings for images in batch.
            2. spatial: Tensor of shape ``(batch_size, embed_dim, spatial_dim,
               spatial_dim)`` giving spatial (patch) features, where spatial
               dimension is ``input_resolution // patch_size``.
        """

        image_key = "clip_image" if "clip_image" in batched_inputs[0] else "image"
        images = [inst[image_key].to(self.device) for inst in batched_inputs]

        # Collect features for every image in these lists.
        output_dict = {"cls_token": [], "spatial": []}

        # Process one image at a time to reduce peak GPU memory usage.
        with torch.inference_mode():
            for image in images:
                # Add fake batch dimension and normalize.
                image = (image[None, ...] - self.pixel_mean) / self.pixel_std

                # Pad image after normalization (this is equivalent to the original
                # image being padded with mean color pixels).
                padh = self.input_resolution - image.shape[-2]
                padw = self.input_resolution - image.shape[-1]
                image = nn.functional.pad(image, (0, padw, 0, padh))

                # pass it through
                hidden = self.vit(pixel_values=image, interpolate_pos_encoding=True)
                hidden = hidden.last_hidden_state

                cls_token = hidden[:, 0]

                # reorganize according to ids_restore -- kinda silly, but whatever
                # ids_restore = ids_restore.unsqueeze(2).repeat(1, 1, self.embed_dim)
                spatial = hidden[:, 1:]
                # spatial = torch.gather(spatial, 1, ids_restore)

                spatial = E.rearrange(spatial, "b (h w) c -> b c h w", h=64)
                spatial = spatial.contiguous()

                output_dict["cls_token"].append(cls_token)
                output_dict["spatial"].append(spatial)

        # Combine features along batch dimension.
        # shape: (batch_size, embed_dim, ...)
        output_dict["cls_token"] = torch.cat(output_dict["cls_token"], dim=0)
        output_dict["spatial"] = torch.cat(output_dict["spatial"], dim=0)
        return output_dict
