import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

from ..registry import register_adapted_sam
from super_gradients.common.decorators.factory_decorator import resolve_param

from src.models.encoders import EncoderFactory
from src.models.prompt_encoders import PromptEncoderFactory
from src.models.decoders import DecoderFactory


@register_adapted_sam()
class AdaptedSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    @resolve_param("mask_decoder", factory=DecoderFactory())
    @resolve_param("prompt_encoder", factory=PromptEncoderFactory())
    @resolve_param("image_encoder", factory=EncoderFactory())
    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.training_state = "training"

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, inputs):
        if self.training_state == "training":
            images, boxes = inputs['images'], inputs['boxes']   # images: list of (N, H, W, 3), boxes: list of (N, 4)
            images = images.permute(0, 3, 1, 2)
            batch_input = []
            for batch_idx in range(len(images)):
                dict_input = {}
                dict_input["image"] = images[batch_idx]
                dict_input["boxes"] = boxes[batch_idx:batch_idx + 1]
                dict_input["original_size"] = inputs["mask_shape"][batch_idx].tolist()
                batch_input.append(dict_input)
            outputs = self.forward_batch(batch_input, multimask_output=False)
            return torch.cat([output["low_res_logits"] for output in outputs], dim=0)
        else:
            images_list, boxes_list = inputs['images'], inputs['boxes']
            results = []
            for batch_id, (images, boxes) in enumerate(zip(images_list, boxes_list)):
                images = images.permute(0, 3, 1, 2)
                output_per_case = []
                for img_idx in range(len(images)):
                    dict_input = {}
                    dict_input["image"] = images[img_idx]
                    dict_input["boxes"] = boxes[img_idx:img_idx + 1]
                    dict_input["original_size"] = inputs["original_size"][batch_id].tolist()
                    outputs = self.forward_batch([dict_input], multimask_output=False)
                    output_per_case.append(outputs[0]["low_res_logits"].squeeze(1))
                outputs = torch.cat(output_per_case, dim=0)
                results.append(outputs)

            return results

    def forward_batch(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        encoder_output = self.image_encoder(input_images)
        image_embeddings = encoder_output["multi_scale_feats"]

        outputs = []
        for batch_id in range(len(batched_input)):
            image_record = batched_input[batch_id]
            if isinstance(image_embeddings, list):
                curr_multi_scale_embedding = [scale_embedding[batch_id].unsqueeze(0) for scale_embedding in image_embeddings]
            else:
                curr_multi_scale_embedding = image_embeddings[batch_id].unsqueeze(0)
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_multi_scale_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def initialize_param_groups(self, lr: float, training_params) -> list:
        sam_adapter_params = {
            'named_params': [(name, param) for name, param in self.image_encoder.named_parameters() if param.requires_grad],
            'lr': lr
        }
        prompt_encoder_params = {
            "named_params": [(name, param) for name, param in self.prompt_encoder.named_parameters() if param.requires_grad],
            "lr": lr,
        }
        mask_decoder_params = {
            'named_params': [(name, param) for name, param in self.mask_decoder.named_parameters() if param.requires_grad],
            'lr': lr
        }

        return [mask_decoder_params, sam_adapter_params, prompt_encoder_params]
