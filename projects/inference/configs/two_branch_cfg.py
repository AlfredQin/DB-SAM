two_branch_cfg = {
    "AdaptedSam": {
        "image_encoder": {
            "AdaptedViTEncoder": {
                "img_size": 1024,
                "patch_size": 16,
                "in_chans": 3,
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "mlp_ratio": 4.0,
                "qkv_bias": True,
                "use_rel_pos": True,
                "global_attn_indexes": [2, 5, 8, 11],
                "window_size": 14,
                "out_chans": 256,
                "pretrained_sam_encoder_weights": None,
                "output_ms_feats": False,
                "finetune_neck": True,
                "adapter": {
                    "MultiStagesTwoBranchAdapter": {
                        "spatial_prior_module": {
                            "LFIP3": {
                                "in_planes": 3, "embed_dim": 768, "ms_feats_levels_index": (1, 2, 3, 4),
                            }
                        },
                        "vit_embed_dim": 768,
                        "ms_feats_levels_index": (1, 2, 3, 4),
                        "interaction_indexes": [[0, 2], [3, 5], [6, 8], [9, 11]],
                        "using_shallow_feats": True,
                        "attn_drop": 0.1,
                        "proj_drop": 0.1,
                        "drop_path_attn": 0.4,
                        "drop_path_ffn": 0.4,
                        "expand_ratio": 4,
                        "vit_conv_position": "before",
                        "aggregation": {"DANE": {"dim": 768, "reduction": 12}},
                    }
                }
            }
        },
        "prompt_encoder": {
            "SAMPromptEncoder": {
                "embed_dim": 256,
                "image_embedding_size": (64, 64),
                "input_image_size": (1024, 1024),
                "mask_in_chans": 16,
            }
        },
        "mask_decoder": {
            "SAMMaskDecoder": {
                "transformer_dim": 256,
                "transformer": {
                    "TwoWayTransformer": {
                        "depth": 2,
                        "embedding_dim": 256,
                        "mlp_dim": 2048,
                        "num_heads": 8,
                    }
                },
                "num_multimask_outputs": 3,
                "iou_head_depth": 3,
                "iou_head_hidden_dim": 256,
                "pretrained_sam_decoder_weights": None
            }
        },
        "pixel_mean": [123.675, 116.28, 103.53],
        "pixel_std": [58.395, 57.12, 57.375]
    }
}