from .nodes import EmptySDXLFlux2LatentImage, TargetTimeConditioning

NODE_CLASS_MAPPINGS = {
    "EmptySDXLFlux2LatentImage": EmptySDXLFlux2LatentImage,
    "TargetTimeConditioning": TargetTimeConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptySDXLFlux2LatentImage": "Empty SDXL Flux2 Latent",
    "TargetTimeConditioning": "Target Time Conditioning (TT)",
}
