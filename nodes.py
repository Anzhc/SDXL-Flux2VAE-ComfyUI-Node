import logging

import comfy.latent_formats
import comfy.model_management
import comfy.sd
import comfy.supported_models
import nodes
import torch

_PATCHED = False


class PackedLatentVAE:
    def __init__(self, inner, packed_channels, spatial_factor=2):
        self._inner = inner
        self.packed_latent_channels = packed_channels
        self.packed_latent_spatial_factor = spatial_factor

    def set_packed_latents(self, packed_channels, spatial_factor=2):
        self.packed_latent_channels = packed_channels
        self.packed_latent_spatial_factor = spatial_factor

    def _to_vae_latent(self, latent):
        packed_channels = self.packed_latent_channels
        sf = self.packed_latent_spatial_factor
        target_channels = getattr(self._inner, "latent_channels", latent.shape[1])
        if packed_channels is None and latent.shape[1] * (sf ** 2) == target_channels:
            packed_channels = latent.shape[1]
        if packed_channels is not None and latent.shape[1] == packed_channels:
            if packed_channels * (sf ** 2) == target_channels and latent.ndim >= 4:
                h = latent.shape[-2]
                w = latent.shape[-1]
                if h % sf != 0 or w % sf != 0:
                    pad_h = (sf - (h % sf)) % sf
                    pad_w = (sf - (w % sf)) % sf
                    latent = torch.nn.functional.pad(latent, (0, pad_w, 0, pad_h))
                    h = latent.shape[-2]
                    w = latent.shape[-1]
                latent = latent.reshape(latent.shape[0], packed_channels, h // sf, sf, w // sf, sf)
                latent = latent.permute(0, 1, 3, 5, 2, 4).reshape(latent.shape[0], target_channels, h // sf, w // sf)
        return latent

    def _from_vae_latent(self, latent):
        packed_channels = self.packed_latent_channels
        sf = self.packed_latent_spatial_factor
        target_channels = getattr(self._inner, "latent_channels", latent.shape[1])
        if packed_channels is None and target_channels == 128 and latent.shape[1] == target_channels:
            packed_channels = 32
        if packed_channels is not None and latent.shape[1] == target_channels:
            if packed_channels * (sf ** 2) == target_channels and latent.ndim >= 4:
                h = latent.shape[-2]
                w = latent.shape[-1]
                latent = latent.reshape(latent.shape[0], packed_channels, sf, sf, h, w)
                latent = latent.permute(0, 1, 4, 2, 5, 3).reshape(latent.shape[0], packed_channels, h * sf, w * sf)
        return latent

    def decode(self, samples_in, vae_options=None):
        if vae_options is None:
            vae_options = {}
        samples_in = self._to_vae_latent(samples_in)
        return self._inner.decode(samples_in, vae_options=vae_options)

    def decode_tiled(self, samples, **kwargs):
        samples = self._to_vae_latent(samples)
        return self._inner.decode_tiled(samples, **kwargs)

    def decode_tiled_(self, samples, *args, **kwargs):
        samples = self._to_vae_latent(samples)
        return self._inner.decode_tiled_(samples, *args, **kwargs)

    def decode_tiled_1d(self, samples, *args, **kwargs):
        samples = self._to_vae_latent(samples)
        return self._inner.decode_tiled_1d(samples, *args, **kwargs)

    def decode_tiled_3d(self, samples, *args, **kwargs):
        samples = self._to_vae_latent(samples)
        return self._inner.decode_tiled_3d(samples, *args, **kwargs)

    def encode(self, pixel_samples):
        return self._from_vae_latent(self._inner.encode(pixel_samples))

    def encode_tiled(self, pixel_samples, **kwargs):
        return self._from_vae_latent(self._inner.encode_tiled(pixel_samples, **kwargs))

    def encode_tiled_(self, pixel_samples, *args, **kwargs):
        return self._from_vae_latent(self._inner.encode_tiled_(pixel_samples, *args, **kwargs))

    def encode_tiled_1d(self, pixel_samples, *args, **kwargs):
        return self._from_vae_latent(self._inner.encode_tiled_1d(pixel_samples, *args, **kwargs))

    def encode_tiled_3d(self, pixel_samples, *args, **kwargs):
        return self._from_vae_latent(self._inner.encode_tiled_3d(pixel_samples, *args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _packed_latent_settings(vae):
    target_channels = getattr(vae, "latent_channels", None)
    if target_channels is None:
        return None

    first_stage = getattr(vae, "first_stage_model", None)
    bn = getattr(first_stage, "bn", None)
    ps = getattr(first_stage, "ps", None)
    if bn is not None and ps is not None and len(ps) == 2 and ps[0] == ps[1]:
        sf = ps[0]
        if target_channels % (sf ** 2) == 0:
            packed_channels = target_channels // (sf ** 2)
            return packed_channels, sf

    latent_dim = getattr(vae, "latent_dim", 2)
    downscale = getattr(vae, "downscale_ratio", None)
    upscale = getattr(vae, "upscale_ratio", None)
    if (
        target_channels == 128
        and latent_dim == 2
        and downscale in (8, 16)
        and upscale in (8, 16)
    ):
        return 32, 2

    return None


def _ensure_latent_format():
    if hasattr(comfy.latent_formats, "SDXL_Flux2"):
        return comfy.latent_formats.SDXL_Flux2
    if not hasattr(comfy.latent_formats, "Flux2"):
        raise RuntimeError("Flux2 latent format is not available in this ComfyUI build.")

    class SDXL_Flux2(comfy.latent_formats.Flux2):
        latent_channels = 32

        def __init__(self):
            super().__init__()
            self.latent_rgb_factors_reshape = None

    comfy.latent_formats.SDXL_Flux2 = SDXL_Flux2
    return SDXL_Flux2


def _ensure_model_class():
    if hasattr(comfy.supported_models, "SDXL_flux2"):
        return comfy.supported_models.SDXL_flux2

    latent_format_cls = _ensure_latent_format()

    class SDXL_flux2(comfy.supported_models.SDXL):
        unet_config = dict(comfy.supported_models.SDXL.unet_config, in_channels=32, out_channels=32)
        latent_format = latent_format_cls
        vae_key_prefix = ["vae.", "first_stage_model."]
        packed_vae_latent_channels = 32
        packed_vae_spatial_factor = 2

        def inpaint_model(self):
            return False

    comfy.supported_models.SDXL_flux2 = SDXL_flux2
    return SDXL_flux2


def _ensure_model_order(model_class):
    models = getattr(comfy.supported_models, "models", None)
    if models is None:
        return

    sdxl_class = getattr(comfy.supported_models, "SDXL", None)
    if model_class not in models:
        if sdxl_class in models:
            models.insert(models.index(sdxl_class), model_class)
        else:
            models.append(model_class)
        return

    if sdxl_class in models and models.index(model_class) > models.index(sdxl_class):
        models.remove(model_class)
        models.insert(models.index(sdxl_class), model_class)


def _wrap_load_state_dict_guess_config():
    if getattr(comfy.sd.load_state_dict_guess_config, "_sdxl_flux2_wrapped", False):
        return

    original = comfy.sd.load_state_dict_guess_config

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        if out is None:
            return out

        model, clip, vae, clipvision = out
        if model is None or vae is None:
            return out

        model_config = getattr(getattr(model, "model", None), "model_config", None)
        packed_channels = getattr(model_config, "packed_vae_latent_channels", None)
        if packed_channels:
            spatial_factor = getattr(model_config, "packed_vae_spatial_factor", 2)
            if hasattr(vae, "_to_vae_latent") and hasattr(vae, "set_packed_latents"):
                vae.set_packed_latents(packed_channels, spatial_factor)
            else:
                vae = PackedLatentVAE(vae, packed_channels, spatial_factor)
        return (model, clip, vae, clipvision)

    wrapped._sdxl_flux2_wrapped = True
    comfy.sd.load_state_dict_guess_config = wrapped


def _wrap_vae_loader():
    loader_class = getattr(nodes, "VAELoader", None)
    if loader_class is None:
        return
    if getattr(loader_class.load_vae, "_sdxl_flux2_wrapped", False):
        return

    original = loader_class.load_vae

    def wrapped(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        if not out:
            return out

        (vae,) = out
        settings = _packed_latent_settings(vae)
        if settings is not None:
            packed_channels, spatial_factor = settings
            if hasattr(vae, "_to_vae_latent") and hasattr(vae, "set_packed_latents"):
                vae.set_packed_latents(packed_channels, spatial_factor)
            else:
                vae = PackedLatentVAE(vae, packed_channels, spatial_factor)
        return (vae,)

    wrapped._sdxl_flux2_wrapped = True
    loader_class.load_vae = wrapped


def _apply_patches():
    global _PATCHED
    if _PATCHED:
        return

    _PATCHED = True
    try:
        model_class = _ensure_model_class()
        _ensure_model_order(model_class)
        _wrap_load_state_dict_guess_config()
        _wrap_vae_loader()
    except Exception:
        logging.exception("SDXL Flux2 support patch failed")


_apply_patches()


class EmptySDXLFlux2LatentImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros(
            [batch_size, 32, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return ({"samples": latent},)
