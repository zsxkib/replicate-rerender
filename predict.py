import os
import shutil
from enum import Enum

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from blendmodes.blend import BlendType, blendLayers
from PIL import Image
from pytorch_lightning import seed_everything
from safetensors.torch import load_file
from skimage import exposure

import src.import_util  # noqa: F401
from ControlNet.annotator.canny import CannyDetector
from ControlNet.annotator.hed import HEDdetector
from ControlNet.annotator.util import HWC3
from ControlNet.cldm.model import create_model, load_state_dict
from gmflow_module.gmflow.gmflow import GMFlow
from flow.flow_utils import get_warped_and_mask
from sd_model_cfg import model_dict
from src.config import RerenderConfig
from src.controller import AttentionControl
from src.ddim_v_hacked import DDIMVSampler
from src.img_util import find_flat_region, numpy2tensor
from src.video_util import (frame_to_video, get_fps, get_frame_count,
                            prepare_frames)

import huggingface_hub

repo_name = 'Anonymous-sub/Rerender'

# huggingface_hub.hf_hub_download(repo_name,
#                                 'pexels-koolshooters-7322716.mp4',
#                                 local_dir='videos')
# huggingface_hub.hf_hub_download(
#     repo_name,
#     'pexels-antoni-shkraba-8048492-540x960-25fps.mp4',
#     local_dir='videos')
# huggingface_hub.hf_hub_download(
#     repo_name,
#     'pexels-cottonbro-studio-6649832-960x506-25fps.mp4',
#     local_dir='videos')

files = [
    'pexels-koolshooters-7322716.mp4',
    'pexels-antoni-shkraba-8048492-540x960-25fps.mp4',
    'pexels-cottonbro-studio-6649832-960x506-25fps.mp4',
]

for file in files:
    if not os.path.exists(f'videos/{file}'):
        huggingface_hub.hf_hub_download(repo_name, file, local_dir='videos')

inversed_model_dict = dict()
for k, v in model_dict.items():
    inversed_model_dict[v] = k

to_tensor = T.PILToTensor()
blur = T.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProcessingState(Enum):
    NULL = 0
    FIRST_IMG = 1
    KEY_IMGS = 2


MAX_KEYFRAME = 300


class GlobalState:

    def __init__(self):
        self.sd_model = None
        self.ddim_v_sampler = None
        self.detector_type = None
        self.detector = None
        self.controller = None
        self.processing_state = ProcessingState.NULL
        flow_model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type='swin',
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        ).to(device)

        checkpoint = torch.load('models/gmflow_sintel-0c07dcb3.pth',
                                map_location=lambda storage, loc: storage)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        flow_model.load_state_dict(weights, strict=False)
        flow_model.eval()
        self.flow_model = flow_model

    def update_controller(self, inner_strength, mask_period, cross_period,
                          ada_period, warp_period):
        self.controller = AttentionControl(inner_strength, mask_period,
                                           cross_period, ada_period,
                                           warp_period)

    def update_sd_model(self, sd_model, control_type):
        if sd_model == self.sd_model:
            return
        self.sd_model = sd_model
        model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
        if control_type == 'HED':
            model.load_state_dict(
                load_state_dict(huggingface_hub.hf_hub_download(
                    'lllyasviel/ControlNet', './models/control_sd15_hed.pth'),
                                location=device))
        elif control_type == 'canny':
            model.load_state_dict(
                load_state_dict(huggingface_hub.hf_hub_download(
                    'lllyasviel/ControlNet', 'models/control_sd15_canny.pth'),
                                location=device))
        model.to(device)
        sd_model_path = model_dict[sd_model]
        if len(sd_model_path) > 0:
            model_ext = os.path.splitext(sd_model_path)[1]
            downloaded_model = huggingface_hub.hf_hub_download(
                repo_name, sd_model_path)
            if model_ext == '.safetensors':
                model.load_state_dict(load_file(downloaded_model),
                                      strict=False)
            elif model_ext == '.ckpt' or model_ext == '.pth':
                model.load_state_dict(
                    torch.load(downloaded_model)['state_dict'], strict=False)

        try:
            model.first_stage_model.load_state_dict(torch.load(
                huggingface_hub.hf_hub_download(
                    'stabilityai/sd-vae-ft-mse-original',
                    'vae-ft-mse-840000-ema-pruned.ckpt'))['state_dict'],
                                                    strict=False)
        except Exception:
            print('Warning: We suggest you download the fine-tuned VAE',
                  'otherwise the generation quality will be degraded')

        self.ddim_v_sampler = DDIMVSampler(model)

    def clear_sd_model(self):
        self.sd_model = None
        self.ddim_v_sampler = None
        if device == 'cuda':
            torch.cuda.empty_cache()

    def update_detector(self, control_type, canny_low=100, canny_high=200):
        if self.detector_type == control_type:
            return
        if control_type == 'HED':
            self.detector = HEDdetector()
        elif control_type == 'canny':
            canny_detector = CannyDetector()
            low_threshold = canny_low
            high_threshold = canny_high

            def apply_canny(x):
                return canny_detector(x, low_threshold, high_threshold)

            self.detector = apply_canny


global_state = GlobalState()
global_video_path = None
video_frame_count = None


def create_cfg(input_path, prompt, image_resolution, control_strength,
               color_preserve, left_crop, right_crop, top_crop, bottom_crop,
               control_type, low_threshold, high_threshold, ddim_steps, scale,
               seed, sd_model, a_prompt, n_prompt, interval, keyframe_count,
               x0_strength, use_constraints, cross_start, cross_end,
               style_update_freq, warp_start, warp_end, mask_start, mask_end,
               ada_start, ada_end, mask_strength, inner_strength,
               smooth_boundary):
    use_warp = 'shape-aware fusion' in use_constraints
    use_mask = 'pixel-aware fusion' in use_constraints
    use_ada = 'color-aware AdaIN' in use_constraints

    if not use_warp:
        warp_start = 1
        warp_end = 0

    if not use_mask:
        mask_start = 1
        mask_end = 0

    if not use_ada:
        ada_start = 1
        ada_end = 0

    input_name = os.path.split(input_path)[-1].split('.')[0]
    frame_count = 2 + keyframe_count * interval
    cfg = RerenderConfig()
    cfg.create_from_parameters(
        input_path,
        os.path.join('result', input_name, 'blend.mp4'),
        prompt,
        a_prompt=a_prompt,
        n_prompt=n_prompt,
        frame_count=frame_count,
        interval=interval,
        crop=[left_crop, right_crop, top_crop, bottom_crop],
        sd_model=sd_model,
        ddim_steps=ddim_steps,
        scale=scale,
        control_type=control_type,
        control_strength=control_strength,
        canny_low=low_threshold,
        canny_high=high_threshold,
        seed=seed,
        image_resolution=image_resolution,
        x0_strength=x0_strength,
        style_update_freq=style_update_freq,
        cross_period=(cross_start, cross_end),
        warp_period=(warp_start, warp_end),
        mask_period=(mask_start, mask_end),
        ada_period=(ada_start, ada_end),
        mask_strength=mask_strength,
        inner_strength=inner_strength,
        smooth_boundary=smooth_boundary,
        color_preserve=color_preserve)
    return cfg


def cfg_to_input(filename):

    cfg = RerenderConfig()
    cfg.create_from_path(filename)
    keyframe_count = (cfg.frame_count - 2) // cfg.interval
    use_constraints = [
        'shape-aware fusion', 'pixel-aware fusion', 'color-aware AdaIN'
    ]

    sd_model = inversed_model_dict.get(cfg.sd_model, 'Stable Diffusion 1.5')

    args = [
        cfg.input_path, cfg.prompt, cfg.image_resolution, cfg.control_strength,
        cfg.color_preserve, *cfg.crop, cfg.control_type, cfg.canny_low,
        cfg.canny_high, cfg.ddim_steps, cfg.scale, cfg.seed, sd_model,
        cfg.a_prompt, cfg.n_prompt, cfg.interval, keyframe_count,
        cfg.x0_strength, use_constraints, *cfg.cross_period,
        cfg.style_update_freq, *cfg.warp_period, *cfg.mask_period,
        *cfg.ada_period, cfg.mask_strength, cfg.inner_strength,
        cfg.smooth_boundary
    ]
    return args


def setup_color_correction(image):
    correction_target = cv2.cvtColor(np.asarray(image.copy()),
                                     cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, original_image):
    image = Image.fromarray(
        cv2.cvtColor(
            exposure.match_histograms(cv2.cvtColor(np.asarray(original_image),
                                                   cv2.COLOR_RGB2LAB),
                                      correction,
                                      channel_axis=2),
            cv2.COLOR_LAB2RGB).astype('uint8'))

    image = blendLayers(image, original_image, BlendType.LUMINOSITY)

    return image


@torch.no_grad()
def process(*args):
    first_frame = process1(*args)

    keypath = process2(*args)

    return first_frame, keypath


@torch.no_grad()
def process0(*args):
    global global_video_path
    global_video_path = args[0]
    return process(*args[1:])


@torch.no_grad()
def process1(*args):

    global global_video_path
    cfg = create_cfg(global_video_path, *args)
    global global_state
    global_state.update_sd_model(cfg.sd_model, cfg.control_type)
    global_state.update_controller(cfg.inner_strength, cfg.mask_period,
                                   cfg.cross_period, cfg.ada_period,
                                   cfg.warp_period)
    global_state.update_detector(cfg.control_type, cfg.canny_low,
                                 cfg.canny_high)
    global_state.processing_state = ProcessingState.FIRST_IMG

    prepare_frames(cfg.input_path, cfg.input_dir, cfg.image_resolution,
                   cfg.crop)

    ddim_v_sampler = global_state.ddim_v_sampler
    model = ddim_v_sampler.model
    detector = global_state.detector
    controller = global_state.controller
    model.control_scales = [cfg.control_strength] * 13
    model.to(device)

    num_samples = 1
    eta = 0.0
    imgs = sorted(os.listdir(cfg.input_dir))
    imgs = [os.path.join(cfg.input_dir, img) for img in imgs]

    model.cond_stage_model.device = device

    with torch.no_grad():
        frame = cv2.imread(imgs[0])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = HWC3(frame)
        H, W, C = img.shape

        img_ = numpy2tensor(img)

        def generate_first_img(img_, strength):
            encoder_posterior = model.encode_first_stage(img_.to(device))
            x0 = model.get_first_stage_encoding(encoder_posterior).detach()

            detected_map = detector(img)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(
                detected_map.copy()).float().to(device) / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            cond = {
                'c_concat': [control],
                'c_crossattn': [
                    model.get_learned_conditioning(
                        [cfg.prompt + ', ' + cfg.a_prompt] * num_samples)
                ]
            }
            un_cond = {
                'c_concat': [control],
                'c_crossattn':
                [model.get_learned_conditioning([cfg.n_prompt] * num_samples)]
            }
            shape = (4, H // 8, W // 8)

            controller.set_task('initfirst')
            seed_everything(cfg.seed)

            samples, _ = ddim_v_sampler.sample(
                cfg.ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=cfg.scale,
                unconditional_conditioning=un_cond,
                controller=controller,
                x0=x0,
                strength=strength)
            x_samples = model.decode_first_stage(samples)
            x_samples_np = (
                einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
                127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            return x_samples, x_samples_np

        # When not preserve color, draw a different frame at first and use its
        # color to redraw the first frame.
        if not cfg.color_preserve:
            first_strength = -1
        else:
            first_strength = 1 - cfg.x0_strength

        x_samples, x_samples_np = generate_first_img(img_, first_strength)

        if not cfg.color_preserve:
            color_corrections = setup_color_correction(
                Image.fromarray(x_samples_np[0]))
            global_state.color_corrections = color_corrections
            img_ = apply_color_correction(color_corrections,
                                          Image.fromarray(img))
            img_ = to_tensor(img_).unsqueeze(0)[:, :3] / 127.5 - 1
            x_samples, x_samples_np = generate_first_img(
                img_, 1 - cfg.x0_strength)

        global_state.first_result = x_samples
        global_state.first_img = img

    Image.fromarray(x_samples_np[0]).save(
        os.path.join(cfg.first_dir, 'first.jpg'))

    return x_samples_np[0]


@torch.no_grad()
def process2(*args):
    global global_state
    global global_video_path

    if global_state.processing_state != ProcessingState.FIRST_IMG:
        raise gr.Error('Please generate the first key image before generating'
                       ' all key images')

    cfg = create_cfg(global_video_path, *args)
    global_state.update_sd_model(cfg.sd_model, cfg.control_type)
    global_state.update_detector(cfg.control_type, cfg.canny_low,
                                 cfg.canny_high)
    global_state.processing_state = ProcessingState.KEY_IMGS

    # reset key dir
    shutil.rmtree(cfg.key_dir)
    os.makedirs(cfg.key_dir, exist_ok=True)

    ddim_v_sampler = global_state.ddim_v_sampler
    model = ddim_v_sampler.model
    detector = global_state.detector
    controller = global_state.controller
    flow_model = global_state.flow_model
    model.control_scales = [cfg.control_strength] * 13

    num_samples = 1
    eta = 0.0
    firstx0 = True
    pixelfusion = cfg.use_mask
    imgs = sorted(os.listdir(cfg.input_dir))
    imgs = [os.path.join(cfg.input_dir, img) for img in imgs]

    first_result = global_state.first_result
    first_img = global_state.first_img
    pre_result = first_result
    pre_img = first_img

    for i in range(0, cfg.frame_count - 1, cfg.interval):
        cid = i + 1
        frame = cv2.imread(imgs[i + 1])
        print(cid)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = HWC3(frame)
        H, W, C = img.shape

        if cfg.color_preserve or global_state.color_corrections is None:
            img_ = numpy2tensor(img)
        else:
            img_ = apply_color_correction(global_state.color_corrections,
                                          Image.fromarray(img))
            img_ = to_tensor(img_).unsqueeze(0)[:, :3] / 127.5 - 1
        encoder_posterior = model.encode_first_stage(img_.to(device))
        x0 = model.get_first_stage_encoding(encoder_posterior).detach()

        detected_map = detector(img)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(
            detected_map.copy()).float().to(device) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        cond = {
            'c_concat': [control],
            'c_crossattn': [
                model.get_learned_conditioning(
                    [cfg.prompt + ', ' + cfg.a_prompt] * num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [model.get_learned_conditioning([cfg.n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        cond['c_concat'] = [control]
        un_cond['c_concat'] = [control]

        image1 = torch.from_numpy(pre_img).permute(2, 0, 1).float()
        image2 = torch.from_numpy(img).permute(2, 0, 1).float()
        warped_pre, bwd_occ_pre, bwd_flow_pre = get_warped_and_mask(
            flow_model, image1, image2, pre_result, False)
        blend_mask_pre = blur(
            F.max_pool2d(bwd_occ_pre, kernel_size=9, stride=1, padding=4))
        blend_mask_pre = torch.clamp(blend_mask_pre + bwd_occ_pre, 0, 1)

        image1 = torch.from_numpy(first_img).permute(2, 0, 1).float()
        warped_0, bwd_occ_0, bwd_flow_0 = get_warped_and_mask(
            flow_model, image1, image2, first_result, False)
        blend_mask_0 = blur(
            F.max_pool2d(bwd_occ_0, kernel_size=9, stride=1, padding=4))
        blend_mask_0 = torch.clamp(blend_mask_0 + bwd_occ_0, 0, 1)

        if firstx0:
            mask = 1 - F.max_pool2d(blend_mask_0, kernel_size=8)
            controller.set_warp(
                F.interpolate(bwd_flow_0 / 8.0,
                              scale_factor=1. / 8,
                              mode='bilinear'), mask)
        else:
            mask = 1 - F.max_pool2d(blend_mask_pre, kernel_size=8)
            controller.set_warp(
                F.interpolate(bwd_flow_pre / 8.0,
                              scale_factor=1. / 8,
                              mode='bilinear'), mask)

        controller.set_task('keepx0, keepstyle')
        seed_everything(cfg.seed)
        samples, intermediates = ddim_v_sampler.sample(
            cfg.ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=cfg.scale,
            unconditional_conditioning=un_cond,
            controller=controller,
            x0=x0,
            strength=1 - cfg.x0_strength)
        direct_result = model.decode_first_stage(samples)

        if not pixelfusion:
            pre_result = direct_result
            pre_img = img
            viz = (
                einops.rearrange(direct_result, 'b c h w -> b h w c') * 127.5 +
                127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        else:

            blend_results = (1 - blend_mask_pre
                             ) * warped_pre + blend_mask_pre * direct_result
            blend_results = (
                1 - blend_mask_0) * warped_0 + blend_mask_0 * blend_results

            bwd_occ = 1 - torch.clamp(1 - bwd_occ_pre + 1 - bwd_occ_0, 0, 1)
            blend_mask = blur(
                F.max_pool2d(bwd_occ, kernel_size=9, stride=1, padding=4))
            blend_mask = 1 - torch.clamp(blend_mask + bwd_occ, 0, 1)

            encoder_posterior = model.encode_first_stage(blend_results)
            xtrg = model.get_first_stage_encoding(
                encoder_posterior).detach()  # * mask
            blend_results_rec = model.decode_first_stage(xtrg)
            encoder_posterior = model.encode_first_stage(blend_results_rec)
            xtrg_rec = model.get_first_stage_encoding(
                encoder_posterior).detach()
            xtrg_ = (xtrg + 1 * (xtrg - xtrg_rec))  # * mask
            blend_results_rec_new = model.decode_first_stage(xtrg_)
            tmp = (abs(blend_results_rec_new - blend_results).mean(
                dim=1, keepdims=True) > 0.25).float()
            mask_x = F.max_pool2d((F.interpolate(tmp,
                                                 scale_factor=1 / 8.,
                                                 mode='bilinear') > 0).float(),
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

            mask = (1 - F.max_pool2d(1 - blend_mask, kernel_size=8)
                    )  # * (1-mask_x)

            if cfg.smooth_boundary:
                noise_rescale = find_flat_region(mask)
            else:
                noise_rescale = torch.ones_like(mask)
            masks = []
            for i in range(cfg.ddim_steps):
                if i <= cfg.ddim_steps * cfg.mask_period[
                        0] or i >= cfg.ddim_steps * cfg.mask_period[1]:
                    masks += [None]
                else:
                    masks += [mask * cfg.mask_strength]

            # mask 3
            # xtrg = ((1-mask_x) *
            #         (xtrg + xtrg - xtrg_rec) + mask_x * samples) * mask
            # mask 2
            # xtrg = (xtrg + 1 * (xtrg - xtrg_rec)) * mask
            xtrg = (xtrg + (1 - mask_x) * (xtrg - xtrg_rec)) * mask  # mask 1

            tasks = 'keepstyle, keepx0'
            if not firstx0:
                tasks += ', updatex0'
            if i % cfg.style_update_freq == 0:
                tasks += ', updatestyle'
            controller.set_task(tasks, 1.0)

            seed_everything(cfg.seed)
            samples, _ = ddim_v_sampler.sample(
                cfg.ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=cfg.scale,
                unconditional_conditioning=un_cond,
                controller=controller,
                x0=x0,
                strength=1 - cfg.x0_strength,
                xtrg=xtrg,
                mask=masks,
                noise_rescale=noise_rescale)
            x_samples = model.decode_first_stage(samples)
            pre_result = x_samples
            pre_img = img

            viz = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
                   127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        Image.fromarray(viz[0]).save(
            os.path.join(cfg.key_dir, f'{cid:04d}.png'))

    key_video_path = os.path.join(cfg.work_dir, 'key.mp4')
    fps = get_fps(cfg.input_path)
    fps //= cfg.interval
    frame_to_video(key_video_path, cfg.key_dir, fps, False)

    return key_video_path
def input_uploaded(path):
    frame_count = get_frame_count(path)
    if frame_count <= 2:
        raise AssertionError(
            "The input video is too short!" "Please input another video."
        )

    default_interval = min(10, frame_count - 2)
    max_keyframe = min((frame_count - 2) // default_interval, MAX_KEYFRAME)

    global video_frame_count
    video_frame_count = frame_count
    global global_video_path
    global_video_path = path

    # return gr.Slider.update(
    #     value=default_interval, maximum=frame_count - 2
    # ), gr.Slider.update(value=max_keyframe, maximum=max_keyframe)

def input_changed(path):
    frame_count = get_frame_count(path)
    # if frame_count <= 2:
    #     return gr.Slider.update(maximum=1), gr.Slider.update(maximum=1)

    default_interval = min(10, frame_count - 2)
    max_keyframe = min((frame_count - 2) // default_interval, MAX_KEYFRAME)

    global video_frame_count
    video_frame_count = frame_count
    global global_video_path
    global_video_path = path

    # return gr.Slider.update(
    #     value=default_interval, maximum=frame_count - 2
    # ), gr.Slider.update(maximum=max_keyframe)

def interval_changed(interval):
    global video_frame_count
    # if video_frame_count is None:
    #     return gr.Slider.update()

    max_keyframe = min((video_frame_count - 2) // interval, MAX_KEYFRAME)

    # return gr.Slider.update(value=max_keyframe, maximum=max_keyframe)

# ==============================================================================================================================================

# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        input_path: Path = Input(description="Input Video", default="videos/pexels-cottonbro-studio-6649832-960x506-25fps.mp4"),
        prompt: str = Input(description="Prompt", default="white ancient Greek sculpture, Venus de Milo, light pink and blue background"),
        image_resolution: int = Input(description="Frame resolution", ge=256, le=512, default=512),
        control_strength: float = Input(description="ControNet strength", ge=0.0, le=2.0, default=1.0),
        color_preserve: bool = Input(description="Preserve color (Keep the color of the input video)", default=True),
        left_crop: int = Input(description="Left crop length", ge=0, le=512, default=0),
        right_crop: int = Input(description="Right crop length", ge=0, le=512, default=0),
        top_crop: int = Input(description="Top crop length", ge=0, le=512, default=0),
        bottom_crop: int = Input(description="Bottom crop length", ge=0, le=512, default=0),
        control_type: str = Input(description="Control type", choices=["HED", "canny"], default="HED"),
        low_threshold: int = Input(description='Canny low threshold (If `Control type` is "canny" Control type)', ge=1, le=255, default=50),
        high_threshold: int = Input(description='Canny high threshold (If `Control type` is "canny" Control type)', ge=1, le=255, default=100),
        ddim_steps: int = Input(description="Steps (To avoid overload, maximum 20)", ge=1, le=20, default=20),
        scale: float = Input(description="CFG scale", ge=0.1, le=30.0, default=7.5),
        seed: int = Input(description="Seed", ge=0, le=2147483647, default=0),
        sd_model: str = Input(description="Base model", default="Stable Diffusion 1.5", choices=['Stable Diffusion 1.5','revAnimated_v11','realisticVisionV20_v20']), # You may want to add choices for the sd_model input.
        a_prompt: str = Input(description="Added prompt", default="RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"),
        n_prompt: str = Input(description="Negative prompt", default="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"),
        interval: int = Input(description="Key frame frequency, K (Uniformly sample the key frames every K frames)", ge=1, le=10, default=1),
        keyframe_count: int = Input(description="Total number of key frames (To avoid overload, maximum 8 key frames)", ge=1, le=8, default=8),
        x0_strength: float = Input(description="Strength of denoising", ge=0.0, le=1.05, default=0.75),
        use_constraints: str = Input(description="Constraints for cross-frame", choices=["shape-aware fusion", "pixel-aware fusion", "color-aware AdaIN"], default="shape-aware fusion"),
        cross_start: float = Input(description="Start of cross-frame attention", ge=0, le=1, default=0),
        cross_end: float = Input(description="End of cross-frame attention", ge=0, le=1, default=1),
        style_update_freq: int = Input(description="Frequency of updating for cross-frame attention (Update the key and value for cross-frame attention every N key frames (recommend N*K>=10))", ge=1, le=100, default=1),
        warp_start: float = Input(description="Start of shape-aware fusion", ge=0, le=1, default=0),
        warp_end: float = Input(description="End of shape-aware fusion", ge=0, le=1, default=1),
        mask_start: float = Input(description="Start of pixel-aware fusion", ge=0, le=1, default=0.5),
        mask_end: float = Input(description="End of pixel-aware fusion", ge=0, le=1, default=0.8),
        mask_strength: float = Input(description="Strength of pixel-aware fusion", ge=0, le=1, default=0.5),
        ada_start: float = Input(description="Start of color-aware AdaIN", ge=0, le=1, default=0.8),
        ada_end: float = Input(description="End of color-aware AdaIN", ge=0, le=1, default=1),
        inner_strength: float = Input(description="Pixel-aware fusion detail level (Use a low value to prevent artifacts)", ge=0.5, le=1, default=0.9),
        smooth_boundary: bool = Input(description="Smooth fusion boundary (Select to prevent artifacts at boundary)", default=True),
        ) -> Path:
            """Run a single prediction on the model"""

            frame_count = 3

            input_path="videos/pexels-cottonbro-studio-6649832-960x506-25fps.mp4"
            prompt="white ancient Greek sculpture, Venus de Milo, light pink and blue background"
            image_resolution=256
            control_strength=0.7
            color_preserve=True
            left_crop=0
            right_crop=180
            top_crop=0
            bottom_crop=0
            control_type="canny"
            low_threshold=50
            high_threshold=100
            ddim_steps=20
            scale=7.5
            seed=0
            sd_model="realisticVisionV20_v20"
            a_prompt="RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
            n_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
            interval=1
            keyframe_count=8#(frame_count - 2) // interval
            x0_strength=0.95
            use_constraints="shape-aware fusion"
            cross_start=0
            cross_end=1
            style_update_freq=1
            warp_start=0
            warp_end=1
            mask_start=0.5
            mask_end=0.8
            ada_start=0.8
            ada_end=1
            mask_strength=0.5
            inner_strength=0.9
            smooth_boundary=True

            input_uploaded(input_path)
            interval_changed(interval)

            global video_frame_count
            video_frame_count = frame_count

            out_first_frame, out_keypath = process(
                *[
                    # input_path,
                    prompt,
                    image_resolution,
                    control_strength,
                    color_preserve,
                    left_crop,
                    right_crop,
                    top_crop,
                    bottom_crop,
                    control_type,
                    low_threshold,
                    high_threshold,
                    ddim_steps,
                    scale,
                    seed,
                    sd_model,
                    a_prompt,
                    n_prompt,
                    interval,
                    keyframe_count,
                    x0_strength,
                    use_constraints,
                    cross_start,
                    cross_end,
                    style_update_freq,
                    warp_start,
                    warp_end,
                    mask_start,
                    mask_end,
                    ada_start,
                    ada_end,
                    mask_strength,
                    inner_strength,
                    smooth_boundary,
                ]
            )
            print(f"Output path: {out_keypath}")
            print(f"First frame: {out_first_frame}")
            
            return out_keypath



