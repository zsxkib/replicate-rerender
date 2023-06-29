# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

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
        # processed_input = preprocess(image)
        # output = self.model(processed_input, scale)
        # return postprocess(output)
        ...



