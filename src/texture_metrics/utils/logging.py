from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import json
from typing import Any, Optional, Sequence
import torch
import wandb
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont

try:
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _RICH_AVAILABLE = False

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from .distributed import get_global_rank


def generate_unique_run_name(base_name: str, save_dir: str) -> str:
    save_path = Path(save_dir)
    i = 0
    while True:
        run_name = f"{base_name}_{i:03d}"
        run_path = save_path / run_name
        if not run_path.exists():
            return run_name
        i += 1


def strip_compile_prefix_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    normalized_state_dict: dict[str, torch.Tensor] = {}
    renamed_keys = 0

    for key, value in state_dict.items():
        normalized_key = key
        if normalized_key.startswith("_orig_mod."):
            normalized_key = normalized_key[len("_orig_mod.") :]
        normalized_key = normalized_key.replace("._orig_mod.", ".")

        if normalized_key != key:
            renamed_keys += 1

        normalized_state_dict[normalized_key] = value

    return normalized_state_dict, renamed_keys


def unwrap_compiled(module: torch.nn.Module) -> torch.nn.Module:
    return module._orig_mod if hasattr(module, "_orig_mod") else module


class CheckpointLogger:
    def __init__(
        self,
        save_dir: Optional[str | Path],
        run_name: str = "run",
        base_save_dir: Optional[str | Path] = None,
        last_ckpt_name: str = "last",
        save_ckpt_every_n_epochs: int = 1,
        snapshot_every_n_epochs: int = 0,
    ):
        if save_dir is not None:
            self.set_save_dir(save_dir)
        else:
            self.run_name = run_name
            self.base_save_dir = base_save_dir
        self.last_ckpt_name = last_ckpt_name
        self.snapshot_every_n_epochs = snapshot_every_n_epochs
        self.save_ckpt_every_n_epochs = save_ckpt_every_n_epochs
        self._config: Optional[dict[str, Any]] = None
        self._resolved_config: Optional[dict[str, Any]] = None

    @property
    def save_dir(self) -> str:
        return Path(self.base_save_dir) / self.run_name

    def set_save_dir(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        self.run_name = save_dir.name
        self.base_save_dir = save_dir.parent

    def setup(self, resume_from_ckpt: Optional[str | Path]) -> None:
        if resume_from_ckpt:
            resume_path = Path(resume_from_ckpt)
            # If a checkpoint file is provided, use its parent run directory.
            if resume_path.suffix == ".ckpt":
                resume_path = resume_path.parent
            self.set_save_dir(resume_path)
        else:
            slurm_job_name = os.environ.get("SLURM_JOB_NAME")
            slurm_job_id = os.environ.get("SLURM_JOB_ID")

            base_name = slurm_job_name or self.run_name

            if slurm_job_id:
                self.run_name = f"{base_name}_{slurm_job_id}"
            else:
                self.run_name = generate_unique_run_name(base_name, self.save_dir)

        os.makedirs(self.save_dir, exist_ok=True)

    def set_config(self, config: Optional[dict[str, Any]]) -> None:
        self._config = config
        self._resolved_config = None

    def _resolve_config(self) -> Optional[dict[str, Any]]:
        if self._resolved_config is not None:
            return self._resolved_config
        if self._config is None:
            return None

        resolved = _normalize_config_value(self._config)
        _apply_logger_runtime_config(
            resolved,
            run_name=self.run_name,
            save_dir=str(self.save_dir),
            base_save_dir=str(self.base_save_dir)
            if self.base_save_dir is not None
            else None,
        )
        self._resolved_config = resolved
        return resolved

    def write_config_yaml(
        self,
        filename: str = "config.yaml",
        overwrite: bool = True,
    ) -> Optional[Path]:
        config = self._resolve_config()
        if config is None:
            return None

        path = Path(self.save_dir) / filename
        if path.exists() and not overwrite:
            return path

        _write_config_yaml(path, config)
        return path

    def save_ckpt(self, trainer) -> None:
        last_flag = (trainer.current_epoch + 1) % self.save_ckpt_every_n_epochs == 0
        snapshot_flag = (
            self.snapshot_every_n_epochs > 0
            and (trainer.current_epoch + 1) % self.snapshot_every_n_epochs == 0
        )
        if not last_flag and not snapshot_flag:
            return

        if get_global_rank() == 0:
            state = trainer.state_dict()
            if last_flag:
                torch.save(state, self.save_dir / f"{self.last_ckpt_name}.ckpt")
            if snapshot_flag:
                torch.save(
                    state,
                    self.save_dir / f"snapshot_epoch_{trainer.current_epoch + 1}.ckpt",
                )


class WandbLogger(CheckpointLogger):
    def __init__(
        self,
        save_dir: Optional[str | Path] = None,
        run_name: str = "run",
        base_save_dir: Optional[str | Path] = None,
        last_ckpt_name: str = "last",
        save_ckpt_every_n_epochs: int = 1,
        snapshot_every_n_epochs: int = 0,
        project: Optional[str] = None,
        **wandb_args: dict[str, Any],
    ):
        super().__init__(
            save_dir,
            run_name,
            base_save_dir,
            last_ckpt_name,
            save_ckpt_every_n_epochs,
            snapshot_every_n_epochs,
        )
        self.project = project
        self._wandb_args = wandb_args
        self.run = None

    @property
    def wandb_args(self) -> dict[str, Any]:
        self._wandb_args["project"] = self.project
        self._wandb_args["dir"] = self.save_dir
        self._wandb_args["name"] = self.run_name
        return self._wandb_args

    def setup(self, resume_from_ckpt: Optional[str | Path] = None) -> None:
        super().setup(resume_from_ckpt)
        if resume_from_ckpt:
            wandb_dir = Path(self.save_dir) / "wandb"
            if wandb_dir.exists():
                run_dirs = sorted(
                    [
                        d
                        for d in wandb_dir.iterdir()
                        if d.is_dir() and d.name.startswith("run-")
                    ]
                )
                if run_dirs:
                    latest_run = run_dirs[-1]
                    run_id = latest_run.name.split("-")[-1]
                    self._wandb_args["id"] = run_id
        if self.run is None:
            init_args = dict(self.wandb_args)
            if "config" not in init_args:
                init_args["config"] = self._resolve_config()
            self.run = wandb.init(**init_args)

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            self.setup()
        wandb.log(data, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalize_config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value(v) for v in value]
    if isinstance(value, set):
        return [_normalize_config_value(v) for v in sorted(value)]
    return value


def _apply_logger_runtime_config(
    config: dict[str, Any],
    *,
    run_name: str,
    save_dir: str,
    base_save_dir: Optional[str],
) -> None:
    logger_cfg = config.get("logger")
    if not isinstance(logger_cfg, dict):
        return

    init_args = logger_cfg.get("init_args")
    if isinstance(init_args, dict):
        init_args["run_name"] = run_name
        init_args["save_dir"] = save_dir
        init_args["base_save_dir"] = base_save_dir
    else:
        logger_cfg["run_name"] = run_name
        logger_cfg["save_dir"] = save_dir
        logger_cfg["base_save_dir"] = base_save_dir


def _write_config_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        payload = json.dumps(config, indent=2, ensure_ascii=True)
        path.write_text(payload, encoding="utf-8")
        return

    payload = yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=False,
    )
    path.write_text(payload, encoding="utf-8")


def numpy2pil(images):
    """Convert a numpy image or a batch of images to a PIL image."""
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def torch2numpy(
    images: Sequence[torch.Tensor] | torch.Tensor, nrow: Optional[int] = None
):
    if not isinstance(images, torch.Tensor):
        images = torch.cat(images)
    imgs = make_grid(images, nrow=nrow or len(images)).unsqueeze(0)
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
    return imgs


def torch2pil(
    images: Sequence[torch.Tensor] | torch.Tensor, nrow: Optional[int] = None
):
    return numpy2pil(torch2numpy(images, nrow=nrow))


def torch2pil_with_captions(
    images: Sequence[torch.Tensor] | torch.Tensor,
    nrow: Optional[int] = None,
    left_captions: Optional[Sequence[str]] = None,
    left_caption_spans: Optional[Sequence[int]] = None,
    bottom_captions: Optional[Sequence[str]] = None,
    title: Optional[str] = None,  # New parameter for title
    font_size: int = 20,
    title_font_size: Optional[int] = None,  # Optional separate font size for title
    caption_color: str = "black",
    title_color: Optional[str] = None,  # Optional separate color for title
    background_color: Optional[str] = "white",  # Set to None for transparent
    padding: int = 10,
    title_padding: Optional[int] = None,  # Optional separate padding for title
    left_captions_vertical: bool = False,
    use_latex: bool = False,  # New parameter to enable LaTeX rendering
):
    """Convert torch tensors to PIL images with optional captions on the left and bottom, and title on top.

    Args:
        images: Sequence of torch.Tensor or single torch.Tensor
        nrow: Number of images per row in the grid
        left_captions: Sequence of captions for row groups (from top to bottom)
        left_caption_spans: Number of rows spanned by each left caption.
            Defaults to 1 for every caption.
        bottom_captions: Sequence of captions for each column (from left to right)
        title: Optional title to display at the top of the image
        font_size: Font size for captions
        title_font_size: Font size for title (defaults to font_size * 1.5 if not specified)
        caption_color: Color of the caption text
        title_color: Color of the title text (defaults to caption_color if not specified)
        background_color: Background color for caption areas. Set to None for transparent.
        padding: Padding around captions in pixels
        title_padding: Padding around title in pixels (defaults to padding if not specified)
        left_captions_vertical: If True, left captions are written vertically (rotated 90° counter-clockwise)
        use_latex: If True, captions are rendered as LaTeX equations using matplotlib

    Returns:
        PIL Image with captions
    """
    # Set defaults for title-specific parameters
    if title_font_size is None:
        title_font_size = int(font_size * 1.5)
    if title_color is None:
        title_color = caption_color
    if title_padding is None:
        title_padding = padding

    # Convert to PIL image first
    pil_images = torch2pil(images, nrow=nrow)
    base_image = pil_images[0]

    if not isinstance(images, torch.Tensor):
        images = torch.cat(images)
    num_images = images.shape[0]
    actual_nrow = nrow or num_images
    num_rows = (num_images + actual_nrow - 1) // actual_nrow
    num_cols = min(actual_nrow, num_images)

    if left_captions is None and bottom_captions is None and title is None:
        return base_image

    if use_latex:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        def render_latex(text, fontsize, color):
            """Render LaTeX text to PIL Image"""
            # Create a figure with transparent background
            fig = Figure(figsize=(10, 2), dpi=300)
            fig.patch.set_alpha(0.0)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            # Render the text
            t = ax.text(
                0.5,
                0.5,
                text,
                fontsize=fontsize,
                ha="center",
                va="center",
                color=color,
            )

            # Draw to get the bbox
            canvas.draw()
            bbox = t.get_window_extent(renderer=canvas.get_renderer())

            # Adjust figure size to fit text
            width_inches = (bbox.width + 20) / fig.dpi
            height_inches = (bbox.height + 20) / fig.dpi
            fig.set_size_inches(width_inches, height_inches)

            # Re-draw with correct size
            canvas.draw()

            # Convert to PIL Image
            buf = canvas.buffer_rgba()
            img = Image.frombytes("RGBA", canvas.get_width_height(), buf)

            plt.close(fig)
            return img
    else:
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
            title_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                title_font_size,
            )
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
                title_font = ImageFont.truetype("arial.ttf", title_font_size)
            except:
                font = ImageFont.load_default()
                title_font = ImageFont.load_default()

    base_width, base_height = base_image.size

    # Calculate dimensions for captions
    left_width = 0
    bottom_height = 0
    title_height = 0

    # Calculate title dimensions
    if title:
        if use_latex:
            title_image = render_latex(title, title_font_size, title_color)
            title_height = title_image.height + 2 * title_padding
        else:
            draw_temp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            bbox = draw_temp.textbbox((0, 0), title, font=title_font, align="center")
            title_height = (bbox[3] - bbox[1]) + 2 * title_padding

    if left_captions:
        if use_latex:
            # Render all left captions to get dimensions
            left_caption_images = [
                render_latex(cap, font_size, caption_color) for cap in left_captions
            ]
            if left_captions_vertical:
                left_width = (
                    max(
                        [
                            img.rotate(90, expand=True).width
                            for img in left_caption_images
                        ]
                    )
                    + 2 * padding
                )
            else:
                left_width = (
                    max([img.width for img in left_caption_images]) + 2 * padding
                )
        else:
            # Calculate the width needed for left captions
            draw_temp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            if left_captions_vertical:
                left_width = (
                    max(
                        [
                            draw_temp.textbbox(
                                (0, 0), caption, font=font, align="center"
                            )[3]
                            for caption in left_captions
                        ]
                    )
                    + 2 * padding
                )
            else:
                left_width = (
                    max(
                        [
                            draw_temp.textbbox(
                                (0, 0), caption, font=font, align="center"
                            )[2]
                            for caption in left_captions
                        ]
                    )
                    + 2 * padding
                )

    if bottom_captions:
        if use_latex:
            # Render all bottom captions to get dimensions
            bottom_caption_images = [
                render_latex(cap, font_size, caption_color) for cap in bottom_captions
            ]
            bottom_height = (
                max([img.height for img in bottom_caption_images]) + 2 * padding
            )
        else:
            # Calculate the height needed for bottom captions
            draw_temp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            bbox = draw_temp.textbbox((0, 0), "Test", font=font, align="center")
            bottom_height = (bbox[3] - bbox[1]) + 2 * padding

    # Create new image with space for captions and title
    new_width = int(base_width + left_width)
    new_height = int(base_height + bottom_height + title_height)

    if background_color is None:
        # Create RGBA image with transparent background
        captioned_image = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 0))
        # Convert base image to RGBA if it isn't already
        if base_image.mode != "RGBA":
            base_image = base_image.convert("RGBA")
    else:
        # Create RGB image with solid background
        captioned_image = Image.new("RGB", (new_width, new_height), background_color)

    # Paste the base image (offset by title_height and left_width)
    captioned_image.paste(base_image, (int(left_width), int(title_height)))

    # Draw on the image (only needed for non-LaTeX)
    if not use_latex:
        draw = ImageDraw.Draw(captioned_image)

    # Add title
    if title:
        if use_latex:
            # Center the title horizontally
            x_pos = int(new_width / 2 - title_image.width / 2)
            y_pos = int(title_padding)

            # Paste the LaTeX image
            if background_color is None or title_image.mode == "RGBA":
                captioned_image.paste(title_image, (x_pos, y_pos), title_image)
            else:
                # Convert RGBA to RGB with background color
                rgb_img = Image.new("RGB", title_image.size, background_color)
                rgb_img.paste(title_image, (0, 0), title_image)
                captioned_image.paste(rgb_img, (x_pos, y_pos))
        else:
            # Non-LaTeX text rendering
            bbox = draw.textbbox((0, 0), title, font=title_font, align="center")
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            x_pos = (new_width - text_width) / 2
            y_pos = title_padding

            draw.text(
                (x_pos, y_pos),
                title,
                fill=title_color,
                font=title_font,
                align="center",
            )

    # Add left captions
    if left_captions:
        caption_spans = (
            list(left_caption_spans)
            if left_caption_spans is not None
            else [1] * len(left_captions)
        )

        if len(caption_spans) != len(left_captions):
            raise ValueError(
                f"Number of left_caption_spans ({len(caption_spans)}) must match "
                f"number of left_captions ({len(left_captions)})"
            )

        if sum(caption_spans) != num_rows:
            raise ValueError(
                f"Sum of left_caption_spans ({sum(caption_spans)}) must match "
                f"number of rows ({num_rows})"
            )

        # Calculate image height per row
        row_height = base_height / num_rows
        row_start = 0

        for i, caption in enumerate(left_captions):
            span = caption_spans[i]
            if use_latex:
                # Render LaTeX caption
                latex_img = left_caption_images[i]

                if left_captions_vertical:
                    latex_img = latex_img.rotate(90, expand=True)

                # Calculate position to center the caption in the row (offset by title_height)
                y_pos = int(
                    title_height
                    + row_start * row_height
                    + (row_height * span) / 2
                    - latex_img.height / 2
                )
                x_pos = int(left_width / 2 - latex_img.width / 2)

                # Paste the LaTeX image
                if background_color is None or latex_img.mode == "RGBA":
                    captioned_image.paste(latex_img, (x_pos, y_pos), latex_img)
                else:
                    # Convert RGBA to RGB with background color
                    rgb_img = Image.new("RGB", latex_img.size, background_color)
                    rgb_img.paste(latex_img, (0, 0), latex_img)
                    captioned_image.paste(rgb_img, (x_pos, y_pos))
            else:
                # Non-LaTeX text rendering
                if left_captions_vertical:
                    # Create a temporary image for the rotated text
                    bbox = draw.textbbox((0, 0), caption, font=font, align="center")
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Create temporary image with text
                    txt_img = Image.new(
                        "RGBA" if background_color is None else "RGB",
                        (text_width + 4, text_height + 4),
                        (255, 255, 255, 0)
                        if background_color is None
                        else background_color,
                    )
                    txt_draw = ImageDraw.Draw(txt_img)
                    txt_draw.text(
                        (2, 2),
                        caption,
                        fill=caption_color,
                        font=font,
                        align="center",
                    )

                    # Rotate the text 90 degrees counter-clockwise
                    rotated = txt_img.rotate(90, expand=True)

                    # Calculate position to center the rotated text in the row (offset by title_height)
                    y_pos = int(
                        title_height
                        + row_start * row_height
                        + (row_height * span) / 2
                        - rotated.height / 2
                    )
                    x_pos = int(left_width / 2 - rotated.width / 2)

                    # Paste the rotated text
                    if background_color is None:
                        captioned_image.paste(rotated, (x_pos, y_pos), rotated)
                    else:
                        captioned_image.paste(rotated, (x_pos, y_pos))
                else:
                    # Horizontal text (offset by title_height)
                    bbox = draw.textbbox((0, 0), caption, font=font, align="center")
                    text_height = bbox[3] - bbox[1]
                    y_pos = (
                        title_height
                        + row_start * row_height
                        + (row_height * span) / 2
                        - text_height / 2
                    )

                    draw.text(
                        (padding, y_pos),
                        caption,
                        fill=caption_color,
                        font=font,
                        align="center",
                    )

            row_start += span

    # Add bottom captions
    if bottom_captions:
        if len(bottom_captions) != num_cols:
            raise ValueError(
                f"Number of bottom_captions ({len(bottom_captions)}) must match "
                f"number of columns ({num_cols})"
            )

        # Calculate image width per column
        col_width = base_width / num_cols

        for i, caption in enumerate(bottom_captions):
            if use_latex:
                # Render LaTeX caption
                latex_img = bottom_caption_images[i]

                # Calculate horizontal center position for this column (offset by title_height)
                x_pos = int(
                    left_width + i * col_width + col_width / 2 - latex_img.width / 2
                )
                y_pos = title_height + base_height + padding

                # Paste the LaTeX image
                if background_color is None or latex_img.mode == "RGBA":
                    captioned_image.paste(latex_img, (x_pos, y_pos), latex_img)
                else:
                    # Convert RGBA to RGB with background color
                    rgb_img = Image.new("RGB", latex_img.size, background_color)
                    rgb_img.paste(latex_img, (0, 0), latex_img)
                    captioned_image.paste(rgb_img, (x_pos, y_pos))
            else:
                # Non-LaTeX text rendering (offset by title_height)
                x_pos = left_width + i * col_width + col_width / 2
                bbox = draw.textbbox((0, 0), caption, font=font, align="center")
                text_width = bbox[2] - bbox[0]
                x_pos = x_pos - text_width / 2

                y_pos = title_height + base_height + padding
                draw.text((x_pos, y_pos), caption, fill=caption_color, font=font)

    return captioned_image


def spectrum_to_viridis(spectrum: torch.Tensor):
    spectrum = spectrum.detach().float()
    if spectrum.ndim == 2:
        spectrum = spectrum.unsqueeze(0)
    if spectrum.ndim != 3:
        raise ValueError("Expected a 2D spectrum or a batched [B, H, W] tensor.")

    spectrum = spectrum - spectrum.min()
    spectrum = spectrum / spectrum.max().clamp_min(1e-8)
    rgb = plt.cm.viridis(spectrum.cpu().numpy())[..., :3]
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()


def logsp(img):
    sp = torch.fft.fft2(img.mean(dim=-3)).abs().log()
    sp = torch.fft.fftshift(sp, dim=(-1, -2))
    return sp


def logsp_color(img):
    sp = logsp(img)
    return 2 * spectrum_to_viridis(sp).to(img.device) - 1


def progress_bar(enable_progress_bar: bool, global_rank: int):
    if not enable_progress_bar:
        return nullcontext()
    if global_rank != 0 or not _RICH_AVAILABLE:
        return nullcontext()
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        transient=False,
    )
