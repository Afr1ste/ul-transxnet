from __future__ import annotations

import base64
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(r"C:\Users\Afr1ste\PycharmProjects\Thyroid")

SOURCE_IMAGE = PROJECT_ROOT / r"busi\busi_voc_v3_square_consistent\JPEGImages\test_benign_0186.png"
ROI_CROP = OUT_DIR / "roi_crop_selected_busi_test_benign_0186.png"

ORIG_W = 741
ORIG_H = 574
IMAGE_SLOT = 224
IMAGE_TOP = 123
IMAGE_PAD_X = 13

# Coordinates measured from the BUSI VOC annotation preview for this sample.
LESION_BOX = (297, 178, 516, 364)
# Current manuscript protocol expands the lesion box in width and height and
# resizes the resulting rectangular ROI to the network input tensor.
EXPANDED_ROI = (264, 150, 549, 392)


def data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def rect(x, y, w, h, cls="", rx=6, extra="") -> str:
    c = f' class="{cls}"' if cls else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"{c} {extra}/>'


def text(x, y, content, cls="", anchor="middle") -> str:
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}"{c}>{content}</text>'


def line(x1, y1, x2, y2, cls="arrow") -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}" marker-end="url(#arrow)"/>'


def image_layout(x: int) -> tuple[float, float, float]:
    scale = min(IMAGE_SLOT / ORIG_W, IMAGE_SLOT / ORIG_H)
    shown_h = ORIG_H * scale
    image_x = x + IMAGE_PAD_X
    image_y = IMAGE_TOP + (IMAGE_SLOT - shown_h) / 2
    return image_x, image_y, scale


def map_box(x: int, box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    image_x, image_y, scale = image_layout(x)
    xmin, ymin, xmax, ymax = box
    return (
        image_x + xmin * scale,
        image_y + ymin * scale,
        (xmax - xmin) * scale,
        (ymax - ymin) * scale,
    )


def box_overlay(x: int, box: tuple[int, int, int, int], cls: str) -> str:
    bx, by, bw, bh = map_box(x, box)
    return f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bw:.2f}" height="{bh:.2f}" rx="1.5" class="{cls}"/>'


def center_of(box: tuple[int, int, int, int]) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = box
    return (xmin + xmax) / 2, (ymin + ymax) / 2


def map_point(x: int, px: float, py: float) -> tuple[float, float]:
    image_x, image_y, scale = image_layout(x)
    return image_x + px * scale, image_y + py * scale


def image_panel(x: int, title: str, subtitle: str, href: str, panel: str, overlay: str = "") -> str:
    return f"""
  <g>
    {rect(x, 110, 250, 250, "img-frame", 8)}
    <image href="{href}" x="{x + 13}" y="123" width="224" height="224" preserveAspectRatio="xMidYMid meet"/>
    {overlay}
    {text(x + 125, 408, title, "panel-title")}
    {text(x + 125, 438, subtitle, "shape")}
    {text(x + 125, 484, panel, "panel-index")}
  </g>
"""


def operation_panel(x: int, href: str) -> str:
    cx0, cy0 = center_of(LESION_BOX)
    cx, cy = map_point(x, cx0, cy0)
    crop_x, crop_y, crop_w, crop_h = map_box(x, EXPANDED_ROI)
    targets = [
        (crop_x + 8, crop_y + 8),
        (crop_x + crop_w - 8, crop_y + 8),
        (crop_x + 8, crop_y + crop_h - 8),
        (crop_x + crop_w - 8, crop_y + crop_h - 8),
    ]
    expansion_arrows = "\n".join(
        f'''
    <line x1="{cx:.2f}" y1="{cy:.2f}" x2="{tx:.2f}" y2="{ty:.2f}" class="expand-arrow-halo"/>
    <line x1="{cx:.2f}" y1="{cy:.2f}" x2="{tx:.2f}" y2="{ty:.2f}" class="expand-arrow" marker-end="url(#arrowExpand)"/>'''
        for tx, ty in targets
    )
    return f"""
  <g>
    {rect(x, 110, 250, 250, "img-frame", 8)}
    <image href="{href}" x="{x + 13}" y="123" width="224" height="224" preserveAspectRatio="xMidYMid meet"/>
    {box_overlay(x, LESION_BOX, "lesion-box")}
    {box_overlay(x, EXPANDED_ROI, "crop-box")}
    {expansion_arrows}
    <circle cx="{cx:.2f}" cy="{cy:.2f}" r="4.6" fill="#fff" opacity="0.95"/>
    <circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.8" class="center-dot"/>
    {text(x + 125, 408, "Expand ROI", "panel-title")}
    {text(x + 125, 438, "aspect ratio preserved", "shape")}
    {text(x + 125, 484, "(c)", "panel-index")}
  </g>
"""


def build_svg() -> str:
    source = data_uri(SOURCE_IMAGE)
    crop = data_uri(ROI_CROP)
    annotation_overlay = f"""
    {box_overlay(410, LESION_BOX, "lesion-box")}
"""

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1420" height="520" viewBox="0 0 1420 520">
  <defs>
    <style>
      text {{ font-family: "Times New Roman", Times, serif; fill: #111; }}
      .panel-title {{ font-size: 27px; font-weight: 700; }}
      .shape {{ font-size: 20px; font-style: italic; }}
      .small {{ font-size: 20px; }}
      .panel-index {{ font-size: 23px; }}
      .img-frame {{ fill: #fff; stroke: #111; stroke-width: 2; }}
      .op-box {{ fill: #fffdf8; stroke: #111; stroke-width: 2; stroke-dasharray: 8 7; }}
      .arrow {{ stroke: #111; stroke-width: 3; stroke-linecap: round; }}
      .lesion-box {{ fill: none; stroke: #0072B2; stroke-width: 2.1; }}
      .crop-box {{ fill: none; stroke: #CC3333; stroke-width: 2.2; stroke-dasharray: 7 5; }}
      .expand-arrow-halo {{ stroke: #fff; stroke-width: 4.6; stroke-linecap: round; opacity: 0.82; }}
      .expand-arrow {{ stroke: #CC3333; stroke-width: 2.1; stroke-linecap: round; }}
      .center-dot {{ fill: #CC3333; }}
    </style>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#111"/>
    </marker>
    <marker id="arrowThin" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 8 4 L 0 8 z" fill="#111"/>
    </marker>
    <marker id="arrowExpand" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 8 4 L 0 8 z" fill="#CC3333"/>
    </marker>
  </defs>

  {image_panel(60, "Original ultrasound", "B × 3 × H × W", source, "(a)")}
  {line(320, 235, 396, 235)}
  {image_panel(410, "Lesion annotation", "lesion bbox", source, "(b)", annotation_overlay)}
  {line(670, 235, 746, 235)}
  {operation_panel(760, source)}

  {line(1028, 235, 1104, 235)}
  {image_panel(1118, "Final ROI crop", "B × 3 × 256 × 256", crop, "(d)")}
</svg>
"""


def main() -> None:
    svg_path = OUT_DIR / "roi_preprocess_topconf_draft_v1.svg"
    png_path = OUT_DIR / "roi_preprocess_topconf_draft_v1_preview.png"
    final_svg_path = OUT_DIR / "fig_roi_preprocessing.svg"
    final_png_path = OUT_DIR / "fig_roi_preprocessing.png"
    svg_path.write_text(build_svg(), encoding="utf-8")
    final_svg_path.write_text(build_svg(), encoding="utf-8")
    print(svg_path)
    try:
        import cairosvg

        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=1420,
            output_height=520,
            background_color="white",
        )
        cairosvg.svg2png(
            url=str(final_svg_path),
            write_to=str(final_png_path),
            output_width=1420,
            output_height=520,
            background_color="white",
        )
        print(png_path)
        print(final_png_path)
    except ImportError:
        print("cairosvg is not installed; SVG was written, PNG preview was not regenerated.")


if __name__ == "__main__":
    main()
