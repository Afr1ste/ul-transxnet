from __future__ import annotations

import base64
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
ROI_CROP = OUT_DIR / "roi_crop_selected_busi_test_benign_0186.png"


def image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def rect(x, y, w, h, cls="", rx=8, extra="") -> str:
    c = f' class="{cls}"' if cls else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"{c} {extra}/>'


def text(x, y, content, cls="", anchor="middle") -> str:
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}"{c}>{content}</text>'


def line(x1, y1, x2, y2, cls="arrow") -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}" marker-end="url(#arrow)"/>'


def stage(x: int, title: str, shape: str, channels: str, repeats: str) -> str:
    return f"""
  <g>
    {rect(x, 96, 245, 220, "stage-box", 18)}
    {text(x + 122, 70, title, "stage-title")}
    {text(x + 122, 92, shape, "shape")}
    {rect(x + 24, 150, 88, 88, "patch", 8)}
    {text(x + 68, 187, "Patch", "small")}
    {text(x + 68, 212, "embed", "small")}
    {rect(x + 142, 140, 76, 112, "block", 12)}
    {text(x + 180, 185, "Hybrid", "small")}
    {text(x + 180, 210, "block", "small")}
    {text(x + 206, 132, repeats, "repeat")}
    {text(x + 122, 342, channels, "shape")}
  </g>
"""


def build_svg() -> str:
    roi = image_data_uri(ROI_CROP)
    stages = [
        stage(318, "Stage 1", "B × 48 × H/4 × W/4", "48 @ 1/4", "×4"),
        stage(628, "Stage 2", "B × 96 × H/8 × W/8", "96 @ 1/8", "×4"),
        stage(938, "Stage 3", "B × 224 × H/16 × W/16", "224 @ 1/16", "×15"),
        stage(1248, "Stage 4", "B × 448 × H/32 × W/32", "448 @ 1/32", "×4"),
    ]

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1760" height="780" viewBox="0 0 1760 780">
  <rect width="1760" height="780" fill="#ffffff"/>
  <defs>
    <style>
      text {{ font-family: "Times New Roman", Times, serif; fill: #111; }}
      .small {{ font-size: 20px; }}
      .shape {{ font-size: 20px; font-style: italic; }}
      .label {{ font-size: 25px; font-weight: 700; }}
      .stage-title {{ font-size: 26px; font-weight: 700; }}
      .repeat {{ font-size: 24px; font-weight: 700; font-style: italic; }}
      .panel-tag {{ font-size: 22px; }}
      .stage-box {{ fill: none; stroke: #111; stroke-width: 2.2; stroke-dasharray: 8 7; }}
      .patch {{ fill: #fff0cf; stroke: #111; stroke-width: 2; }}
      .block {{ fill: #dcebd2; stroke: #111; stroke-width: 2; }}
      .head {{ fill: #dde8f5; stroke: #275b9f; stroke-width: 2; }}
      .op {{ fill: #f8f8f8; stroke: #111; stroke-width: 2; }}
      .dpe {{ fill: #f7e5d5; stroke: #111; stroke-width: 2; }}
      .mixer {{ fill: #dfcae8; stroke: #111; stroke-width: 2; }}
      .ffn {{ fill: #d8e8f6; stroke: #111; stroke-width: 2; }}
      .note {{ fill: #fff; stroke: #777; stroke-width: 1.3; }}
      .roi-frame {{ fill: #fff; stroke: #111; stroke-width: 1.8; }}
      .arrow {{ stroke: #111; stroke-width: 3; stroke-linecap: round; }}
      .thin {{ stroke: #111; stroke-width: 2; fill: none; }}
    </style>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#111"/>
    </marker>
  </defs>

  <!-- Top row: actual lesion-classification pipeline. -->
  <g>
    {rect(48, 136, 142, 142, "roi-frame", 6)}
    <image href="{roi}" x="57" y="145" width="124" height="124" preserveAspectRatio="xMidYMid slice"/>
    {text(119, 312, "ROI crop", "label")}
    {text(119, 338, "B × 3 × 256 × 256", "shape")}
    {line(202, 207, 286, 207)}
    {stage(318, "Stage 1", "B × 48 × H/4 × W/4", "48 @ 1/4", "×4")}
    {line(563, 207, 616, 207)}
    {stage(628, "Stage 2", "B × 96 × H/8 × W/8", "96 @ 1/8", "×4")}
    {line(873, 207, 926, 207)}
    {stage(938, "Stage 3", "B × 224 × H/16 × W/16", "224 @ 1/16", "×15")}
    {line(1183, 207, 1236, 207)}
    {stage(1248, "Stage 4", "B × 448 × H/32 × W/32", "448 @ 1/32", "×4")}
    {line(1493, 207, 1548, 207)}
    {rect(1566, 137, 118, 56, "head", 8)}
    {text(1625, 172, "GlobalPool", "small")}
    {line(1625, 193, 1625, 228)}
    {rect(1566, 228, 118, 52, "head", 8)}
    {text(1625, 262, "MLP", "small")}
    {text(1625, 314, "1000 → 512 → 2", "shape")}
  </g>

  <!-- Bottom row: representative block only, not a second full network. -->
  <g>
    {rect(180, 430, 1400, 250, "", 22, 'fill="#fffdf8" stroke="#111" stroke-width="2.2"')}
    {text(880, 466, "Representative UL-TransXNet block", "stage-title")}
    {rect(245, 528, 135, 58, "dpe", 9)}
    {text(312, 565, "DPE", "label")}
    {line(380, 557, 430, 557)}
    {rect(440, 528, 110, 58, "op", 9)}
    {text(495, 565, "Norm", "small")}
    {line(550, 557, 590, 557)}
    {rect(600, 528, 210, 58, "mixer", 9)}
    {text(705, 565, "Local / Global mixer", "small")}
    {line(810, 557, 846, 557)}
    <circle cx="858" cy="557" r="13" fill="#fff" stroke="#111" stroke-width="2"/>
    <line x1="846" y1="557" x2="870" y2="557" class="thin"/>
    <line x1="858" y1="545" x2="858" y2="569" class="thin"/>
    {line(870, 557, 925, 557)}
    {rect(936, 528, 110, 58, "op", 9)}
    {text(991, 565, "Norm", "small")}
    {line(1046, 557, 1100, 557)}
    {rect(1112, 528, 170, 58, "ffn", 9)}
    {text(1197, 565, "MS-FFN", "label")}
    {line(1282, 557, 1360, 557)}
    <circle cx="1372" cy="557" r="13" fill="#fff" stroke="#111" stroke-width="2"/>
    <line x1="1360" y1="557" x2="1384" y2="557" class="thin"/>
    <line x1="1372" y1="545" x2="1372" y2="569" class="thin"/>
    {line(1384, 557, 1450, 557)}
    <path d="M 230 557 H 245" class="thin"/>
    <path d="M 380 586 V 630 H 858 V 570" class="thin"/>
    <path d="M 858 570 V 630 H 1372 V 570" class="thin"/>
    {rect(430, 622, 245, 34, "note", 14)}
    {text(552, 645, "G: MCA", "small")}
    {rect(720, 622, 260, 34, "note", 14)}
    {text(850, 645, "GG: MUDD", "small")}
    {rect(1028, 622, 255, 34, "note", 14)}
    {text(1156, 645, "GGG: Diff. attention", "small")}
  </g>

</svg>
"""


def main() -> None:
    svg = build_svg()
    out = OUT_DIR / "architecture_topconf_draft_v1.svg"
    out.write_text(svg, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
