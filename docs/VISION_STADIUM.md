# KASC-inspired stadium visual reconstruction and computer vision

This is an original procedural **illustration** of a football stadium, inspired by the faceted, triangulated roof and seating bowl of King Abdullah Sports City (Al Jawhara) in Jeddah. It does not copy photographs, architectural drawings or original academic project assets. The roof uses 36 repeated geometric facets; the interior includes stylized spectator tiers and a generic football pitch.

## Source references for architectural inspiration

- [Arup — King Abdullah Sports City](https://www.arup.com/en-us/projects/king-abdullah-sports-city/)
- [Saudi Press Agency — stadium imagery](https://www.spa.gov.sa/N2112911)

The six simulated gates, 12 sections, path lengths, obstruction options, and their SVG overlay come from the project's **synthetic routing model**. They do not represent approved entrances, official sections, concourses, accessible routes, event operations or emergency exits. The illustration is not to scale and is not georeferenced.

## Computer Vision Studio

An optional local OpenCV endpoint accepts a PNG, JPEG or WEBP photo that you have permission to analyze. Images are limited to 8 MB and processed in memory.

The algorithm uses:

1. HSV green color thresholding for a possible grass-pitch region.
2. Morphological filtering and connected-component scoring.
3. Gaussian smoothing and Canny edge detection.
4. An annotated output and a normalized green-region bounding-box estimate.

This is **classical color/edge processing**, not a deep neural-network segmentation model, model-trained stadium recognition system, or a verified site survey. Grass-colored spectator areas, landscaping, lighting, and camera angle may produce incorrect detections. The resulting bounding box is **not used to position routes** or change the routing graph.

To enable CV locally:

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[vision]"
```

Launch the app, select **Route intelligence**, and use **Computer Vision Studio**. Avoid exposing the local development server publicly; this experimental endpoint has no authentication or operational upload hardening.

## Implementation

- `static/stadium-kasc-inspired.svg` — original vector stadium illustration.
- `static/app.js` — route overlay, visual-mode switch, and photo-upload interface.
- `src/kasc_v2/vision.py` — OpenCV pitch-region and image-edge analysis.
- `POST /api/vision/analyze` — optional local image-analysis API.
- `tests/test_vision.py` — two optional image-analysis checks.

**To achieve physical accuracy**, a rights-cleared BIM/CAD/floor-plan or survey dataset, official route permissions, scale and coordinate calibration, on-site validation, accessible-path constraints and venue operator approval would be needed.
