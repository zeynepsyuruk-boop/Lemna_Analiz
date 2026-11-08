"""Single-image analysis: dish detection, plant segmentation, frond counting, colour indices."""

import cv2
import numpy as np


_REDUCED_FLAGS = [(8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4), (2, cv2.IMREAD_REDUCED_COLOR_2)]


def read_image(path, max_side=None):
    """Read an image, including paths with non-ASCII characters.

    With max_side, large images are decoded directly at 1/2, 1/4 or 1/8 size as long
    as the long side stays at or above max_side. This is much faster for phone photos.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_COLOR
    if max_side:
        try:
            from PIL import Image
            with Image.open(path) as im:
                long_side = max(im.size)
            for factor, reduced in _REDUCED_FLAGS:
                if long_side / factor >= max_side:
                    flag = reduced
                    break
        except Exception:
            pass
    image = cv2.imdecode(data, flag)
    if image is None:
        raise ValueError(f"Image could not be read: {path}")
    return image


def resize_to_work(image, max_side):
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return image, scale


def gray_world(image):
    img = image.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1e-6)
    return np.clip(img * gain, 0, 255).astype(np.uint8)


def find_dish(image, dish_cfg):
    """Return (cx, cy, r) of the analysed circle in pixels and the detection method used."""
    h, w = image.shape[:2]
    mode = dish_cfg.get("mode", "auto")
    if mode == "fixed":
        r = dish_cfg["fixed_r"] * min(h, w)
        return (dish_cfg["fixed_cx"] * w, dish_cfg["fixed_cy"] * h, r), "fixed"
    if mode == "full":
        return None, "full"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small_scale = 600.0 / max(h, w)
    small = cv2.resize(gray, None, fx=small_scale, fy=small_scale, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    m = min(small.shape)
    circles = None
    # A low-contrast dish rim gives fewer votes, so the accumulator threshold is lowered stepwise.
    for votes in (40, 30, 22):
        circles = cv2.HoughCircles(
            small, cv2.HOUGH_GRADIENT, dp=1.5, minDist=m,
            param1=60, param2=votes, minRadius=int(0.25 * m), maxRadius=int(0.55 * m),
        )
        if circles is not None:
            break
    if circles is None:
        r = dish_cfg["fixed_r"] * min(h, w)
        return (dish_cfg["fixed_cx"] * w, dish_cfg["fixed_cy"] * h, r), "fixed (auto failed)"
    cx, cy, r = (float(v) for v in circles[0][0])
    return (cx / small_scale, cy / small_scale, r / small_scale), "auto"


def roi_mask(shape, circle, margin):
    h, w = shape[:2]
    if circle is None:
        return np.full((h, w), 255, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cx, cy, r = circle
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r * margin)), 255, -1)
    return mask


def _clean(mask, morph):
    k_open = int(morph["open_kernel_px"])
    k_close = int(morph["close_kernel_px"])
    if k_open > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    if k_close > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(morph["min_object_area_px"])
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def refine_edges(image, plant, roi):
    """Reclassify the outer pixels of the plant mask with a 50 % mixing rule.

    Edge pixels mix plant and background colour (and JPEG stores colour at reduced
    resolution), so HSV thresholds also accept pixels that are mostly background.
    Each edge pixel is projected in Lab space onto the line between the local
    background colour and the local plant colour and kept when it lies at least
    half-way towards the plant colour.
    """
    core = cv2.erode(plant, np.ones((3, 3), np.uint8), iterations=2)
    ring = (plant > 0) & (core == 0)
    background = (roi > 0) & (cv2.dilate(plant, np.ones((7, 7), np.uint8), iterations=2) == 0)
    if not ring.any() or core.max() == 0 or background.sum() < 100:
        return plant

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    def local_mean(mask, size):
        m = mask.astype(np.float32)
        weight = cv2.boxFilter(m, -1, (size, size), normalize=False)
        total = cv2.boxFilter(lab * m[..., None], -1, (size, size), normalize=False)
        return total / np.maximum(weight, 1e-6)[..., None], weight

    bg_ref, bg_w = local_mean(background, 41)
    bg_ref[bg_w < 1] = np.median(lab[background], axis=0)
    pl_ref, pl_w = local_mean(core > 0, 15)

    ys, xs = np.nonzero(ring)
    px, bg, pl = lab[ys, xs], bg_ref[ys, xs], pl_ref[ys, xs]
    span = pl - bg
    span2 = (span ** 2).sum(axis=1)
    frac = ((px - bg) * span).sum(axis=1) / np.maximum(span2, 1e-6)
    # Keep the pixel when no plant reference is nearby or plant and background are too similar to judge.
    keep = (frac >= 0.5) | (pl_w[ys, xs] < 1) | (span2 < 100)
    out = core.copy()
    out[ys[keep], xs[keep]] = 255
    return out


def segment(image, roi, hsv_cfg, morph):
    """Return masks for green, yellow and brown tissue and for the whole plant."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def rng(name):
        return cv2.inRange(hsv, np.array(hsv_cfg[f"{name}_low"]), np.array(hsv_cfg[f"{name}_high"]))

    green = cv2.bitwise_and(rng("green"), roi)
    yellow = cv2.bitwise_and(rng("yellow"), roi)
    brown = cv2.bitwise_and(rng("brown"), roi)
    plant = _clean(cv2.bitwise_or(cv2.bitwise_or(green, yellow), brown), morph)
    if morph.get("refine_edges", True):
        plant = refine_edges(image, plant, roi)
    green = cv2.bitwise_and(green, plant)
    yellow = cv2.bitwise_and(cv2.bitwise_and(yellow, plant), cv2.bitwise_not(green))
    brown = cv2.bitwise_and(brown, plant)
    brown = cv2.bitwise_and(brown, cv2.bitwise_not(cv2.bitwise_or(green, yellow)))
    return {"plant": plant, "green": green, "yellow": yellow, "brown": brown}


def count_fronds(plant_mask, peak_ratio, min_frond_area, seed_window=1.2):
    """Estimate the number of fronds with a distance-transform watershed.

    The typical frond half-width r0 is taken as the median of the largest
    distance-transform value of each object. Local maxima of the distance map that
    reach peak_ratio * r0 within a window of seed_window * r0 are used as frond seeds,
    so touching or partly overlapping fronds are separated.
    Returns the count and a label image (0 = background).
    """
    n, comp_labels, stats, _ = cv2.connectedComponentsWithStats(plant_mask, connectivity=8)
    if n <= 1:
        return 0, np.zeros(plant_mask.shape, np.int32)
    dist = cv2.distanceTransform(plant_mask, cv2.DIST_L2, 5)

    # Largest distance value of each object and its position, using object pixels only.
    ys, xs = np.nonzero(comp_labels)
    lab = comp_labels[ys, xs]
    order = np.lexsort((dist[ys, xs], lab))
    last = np.r_[np.nonzero(np.diff(lab[order]))[0], len(order) - 1]
    top = order[last]
    comp_max = dist[ys[top], xs[top]]
    r0 = float(np.median(comp_max))
    k = max(3, int(round(seed_window * r0)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    local_max = dist >= cv2.dilate(dist, kernel) - 0.5
    peaks = (local_max & (dist >= peak_ratio * r0) & (plant_mask > 0)).astype(np.uint8)
    peaks = cv2.dilate(peaks, np.ones((3, 3), np.uint8)) & (plant_mask > 0)
    n_peaks, markers = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)

    # Objects without a seed (very small fronds) receive one at their maximum.
    seeded = np.zeros(n, bool)
    seeded[np.unique(comp_labels[markers > 0])] = True
    next_label = n_peaks
    for i in np.where(~seeded[1:])[0]:
        markers[ys[top[i]], xs[top[i]]] = next_label
        next_label += 1

    background = next_label
    markers[plant_mask == 0] = background
    relief = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.watershed(cv2.cvtColor(255 - relief, cv2.COLOR_GRAY2BGR), markers)
    labels = np.where((markers > 0) & (markers < background) & (plant_mask > 0), markers, 0)

    # Regions smaller than min_frond_area count only when they form an object of their own.
    areas = np.bincount(labels.ravel(), minlength=background)
    count = int(np.count_nonzero(areas[1:] >= min_frond_area))
    small = np.where((areas > 0) & (areas < min_frond_area))[0]
    small = small[small > 0]
    if len(small):
        inside = labels > 0
        pairs = np.unique(comp_labels[inside].astype(np.int64) * background + labels[inside])
        comp_of_label = np.zeros(background, np.int64)
        comp_of_label[pairs % background] = pairs // background
        regions_per_comp = np.bincount(pairs // background, minlength=n)
        count += int(np.count_nonzero(regions_per_comp[comp_of_label[small]] == 1))
    return count, labels


def colour_indices(image, masks):
    """Mean colour descriptors of the plant tissue."""
    plant = masks["plant"] > 0
    n = int(plant.sum())
    if n == 0:
        return {"mean_hue": np.nan, "mean_saturation": np.nan, "mean_value": np.nan, "exg_index": np.nan}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = [c[plant].astype(np.float64) for c in cv2.split(image)]
    total = np.maximum(r + g + b, 1.0)
    exg = 2 * g / total - r / total - b / total
    return {
        "mean_hue": float(hsv[..., 0][plant].mean()),
        "mean_saturation": float(hsv[..., 1][plant].mean()),
        "mean_value": float(hsv[..., 2][plant].mean()),
        "exg_index": float(exg.mean()),
    }


def analyse_image(image_bgr, cfg):
    """Analyse one dish photo. Returns (measurements dict, overlay image, masks)."""
    image, _ = resize_to_work(image_bgr, cfg["work_max_side_px"])
    if cfg.get("white_balance") == "grayworld":
        image = gray_world(image)

    dish_cfg = cfg["dish"]
    circle, method = find_dish(image, dish_cfg)
    roi = roi_mask(image.shape, circle, dish_cfg["margin_fraction"])
    masks = segment(image, roi, cfg["hsv"], cfg["morphology"])

    roi_px = int(np.count_nonzero(roi))
    plant_px = int(np.count_nonzero(masks["plant"]))
    green_px = int(np.count_nonzero(masks["green"]))
    yellow_px = int(np.count_nonzero(masks["yellow"]))
    brown_px = int(np.count_nonzero(masks["brown"]))

    mm_per_px = np.nan
    if circle is not None and dish_cfg.get("inner_diameter_mm"):
        mm_per_px = float(dish_cfg["inner_diameter_mm"]) / (2.0 * circle[2])
    px_area_mm2 = mm_per_px ** 2

    fc = cfg["frond_count"]
    frond_count, labels = count_fronds(masks["plant"], fc["peak_ratio"], fc["min_frond_area_px"],
                                       fc.get("seed_window", 1.2))

    frond_count_area = np.nan
    if fc.get("reference_frond_area_mm2") and np.isfinite(px_area_mm2):
        frond_count_area = plant_px * px_area_mm2 / float(fc["reference_frond_area_mm2"])

    result = {
        "dish_method": method,
        "image_width_px": image.shape[1],
        "image_height_px": image.shape[0],
        "roi_area_px": roi_px,
        "plant_area_px": plant_px,
        "green_area_px": green_px,
        "yellow_area_px": yellow_px,
        "brown_area_px": brown_px,
        "green_pct_of_roi": 100.0 * green_px / roi_px if roi_px else np.nan,
        "plant_pct_of_roi": 100.0 * plant_px / roi_px if roi_px else np.nan,
        "yellow_pct_of_plant": 100.0 * yellow_px / plant_px if plant_px else np.nan,
        "brown_pct_of_plant": 100.0 * brown_px / plant_px if plant_px else np.nan,
        "mm_per_px": mm_per_px,
        "plant_area_mm2": plant_px * px_area_mm2,
        "green_area_mm2": green_px * px_area_mm2,
        "frond_count_auto": frond_count,
        "frond_count_from_area": frond_count_area,
    }
    result.update(colour_indices(image, masks))
    overlay = draw_overlay(image, circle, dish_cfg["margin_fraction"], masks, labels, result)
    return result, overlay, masks


def draw_overlay(image, circle, margin, masks, labels, result):
    out = image.copy()
    tint = out.copy()
    tint[masks["green"] > 0] = (0, 200, 0)
    tint[masks["yellow"] > 0] = (0, 220, 255)
    tint[masks["brown"] > 0] = (30, 60, 140)
    out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)

    if labels is not None and labels.max() > 0:
        edges = np.zeros(labels.shape, np.uint8)
        lab16 = labels.astype(np.uint16)
        grad = cv2.morphologyEx(lab16, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        edges[(grad > 0) & (labels > 0)] = 255
        out[edges > 0] = (255, 0, 255)
        ys, xs = np.nonzero(labels)
        lab = labels[ys, xs]
        npx = np.bincount(lab)
        keep = np.nonzero(npx)[0]
        cx = np.bincount(lab, weights=xs)[keep] / npx[keep]
        cy = np.bincount(lab, weights=ys)[keep] / npx[keep]
        for x, y in zip(cx, cy):
            cv2.circle(out, (int(x), int(y)), 3, (255, 255, 255), -1)

    if circle is not None:
        cx, cy, r = circle
        cv2.circle(out, (int(cx), int(cy)), int(r), (255, 128, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), int(r * margin), (255, 255, 0), 1)

    text = [
        f"plant area: {result['plant_area_px']} px",
        f"green % of dish: {result['green_pct_of_roi']:.2f}",
        f"yellow % of plant: {result['yellow_pct_of_plant']:.1f}" if np.isfinite(result["yellow_pct_of_plant"]) else "yellow % of plant: -",
        f"fronds (auto): {result['frond_count_auto']}",
    ]
    if np.isfinite(result["plant_area_mm2"]):
        text.insert(1, f"plant area: {result['plant_area_mm2']:.1f} mm2")
    draw_text_panel(out, text)
    return out


def draw_text_panel(image, lines):
    """White text on a semi-transparent dark panel in the top-left corner.

    The font size follows the image width so that the text stays readable when the
    image is shown scaled down.
    """
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.5, w / 1400.0)
    thick = max(1, int(round(1.6 * scale)))
    sizes = [cv2.getTextSize(line, font, scale, thick)[0] for line in lines]
    line_h = int(max(s[1] for s in sizes) * 1.8)
    pad = int(10 * scale)
    x1 = min(w, pad * 2 + max(s[0] for s in sizes))
    y1 = min(h, pad * 2 + line_h * len(lines))
    roi = image[0:y1, 0:x1]
    image[0:y1, 0:x1] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)
    y = pad + int(line_h * 0.75)
    for line in lines:
        cv2.putText(image, line, (pad, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += line_h
    return image
