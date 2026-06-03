import os
import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import app.processing.image_core as ic

def classify_zurich_advanced(group_spots, radius):
    count = sum(s.get('umbra_count', 1) for s in group_spots)
    if count == 0: return ""
    has_penumbra = any(s.get('has_penumbra', False) for s in group_spots)
    if len(group_spots) == 1 and count == 1: return "H" if has_penumbra else "A"
    if not has_penumbra: return "B"
    min_x = min(cv2.boundingRect(s['contour'])[0] for s in group_spots)
    max_x = max(cv2.boundingRect(s['contour'])[0] + cv2.boundingRect(s['contour'])[2] for s in group_spots)
    extent_deg = ((max_x - min_x) / float(radius)) * 57.2958
    if extent_deg > 15: return "F"
    elif extent_deg > 10: return "E"
    elif extent_deg > 5: return "D"
    else: return "C"

def detect_sunspots_advanced(img, radius, sensitivity="auto", sun_center=None):
    if ic.BASE_CONFIG is None: ic.BASE_CONFIG = ic.load_base_config()
    annotated_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center = sun_center if sun_center else (w//2, h//2)
    mask_disk = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask_disk, center, int(radius * ic.BASE_CONFIG.get("EDGE_MASK_RADIUS", 0.97)), 255, -1)
    
    background = cv2.GaussianBlur(cv2.medianBlur(gray, 5), (int(radius * 0.3) | 1, int(radius * 0.3) | 1), 0)
    sun_details = cv2.bitwise_and(cv2.subtract(background, gray), mask_disk)
    m_val, std_val = cv2.meanStdDev(sun_details, mask=mask_disk)[0][0][0], cv2.meanStdDev(sun_details, mask=mask_disk)[1][0][0]
    
    if str(sensitivity).strip().lower() == "auto":
        mult_halo, used_sens_str = 1.5, "auto (1.5)"
    else:
        try:
            sens_val = float(sensitivity)
            mult_halo, used_sens_str = max(0.5, 1.5 - (sens_val - 50) / 20.0), f"manual ({max(0.5, 1.5 - (sens_val - 50) / 20.0):.1f})"
        except: mult_halo, used_sens_str = 1.5, "auto (1.5)"

    _, mask_halo = cv2.threshold(sun_details, max(5, m_val + (std_val * mult_halo)), 255, cv2.THRESH_BINARY)
    k_open = max(3, int(radius * 0.003) | 1); k_close = max(5, int(radius * 0.008) | 1)
    mask_halo = cv2.morphologyEx(cv2.morphologyEx(mask_halo, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))

    contours_halo, _ = cv2.findContours(mask_halo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask, spots_data, spots_centers = np.zeros_like(mask_halo), [], []
    
    for cnt in contours_halo:
        area_px = float(cv2.contourArea(cnt))
        if area_px <= 0: continue
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        area_msh = float(((area_px / np.sqrt(1.0 - min(0.98, ((cx - center[0])**2 + (cy - center[1])**2) / (radius**2)))) / (2.0 * np.pi * (radius ** 2))) * 1_000_000.0)
        if area_msh < ic.BASE_CONFIG.get("MIN_AREA_MSH", 3.0): continue

        x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(cnt)
        roi_details = sun_details[y_bb:y_bb+h_bb, x_bb:x_bb+w_bb]
        mask_cnt_roi = np.zeros((h_bb, w_bb), dtype=np.uint8)
        cv2.drawContours(mask_cnt_roi, [cnt - [x_bb, y_bb]], -1, 255, -1)
        _, max_val, _, _ = cv2.minMaxLoc(roi_details, mask=mask_cnt_roi)
        
        valid_core_count, core_contours = 0, []
        if max_val >= m_val + (std_val * 3.5):
            _, local_core_mask_roi = cv2.threshold(roi_details, max(m_val + (std_val * 3.5), max_val * 0.55), 255, cv2.THRESH_BINARY)
            local_core_roi = cv2.bitwise_and(local_core_mask_roi, mask_cnt_roi)
            k_core_close = max(3, int(radius * 0.005) | 1)
            local_core_roi = cv2.morphologyEx(cv2.morphologyEx(local_core_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_core_close, k_core_close))), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))
            core_contours_roi, _ = cv2.findContours(local_core_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for uc_roi in core_contours_roi:
                if cv2.contourArea(uc_roi) >= 1.0:
                    valid_core_count += 1; core_contours.append(uc_roi + [x_bb, y_bb])

        if valid_core_count == 0:
            perimeter = cv2.arcLength(cnt, True)
            if max_val < m_val + (std_val * 2.5) or (4 * np.pi * (area_px / (perimeter * perimeter)) if perimeter > 0 else 0) < 0.25: continue

        cv2.drawContours(final_mask, [cnt], -1, 255, -1)
        spots_centers.append([cx, cy])
        spots_data.append({"x": cx, "y": cy, "area": area_msh, "umbra": valid_core_count > 0, "has_penumbra": valid_core_count > 0 and (area_px > sum(cv2.contourArea(uc) for uc in core_contours) * 1.5), "umbra_count": max(1, valid_core_count), "contour": cnt, "class": ""})
        cv2.drawContours(annotated_img, [cnt], -1, (0, 255, 0), 1)
        cv2.drawContours(annotated_img, core_contours, -1, (0, 0, 255), 1)

    total_S, g, group_rects = len(spots_data), 0, []
    _, _, grp_scl, grp_thk, _, _ = ic.get_visual_scales(radius)
    pad = max(5, int(15 * (radius / 600.0)))
    font = cv2.FONT_HERSHEY_SIMPLEX

    if len(spots_centers) > 0:
        labels = fcluster(linkage(spots_centers, method='single'), t=radius*0.10, criterion='distance') if len(spots_centers) > 1 else [1]
        g = len(set(labels))
        groups = defaultdict(list)
        for i, spot in enumerate(spots_data):
            spot["group_id"] = labels[i]; groups[labels[i]].append(spot)

        for grp_id, group_spots in groups.items():
            grp_class = classify_zurich_advanced(group_spots, radius)
            all_pts = []
            for spot in group_spots:
                spot["class"] = grp_class; all_pts.append(spot["contour"])
                
            if all_pts:
                gx, gy, gw, gh = cv2.boundingRect(np.vstack(all_pts))
                x1, y1, x2, y2 = max(0, gx - pad), max(0, gy - pad), min(w, gx + gw + pad), min(h, gy + gh + pad)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 255, 0), grp_thk, cv2.LINE_AA)
                grp_txt = f"Group: {grp_class}"
                tw, th = cv2.getTextSize(grp_txt, font, grp_scl, grp_thk)[0]
                tx, ty = x1, y1 - int(6 * grp_scl)
                ic.draw_shadow_text(annotated_img, grp_txt, (tx, ty), font, grp_scl, (255, 255, 0), grp_thk)
                group_rects.extend([(x1, y2, x2 - x1, y2 - y1), (tx, ty, tw, th)])

    return annotated_img, total_S, g, int(10 * g + total_S), final_mask, spots_data, float(sum(spot["area"] for spot in spots_data)), used_sens_str, group_rects

def process_sun(input_path, task_dir, photo_idx, sensitivity):
    img = cv2.imread(input_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, disk_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(disk_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    center, rad, m = (int(x), int(y)), int(r), int(r * 0.05) 
    crop = img[max(0, center[1]-rad-m):center[1]+rad+m, max(0, center[0]-rad-m):center[0]+rad+m]
    p1 = os.path.join(task_dir, f"img_{photo_idx}_1_cropped.png")
    cv2.imwrite(p1, crop)
    final_img_raw, s, g, wolf, mask_img, spots_list, total_area, used_sens_str, grp_rects = detect_sunspots_advanced(crop, rad, sensitivity)
    p2, p3 = os.path.join(task_dir, f"img_{photo_idx}_2_final.png"), os.path.join(task_dir, f"img_{photo_idx}_3_mask.png")
    cv2.imwrite(p3, mask_img)
    return p1, p2, p3, rad, crop.shape[1]//2, crop.shape[0]//2, s, g, wolf, float(total_area), spots_list, final_img_raw, used_sens_str, grp_rects