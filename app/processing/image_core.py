import io
import os
import cv2
import json
import re
from datetime import datetime, timezone
from PIL import Image, ExifTags
from fastapi import UploadFile

CONFIG_FILE = "config.json"
BASE_CONFIG = None

def load_base_config():
    default = {
        "CONTRAST_LIMIT": 2.5,
        "EDGE_MASK_RADIUS": 0.97,
        "MIN_AREA_MSH": 3.0,  
        "SENSITIVITY_MODIFIER": 0.35, 
        "DENOISE_H": 5
    }
    if not os.path.exists(CONFIG_FILE): return default
    try:
        with open(CONFIG_FILE, 'r') as f:
            user_config = json.load(f)
            if "MIN_AREA_MSH" not in user_config: user_config["MIN_AREA_MSH"] = 3.0
            return {**default, **user_config}
    except: return default

def get_photo_time(path_or_bytes, original_filename, default_time):
    # 1. Приоритет: Поиск даты в названии файла (формат YYYYMMDD_HHMMSS)
    match = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', original_filename)
    if match:
        y, m, d, hh, mm, ss = map(int, match.groups())
        return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)

    # 2. Попытка извлечь EXIF (из пути или из байтов в памяти)
    try:
        if isinstance(path_or_bytes, bytes):
            img = Image.open(io.BytesIO(path_or_bytes))
        else:
            img = Image.open(path_or_bytes)
            
        exif = img.getexif()
        if exif:
            for k, v in exif.items():
                tag = ExifTags.TAGS.get(k, k)
                if tag in ['DateTimeOriginal', 'DateTimeDigitized', 'DateTime']:
                    try:
                        # Стандарт EXIF: YYYY:MM:DD HH:MM:SS
                        dt = datetime.strptime(str(v).strip(), '%Y:%m:%d %H:%M:%S')
                        return dt.replace(tzinfo=timezone.utc)
                    except ValueError: 
                        continue
    except Exception:
        pass

    # 3. Фолбек: если метаданных нет, возвращаем время обработки (default_time)
    return default_time

def natural_sort_key(file: UploadFile): 
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file.filename)]

def get_visual_scales(radius):
    ratio = radius / 600.0
    leg_scl = max(0.5, 0.65 * ratio) 
    grp_scl = max(0.35, 0.5 * ratio)
    spot_scl = max(0.3, 0.45 * ratio)
    leg_thk = 1 if leg_scl < 0.8 else max(2, int(leg_scl * 1.5))
    grp_thk = 1 if grp_scl < 0.8 else max(2, int(grp_scl * 1.5))
    spot_thk = 1 if spot_scl < 0.8 else max(2, int(spot_scl * 1.5))
    return leg_scl, leg_thk, grp_scl, grp_thk, spot_scl, spot_thk

def draw_shadow_text(img, text, pos, font, scale, color, thick):
    x, y = pos
    offset = max(1, int(scale * 1.5)) 
    cv2.putText(img, text, (x + offset, y + offset), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

def draw_legend(img, data_table, radius):
    leg_scl, leg_thk, _, _, _, _ = get_visual_scales(radius)
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = max(18, int(35 * leg_scl))
    padding = max(12, int(20 * leg_scl))
    
    max_w = max([cv2.getTextSize(line, font, leg_scl, leg_thk)[0][0] for line in data_table])
    x0, y0 = padding, padding
    rect_w, rect_h = max_w + padding * 2, len(data_table) * line_height + padding + int(5 * leg_scl)
    
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + rect_w, y0 + rect_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    cy = y0 + padding + int(15 * leg_scl)
    for line in data_table:
        if "|" in line:
            prefix, value = line.split("|")
            pw = cv2.getTextSize(prefix, font, leg_scl, leg_thk)[0][0]
            draw_shadow_text(img, prefix, (x0 + padding, cy), font, leg_scl, (0, 255, 255), leg_thk)
            draw_shadow_text(img, value, (x0 + padding + pw, cy), font, leg_scl, (255, 255, 255), leg_thk)
        else:
            draw_shadow_text(img, line, (x0 + padding, cy), font, leg_scl, (255, 255, 255), leg_thk)
        cy += line_height
    cv2.line(img, (x0 + padding, y0 + rect_h - 2), (x0 + rect_w - padding, y0 + rect_h - 2), (0, 255, 255), leg_thk, cv2.LINE_AA)