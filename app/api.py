import os
import cv2
import numpy as np
import math
import traceback
from datetime import datetime, timedelta, timezone
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
import psycopg2.extras

from app.database import init_db_pool, close_db_pool, init_db_tables, get_db_connection
from app.auth import verify_token
from app.utils.logger import log_action
import app.processing.image_core as ic
from app.processing.sunspots import process_sun, detect_sunspots_advanced
from app.processing.tracking import SpotTracker

BASE_RESULT_DIR = "results"
os.makedirs(BASE_RESULT_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ic.BASE_CONFIG = ic.load_base_config()
    init_db_pool()
    init_db_tables()
    yield
    close_db_pool()

app = FastAPI(title="Solar Spotter Pro", lifespan=lifespan)

@app.post("/tasks")
async def create_task(
    photos: List[UploadFile] = File(...), msg: str = Form(""), sensitivity: str = Form("auto"), 
    save_to_db: bool = Form(True), current_user: dict = Depends(verify_token)
):
    tid = int(datetime.now(timezone.utc).timestamp()) 
    user_id = current_user['id']
    
    if save_to_db:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO tasks (user_id, message) VALUES (%s, %s) RETURNING id", (user_id, msg))
                tid = cur.fetchone()[0]
            conn.commit()

    try:
        task_folder = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_task_{tid}"
        task_dir = os.path.join(BASE_RESULT_DIR, task_folder)
        os.makedirs(task_dir, exist_ok=True)

        batch_results, prev_spots_data, last_photo_time, mock_track_id = [], [], None, 1

        if save_to_db:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, photo_time, sun_radius FROM photo_results ORDER BY photo_time DESC LIMIT 1")
                    last_record = cur.fetchone()
                    if last_record:
                        last_photo_time = last_record[1]
                        cur.execute("SELECT track_id, x, y FROM sunspots WHERE photo_result_id = %s", (last_record[0],))
                        for row in cur.fetchall(): prev_spots_data.append({'track_id': row[0], 'x': row[1], 'y': row[2], 'rad': last_record[2]})
                        cur.execute("SELECT MAX(id) FROM spot_tracks")
                        if max_t := cur.fetchone()[0]: mock_track_id = max_t + 1

        for idx, photo in enumerate(sorted(photos, key=ic.natural_sort_key)):
            temp_path = f"temp_{tid}_{idx}.jpg"
            with open(temp_path, "wb") as f: f.write(await photo.read())
            try:
                current_time = ic.get_photo_time(temp_path, photo.filename, last_photo_time + timedelta(days=1) if last_photo_time else datetime.now(timezone.utc))
                delta_t = (current_time - last_photo_time).total_seconds() / 86400.0 if last_photo_time else 0.0
                last_photo_time = current_time

                res = process_sun(temp_path, task_dir, idx, sensitivity)
                if not res:
                    if save_to_db: log_action(user_id, "ERROR", f"File {photo.filename}: sun not found", tid)
                    continue

                p1, p2, p3, rad, lcx, lcy, s, g, wolf, total_area, spots_list, final_img, used_sens_str, grp_rects = res
                
                rid = -1
                if save_to_db:
                    with get_db_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("INSERT INTO photo_results (task_id, file_name, status, cropped_path, final_path, mask_path, sun_radius, total_area, wolf_number, spots_count, groups_count, photo_index, photo_time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id", (tid, photo.filename, "COMPLETED", p1, p2, p3, rad, total_area, wolf, s, g, idx, current_time))
                            rid = cur.fetchone()[0]
                        conn.commit()

                for ps in prev_spots_data:
                    if 'rad' in ps and ps['rad'] != rad:
                        prev_cx = prev_cy = int(ps['rad'] * 1.05)
                        ps['x'], ps['y'], ps['rad'] = int(lcx + ((ps['x'] - prev_cx) / float(ps['rad'])) * rad), int(lcy + ((ps['y'] - prev_cy) / float(ps['rad'])) * rad), rad 

                curr_spots = SpotTracker(rad, (lcx, lcy)).match_frames(prev_spots_data, spots_list, delta_t)
                split_events, drawn_text_rects = {}, grp_rects.copy() 
                _, _, _, _, spot_scl, spot_thk = ic.get_visual_scales(rad)
                font = cv2.FONT_HERSHEY_SIMPLEX

                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        for spot in curr_spots:
                            track_id = spot.get('track_id')
                            if not track_id:
                                if save_to_db:
                                    cur.execute("INSERT INTO spot_tracks (task_id, first_photo_id) VALUES (%s, %s) RETURNING id", (tid, rid))
                                    track_id = cur.fetchone()[0]
                                else: track_id, mock_track_id = mock_track_id, mock_track_id + 1
                                spot['track_id'] = track_id
                                if 'parent_track' in spot: split_events.setdefault(spot['parent_track'], []).append(track_id)

                            if save_to_db:
                                if 'parent_tracks' in spot: cur.execute("INSERT INTO spot_events (task_id, photo_result_id, event_type, involved_tracks, description) VALUES (%s, %s, %s, %s, %s)", (tid, rid, 'merge', spot['parent_tracks'] + [track_id], f"Merge into {track_id}"))
                                cur.execute("INSERT INTO sunspots (photo_result_id, track_id, x, y, area, is_umbra, class) VALUES (%s, %s, %s, %s, %s, %s, %s)", (rid, track_id, int(spot['x']), int(spot['y']), float(spot['area']), bool(spot['umbra']), spot.get('class', '')))

                            label_text = f"#{track_id} ({spot.get('class', 'A')}) {spot['area']:.1f}" + (f" (merge: #{', #'.join(map(str, spot['parent_tracks']))})" if 'parent_tracks' in spot else (f" (split: #{spot['parent_track']})" if 'parent_track' in spot else ""))
                            tw, th = cv2.getTextSize(label_text, font, spot_scl, spot_thk)[0]
                            tx, ty, placed = spot['x'], spot['y'], False
                            
                            for r_dist in range(max(10, int(20 * spot_scl)), int(rad * 0.8), max(10, int(20 * spot_scl))): 
                                for angle_deg in range(0, 360, 20):
                                    cand_tx, cand_ty, overlap, padd = int(spot['x'] + r_dist * math.cos(math.radians(angle_deg))), int(spot['y'] + r_dist * math.sin(math.radians(angle_deg))), False, max(2, int(4 * spot_scl))
                                    for (rx, ry, rw, rh) in drawn_text_rects:
                                        if not (cand_tx + tw + padd < rx or cand_tx - padd > rx + rw or cand_ty - th - padd > ry or cand_ty + padd < ry - rh): overlap = True; break
                                    if not overlap:
                                        for os_ in curr_spots:
                                            if (cand_tx - padd < os_['x'] < cand_tx + tw + padd) and (cand_ty - th - padd < os_['y'] < cand_ty + padd): overlap = True; break
                                    if not overlap: tx, ty, placed = cand_tx, cand_ty, True; break
                                if placed: break
                            if not placed: tx, ty = spot['x'] + int(15 * spot_scl), spot['y'] - int(15 * spot_scl)

                            drawn_text_rects.append((tx, ty, tw, th))
                            cv2.line(final_img, (spot['x'], spot['y']), (tx, ty), (120, 120, 120), spot_thk, cv2.LINE_AA)
                            ic.draw_shadow_text(final_img, label_text, (tx, ty), font, spot_scl, (0, 255, 255), spot_thk)

                        if save_to_db:
                            for parent, children in split_events.items(): cur.execute("INSERT INTO spot_events (task_id, photo_result_id, event_type, involved_tracks, description) VALUES (%s, %s, %s, %s, %s)", (tid, rid, 'split', [parent] + children, f"Split {parent}"))
                            if any('parent_tracks' in sp for sp in curr_spots): log_action(user_id, "INFO", f"Merge in {photo.filename}", tid)
                    conn.commit()

                data_table = [f"FILE:| {photo.filename}", f"DATE (UTC):| {current_time.strftime('%Y-%m-%d')}", f"TIME (UTC):| {current_time.strftime('%H:%M:%S')}", f"SIZE:| {total_area:.1f} MSH", f"WOLF NUMBER:| {wolf}", f"SUN RADIUS:| {rad} px", f"SPOTS:| {s}", f"GROUPS:| {g}", f"SENSITIVITY:| {used_sens_str}"]
                if not save_to_db: data_table.append("DB SAVE:| OFF (Test Mode)")
                ic.draw_legend(final_img, data_table, rad)
                cv2.imwrite(p2, final_img)
                prev_spots_data = [{'track_id': sp['track_id'], 'x': sp['x'], 'y': sp['y'], 'rad': rad} for sp in curr_spots]
                batch_results.append({"file": photo.filename, "wolf": wolf, "total_area": round(total_area, 2), "spots": s})
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)

        if save_to_db: log_action(user_id, "SUCCESS", f"Done {len(batch_results)}", tid)
        return {"task_id": tid, "total_spots_found": sum(r['spots'] for r in batch_results), "summary": batch_results}
    except Exception as e:
        print(traceback.format_exc()) 
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/diurnal_parallel")
async def calculate_diurnal_parallel(photos: List[UploadFile] = File(...), current_user: dict = Depends(verify_token)):
    if len(photos) < 2: raise HTTPException(status_code=400, detail="Requires >= 2 photos")
    tid = int(datetime.now(timezone.utc).timestamp())
    task_dir = os.path.join(BASE_RESULT_DIR, f"diurnal_task_{tid}"); os.makedirs(task_dir, exist_ok=True)
    sorted_photos = sorted(photos, key=ic.natural_sort_key)
    centers, radii, images_in_ram, photo_times = [], [], [], []

    for photo in sorted_photos:
        content = await photo.read()
        photo_times.append(ic.get_photo_time(content, photo.filename, datetime.now(timezone.utc)))
        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if img is None: centers.append(None); radii.append(None); images_in_ram.append(None); continue
        images_in_ram.append(img)
        cnts, _ = cv2.findContours(cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            (x, y), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
            centers.append((int(x), int(y))); radii.append(int(r))
        else: centers.append(None); radii.append(None)

    valid_centers = [c for c in centers if c is not None]
    if len(valid_centers) < 2: raise HTTPException(status_code=400, detail="Not enough valid sun disks")
    angle_deg, results_info = np.degrees(np.arctan2(valid_centers[-1][1] - valid_centers[0][1], valid_centers[-1][0] - valid_centers[0][0])), []

    for idx, photo in enumerate(sorted_photos):
        if centers[idx] is None: continue
        img, cx, cy, R = images_in_ram[idx], centers[idx][0], centers[idx][1], radii[idx]
        h, w = img.shape[:2]
        pad_rot = int(R * 1.3)
        raw_crop = img[max(0, cy - pad_rot):min(h, cy + pad_rot), max(0, cx - pad_rot):min(w, cx + pad_rot)]
        images_in_ram[idx], new_cx_raw, new_cy_raw = None, cx - max(0, cx - pad_rot), cy - max(0, cy - pad_rot)
        
        try: _, s_c, g_c, wolf_num, _, _, _, used_sens_str, _ = detect_sunspots_advanced(raw_crop, R, "auto", sun_center=(new_cx_raw, new_cy_raw))
        except: s_c, g_c, wolf_num, used_sens_str = 0, 0, 0, "auto (1.5)"
        
        rotated_crop = cv2.warpAffine(raw_crop, cv2.getRotationMatrix2D((new_cx_raw, new_cy_raw), angle_deg, 1.0), (raw_crop.shape[1], raw_crop.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        padding = int(R * 0.4) 
        y1, x1 = max(0, new_cy_raw - R - padding), max(0, new_cx_raw - R - padding)
        cropped, new_cx, new_cy, ptime = rotated_crop[y1:min(rotated_crop.shape[0], new_cy_raw + R + padding), x1:min(rotated_crop.shape[1], new_cx_raw + R + padding)], new_cx_raw - x1, new_cy_raw - y1, photo_times[idx]
        
        b_s, line_color = max(0.5, R / 600.0), (150, 150, 150)
        cv2.line(cropped, (0, new_cy), (new_cx - R, new_cy), line_color, max(1, int(1 * b_s)), cv2.LINE_AA)
        cv2.line(cropped, (new_cx + R, new_cy), (cropped.shape[1], new_cy), line_color, max(1, int(1 * b_s)), cv2.LINE_AA)
        
        font, dir_scale = cv2.FONT_HERSHEY_COMPLEX, max(0.6, b_s * 1.2)
        dir_thick, off = 1 if dir_scale < 0.8 else max(2, int(dir_scale * 1.5)), int(R + 40 * dir_scale) 
        
        for t, pt in [("E", (new_cx - off, new_cy)), ("W", (new_cx + off, new_cy)), ("N", (new_cx, new_cy - off)), ("S", (new_cx, new_cy + off))]:
            tw, th = cv2.getTextSize(t, font, dir_scale, dir_thick)[0]
            ic.draw_shadow_text(cropped, t, (pt[0] - tw//2, pt[1] + th//2), font, dir_scale, line_color, dir_thick)
            
        dt_scale, dt_thick = max(0.5, b_s * 0.9), 1 if max(0.5, b_s * 0.9) < 0.8 else max(2, int(max(0.5, b_s * 0.9) * 1.5))
        d_str, t_str = ptime.strftime("%Y-%m-%d"), ptime.strftime("%H:%M:%S UTC")
        tw1, th1 = cv2.getTextSize(d_str, font, dt_scale, dt_thick)[0]
        tw2, th2 = cv2.getTextSize(t_str, font, dt_scale, dt_thick)[0]
        pad_x, pad_y, gap = int(30 * b_s), int(30 * b_s), int(12 * b_s)
        
        ic.draw_shadow_text(cropped, d_str, (cropped.shape[1] - tw1 - pad_x, cropped.shape[0] - th2 - pad_y - gap - th1), font, dt_scale, line_color, dt_thick)
        ic.draw_shadow_text(cropped, t_str, (cropped.shape[1] - tw2 - pad_x, cropped.shape[0] - pad_y), font, dt_scale, line_color, dt_thick)
        
        ic.draw_legend(cropped, [f"FILE:| {photo.filename}", f"DRIFT ANGLE:| {angle_deg:.2f} DEG", f"WOLF NUMBER:| {wolf_num}", f"SPOTS/GROUPS:| {s_c}/{g_c}", f"SUN RADIUS:| {R} px", f"SENSITIVITY:| {used_sens_str}", f"STATUS:| De-rotated"], R)
        out_filename = f"diurnal_{idx}_final.jpg"
        cv2.imwrite(os.path.join(task_dir, out_filename), cropped)
        results_info.append({"file": photo.filename, "url": f"/images/diurnal_task_{tid}/{out_filename}", "center": {"x": new_cx, "y": new_cy}})

    log_action(current_user['id'], "INFO", f"Diurnal parallel calculated by {current_user['username']}", tid)
    return {"message": "Diurnal parallel calculated", "drift_angle_degrees": float(round(angle_deg, 4)), "task_id": f"diurnal_task_{tid}", "images": results_info}

@app.get("/images/{task_folder}/{filename}")
async def get_image(task_folder: str, filename: str, current_user: dict = Depends(verify_token)):
    path = os.path.join(BASE_RESULT_DIR, task_folder, filename)
    if os.path.exists(path): return FileResponse(path)
    raise HTTPException(status_code=404)

@app.get("/tasks/{task_id}")
async def get_task(task_id: int, current_user: dict = Depends(verify_token)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM tasks WHERE id = %s", (task_id,))
                if not (task := cur.fetchone()): raise HTTPException(status_code=404)
                cur.execute("SELECT * FROM photo_results WHERE task_id = %s ORDER BY photo_index", (task_id,))
                return {"task": task, "results": cur.fetchall()}
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail="Database error")