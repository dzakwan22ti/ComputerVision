from flask import Flask, render_template, Response, jsonify
import cv2
import time
import threading
import platform
import subprocess
import tempfile
import os
import numpy as np

app = Flask(__name__)

TTS_AVAILABLE = False
_voice_files  = {}

def play_async(key):
    """Putar file suara (.wav) yang sudah di-generate sebelumnya berdasarkan key
    ('bungkuk' / 'peregangan'). Dipanggil berkali-kali TIDAK masalah — beda dari
    TTS live (pyttsx3.say()+runAndWait()) yang gampang macet kalau dipanggil
    berulang, sini murni memutar file audio."""
    if not TTS_AVAILABLE or key not in _voice_files:
        return
    path = _voice_files[key]
    system = platform.system()
    try:
        if system == "Windows":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif system == "Darwin":
            subprocess.Popen(["afplay", path])
        else:
            subprocess.Popen(["aplay", path])
    except Exception as e:
        print(f"[VOICE PLAY ERROR] {e}")

from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   
                                           
SPINE_BUNGKUK_THRESHOLD  = 15.0   
NECK_BUNGKUK_THRESHOLD   = 25.0   
HIP_CONF_MIN             = 0.25  
TEGAP_DEAD_ZONE          = 3.0    
RAW_STATUS_MIN_HOLD_SEC  = 0.6    

HADAP_DEPAN_RATIO        = 0.30

BUNGKUK_CONFIRM_SEC      = 5.0   
TEGAP_CONFIRM_SEC        = 2.0   
DUDUK_ABSENT_FREEZE_SEC  = 5.0    
DUDUK_CONFIRM_SEC        = 1.5   

KAMERA_TOLERANSI_SEC     = 3.0

PEREGANGAN_FIRST_SEC     = 60     
PEREGANGAN_REPEAT_SEC    = 30    
PEREGANGAN_TEXT          = "Sudah waktunya peregangan, silakan berdiri sejenak."

SUARA_BUNGKUK_TEXT       = "Anda membungkuk, tegakkan punggung Anda"
SUARA_DELAY_SEC          = 6.0   

COLOR_GREEN  = (0, 210, 0)
COLOR_RED    = (0, 0, 220)
COLOR_CYAN   = (200, 200, 0)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (160, 160, 160)
COLOR_ORANGE = (0, 140, 220)

def _generate_voice_files():
    global TTS_AVAILABLE, _voice_files
    try:
        import pyttsx3
        voice_dir = tempfile.mkdtemp(prefix="posture_voice_")
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)

        phrases = {
            "bungkuk"   : SUARA_BUNGKUK_TEXT,
            "peregangan": PEREGANGAN_TEXT,
        }
        for key, text in phrases.items():
            path = os.path.join(voice_dir, f"{key}.wav")
            engine.save_to_file(text, path)
        engine.runAndWait()  

        for key in phrases:
            path = os.path.join(voice_dir, f"{key}.wav")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                _voice_files[key] = path
            else:
                print(f"[TTS WARNING] File suara '{key}' gagal dibuat: {path}")

        TTS_AVAILABLE = len(_voice_files) > 0
        if TTS_AVAILABLE:
            print(f"[TTS] Berhasil generate {len(_voice_files)} file suara di {voice_dir}")
        else:
            print("[TTS WARNING] Tidak ada file suara yang berhasil dibuat — suara dinonaktifkan.")
    except Exception as e:
        print(f"[TTS INIT ERROR] {e}")
        TTS_AVAILABLE = False

_generate_voice_files()

_last_seen_time      = None
_user_present        = False

_raw_bungkuk_since   = None
_raw_tegap_since     = None
confirmed_ergonomi   = "TEGAP"   

_duduk_start         = None
_duduk_absent_since  = None
_duduk_pending_since = None      
_next_peregangan_at  = None     

_last_confident_posisi = None    

status_ergonomi      = "-"  
postur_score_global  = 0.0
_skor_ema            = None  
_pending_status       = None  
_pending_since        = None  
lama_duduk_global    = 0.0

_last_postur_debug   = {"skor": 0.0, "sudut_spine": 0.0, "sudut_neck": 0.0, "arah": "-"}

latest_data = {
    "score"    : 0.0,
    "durasi"   : "00:00:00",
    "ui_status": "no_object"
}

camera_active = False

def _get_best_person_index(result):
    if result.boxes is None or len(result.boxes) == 0:
        return 0
    return int(np.argmax(result.boxes.conf.cpu().numpy()))

def _sudut_vertikal(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if abs(dy) < 1e-3:
        return 90.0
    return abs(np.degrees(np.arctan2(abs(dx), abs(dy))))

def _postur_score(skor_raw, threshold):
    return float(np.clip((skor_raw / (threshold * 2.0)) * 100.0, 0.0, 100.0))

def _format_waktu(detik):
    detik = max(0.0, detik)
    h = int(detik) // 3600
    m = (int(detik) % 3600) // 60
    s = int(detik) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def deteksi_posisi_dan_postur(kp_data, box_coords, now):
    """
    Return: (posisi, arah)
    posisi : "DUDUK" | "BERDIRI" | "-"
    arah   : "SAMPING" | "DEPAN" | "-"

    Bungkuk hanya dievaluasi saat DUDUK + SAMPING.
    Warna border langsung dari status_ergonomi RAW (tidak perlu tunggu confirm).
    """
    global status_ergonomi, postur_score_global, _last_postur_debug, _skor_ema
    global _pending_status, _pending_since
    global _last_confident_posisi

    x1, y1, x2, y2 = box_coords
    box_w = x2 - x1
    box_h = y2 - y1

    if box_h < 1:
        status_ergonomi = "-"
        postur_score_global = 0.0
        _skor_ema = None
        _pending_status = None
        _pending_since = None
        return "-", "-"

    aspect_ratio = box_w / box_h

    if aspect_ratio > 1.2:
        status_ergonomi = "-"
        postur_score_global = 0.0
        _skor_ema = None
        _pending_status = None
        _pending_since = None
        return "-", "-"

    hip_l      = kp_data[11]
    hip_r      = kp_data[12]
    shoulder_l = kp_data[5]
    shoulder_r = kp_data[6]
    head       = kp_data[0]
    ankle_l    = kp_data[15]
    ankle_r    = kp_data[16]
    knee_l     = kp_data[13]
    knee_r     = kp_data[14]

    if hip_l[2] > 0.4:
        hip_y = hip_l[1]
    elif hip_r[2] > 0.4:
        hip_y = hip_r[1]
    else:
        status_ergonomi = "-"
        postur_score_global = 0.0
        _skor_ema = None
        _pending_status = None
        _pending_since = None
        return "BERDIRI", "-"

    hip_rel = (hip_y - y1) / box_h

    ankle_y = None
    if ankle_l[2] > 0.3:
        ankle_y = ankle_l[1]
    elif ankle_r[2] > 0.3:
        ankle_y = ankle_r[1]

    knee_y = None
    if knee_l[2] > 0.3:
        knee_y = knee_l[1]
    elif knee_r[2] > 0.3:
        knee_y = knee_r[1]

    posisi = "-"

    if ankle_y is not None:
        ankle_hip_dist = (ankle_y - hip_y) / box_h if box_h > 0 else 0
        if ankle_hip_dist > 0.35:
            posisi = "BERDIRI"
        else:
            posisi = "DUDUK"
        _last_confident_posisi = posisi
    elif knee_y is not None:
        knee_rel = (knee_y - y1) / box_h
        if knee_rel > 0.70:
            posisi = "DUDUK"
        else:
            posisi = "BERDIRI"
        _last_confident_posisi = posisi
    else:
        posisi = _last_confident_posisi if _last_confident_posisi is not None else "BERDIRI"

    if posisi != "DUDUK":
        status_ergonomi = "-"
        postur_score_global = 0.0
        _skor_ema = None
        _pending_status = None
        _pending_since = None
        return posisi, "-"

    arah = "-"
    if shoulder_l[2] > 0.4 and shoulder_r[2] > 0.4:
        rasio = abs(shoulder_l[0] - shoulder_r[0]) / box_w if box_w > 0 else 0
        arah = "DEPAN" if rasio > HADAP_DEPAN_RATIO else "SAMPING"
    else:
        arah = "SAMPING"

    if arah == "DEPAN":
        status_ergonomi = "DEPAN"
        postur_score_global = 0.0
        _skor_ema = None
        _pending_status = None
        _pending_since = None
        _last_postur_debug = {"skor": 0.0, "sudut_spine": 0.0, "sudut_neck": 0.0, "arah": "DEPAN"}
        return posisi, arah

    hip_avg  = (kp_data[11] + kp_data[12]) / 2
    hip_conf = max(kp_data[11][2], kp_data[12][2])

    if shoulder_l[2] >= shoulder_r[2] and shoulder_l[2] > 0.25:
        shoulder = shoulder_l
    elif shoulder_r[2] > 0.25:
        shoulder = shoulder_r
    else:
        status_ergonomi = "-"
        postur_score_global = 0.0
        _skor_ema = None
        _pending_status = None
        _pending_since = None
        return posisi, arah

    sudut_neck  = _sudut_vertikal(head, shoulder) if head[2] > 0.25 else 0.0
    sudut_spine = _sudut_vertikal(shoulder, hip_avg) if hip_conf > 0.25 else 0.0

    spine_ok = hip_conf > HIP_CONF_MIN

    if spine_ok:
        if sudut_spine < 3.0:
            w_neck = 0.15
        else:
            w_neck = 0.40
        skor      = sudut_spine * (1 - w_neck) + sudut_neck * w_neck
        threshold = SPINE_BUNGKUK_THRESHOLD
    else:
        skor      = sudut_neck
        threshold = NECK_BUNGKUK_THRESHOLD

    _last_postur_debug = {
        "skor"       : skor,
        "sudut_spine": sudut_spine,
        "sudut_neck" : sudut_neck,
        "arah"       : arah
    }

    EMA_ALPHA = 0.35
    _skor_ema = skor if _skor_ema is None else (EMA_ALPHA * skor + (1 - EMA_ALPHA) * _skor_ema)
    skor_smoothed = _skor_ema

    postur_score_global = _postur_score(skor_smoothed, threshold)

    if status_ergonomi == "BUNGKUK":
        candidate = "TEGAP" if skor_smoothed < threshold else "BUNGKUK"
    else:
        candidate = "BUNGKUK" if skor_smoothed >= (threshold + TEGAP_DEAD_ZONE) else "TEGAP"

    if candidate == status_ergonomi:
        _pending_status = None
        _pending_since  = None
    else:
        if _pending_status != candidate:
            _pending_status = candidate
            _pending_since  = now
        elif (now - _pending_since) >= RAW_STATUS_MIN_HOLD_SEC:
            status_ergonomi = candidate
            _pending_status = None
            _pending_since  = None

    return posisi, arah


def _update_confirmed_ergonomi(raw_ergonomi, now):
    """
    Hanya dipakai untuk memicu SUARA setelah 5 detik.
    Border warna langsung dari status_ergonomi RAW.
    """
    global _raw_bungkuk_since, _raw_tegap_since, confirmed_ergonomi

    if raw_ergonomi == "BUNGKUK":
        _raw_tegap_since = None
        if _raw_bungkuk_since is None:
            _raw_bungkuk_since = now
        elif (now - _raw_bungkuk_since) >= BUNGKUK_CONFIRM_SEC:
            confirmed_ergonomi = "BUNGKUK"
    elif raw_ergonomi == "TEGAP":
        _raw_bungkuk_since = None
        if _raw_tegap_since is None:
            _raw_tegap_since = now
        elif (now - _raw_tegap_since) >= TEGAP_CONFIRM_SEC:
            confirmed_ergonomi = "TEGAP"
    else:
        _raw_bungkuk_since = None
        _raw_tegap_since   = None

    return confirmed_ergonomi


def _update_user_presence(detected, now):
    global _last_seen_time, _user_present
    if detected:
        _last_seen_time = now
        _user_present   = True
    else:
        if _last_seen_time is not None:
            if (now - _last_seen_time) >= KAMERA_TOLERANSI_SEC:
                _user_present = False
        else:
            _user_present = False
    return _user_present


def _update_duduk_timer(is_duduk_now, user_present, now):
    global _duduk_start, _duduk_absent_since, _next_peregangan_at, _duduk_pending_since
    global _last_confident_posisi

    sedang_duduk = user_present and is_duduk_now

    if sedang_duduk:
        if _duduk_start is None:
            if _duduk_pending_since is None:
                _duduk_pending_since = now
                return 0.0
            elif (now - _duduk_pending_since) < DUDUK_CONFIRM_SEC:
                return 0.0
            else:
                _duduk_start        = now
                _duduk_absent_since = None
                _next_peregangan_at = PEREGANGAN_FIRST_SEC
                _duduk_pending_since = None
        elif _duduk_absent_since is not None:
            elapsed_absent      = now - _duduk_absent_since
            _duduk_start       += elapsed_absent
            _duduk_absent_since = None

        lama = now - _duduk_start

        if _next_peregangan_at is not None and lama >= _next_peregangan_at:
            play_async('peregangan')
            _next_peregangan_at += PEREGANGAN_REPEAT_SEC

        return lama

    else:
        _duduk_pending_since = None

        if _duduk_start is not None:
            if _duduk_absent_since is None:
                _duduk_absent_since = now
            elapsed_absent = now - _duduk_absent_since
            if elapsed_absent < DUDUK_ABSENT_FREEZE_SEC:
                return _duduk_absent_since - _duduk_start
            else:
                _duduk_start           = None
                _duduk_absent_since    = None
                _next_peregangan_at    = None
                _last_confident_posisi = None

        return 0.0



def _gambar_keypoint_overlay(frame, kp_data, raw_ergonomi, score):
    if kp_data is None:
        return frame

    _, w_f = frame.shape[:2]

    def valid(kp, conf=0.25):
        return kp[2] > conf

    def pt(kp):
        return (int(kp[0]), int(kp[1]))

    head       = kp_data[0]
    shoulder_l = kp_data[5]
    shoulder_r = kp_data[6]
    hip_l      = kp_data[11]
    hip_r      = kp_data[12]
    hip_avg    = (kp_data[11] + kp_data[12]) / 2

    is_bungkuk   = (raw_ergonomi == "BUNGKUK")
    col_line     = COLOR_RED   if is_bungkuk else COLOR_GREEN
    col_shoulder = COLOR_RED   if is_bungkuk else COLOR_GREEN
    col_head     = COLOR_CYAN
    col_hip      = COLOR_ORANGE
    col_white    = COLOR_WHITE

    if valid(shoulder_l) and shoulder_l[2] >= (shoulder_r[2] if valid(shoulder_r) else 0):
        shoulder = shoulder_l
    elif valid(shoulder_r):
        shoulder = shoulder_r
    else:
        shoulder = None

    if shoulder is not None and (valid(hip_l) or valid(hip_r)):
        cv2.line(frame, pt(shoulder), pt(hip_avg), col_line, 2, cv2.LINE_AA)
    if valid(head) and shoulder is not None:
        cv2.line(frame, pt(head), pt(shoulder), col_line, 2, cv2.LINE_AA)
    if valid(shoulder_l) and valid(shoulder_r):
        cv2.line(frame, pt(shoulder_l), pt(shoulder_r), col_shoulder, 1, cv2.LINE_AA)
    if valid(hip_l) and valid(hip_r):
        cv2.line(frame, pt(hip_l), pt(hip_r), col_hip, 1, cv2.LINE_AA)

    for kp, label, col in [
        (head,       "K",  col_head),
        (shoulder_l, "BL", col_shoulder),
        (shoulder_r, "BR", col_shoulder),
        (hip_l,      "PL", col_hip),
        (hip_r,      "PR", col_hip),
    ]:
        if valid(kp):
            cv2.circle(frame, pt(kp), 7, col, -1, cv2.LINE_AA)
            cv2.circle(frame, pt(kp), 7, col_white, 1, cv2.LINE_AA)
            cv2.putText(frame, label, (pt(kp)[0]+9, pt(kp)[1]+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    bw, bh = 110, 12
    bx = w_f - bw - 8
    by = 8
    filled = int(bw * score / 100.0)
    cv2.rectangle(frame, (bx-2, by-2), (bx+bw+2, by+bh+2), (25, 25, 25), -1)
    if filled > 0:
        r = int(min(255, 510 * score / 100))
        g = int(min(255, 510 * (1 - score / 100)))
        cv2.rectangle(frame, (bx, by), (bx+filled, by+bh), (0, g, r), -1)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (140, 140, 140), 1)

    return frame


def _black_frame_bytes():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Camera Off", (210, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2, cv2.LINE_AA)
    _, buf = cv2.imencode('.jpg', frame)
    return buf.tobytes()


def _SET_FRAME():
    global latest_data, status_ergonomi, confirmed_ergonomi
    global lama_duduk_global, postur_score_global, _skor_ema

    TERAKHIR_BERSUARA = 0

    while True:
        if not camera_active:
            blk = _black_frame_bytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + blk + b'\r\n')
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue

        frame     = cv2.flip(frame, 1)
        now       = time.time()
        ann_frame = frame.copy()

        results         = model(frame, conf=0.35, verbose=False)
        kp_data         = None
        box_coords      = (0, 0, 0, 0)
        person_detected = False

        if results and len(results[0].boxes) > 0:
            result   = results[0]
            best_idx = _get_best_person_index(result)
            if result.keypoints is not None and len(result.keypoints) > best_idx:
                try:
                    kp      = result.keypoints[best_idx]
                    kp_data = kp.data[0].cpu().numpy() if hasattr(kp, 'data') else None
                    if result.boxes is not None and len(result.boxes) > best_idx:
                        x1, y1, x2, y2 = result.boxes[best_idx].xyxy[0].cpu().numpy()
                        box_coords      = (int(x1), int(y1), int(x2), int(y2))
                        person_detected = True
                except Exception as e:
                    print(f"[ERROR kp] {e}")

        user_present = _update_user_presence(person_detected, now)

        posisi_frame = "-"
        arah_frame   = "-"
        if user_present and kp_data is not None:
            posisi_frame, arah_frame = deteksi_posisi_dan_postur(kp_data, box_coords, now)
        else:
            status_ergonomi     = "-"
            postur_score_global = 0.0
            _skor_ema = None

        is_duduk = (posisi_frame == "DUDUK")

        raw_erg_input = status_ergonomi if (is_duduk and arah_frame == "SAMPING") else "-"
        confirmed_ergonomi = _update_confirmed_ergonomi(raw_erg_input, now)

        lama_duduk        = _update_duduk_timer(is_duduk, user_present, now)
        lama_duduk_global = lama_duduk

        if is_duduk and user_present and arah_frame == "SAMPING" and confirmed_ergonomi == "BUNGKUK":
            if now - TERAKHIR_BERSUARA > SUARA_DELAY_SEC:
                play_async('bungkuk')
                TERAKHIR_BERSUARA = now
        else:
            TERAKHIR_BERSUARA = 0

        if user_present and kp_data is not None and is_duduk and arah_frame == "SAMPING":
            ann_frame = _gambar_keypoint_overlay(
                ann_frame, kp_data, status_ergonomi, postur_score_global
            )

        if person_detected and posisi_frame != "-":
            cv2.rectangle(ann_frame,
                          (box_coords[0], box_coords[1]),
                          (box_coords[2], box_coords[3]),
                          (200, 100, 0), 2)

        dbg = _last_postur_debug  

        if not user_present or not is_duduk:
            label_ui = "no_object"
        elif arah_frame == "DEPAN":
            label_ui = "person"
        elif status_ergonomi == "BUNGKUK":   
            label_ui = "bungkuk"
        else:
            label_ui = "person"

        latest_data["durasi"]    = _format_waktu(lama_duduk)
        latest_data["ui_status"] = label_ui
        latest_data["score"]     = round(postur_score_global, 1)

        detik_menuju_confirm = ""
        if _raw_bungkuk_since is not None and confirmed_ergonomi != "BUNGKUK":
            sisa = BUNGKUK_CONFIRM_SEC - (now - _raw_bungkuk_since)
            detik_menuju_confirm = f" [confirm in {max(0,sisa):.1f}s]"

        detik_menuju_peregangan = ""
        if _next_peregangan_at is not None:
            sisa_p = _next_peregangan_at - lama_duduk
            detik_menuju_peregangan = f" [peregangan in {max(0,sisa_p):.0f}s]"

        ret, buf = cv2.imencode('.jpg', ann_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(_SET_FRAME(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    return jsonify({
        'durasi'   : latest_data["durasi"],
        'ui_status': latest_data["ui_status"],
        'score'    : latest_data["score"]
    })

@app.route('/camera/on', methods=['POST'])
def camera_on():
    global camera_active
    camera_active = True
    return jsonify({'camera': 'on'})

@app.route('/camera/off', methods=['POST'])
def camera_off():
    global camera_active, status_ergonomi, confirmed_ergonomi
    global lama_duduk_global, postur_score_global, _skor_ema
    global _duduk_start, _duduk_absent_since, _next_peregangan_at, _duduk_pending_since
    global _raw_bungkuk_since, _raw_tegap_since
    global _user_present, _last_seen_time
    global _last_confident_posisi

    camera_active        = False
    status_ergonomi      = "-"
    confirmed_ergonomi   = "TEGAP"
    lama_duduk_global    = 0.0
    postur_score_global  = 0.0
    _skor_ema             = None
    _duduk_start         = None
    _duduk_absent_since  = None
    _duduk_pending_since = None
    _next_peregangan_at  = None
    _raw_bungkuk_since   = None
    _raw_tegap_since     = None
    _user_present        = False
    _last_seen_time      = None
    _last_confident_posisi = None
    latest_data["score"]     = 0.0
    latest_data["durasi"]    = "00:00:00"
    latest_data["ui_status"] = "no_object"

    return jsonify({'camera': 'off'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)