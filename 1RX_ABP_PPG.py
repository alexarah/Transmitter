"""
rx_abp_ppg.py — SISI RX SAJA (berdiri sendiri, tidak butuh proses TX)
======================================================================
Skema:
  RX (PORT_RX) : ESP32 terima ABP + PPG sekaligus, dibedakan header TYPE.

  Script ini BERDIRI SENDIRI — jalan di proses/komputer terpisah dari
  tx_abp_ppg.py. Format paket dari radio TIDAK berubah (firmware Arduino
  TIDAK perlu diubah):

    ABP: START|TYPE:ABP|ABP:<raw>|SBP:<sbp>|DBP:<dbp>|SEQ:<seq>|END
    PPG: START|TYPE:PPG|PPG:<raw>|SEQ:<seq>|END

  Karena tidak ada koneksi memori bersama ke proses TX, RX menyimpulkan
  sendiri:
    1) Segmen & label yang sedang aktif -> dari SEQ:
         segmen_ke = SEQ // PAKET_PER_SEGMEN  (lalu di-index ke SEGMENTS)
    2) Nilai sinyal asli (untuk hitung SNR & error TX-RX) -> RX ikut
       memuat file .npy yang SAMA (ABP_FILE/PPG_FILE) dan mengambil
       sample yang tepat sesuai SEQ. Ini malah lebih akurat daripada
       korelasi silang, karena pemetaan SEQ -> sample sudah pasti.

  SYARAT: FS, SEGMENTS, PAKET_PER_SEGMEN, ABP_FILE, PPG_FILE di sini
  HARUS SAMA PERSIS dengan yang dipakai tx_abp_ppg.py.

  Output:
    hasil_rx/rx_abp.csv
    hasil_rx/rx_ppg.csv
    hasil_rx/ringkasan_rx.csv
"""

import serial
import threading
import time
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from scipy.signal import find_peaks
from collections import deque

# ─── KONFIGURASI (WAJIB SAMA DENGAN tx_abp_ppg.py: FS, SEGMENTS, PAKET_PER_SEGMEN) ─
PORT_RX        = 'COM14'
BAUD_RATE      = 115200
FS             = 125
WINDOW_S       = 10
ABP_FILE       = 'p000188_abp.npy'
PPG_FILE       = 'p000188_ppg.npy'

SEGMENTS = [
    {'idx': 0,  'label': 'Normal'},
    {'idx': 20, 'label': 'BP Tinggi'},
    {'idx': 25, 'label': 'BP Rendah'},
]

PAKET_PER_SEGMEN = 3750          # HARUS SAMA dengan tx_abp_ppg.py

OUTPUT_DIR     = 'hasil_rx'

PEAK_DIST_ABP  = 40
PEAK_PROM_ABP  = 3
PEAK_DIST_PPG  = 40
PEAK_PROM_PPG  = 0.1

THROUGHPUT_WIN_S = 3
SNR_WIN_S        = 5

# Berhenti otomatis kalau tidak ada data sama sekali selama sekian detik
# (setelah minimal 1 paket pernah diterima). Tutup jendela dashboard juga bisa.
IDLE_STOP_S = 30.0
# ─────────────────────────────────────────────────────────────

WINDOW_N = WINDOW_S * FS
TOTAL_EXPECTED_PER_SIGNAL = PAKET_PER_SEGMEN * len(SEGMENTS)
os.makedirs(OUTPUT_DIR, exist_ok=True)

buf_rx_abp = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)
buf_rx_ppg = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)

rx_rows_abp = []
rx_rows_ppg = []
lock        = threading.Lock()
stop_evt    = threading.Event()

stats_common = {'segmen': 0, 'segmen_label': '', 'channel_aktif': '-'}

stats_abp = {
    'rx_pkt': 0, 'loss_pkt': 0, 'loss_pct': 0.0,
    'snr_db': 0.0, 'sbp_tx': 0.0, 'dbp_tx': 0.0,
    'sbp_rx': 0.0, 'dbp_rx': 0.0, 'map_rx': 0.0, 'pp_rx': 0.0,
    'bytes_rx': 0, 'thr_rx_bps': 0.0, 'thr_rx_kbps': 0.0, 'pkt_rate_rx': 0.0,
}
stats_ppg = {
    'rx_pkt': 0, 'loss_pkt': 0, 'loss_pct': 0.0,
    'snr_db': 0.0, 'hr_rx': 0.0,
    'bytes_rx': 0, 'thr_rx_bps': 0.0, 'thr_rx_kbps': 0.0, 'pkt_rate_rx': 0.0,
}


class ThroughputMeter:
    def __init__(self, window_s=THROUGHPUT_WIN_S):
        self.window_s = window_s
        self._samples = deque()
        self._lock    = threading.Lock()

    def update(self, n_bytes, n_pkts=1):
        now = time.perf_counter()
        with self._lock:
            self._samples.append((now, n_bytes, n_pkts))
            cutoff = now - self.window_s
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

    def get(self):
        now = time.perf_counter()
        with self._lock:
            if len(self._samples) < 2:
                return 0.0, 0.0
            elapsed = now - self._samples[0][0]
            if elapsed <= 0:
                return 0.0, 0.0
            total_b = sum(s[1] for s in self._samples)
            total_p = sum(s[2] for s in self._samples)
            return (total_b * 8) / elapsed, total_p / elapsed


meter_rx_abp = ThroughputMeter()
meter_rx_ppg = ThroughputMeter()


class ABPDetector:
    def __init__(self, fs=FS, win_s=5):
        self.fs = fs
        self.buf = deque(maxlen=win_s * fs)
        self.sbp = self.dbp = self.map_ = self.pp = 0.0
        self.n = 0

    def update(self, val):
        self.buf.append(val)
        self.n += 1
        if self.n % (self.fs // 2) != 0 or len(self.buf) < PEAK_DIST_ABP * 2:
            return self.sbp, self.dbp, self.map_, self.pp
        arr = np.array(self.buf)
        peaks,   _ = find_peaks( arr, distance=PEAK_DIST_ABP, prominence=PEAK_PROM_ABP)
        valleys, _ = find_peaks(-arr, distance=PEAK_DIST_ABP, prominence=PEAK_PROM_ABP)
        if len(peaks)   > 0: self.sbp = float(arr[peaks].mean())
        if len(valleys) > 0: self.dbp = float(arr[valleys].mean())
        if self.sbp > 0 and self.dbp > 0:
            self.pp = self.sbp - self.dbp
            self.map_ = self.dbp + self.pp / 3.0
        return self.sbp, self.dbp, self.map_, self.pp


class HRDetector:
    def __init__(self, fs=FS, win_s=5):
        self.fs = fs
        self.buf = deque(maxlen=win_s * fs)
        self.hr = 0.0
        self.n = 0

    def update(self, val):
        self.buf.append(val)
        self.n += 1
        if self.n % (self.fs // 2) != 0 or len(self.buf) < PEAK_DIST_PPG * 2:
            return self.hr
        arr = np.array(self.buf)
        peaks, _ = find_peaks(arr, distance=PEAK_DIST_PPG, prominence=PEAK_PROM_PPG)
        if len(peaks) >= 2:
            self.hr = 60.0 / (np.diff(peaks) / self.fs).mean()
        return self.hr


# ─── Muat sinyal asli utuh, untuk rekonstruksi nilai TX dari SEQ ──
def _load_full(filepath, label):
    data = np.load(filepath, allow_pickle=True)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


_abp_data = _load_full(ABP_FILE, 'ABP')
_ppg_data = _load_full(PPG_FILE, 'PPG')


def segmen_dari_seq(seq):
    seg_number = seq // PAKET_PER_SEGMEN
    if seg_number >= len(SEGMENTS):
        seg_number = len(SEGMENTS) - 1
    return SEGMENTS[seg_number]


def nilai_asli_dari_seq(data, seq):
    seg = segmen_dari_seq(seq)
    i   = seq % PAKET_PER_SEGMEN
    sig = data[seg['idx']].astype(float)
    n   = len(sig)
    return float(sig[i % n]), seg['idx'], seg['label']


# ─── THREAD RX ────────────────────────────────────────────────
def thread_rx():
    try:
        ser = serial.Serial(PORT_RX, BAUD_RATE, timeout=1)
        time.sleep(1.5)
        print(f"[RX] Terhubung ke {PORT_RX}")
    except Exception as e:
        print(f"[RX] ERROR: {e} — RX dinonaktifkan.")
        stop_evt.set()
        return

    det_abp = ABPDetector()
    det_ppg = HRDetector()

    snr_buf_abp = deque(maxlen=SNR_WIN_S * FS)   # (expected, received)
    snr_buf_ppg = deque(maxlen=SNR_WIN_S * FS)

    csv_abp_path = os.path.join(OUTPUT_DIR, 'rx_abp.csv')
    f_abp = open(csv_abp_path, 'w', newline='', encoding='utf-8')
    w_abp = csv.writer(f_abp)
    w_abp.writerow([
        'timestamp_s', 'segmen_idx', 'segmen_label', 'seq',
        'abp_raw', 'abp_mmhg', 'abp_asli',
        'sbp_rx', 'dbp_rx', 'map_rx', 'pp_rx',
        'sbp_tx', 'dbp_tx', 'selisih_sbp', 'selisih_dbp',
        'loss_seq_pct', 'snr_db',
        'thr_rx_bps', 'thr_rx_kbps', 'pkt_rate_rx',
        'valid', 'corrupt_reason'
    ])

    csv_ppg_path = os.path.join(OUTPUT_DIR, 'rx_ppg.csv')
    f_ppg = open(csv_ppg_path, 'w', newline='', encoding='utf-8')
    w_ppg = csv.writer(f_ppg)
    w_ppg.writerow([
        'timestamp_s', 'segmen_idx', 'segmen_label', 'seq',
        'ppg_raw', 'ppg_val', 'ppg_asli', 'hr_rx',
        'loss_seq_pct', 'snr_db',
        'thr_rx_bps', 'thr_rx_kbps', 'pkt_rate_rx',
        'valid', 'corrupt_reason'
    ])

    t_start      = time.time()
    last_seq_abp = -1
    last_seq_ppg = -1
    buf          = b""
    last_data_t  = time.perf_counter()
    got_any_data = False

    while True:
        now_t = time.perf_counter()
        if got_any_data and (now_t - last_data_t >= IDLE_STOP_S):
            print(f"[RX] Tidak ada data selama {IDLE_STOP_S:.0f}s — dianggap selesai.")
            break
        if stop_evt.is_set():
            break

        try:
            chunk = ser.read(ser.in_waiting or 1)
        except serial.SerialException as e:
            print(f"[RX] Read error: {e}"); break

        if not chunk:
            time.sleep(0.001); continue

        got_any_data = True
        last_data_t  = time.perf_counter()
        buf         += chunk

        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line: continue

            try:
                text = line.decode('utf-8', errors='replace')
                if not (text.startswith('START') and 'END' in text):
                    continue

                fields = {}
                for part in text.replace('START|', '').replace('|END', '').split('|'):
                    if ':' in part:
                        k, v = part.split(':', 1)
                        fields[k] = v

                pkt_type = fields.get('TYPE', '')
                ts_s     = round(time.time() - t_start, 3)

                # ── Paket ABP ──────────────────────────────
                if pkt_type == 'ABP':
                    is_valid, corrupt_reason = True, ''

                    if not all(k in fields for k in ('ABP', 'SEQ')):
                        w_abp.writerow([ts_s, '', '', ''] + [''] * 17 + [0, 'missing_field'])
                        f_abp.flush(); continue

                    raw        = int(fields['ABP'])
                    seq        = int(fields['SEQ'])
                    sbp_tx_now = float(fields.get('SBP', 0))
                    dbp_tx_now = float(fields.get('DBP', 0))
                    val_mm     = raw / 100.0

                    seg_idx_now, seg_label_now = segmen_dari_seq(seq)['idx'], segmen_dari_seq(seq)['label']
                    with lock:
                        stats_common['segmen'] = seg_idx_now
                        stats_common['segmen_label'] = seg_label_now
                        stats_common['channel_aktif'] = 'ABP'

                    # FIX: pakai panjang baris (len(line)+1), bukan chunk_len,
                    # dan cukup 1x update per paket -> byte count & throughput
                    # gak lagi ke-hitung berkali-kali kalau 1 chunk read berisi
                    # banyak paket sekaligus.
                    meter_rx_abp.update(len(line) + 1, n_pkts=1)
                    bps_rx, pps_rx = meter_rx_abp.get()

                    if not (2000 <= raw <= 25000):
                        is_valid, corrupt_reason = False, f'abp_out_of_range({raw})'

                    if is_valid and last_seq_abp >= 0:
                        gap = seq - last_seq_abp - 1
                        if 0 < gap <= 1000:
                            with lock: stats_abp['loss_pkt'] += gap
                    if is_valid: last_seq_abp = seq

                    sbp_rx = dbp_rx = map_rx = pp_rx = 0.0
                    sel_sbp = sel_dbp = ''
                    asli_val = ''

                    with lock:
                        stats_abp['rx_pkt']   += 1
                        stats_abp['bytes_rx'] += len(line) + 1
                        total_exp = stats_abp['rx_pkt'] + stats_abp['loss_pkt']
                        stats_abp['loss_pct'] = (stats_abp['loss_pkt'] / total_exp * 100
                                                  if total_exp > 0 else 0.0)
                        stats_abp['thr_rx_bps']  = bps_rx
                        stats_abp['thr_rx_kbps'] = bps_rx / 1000.0
                        stats_abp['pkt_rate_rx'] = pps_rx
                        lp_seq_snap = stats_abp['loss_pct']

                    if is_valid:
                        sbp_rx, dbp_rx, map_rx, pp_rx = det_abp.update(val_mm)

                        asli_val, _, _ = nilai_asli_dari_seq(_abp_data, seq)
                        snr_buf_abp.append((asli_val, val_mm))
                        if len(snr_buf_abp) >= FS:
                            exp_arr = np.array([p[0] for p in snr_buf_abp])
                            rx_arr  = np.array([p[1] for p in snr_buf_abp])
                            exp_c   = exp_arr - exp_arr.mean()
                            err     = exp_arr - rx_arr
                            ps, pn  = np.mean(exp_c ** 2), np.mean(err ** 2)
                            snr = float(np.clip(10 * np.log10(ps / pn) if pn > 1e-12 else 99.0, -10, 60))
                            with lock: stats_abp['snr_db'] = snr

                        if sbp_rx > 0 and dbp_rx > 0:
                            with lock:
                                stats_abp['sbp_rx'] = sbp_rx
                                stats_abp['dbp_rx'] = dbp_rx
                                stats_abp['map_rx'] = map_rx
                                stats_abp['pp_rx']  = pp_rx
                        with lock:
                            stats_abp['sbp_tx'] = sbp_tx_now
                            stats_abp['dbp_tx'] = dbp_tx_now
                            buf_rx_abp.append(val_mm)

                        sel_sbp = round(sbp_rx - sbp_tx_now, 2) if sbp_rx > 0 and sbp_tx_now > 0 else ''
                        sel_dbp = round(dbp_rx - dbp_tx_now, 2) if dbp_rx > 0 and dbp_tx_now > 0 else ''

                    with lock:
                        snr_snap = stats_abp['snr_db']

                    w_abp.writerow([
                        ts_s, seg_idx_now, seg_label_now, seq,
                        raw, round(val_mm, 2) if is_valid else '',
                        round(asli_val, 2) if asli_val != '' else '',
                        round(sbp_rx, 1) if sbp_rx else '', round(dbp_rx, 1) if dbp_rx else '',
                        round(map_rx, 1) if map_rx else '', round(pp_rx, 1) if pp_rx else '',
                        round(sbp_tx_now, 1), round(dbp_tx_now, 1), sel_sbp, sel_dbp,
                        round(lp_seq_snap, 2), round(snr_snap, 2),
                        round(bps_rx, 1), round(bps_rx / 1000.0, 3), round(pps_rx, 2),
                        1 if is_valid else 0, corrupt_reason
                    ])
                    f_abp.flush()

                    with lock:
                        rx_rows_abp.append({'seq': seq, 'loss_pct': lp_seq_snap,
                                             'thr_rx_bps': bps_rx, 'pkt_rate_rx': pps_rx,
                                             'valid': 1 if is_valid else 0, 'snr': snr_snap})

                # ── Paket PPG ──────────────────────────────
                elif pkt_type == 'PPG':
                    is_valid, corrupt_reason = True, ''

                    if not all(k in fields for k in ('PPG', 'SEQ')):
                        w_ppg.writerow([ts_s, '', '', ''] + [''] * 10 + [0, 'missing_field'])
                        f_ppg.flush(); continue

                    raw = int(fields['PPG'])
                    seq = int(fields['SEQ'])
                    val = raw / 10000.0

                    seg_idx_now, seg_label_now = segmen_dari_seq(seq)['idx'], segmen_dari_seq(seq)['label']
                    with lock:
                        stats_common['segmen'] = seg_idx_now
                        stats_common['segmen_label'] = seg_label_now
                        stats_common['channel_aktif'] = 'PPG'

                    # FIX: sama seperti ABP — pakai panjang baris, 1x update per paket.
                    meter_rx_ppg.update(len(line) + 1, n_pkts=1)
                    bps_rx, pps_rx = meter_rx_ppg.get()

                    if not (0 <= raw <= 40020):
                        is_valid, corrupt_reason = False, f'ppg_out_of_range({raw})'

                    if is_valid and last_seq_ppg >= 0:
                        gap = seq - last_seq_ppg - 1
                        if 0 < gap <= 1000:
                            with lock: stats_ppg['loss_pkt'] += gap
                    if is_valid: last_seq_ppg = seq

                    hr_rx = 0.0
                    asli_val = ''

                    with lock:
                        stats_ppg['rx_pkt']   += 1
                        stats_ppg['bytes_rx'] += len(line) + 1
                        total_exp = stats_ppg['rx_pkt'] + stats_ppg['loss_pkt']
                        stats_ppg['loss_pct'] = (stats_ppg['loss_pkt'] / total_exp * 100
                                                  if total_exp > 0 else 0.0)
                        stats_ppg['thr_rx_bps']  = bps_rx
                        stats_ppg['thr_rx_kbps'] = bps_rx / 1000.0
                        stats_ppg['pkt_rate_rx'] = pps_rx
                        lp_seq_snap = stats_ppg['loss_pct']

                    if is_valid:
                        hr_rx = det_ppg.update(val)

                        asli_val, _, _ = nilai_asli_dari_seq(_ppg_data, seq)
                        snr_buf_ppg.append((asli_val, val))
                        if len(snr_buf_ppg) >= FS:
                            exp_arr = np.array([p[0] for p in snr_buf_ppg])
                            rx_arr  = np.array([p[1] for p in snr_buf_ppg])
                            exp_c   = exp_arr - exp_arr.mean()
                            err     = exp_arr - rx_arr
                            ps, pn  = np.mean(exp_c ** 2), np.mean(err ** 2)
                            snr = float(np.clip(10 * np.log10(ps / pn) if pn > 1e-12 else 99.0, -10, 60))
                            with lock: stats_ppg['snr_db'] = snr

                        if hr_rx > 0:
                            with lock: stats_ppg['hr_rx'] = hr_rx
                        with lock: buf_rx_ppg.append(val)

                    with lock:
                        snr_snap = stats_ppg['snr_db']

                    w_ppg.writerow([
                        ts_s, seg_idx_now, seg_label_now, seq,
                        raw, round(val, 4) if is_valid else '',
                        round(asli_val, 4) if asli_val != '' else '',
                        round(hr_rx, 1) if hr_rx else '',
                        round(lp_seq_snap, 2), round(snr_snap, 2),
                        round(bps_rx, 1), round(bps_rx / 1000.0, 3), round(pps_rx, 2),
                        1 if is_valid else 0, corrupt_reason
                    ])
                    f_ppg.flush()

                    with lock:
                        rx_rows_ppg.append({'seq': seq, 'loss_pct': lp_seq_snap,
                                             'thr_rx_bps': bps_rx, 'pkt_rate_rx': pps_rx,
                                             'valid': 1 if is_valid else 0, 'snr': snr_snap})

            except Exception:
                continue

    f_abp.close(); f_ppg.close(); ser.close()
    stop_evt.set()
    print(f"[RX] Selesai. ABP -> {csv_abp_path}  PPG -> {csv_ppg_path}")


# ─── RINGKASAN ───────────────────────────────────────────────
def cetak_ringkasan():
    with lock:
        rows_a = list(rx_rows_abp)
        rows_p = list(rx_rows_ppg)
        sa     = dict(stats_abp)
        sp     = dict(stats_ppg)

    def _ringkas(label, rows, s):
        if not rows:
            print(f"\n[!] Tidak ada data RX {label}."); return None
        total          = len(rows)
        loss_seq_avg   = np.mean([r['loss_pct'] for r in rows])
        thr_rx = [r['thr_rx_bps']  for r in rows if r['thr_rx_bps']  > 0]
        pps_rx = [r['pkt_rate_rx'] for r in rows if r['pkt_rate_rx'] > 0]
        snr_avg = np.mean([r['snr'] for r in rows if r['snr'] > 0]) if any(r['snr'] > 0 for r in rows) else 0.0
        # Rasio keberhasilan berbasis JUMLAH PAKET, dibandingkan target
        # total paket per sinyal (asumsi TX menyelesaikan semua segmen).
        sukses_pct = (total / TOTAL_EXPECTED_PER_SIGNAL * 100) if TOTAL_EXPECTED_PER_SIGNAL > 0 else 0.0

        print(f"\n{'='*60}\n  RINGKASAN RX — {label}\n{'='*60}")
        print(f"  Segmen              : {[sg['label'] for sg in SEGMENTS]}")
        print(f"  Paket/segmen        : {PAKET_PER_SEGMEN:,}")
        print(f"  Target total paket  : {TOTAL_EXPECTED_PER_SIGNAL:,}  (asumsi, dari konfigurasi)")
        print(f"  Total paket RX      : {total:,}")
        print(f"  Loss jaringan (SEQ) : {loss_seq_avg:.2f}%")
        print(f"  Rasio keberhasilan  : {sukses_pct:.1f}%  ({total:,}/{TOTAL_EXPECTED_PER_SIGNAL:,} paket)")
        print(f"  SNR rata-rata       : {snr_avg:.2f} dB")
        if thr_rx:
            print(f"  Throughput RX       : {np.mean(thr_rx):.1f} bps ({np.mean(thr_rx)/1000:.3f} kbps)"
                  f" | puncak: {np.max(thr_rx):.1f} bps")
        if pps_rx: print(f"  Laju paket RX       : {np.mean(pps_rx):.1f} pkt/s")
        print(f"  Byte RX             : {s['bytes_rx']:,} byte")

        return {'total': total, 'loss_seq_avg': loss_seq_avg, 'sukses_pct': sukses_pct,
                'snr_avg': snr_avg, 'thr_rx': thr_rx, 'pps_rx': pps_rx, 'bytes_rx': s['bytes_rx']}

    res_a = _ringkas('ABP', rows_a, sa)
    res_p = _ringkas('PPG', rows_p, sp)

    ring_path = os.path.join(OUTPUT_DIR, 'ringkasan_rx.csv')
    with open(ring_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sinyal', 'metrik', 'nilai'])
        for label, res in [('ABP', res_a), ('PPG', res_p)]:
            if res is None: continue
            w.writerow([label, 'paket_per_segmen', PAKET_PER_SEGMEN])
            w.writerow([label, 'target_total_paket', TOTAL_EXPECTED_PER_SIGNAL])
            w.writerow([label, 'total_paket_rx', res['total']])
            w.writerow([label, 'loss_jaringan_seq_pct', round(res['loss_seq_avg'], 2)])
            w.writerow([label, 'rasio_keberhasilan_pct', round(res['sukses_pct'], 1)])
            w.writerow([label, 'snr_db_avg', round(res['snr_avg'], 2)])
            w.writerow([label, 'bytes_rx', res['bytes_rx']])
            if res['thr_rx']:
                w.writerow([label, 'thr_rx_bps_avg', round(float(np.mean(res['thr_rx'])), 1)])
                w.writerow([label, 'thr_rx_kbps_avg', round(float(np.mean(res['thr_rx']))/1000, 3)])
                w.writerow([label, 'thr_rx_bps_peak', round(float(np.max(res['thr_rx'])), 1)])
            if res['pps_rx']:
                w.writerow([label, 'pkt_rate_rx_avg', round(float(np.mean(res['pps_rx'])), 2)])
    print(f"\n[OK] Ringkasan RX -> {ring_path}")
    print(f"[OK] Semua file CSV di folder: {os.path.abspath(OUTPUT_DIR)}")


# ─── DASHBOARD ───────────────────────────────────────────────
def run_dashboard():
    plt.rcParams.update({
        'figure.facecolor': '#0D1117', 'axes.facecolor': '#161B22',
        'axes.edgecolor':   '#30363D', 'axes.labelcolor': '#C9D1D9',
        'xtick.color':      '#8B949E', 'ytick.color':     '#8B949E',
        'grid.color':       '#21262D', 'text.color':      '#C9D1D9',
        'font.family':      'monospace', 'font.size': 10,
    })

    t_axis = np.linspace(0, WINDOW_S, WINDOW_N)

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f'ABP + PPG — SISI RX\nRX: {PORT_RX}', color='#3FB950',
                 fontsize=14, fontweight='bold', linespacing=1.6)

    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[0.35, 1, 1, 1.6],
                            hspace=0.7, wspace=0.25, left=0.08, right=0.97, top=0.85, bottom=0.06)

    ax_slot = fig.add_subplot(gs[0, :]); ax_slot.axis('off')
    slot_text = ax_slot.text(0.5, 0.5, '', ha='center', va='center', transform=ax_slot.transAxes,
                              fontsize=15, fontweight='bold')

    ax_rx_abp = fig.add_subplot(gs[1, :])
    ax_rx_ppg = fig.add_subplot(gs[2, :])
    ax_info_abp = fig.add_subplot(gs[3, 0]); ax_info_abp.axis('off')
    ax_info_ppg = fig.add_subplot(gs[3, 1]); ax_info_ppg.axis('off')

    ax_rx_abp.set_title(f'ABP — RX ({PORT_RX})', color='#3FB950', fontsize=11, fontweight='bold', loc='left')
    ax_rx_ppg.set_title(f'PPG — RX ({PORT_RX})', color='#56D364', fontsize=11, fontweight='bold', loc='left')

    for ax in [ax_rx_abp, ax_rx_ppg]:
        ax.set_xlim(0, WINDOW_S)
        ax.set_xlabel('Waktu (detik)', fontsize=9)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)
    ax_rx_abp.set_ylabel('mmHg', fontsize=9)
    ax_rx_ppg.set_ylabel('Amplitudo', fontsize=9)

    line_rx_abp, = ax_rx_abp.plot(t_axis, list(buf_rx_abp), color='#3FB950', lw=1.1)
    line_rx_ppg, = ax_rx_ppg.plot(t_axis, list(buf_rx_ppg), color='#56D364', lw=1.1)

    box_style = dict(boxstyle='round,pad=0.35', facecolor='#0D1117', alpha=0.9, edgecolor='#30363D')
    lbl_sbp = ax_rx_abp.text(0.99, 0.90, 'SBP: -- mmHg', transform=ax_rx_abp.transAxes, fontsize=10,
                              fontweight='bold', color='#F4C275', ha='right', va='top', bbox=box_style)
    lbl_dbp = ax_rx_abp.text(0.99, 0.66, 'DBP: -- mmHg', transform=ax_rx_abp.transAxes, fontsize=10,
                              fontweight='bold', color='#5DCAA5', ha='right', va='top', bbox=box_style)
    lbl_hr  = ax_rx_ppg.text(0.99, 0.90, 'HR: -- BPM', transform=ax_rx_ppg.transAxes, fontsize=10,
                              fontweight='bold', color='#F4C275', ha='right', va='top', bbox=box_style)

    info_box = dict(boxstyle='round,pad=0.5', facecolor='#161B22', alpha=0.9, edgecolor='#30363D')
    stats_text_abp = ax_info_abp.text(0.03, 0.97, '', transform=ax_info_abp.transAxes, fontsize=10.5,
                                       va='top', ha='left', linespacing=1.7, fontfamily='monospace',
                                       color='#E3B341', bbox=info_box)
    stats_text_ppg = ax_info_ppg.text(0.03, 0.97, '', transform=ax_info_ppg.transAxes, fontsize=10.5,
                                       va='top', ha='left', linespacing=1.7, fontfamily='monospace',
                                       color='#79C0FF', bbox=info_box)

    def _fmt_abp(sa):
        sukses = f"{sa['rx_pkt']/TOTAL_EXPECTED_PER_SIGNAL*100:5.1f}%" if TOTAL_EXPECTED_PER_SIGNAL > 0 else "  -- %"
        return ("── ABP (RX) ─────────────────\n"
                f"RX pkt      : {sa['rx_pkt']:>8,}\n"
                f"Loss SEQ    : {sa['loss_pct']:>7.2f} %\n"
                f"SBP TX/RX   : {sa['sbp_tx']:>5.1f} / {sa['sbp_rx']:<5.1f} mmHg\n"
                f"DBP TX/RX   : {sa['dbp_tx']:>5.1f} / {sa['dbp_rx']:<5.1f} mmHg\n"
                f"MAP / PP    : {sa['map_rx']:>5.1f} / {sa['pp_rx']:<5.1f} mmHg\n"
                f"SNR         : {sa['snr_db']:>7.2f} dB\n"
                f"Rasio berhasil: {sukses}\n"
                f"Thr RX      : {sa['thr_rx_kbps']:>7.2f} kbps\n"
                f"Rate RX     : {sa['pkt_rate_rx']:>7.1f} pkt/s\n"
                f"Byte RX     : {sa['bytes_rx']/1024:>7.1f} KB")

    def _fmt_ppg(sp):
        sukses = f"{sp['rx_pkt']/TOTAL_EXPECTED_PER_SIGNAL*100:5.1f}%" if TOTAL_EXPECTED_PER_SIGNAL > 0 else "  -- %"
        return ("── PPG (RX) ─────────────────\n"
                f"RX pkt      : {sp['rx_pkt']:>8,}\n"
                f"Loss SEQ    : {sp['loss_pct']:>7.2f} %\n"
                f"HR RX       : {sp['hr_rx']:>7.1f} BPM\n"
                f"SNR         : {sp['snr_db']:>7.2f} dB\n"
                f"Rasio berhasil: {sukses}\n"
                f"Thr RX      : {sp['thr_rx_kbps']:>7.2f} kbps\n"
                f"Rate RX     : {sp['pkt_rate_rx']:>7.1f} pkt/s\n"
                f"Byte RX     : {sp['bytes_rx']/1024:>7.1f} KB")

    def update(_frame):
        if stop_evt.is_set():
            try: ani.event_source.stop()
            except Exception: pass
            plt.close(fig)
            return

        with lock:
            ra = np.array(buf_rx_abp); rp = np.array(buf_rx_ppg)
            sa = dict(stats_abp); sp = dict(stats_ppg); sc = dict(stats_common)

        for line, arr, ax in [(line_rx_abp, ra, ax_rx_abp), (line_rx_ppg, rp, ax_rx_ppg)]:
            line.set_ydata(arr)
            rng = np.ptp(arr)
            if rng > 0:
                m = rng * 0.15
                ax.set_ylim(arr.min() - m, arr.max() + m)

        lbl_sbp.set_text(f"SBP: {sa['sbp_rx']:.0f} mmHg" if sa['sbp_rx'] > 0 else 'SBP: -- mmHg')
        lbl_dbp.set_text(f"DBP: {sa['dbp_rx']:.0f} mmHg" if sa['dbp_rx'] > 0 else 'DBP: -- mmHg')
        lbl_hr.set_text( f"HR: {sp['hr_rx']:.0f} BPM"    if sp['hr_rx']  > 0 else 'HR: -- BPM')

        ch = sc.get('channel_aktif', '-')
        slot_text.set_text(f"MENERIMA: {ch}   |   Segmen {sc.get('segmen')} ({sc.get('segmen_label')})")
        slot_text.set_color('#FF6B6B' if ch == 'ABP' else '#51CF66')

        stats_text_abp.set_text(_fmt_abp(sa))
        stats_text_ppg.set_text(_fmt_ppg(sp))
        fig.canvas.draw_idle()

    ani = animation.FuncAnimation(fig, update, interval=150, blit=False, cache_frame_data=False)

    def on_close(event):
        try: ani.event_source.stop()
        except Exception: pass
        stop_evt.set()

    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()


def main():
    total_pkt = TOTAL_EXPECTED_PER_SIGNAL
    print('=' * 60)
    print('  RX ABP + PPG (proses RX terpisah dari TX)')
    print('=' * 60)
    print(f'  RX               : {PORT_RX}')
    print(f'  Segmen           : {[s["label"] for s in SEGMENTS]}')
    print(f'  Paket/segmen     : {PAKET_PER_SEGMEN:,}')
    print(f'  Target total/sinyal : {total_pkt:,} paket')
    print(f'  Idle-stop        : {IDLE_STOP_S:.0f}s tanpa data -> berhenti otomatis')
    print(f'  Output           : {OUTPUT_DIR}/ (rx_abp.csv, rx_ppg.csv, ringkasan_rx.csv)')
    print('=' * 60)
    print('\nTutup jendela dashboard atau tunggu idle-stop untuk berhenti.\n')

    t_rx = threading.Thread(target=thread_rx, daemon=True)
    t_rx.start()
    time.sleep(0.5)

    try:
        run_dashboard()
    except KeyboardInterrupt:
        print('\n[Main] Dihentikan.')
        stop_evt.set()

    t_rx.join(timeout=IDLE_STOP_S + 5)

    cetak_ringkasan()
    print('[Main] Selesai.')


if __name__ == '__main__':
    main()