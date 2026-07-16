"""
rx_ppg_only.py — SISI RX SAJA (untuk setup 2 laptop/proses terpisah)
======================================================================
Jalankan skrip ini di laptop/terminal yang tersambung ke ESP32 RX (USB serial).
Pasangannya: tx_ppg_only.py (dijalankan di laptop/terminal LAIN, tersambung
ke ESP32 TX).


Output:
    hasil_ppg_rx/rx_data.csv      <- detail tiap paket yang diterima
    hasil_ppg_rx/ringkasan_rx.csv <- ringkasan sesi (ditulis saat RX selesai)
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

# ─── KONFIGURASI (harus SAMA dengan tx_ppg_only.py, kecuali PORT) ───
PORT           = 'COM14'          # ganti sesuai port ESP32 RX di laptop ini
BAUD_RATE      = 115200
FS             = 125
PPG_FILE       = 'p000188_ppg.npy'
SEGMEN_DIPILIH = [0, 20, 25]      # HARUS SAMA dengan tx_ppg_only.py
SEG_DURATION_S = 30               # HARUS SAMA dengan tx_ppg_only.py — durasi tiap
                                   # segmen (detik) yang BENAR-BENAR dikirim TX.
                                   # Kalau ini beda dengan TX, TOTAL_TARGET di RX
                                   # jadi salah dan rasio keberhasilan meleset jauh.
# ──────────────────────────────────────────────────────────────

OUTPUT_DIR = 'hasil_ppg_rx'
WINDOW_S   = 10

PEAK_DIST  = 40            # jarak min antar puncak (sampel) untuk deteksi HR
PEAK_PROM  = 0.1           # prominensi min (unit PPG)
SNR_WIN_S  = 5

THROUGHPUT_WIN_S = 3
# Berhenti otomatis kalau tidak ada data sama sekali selama sekian detik
# (setelah minimal 1 paket pernah diterima). Tutup jendela dashboard juga bisa.
IDLE_STOP_S = 30.0

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PPG_PATH    = os.path.join(SCRIPT_DIR, PPG_FILE)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
os.makedirs(OUTPUT_PATH, exist_ok=True)

WINDOW_N = WINDOW_S * FS

buf_rx  = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)
rx_rows = []
lock     = threading.Lock()
stop_evt = threading.Event()

stats = {
    'rx_pkt': 0, 'loss_pkt': 0, 'loss_pct': 0.0,
    'snr_db': 0.0, 'hr_rx': 0.0, 'segment_rx': -1,
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


meter_rx = ThroughputMeter()


class HRDetector:
    """Estimasi Heart Rate dari puncak gelombang PPG."""
    def __init__(self, fs=FS, win_s=5):
        self.fs  = fs
        self.buf = deque(maxlen=win_s * fs)
        self.hr  = 0.0
        self.n   = 0

    def update(self, val):
        self.buf.append(val)
        self.n += 1
        if self.n % (self.fs // 2) != 0 or len(self.buf) < PEAK_DIST * 2:
            return self.hr
        arr = np.array(self.buf)
        peaks, _ = find_peaks(arr, distance=PEAK_DIST, prominence=PEAK_PROM)
        if len(peaks) >= 2:
            self.hr = 60.0 / (np.diff(peaks) / self.fs).mean()
        return self.hr


def load_ppg(filepath, segmen_dipilih=None, seg_duration_s=None, fs=FS):
    """Mengembalikan (sig, seg_boundaries, seg_labels).
    seg_boundaries/seg_labels dipakai untuk menentukan SEGMEN AKTIF dari
    nomor SEQ yang diterima (lihat cari_segmen_dari_seq), supaya dashboard
    bisa menampilkan segmen mana (0/20/25) yang sedang ditransmisikan."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    print(f"[PPG] Loading '{filepath}'...")
    data = np.load(filepath, allow_pickle=True)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_segmen_total = data.shape[0]

    if segmen_dipilih is not None:
        idx_valid = [i for i in segmen_dipilih if 0 <= i < n_segmen_total]
        if not idx_valid:
            raise ValueError(f"[PPG] Tidak ada segmen valid dari {segmen_dipilih}")
    else:
        idx_valid = list(range(n_segmen_total))

    # FIX: potong/ulang tiap segmen ke seg_duration_s * fs sampel, SAMA PERSIS
    # dengan cara tx_ppg_only.py memotong sinyalnya sebelum dikirim. Tanpa ini,
    # TOTAL_TARGET bisa jauh lebih besar dari jumlah paket yang benar-benar
    # dikirim TX, sehingga rasio keberhasilan di RX kelihatan anjlok padahal
    # transmisinya sendiri baik-baik saja.
    seg_len_target = int(seg_duration_s * fs) if seg_duration_s else None

    segments = []
    for i in idx_valid:
        s = data[i].astype(float)
        if seg_len_target is not None:
            if len(s) >= seg_len_target:
                s = s[:seg_len_target]
            else:
                reps = int(np.ceil(seg_len_target / len(s)))
                s = np.tile(s, reps)[:seg_len_target]
        segments.append(s)

    seg_lengths    = [len(s) for s in segments]
    seg_labels     = idx_valid                     # nomor segmen asli, mis. [0, 20, 25]
    seg_boundaries = np.cumsum(seg_lengths)        # index akhir kumulatif tiap segmen

    sig = np.concatenate(segments)
    print(f"[PPG] {len(sig):,} sampel siap sebagai referensi (target total paket)")
    return sig, seg_boundaries, seg_labels


def cari_segmen_dari_seq(seq, seg_boundaries, seg_labels):
    """Tentukan segmen aktif (nomor segmen asli, mis. 0/20/25) HANYA dari
    nomor SEQ + konfigurasi lokal RX (SEGMEN_DIPILIH/SEG_DURATION_S), tanpa
    perlu field tambahan apa pun dari paket radio."""
    pos = int(np.searchsorted(seg_boundaries, seq, side='right'))
    pos = min(max(pos, 0), len(seg_labels) - 1)
    return seg_labels[pos]


# Sinyal asli, dipakai untuk hitung SNR & sebagai target total paket.
# HARUS identik urutan/konstruksinya dengan yang dipakai tx_ppg_only.py.
_sig_asli, _seg_boundaries, _seg_labels = load_ppg(
    PPG_PATH, segmen_dipilih=SEGMEN_DIPILIH, seg_duration_s=SEG_DURATION_S, fs=FS
)
TOTAL_TARGET = len(_sig_asli)


def nilai_asli_dari_seq(seq):
    if 0 <= seq < TOTAL_TARGET:
        return float(_sig_asli[seq])
    return float(_sig_asli[seq % TOTAL_TARGET])   # jaga-jaga kalau SEQ melebihi target


# ─── RINGKASAN CSV ───────────────────────────────────────────
def simpan_ringkasan(rows, snapshot, elapsed_total_s):
    ring_path = os.path.join(OUTPUT_PATH, 'ringkasan_rx.csv')

    if not rows:
        with open(ring_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['metrik', 'nilai'])
            w.writerow(['catatan', 'tidak ada paket diterima'])
        print(f"[RX] Ringkasan (kosong) -> {ring_path}")
        return

    thr_rx   = [r['thr_rx_bps']  for r in rows if r['thr_rx_bps']  > 0]
    pps_rx   = [r['pkt_rate_rx'] for r in rows if r['pkt_rate_rx'] > 0]
    snr_avg  = np.mean([r['snr'] for r in rows if r['snr'] > 0]) if any(r['snr'] > 0 for r in rows) else 0.0
    total    = len(rows)
    loss_final = snapshot['loss_pct']
    sukses_pct = (total / TOTAL_TARGET * 100) if TOTAL_TARGET > 0 else 0.0

    with open(ring_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metrik', 'nilai'])
        w.writerow(['port', PORT])
        w.writerow(['sumber_file', PPG_FILE])
        w.writerow(['segmen_dipilih', SEGMEN_DIPILIH])
        w.writerow(['fs_hz', FS])
        w.writerow(['target_paket', TOTAL_TARGET])
        w.writerow(['paket_diterima', total])
        w.writerow(['durasi_sesi_s', round(elapsed_total_s, 3)])
        w.writerow(['loss_jaringan_seq_pct', round(loss_final, 2)])
        w.writerow(['rasio_keberhasilan_pct', round(sukses_pct, 1)])
        w.writerow(['snr_db_avg', round(snr_avg, 2)])
        w.writerow(['hr_rx_terakhir_bpm', round(snapshot['hr_rx'], 1)])
        w.writerow(['bytes_rx', snapshot['bytes_rx']])
        if thr_rx:
            w.writerow(['thr_rx_bps_avg', round(float(np.mean(thr_rx)), 1)])
            w.writerow(['thr_rx_kbps_avg', round(float(np.mean(thr_rx)) / 1000, 3)])
            w.writerow(['thr_rx_bps_peak', round(float(np.max(thr_rx)), 1)])
        if pps_rx:
            w.writerow(['pkt_rate_rx_avg', round(float(np.mean(pps_rx)), 2)])

    print(f"[RX] Ringkasan -> {ring_path}")


# ─── THREAD RX ────────────────────────────────────────────────
def thread_rx():
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(1.5)
        print(f"[RX] Terhubung ke {PORT}")
    except Exception as e:
        print(f"[RX] ERROR buka port: {e} — RX dinonaktifkan.")
        stop_evt.set()
        return

    detector = HRDetector()
    snr_buf  = deque(maxlen=SNR_WIN_S * FS)   # (expected, received)

    csv_path = os.path.join(OUTPUT_PATH, 'rx_data.csv')
    csv_f    = open(csv_path, 'w', newline='', encoding='utf-8')
    writer   = csv.writer(csv_f)
    writer.writerow([
        'timestamp_s', 'seq', 'segment', 'ppg_raw', 'ppg_val', 'ppg_asli', 'hr_rx',
        'loss_pct', 'snr_db',
        'thr_rx_bps', 'thr_rx_kbps', 'pkt_rate_rx',
        'valid', 'corrupt_reason'
    ])

    t_start      = time.time()
    last_seq     = -1
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
            print(f"[RX] Read error: {e}")
            break

        if not chunk:
            time.sleep(0.001)
            continue

        got_any_data = True
        last_data_t  = time.perf_counter()
        buf         += chunk

        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue

            try:
                text = line.decode('utf-8', errors='replace')
                if not (text.startswith('START') and 'END' in text):
                    continue

                fields = {}
                for part in text.replace('START|', '').replace('|END', '').split('|'):
                    if ':' in part:
                        k, v = part.split(':', 1)
                        fields[k] = v

                ts_s = round(time.time() - t_start, 3)

                if not all(k in fields for k in ('PPG', 'SEQ')):
                    writer.writerow([ts_s, '', '', '', '', '', '', '', '', '', '', '', 0, 'missing_field'])
                    csv_f.flush()
                    continue

                raw = int(fields['PPG'])
                seq = int(fields['SEQ'])
                val = raw / 10000.0

                is_valid, corrupt_reason = True, ''
                if not (0 <= raw <= 40020):
                    is_valid, corrupt_reason = False, f'ppg_out_of_range({raw})'

                if is_valid and last_seq >= 0:
                    gap = seq - last_seq - 1
                    if gap > 0:
                        with lock:
                            stats['loss_pkt'] += gap
                if is_valid:
                    last_seq = max(last_seq, seq)

                seg_idx = cari_segmen_dari_seq(seq, _seg_boundaries, _seg_labels)

                hr_rx = 0.0
                asli_val = ''
                bps_rx, pps_rx = 0.0, 0.0

                with lock:
                    stats['rx_pkt']   += 1
                    stats['bytes_rx'] += len(line) + 1
                    stats['segment_rx'] = seg_idx
                    total_exp = stats['rx_pkt'] + stats['loss_pkt']
                    stats['loss_pct'] = (stats['loss_pkt'] / total_exp * 100
                                          if total_exp > 0 else 0.0)
                    lp_snap = stats['loss_pct']

                if is_valid:
                    # Throughput dihitung HANYA dari paket yang valid, sama
                    # seperti versi ABP — bukan dari semua byte mentah yang
                    # masuk lewat serial (termasuk paket corrupt/invalid),
                    # supaya angkanya mencerminkan data yang benar-benar
                    # berguna, bukan trafik mentah.
                    meter_rx.update(len(line) + 1, n_pkts=1)
                    bps_rx, pps_rx = meter_rx.get()
                    with lock:
                        stats['thr_rx_bps']  = bps_rx
                        stats['thr_rx_kbps'] = bps_rx / 1000.0
                        stats['pkt_rate_rx'] = pps_rx

                    hr_rx = detector.update(val)

                    asli_val = nilai_asli_dari_seq(seq)
                    snr_buf.append((asli_val, val))
                    if len(snr_buf) >= FS:
                        exp_arr = np.array([p[0] for p in snr_buf])
                        rx_arr  = np.array([p[1] for p in snr_buf])
                        exp_c   = exp_arr - exp_arr.mean()
                        err     = exp_arr - rx_arr
                        ps, pn  = np.mean(exp_c ** 2), np.mean(err ** 2)
                        snr = float(np.clip(10 * np.log10(ps / pn) if pn > 1e-12 else 99.0, -10, 60))
                        with lock:
                            stats['snr_db'] = snr

                    if hr_rx > 0:
                        with lock:
                            stats['hr_rx'] = hr_rx
                    with lock:
                        buf_rx.append(val)

                with lock:
                    snr_snap = stats['snr_db']

                writer.writerow([
                    ts_s, seq, seg_idx, raw, round(val, 4) if is_valid else '',
                    round(asli_val, 4) if asli_val != '' else '',
                    round(hr_rx, 1) if hr_rx else '',
                    round(lp_snap, 2), round(snr_snap, 2),
                    round(bps_rx, 1), round(bps_rx / 1000.0, 3), round(pps_rx, 2),
                    1 if is_valid else 0, corrupt_reason
                ])
                csv_f.flush()

                with lock:
                    rx_rows.append({'seq': seq, 'loss_pct': lp_snap,
                                     'thr_rx_bps': bps_rx, 'pkt_rate_rx': pps_rx,
                                     'valid': 1 if is_valid else 0, 'snr': snr_snap})

            except Exception:
                continue

    csv_f.close()
    ser.close()
    stop_evt.set()
    print(f"[RX] Data disimpan ke {csv_path}")


# ─── RINGKASAN KONSOLE + CSV ──────────────────────────────────
def cetak_ringkasan(elapsed_total_s):
    with lock:
        rows = list(rx_rows)
        s    = dict(stats)

    if not rows:
        print("\n[!] Tidak ada data RX.")
        simpan_ringkasan(rows, s, elapsed_total_s)
        return

    total    = len(rows)
    loss_final = s['loss_pct']
    thr_rx   = [r['thr_rx_bps']  for r in rows if r['thr_rx_bps']  > 0]
    pps_rx   = [r['pkt_rate_rx'] for r in rows if r['pkt_rate_rx'] > 0]
    snr_avg  = np.mean([r['snr'] for r in rows if r['snr'] > 0]) if any(r['snr'] > 0 for r in rows) else 0.0
    sukses_pct = (total / TOTAL_TARGET * 100) if TOTAL_TARGET > 0 else 0.0

    print('\n' + '=' * 60)
    print('  RINGKASAN AKHIR — PPG Monitor (RX)')
    print('=' * 60)
    print(f"  Segmen ditransmisikan (asumsi) : {SEGMEN_DIPILIH}")
    print(f"  Target total paket  : {TOTAL_TARGET:,}")
    print(f"  Total paket RX      : {total:,}")
    print(f"  Loss jaringan (SEQ) : {loss_final:.2f}%")
    print(f"  Rasio keberhasilan  : {sukses_pct:.1f}%  ({total:,}/{TOTAL_TARGET:,} paket)")
    print(f"  SNR rata-rata       : {snr_avg:.2f} dB")
    print(f"  HR terakhir         : {s['hr_rx']:.1f} BPM")
    print(f"  Segmen terakhir     : {s['segment_rx'] if s['segment_rx'] >= 0 else '--'}")
    if thr_rx:
        print(f"  Throughput RX       : {np.mean(thr_rx):.1f} bps ({np.mean(thr_rx)/1000:.3f} kbps)")
    if pps_rx:
        print(f"  Laju paket RX       : {np.mean(pps_rx):.1f} pkt/s")
    print(f"  Byte RX             : {s['bytes_rx']:,} byte")
    print('=' * 60)

    simpan_ringkasan(rows, s, elapsed_total_s)
    print(f"[✓] Semua file CSV ada di folder: {os.path.abspath(OUTPUT_PATH)}")
    print("    - rx_data.csv       (detail paket RX)")
    print("    - ringkasan_rx.csv  (rekap)\n")


# ─── DASHBOARD ───────────────────────────────────────────────
def run_dashboard():
    plt.rcParams.update({
        'figure.facecolor': '#0D1117', 'axes.facecolor': '#161B22',
        'axes.edgecolor': '#30363D',   'axes.labelcolor': '#C9D1D9',
        'xtick.color': '#8B949E',      'ytick.color': '#8B949E',
        'grid.color': '#21262D',       'text.color': '#C9D1D9',
        'font.family': 'monospace',
    })

    t_axis = np.linspace(0, WINDOW_S, WINDOW_N)
    fig = plt.figure(figsize=(11, 7))
    sup_title = fig.suptitle(f'PPG Monitor — RX SAJA  |  Port: {PORT}\nSegmen aktif: --',
                              color='#3FB950', fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                            left=0.09, right=0.96, top=0.88, bottom=0.08,
                            height_ratios=[1.4, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    ax1.set_title('Sinyal PPG — RX (yang diterima)', color='#3FB950', fontsize=10)
    ax1.set_xlim(0, WINDOW_S)
    ax1.set_ylim(-0.1, 4.5)
    ax1.set_xlabel('Waktu (detik)', fontsize=9)
    ax1.set_ylabel('Amplitudo', fontsize=9)
    ax1.grid(True, alpha=0.3)

    line_rx, = ax1.plot(t_axis, list(buf_rx), color='#3FB950', lw=1.0)

    lbl_hr = ax1.text(0.985, 0.93, 'HR: -- BPM', transform=ax1.transAxes, fontsize=11,
                       fontweight='bold', color='#F4C275', ha='right', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#0D1117', alpha=0.85))

    stats_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes,
                          fontsize=10.5, verticalalignment='top',
                          fontfamily='monospace', color='#E3B341',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='#161B22', alpha=0.85))

    def update(_frame):
        if stop_evt.is_set():
            try:
                ani.event_source.stop()
            except Exception:
                pass
            plt.close(fig)
            return line_rx, lbl_hr, sup_title, stats_text

        with lock:
            rx = np.array(buf_rx)
            s = dict(stats)

        line_rx.set_ydata(rx)
        if np.ptp(rx) > 0:
            margin = np.ptp(rx) * 0.12
            ax1.set_ylim(np.min(rx) - margin, np.max(rx) + margin)

        hr_val = s['hr_rx']
        lbl_hr.set_text(f"HR: {hr_val:.0f} BPM" if hr_val > 0 else 'HR: -- BPM')
        lbl_hr.set_color('#D85A30' if hr_val > 100 else
                         '#E24B4A' if 0 < hr_val < 60 else '#F4C275')

        sukses_str = f"{s['rx_pkt']/TOTAL_TARGET*100:.1f}%" if TOTAL_TARGET > 0 else "--"
        seg_now = s['segment_rx']
        sup_title.set_text(
            f'PPG Monitor — RX SAJA  |  Port: {PORT}\n'
            f'Segmen aktif: {seg_now if seg_now >= 0 else "--"}'
        )
        txt = (
            f"RX paket    : {s['rx_pkt']:>8,} / {TOTAL_TARGET:,}  ({sukses_str})\n"
            f"Loss SEQ    : {s['loss_pct']:>7.2f} %\n"
            f"SNR         : {s['snr_db']:>7.2f} dB\n"
            f"Thr RX      : {s['thr_rx_kbps']:>7.2f} kbps\n"
            f"Laju RX     : {s['pkt_rate_rx']:>7.1f} pkt/s\n"
            f"Byte RX     : {s['bytes_rx']/1024:>7.1f} KB"
        )
        stats_text.set_text(txt)
        fig.canvas.draw_idle()
        return line_rx, lbl_hr, sup_title, stats_text

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)

    def on_close(event):
        try:
            ani.event_source.stop()
        except Exception:
            pass
        stop_evt.set()

    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()


# ─── MAIN ────────────────────────────────────────────────────
def main():
    print('=' * 70)
    print('  PPG Serial Monitor — RX SAJA (proses/laptop terpisah)')
    print('=' * 70)
    print(f'  Port         : {PORT}')
    print(f'  Sumber (ref) : {PPG_PATH}')
    print(f'  Segmen       : {SEGMEN_DIPILIH}')
    print(f'  Target paket : {TOTAL_TARGET:,}')
    print(f'  Idle-stop    : {IDLE_STOP_S:.0f}s tanpa data -> berhenti otomatis')
    print(f'  Output       : {OUTPUT_PATH}/  (rx_data.csv, ringkasan_rx.csv)')
    print('=' * 70)
    print('\nTutup jendela dashboard atau tunggu idle-stop untuk berhenti.\n')

    t_start_global = time.time()
    t = threading.Thread(target=thread_rx, daemon=True)
    t.start()

    try:
        run_dashboard()
    except KeyboardInterrupt:
        print('\n[Main] Dihentikan.')
        stop_evt.set()

    t.join(timeout=IDLE_STOP_S + 5)

    elapsed_total_s = time.time() - t_start_global
    cetak_ringkasan(elapsed_total_s)
    print('[Main] Selesai.')


if __name__ == '__main__':
    main()