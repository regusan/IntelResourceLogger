#!/usr/bin/env python3

"""
monitor_gui.py

各種システムリソース（CPU使用率, RAPL電力, 周波数）を
高頻度で監視し、CSVファイルへの出力とリアルタイムグラフ描画を行う。

グラフの凡例（レジェンド）をクリックすると、
そのグラフの表示/非表示を切り替えられます。

実行方法:
python3 monitor_gui.py

(事前に .../energy_uj, .../gt_act_freq_mhz 等のパーミッションを変更すること)

オプション:
  -int, --interval  計測間隔 (ミリ秒) (デフォルト: 20)
  -dur, --duration  計測時間 (秒) (デフォルト: 600)
  -o, --output      出力CSVファイル名 (デフォルト: monitor_log.csv)
  -p, --points      グラフに描画する最大データ点数 (デフォルト: 500)
"""

import sys
import time
import signal
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque

# PyQtGraph と Qt
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np

# --- 監視対象の設定 ---

RAPL_TARGETS: List[Tuple[str, str]] = [
    ("/sys/class/powercap/intel-rapl:0/energy_uj", "Power_Package_W"),
    ("/sys/class/powercap/intel-rapl:0:0/energy_uj", "Power_Core_W"),
    ("/sys/class/powercap/intel-rapl:0:1/energy_uj", "Power_Uncore_W"),
]

PROC_STAT_PATH = "/proc/stat"
IGPU_FREQ_PATH = "/sys/class/drm/card0/gt_act_freq_mhz"
CPU_FREQ_PATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"

# --- CPU/RAPL/Freq 読み取り ---

@dataclass
class CpuTimes:
    user: int = 0
    nice: int = 0
    system: int = 0
    idle: int = 0
    iowait: int = 0
    irq: int = 0
    softirq: int = 0
    steal: int = 0

    def total(self) -> int:
        return (self.user + self.nice + self.system + self.idle +
                self.iowait + self.irq + self.softirq + self.steal)

    def idle_total(self) -> int:
        return self.idle + self.iowait

def get_cpu_times(path: str) -> Optional[CpuTimes]:
    try:
        with open(path, 'r') as f:
            line = f.readline()
            parts = line.split()
            if not parts or parts[0] != 'cpu':
                return None
            times = [int(p) for p in parts[1:9]]
            return CpuTimes(*times)
    except (FileNotFoundError, IOError, ValueError, PermissionError) as e:
        print(f"Error reading {path}: {e}", file=sys.stderr, flush=True)
        return None

def get_rapl_energy(path: str) -> int:
    try:
        with open(path, 'r') as f:
            return int(f.read())
    except (FileNotFoundError, IOError, ValueError, PermissionError):
        return -1

def get_freq_mhz(path: str, divisor: int = 1) -> float:
    try:
        with open(path, 'r') as f:
            value_str = f.read().strip()
            if not value_str:
                return -1.0
            return float(value_str) / divisor
    except (FileNotFoundError, IOError, ValueError, PermissionError):
        return -1.0


# --- 監視スレッド ---

class MonitorThread(QtCore.QThread):
    """
    別スレッドで監視を実行し、結果をシグナルでGUIに送信する
    """
    # { 'Time_ms': 123.4, 'CPU_Usage_Percent': 15.2, ... }
    data_ready = QtCore.Signal(dict) 

    def __init__(self, interval_ms, duration_sec, output_file):
        super().__init__()
        self.interval_sec = interval_ms / 1000.0
        self.duration_sec = duration_sec
        self.output_file = output_file
        self.running = True
        self.csv_file = None
        self.header_cols = [
            "Time_ms", "DeltaT_ms", "CPU_Usage_Percent",
            "iGPU_Freq_MHz", "CPU0_Freq_MHz"
        ]
        self.header_cols.extend([name for _, name in RAPL_TARGETS])

    def stop(self):
        self.running = False

    def run(self):
        try:
            self.csv_file = open(self.output_file, 'w', encoding='utf-8')
            self.csv_file.write(",".join(self.header_cols) + "\n")
        except IOError as e:
            print(f"Error opening CSV: {e}", file=sys.stderr, flush=True)
            self.running = False
            return

        # 初期値の取得
        prev_cpu_times = get_cpu_times(PROC_STAT_PATH)
        if prev_cpu_times is None:
            self.running = False
            return
            
        prev_rapl_energy = [get_rapl_energy(path) for path, _ in RAPL_TARGETS]

        start_time = time.monotonic()
        prev_tick_time = start_time
        next_tick = start_time + self.interval_sec

        while self.running:
            # 1. スリープ
            current_time = time.monotonic()
            sleep_duration = next_tick - current_time
            if sleep_duration > 0:
                time.sleep(sleep_duration) # QThread.msleep() よりも高精度

            current_tick_time = time.monotonic()
            elapsed_ms = (current_tick_time - start_time) * 1000.0
            delta_t_sec = current_tick_time - prev_tick_time

            if (current_tick_time - start_time) >= self.duration_sec:
                self.running = False
                break

            # 2. データ取得と計算
            current_data = {"Time_ms": elapsed_ms, "DeltaT_ms": delta_t_sec * 1000.0}

            # CPU
            current_cpu_times = get_cpu_times(PROC_STAT_PATH)
            cpu_usage = 0.0
            if current_cpu_times:
                delta_total = current_cpu_times.total() - prev_cpu_times.total()
                delta_idle = current_cpu_times.idle_total() - prev_cpu_times.idle_total()
                if delta_total > 0:
                    cpu_usage = 100.0 * (delta_total - delta_idle) / delta_total
                prev_cpu_times = current_cpu_times
            current_data["CPU_Usage_Percent"] = cpu_usage

            # Freq
            current_data["iGPU_Freq_MHz"] = get_freq_mhz(IGPU_FREQ_PATH)
            current_data["CPU0_Freq_MHz"] = get_freq_mhz(CPU_FREQ_PATH, divisor=1000)

            # RAPL
            current_rapl_energy: List[int] = []
            for i, (path, name) in enumerate(RAPL_TARGETS):
                current_energy = get_rapl_energy(path)
                power_w = 0.0
                if (current_energy != -1 and prev_rapl_energy[i] != -1 and delta_t_sec > 0):
                    delta_energy_uj = current_energy - prev_rapl_energy[i]
                    if delta_energy_uj < 0: delta_energy_uj = 0
                    power_w = (delta_energy_uj / 1_000_000.0) / delta_t_sec
                current_data[name] = power_w
                current_rapl_energy.append(current_energy)
            prev_rapl_energy = current_rapl_energy

            # 3. CSV書き込み
            try:
                # header_cols の順序でCSV行を構築
                row_values = [f"{current_data[col]:.3f}" for col in self.header_cols]
                self.csv_file.write(",".join(row_values) + "\n")
            except (IOError, ValueError) as e:
                print(f"Error writing CSV: {e}", file=sys.stderr, flush=True)
                self.running = False
            
            # 4. GUIへシグナル送信
            self.data_ready.emit(current_data)

            # 5. 次回時刻の更新
            prev_tick_time = current_tick_time
            next_tick += self.interval_sec
        
        # 終了処理
        if self.csv_file:
            self.csv_file.close()
            print(f"\nMonitoring stopped. Log saved to {self.output_file}", flush=True)

# --- GUI / メイン処理 ---

class MonitorWindow(pg.GraphicsLayoutWidget):
    
    def __init__(self, max_points: int, interval_ms: int):
        super().__init__(show=True)
        self.setWindowTitle(f'Real-time Monitor (Interval: {interval_ms}ms)')
        self.resize(1000, 600)
        
        self.max_points = max_points
        self.data: Dict[str, deque] = {}
        self.curves: Dict[str, pg.PlotDataItem] = {}
        self.time_deque = deque(maxlen=self.max_points)
        
        self.plot_item = self.addPlot(row=0, col=0)
        self.plot_item.showGrid(x=True, y=True, alpha=0.5)
        self.plot_item.setLabel('bottom', 'Time (s)')
        self.plot_item.addLegend()

        # 凡例で切り替えるプロット対象
        # (名前, 色)
        self.plot_targets = [
            ("CPU_Usage_Percent", 'r'),
            ("Power_Package_W", 'c'),
            ("Power_Core_W", 'm'),
            ("iGPU_Freq_MHz", 'g'),
            ("CPU0_Freq_MHz", 'y'),
        ]

        # データバッファとプロット曲線を初期化
        for name, color in self.plot_targets:
            self.data[name] = deque(maxlen=self.max_points)
            self.curves[name] = self.plot_item.plot(
                pen=pg.mkPen(color, width=2), 
                name=name
            )
    @QtCore.pyqtSlot(dict)
    def update_plot(self, data: dict):
        """シグナル受信時にグラフを更新するスロット"""
        
        # データをデキューに追加
        self.time_deque.append(data["Time_ms"] / 1000.0) # ms -> s
        for name, _ in self.plot_targets:
            if name in data:
                self.data[name].append(data[name])

        # NumPy配列に変換 (高速化のため)
        x_data = np.array(self.time_deque, dtype=float)
        
        # 各曲線のデータを更新
        for name, _ in self.plot_targets:
            y_data = np.array(self.data[name], dtype=float)
            self.curves[name].setData(x=x_data, y=y_data)


def main():
    # --- 引数解析 ---
    parser = argparse.ArgumentParser(
        description="CPU/RAPL/Freq Monitor with GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-int", "--interval", type=int, default=20,
        help="計測間F (ミリ秒) (デフォルト: 20ms)"
    )
    parser.add_argument(
        "-dur", "--duration", type=int, default=600,
        help="計測時間 (秒) (デフォルト: 600)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="monitor_gui_log.csv",
        help="出力CSVファイル名 (デフォルト: monitor_gui_log.csv)"
    )
    parser.add_argument(
        "-p", "--points", type=int, default=500,
        help="グラフに描画する最大データ点数 (デフォルト: 500)"
    )
    # console出力はGUIが担当するので -c は削除
    args = parser.parse_args()

    # 高頻度すぎるインターバルの警告
    if args.interval < 10:
        print(f"Warning: Interval {args.interval}ms is very fast and may "
              "impact performance or stability.", file=sys.stderr, flush=True)

    # --- アプリケーション起動 ---
    
    # pg.mkQApp() は QApplication.instance() があればそれを返し、なければ作成する
    app = pg.mkQApp("Monitor GUI")
    
    # メインウィンドウ作成
    window = MonitorWindow(max_points=args.points, interval_ms=args.interval)

    # 監視スレッド作成
    monitor_thread = MonitorThread(
        interval_ms=args.interval,
        duration_sec=args.duration,
        output_file=args.output
    )

    # スレッドのシグナルをGUIのスロットに接続
    monitor_thread.data_ready.connect(window.update_plot)
    
    # アプリ終了時にスレッドも停止する
    app.aboutToQuit.connect(monitor_thread.stop)

    # Ctrl+C ハンドラ (ターミナルとGUI両方で動作)
    def sigint_handler(*_):
        print("\nCtrl+C detected. Stopping...", flush=True)
        monitor_thread.stop()
        app.quit()
        
    signal.signal(signal.SIGINT, sigint_handler)
    # GUIがCtrl+Cを検知できるようにタイマーを設定
    timer = QtCore.QTimer()
    timer.start(500) # 500msごとにPythonインタプリタを起動
    timer.timeout.connect(lambda: None) 

    # 監視スレッド開始
    monitor_thread.start()

    print(f"Monitoring started. (Interval: {args.interval}ms, Duration: {args.duration}s)")
    print(f"Saving log to: {args.output}")
    print("Press Ctrl+C in terminal or close window to stop.")

    # GUIイベントループ実行
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app.exec_()
    
    # 終了待機
    monitor_thread.wait(2000) # 最大2秒待機
    print("Application finished.", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())