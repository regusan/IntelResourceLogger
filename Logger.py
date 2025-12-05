#!/usr/bin/env python3

"""
monitor_gui_v3.py

システムリソースを監視し、CSV出力とリアルタイムグラフ描画を行う。
グラフ領域を「周波数/使用率」と「電力」の2つに分割。

実行方法:
python3 monitor_gui_v3.py

(事前に .../energy_uj, .../gt_act_freq_mhz 等のパーミッションを変更すること)
"""

import sys
import time
import signal
import argparse
import subprocess
import os
import glob
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque

# PyQtGraph と Qt
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np

# --- テーマ設定 (白背景) ---
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# --- 監視対象の設定 (変更なし) ---

RAPL_TARGETS: List[Tuple[str, str]] = [
    ("/sys/class/powercap/intel-rapl:0/energy_uj", "Power_Package_W"),
    ("/sys/class/powercap/intel-rapl:0:0/energy_uj", "Power_Core_W"),
    ("/sys/class/powercap/intel-rapl:0:1/energy_uj", "Power_Uncore_W"),
]

PROC_STAT_PATH = "/proc/stat"

# --- 周波数ターゲットのクラス定義 ---

class FreqTarget(ABC):
    """周波数取得の基底クラス"""
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def get_freq_mhz(self) -> float:
        """周波数をMHz単位で返す。エラー時は-1.0を返す"""
        pass

class SingleFileFreqTarget(FreqTarget):
    """単一ファイルから周波数を読み取る (iGPU, NPUなど)"""
    def __init__(self, path: str, name: str, divisor: int = 1):
        super().__init__(name)
        self.path = path
        self.divisor = divisor
    
    def get_freq_mhz(self) -> float:
        try:
            with open(self.path, 'r') as f:
                value_str = f.read().strip()
                if not value_str:
                    return -1.0
                return float(value_str) / self.divisor
        except (FileNotFoundError, IOError, ValueError, PermissionError):
            return -1.0

class GroupedCpuFreqTarget(FreqTarget):
    """特定の最大周波数を持つCPUコアグループの平均周波数を計算"""
    
    def __init__(self, name: str, paths: List[str]):
        super().__init__(name)
        self.paths = paths
    
    def get_freq_mhz(self) -> float:
        values = []
        for path in self.paths:
            try:
                with open(path, 'r') as f:
                    value_str = f.read().strip()
                    if value_str:
                        values.append(float(value_str) / 1000)  # kHz → MHz
            except (FileNotFoundError, IOError, ValueError, PermissionError):
                pass
        if not values:
            return -1.0
        return sum(values) / len(values)  # avg


def discover_cpu_freq_groups() -> Dict[int, List[str]]:
    """CPUコアを最大周波数でグループ分けし、{最大周波数(MHz): [パスリスト]} を返す"""
    all_cpu_dirs = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*"))
    groups: Dict[int, List[str]] = {}
    
    for cpu_dir in all_cpu_dirs:
        freq_path = f"{cpu_dir}/cpufreq/scaling_cur_freq"
        max_freq_path = f"{cpu_dir}/cpufreq/cpuinfo_max_freq"
        
        if not os.path.exists(freq_path):
            continue
        
        try:
            with open(max_freq_path, 'r') as f:
                max_freq_khz = int(f.read().strip())
            max_freq_mhz = max_freq_khz // 1000  # kHz → MHz (整数)
            
            if max_freq_mhz not in groups:
                groups[max_freq_mhz] = []
            groups[max_freq_mhz].append(freq_path)
        except (FileNotFoundError, IOError, ValueError, PermissionError):
            pass
    
    return groups


def create_cpu_freq_targets() -> List[FreqTarget]:
    """最大周波数グループごとにFreqTargetを動的に生成"""
    targets: List[FreqTarget] = []
    groups = discover_cpu_freq_groups()
    
    # 最大周波数の降順でソート (高いクロックが先)
    for max_freq_mhz in sorted(groups.keys(), reverse=True):
        paths = groups[max_freq_mhz]
        name = f"CPU_{max_freq_mhz}MHz_Avg"
        targets.append(GroupedCpuFreqTarget(name, paths))
    
    return targets


# 周波数監視ターゲット (動的に生成)
FREQ_TARGETS: List[FreqTarget] = [
    SingleFileFreqTarget("/sys/class/drm/card1/gt_act_freq_mhz", "iGPU_Freq_MHz"),
    *create_cpu_freq_targets(),  # CPUグループを動的に追加
    SingleFileFreqTarget("/sys/devices/pci0000:00/0000:00:0b.0/npu_current_frequency_mhz", "NPU_Freq_MHz"),
]

# --- CPU/RAPL/Freq 読み取り (変更なし) ---

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
            if not parts or parts[0] != 'cpu': return None
            times = [int(p) for p in parts[1:9]]
            return CpuTimes(*times)
    except (FileNotFoundError, IOError, ValueError, PermissionError) as e:
        print(f"Error reading {path}: {e}", file=sys.stderr, flush=True)
        return None

def get_rapl_energy(path: str) -> int:
    try:
        with open(path, 'r') as f: return int(f.read())
    except (FileNotFoundError, IOError, ValueError, PermissionError):
        return -1

def get_freq_mhz(path: str, divisor: int = 1) -> float:
    try:
        with open(path, 'r') as f:
            value_str = f.read().strip()
            if not value_str: return -1.0
            return float(value_str) / divisor
    except (FileNotFoundError, IOError, ValueError, PermissionError):
        return -1.0

# --- 監視スレッド (変更なし) ---

class ExponentialMovingAverage:
    """指数移動平均 (EMA) を計算するクラス
    S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
    """
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

class MonitorThread(QtCore.QThread):
    data_ready = QtCore.Signal(dict) 

    def __init__(self, interval_ms, duration_sec, output_file):
        super().__init__()
        self.interval_sec = interval_ms / 1000.0
        self.duration_sec = duration_sec
        self.output_file = output_file
        self.running = True
        self.is_paused = False
        self.reset_on_resume = False
        self.csv_file = None
        self.header_cols = [
            "Time_ms", "DeltaT_ms",
        ]
        self.header_cols.extend([target.name for target in FREQ_TARGETS])
        self.header_cols.extend([name for _, name in RAPL_TARGETS])
        
        # 電力値の指数移動平均用 (alpha=0.1)
        self.power_smoothers = {
            name: ExponentialMovingAverage(alpha=0.2) 
            for _, name in RAPL_TARGETS
        }

    def stop(self):
        self.is_paused = False
        self.running = False

    def pause(self):
        print("Pausing monitor thread...", flush=True)
        self.is_paused = True
    
    def resume(self):
        print("Resuming monitor thread...", flush=True)
        self.is_paused = False
        self.reset_on_resume = True

    def run(self):
        try:
            self.csv_file = open(self.output_file, 'w', encoding='utf-8')
            self.csv_file.write(",".join(self.header_cols) + "\n")
        except IOError as e:
            print(f"Error opening CSV: {e}", file=sys.stderr, flush=True)
            self.running = False
            return

        prev_cpu_times = get_cpu_times(PROC_STAT_PATH)
        if prev_cpu_times is None:
            self.running = False
            return
        prev_rapl_energy = [get_rapl_energy(path) for path, _ in RAPL_TARGETS]

        start_time = time.monotonic()
        prev_tick_time = start_time
        next_tick = start_time + self.interval_sec

        while self.running:
            current_time = time.monotonic()
            sleep_duration = next_tick - current_time
            if sleep_duration > 0:
                time.sleep(sleep_duration)

            while self.is_paused and self.running:
                time.sleep(0.1)
            if not self.running: break
                
            if self.reset_on_resume:
                print("Resuming... recalibrating previous values.", flush=True)
                prev_cpu_times = get_cpu_times(PROC_STAT_PATH)
                if prev_cpu_times is None:
                    self.running = False
                    break
                prev_rapl_energy = [get_rapl_energy(path) for path, _ in RAPL_TARGETS]
                current_tick_time = time.monotonic()
                prev_tick_time = current_tick_time
                next_tick = current_tick_time + self.interval_sec
                self.reset_on_resume = False
                continue 

            current_tick_time = time.monotonic()
            elapsed_ms = (current_tick_time - start_time) * 1000.0
            delta_t_sec = current_tick_time - prev_tick_time

            if (current_tick_time - start_time) >= self.duration_sec:
                self.running = False
                break

            current_data = {"Time_ms": elapsed_ms, "DeltaT_ms": delta_t_sec * 1000.0}

            current_cpu_times = get_cpu_times(PROC_STAT_PATH)
            cpu_usage = 0.0
            if current_cpu_times:
                delta_total = current_cpu_times.total() - prev_cpu_times.total()
                delta_idle = current_cpu_times.idle_total() - prev_cpu_times.idle_total()
                if delta_total > 0:
                    cpu_usage = 100.0 * (delta_total - delta_idle) / delta_total
                prev_cpu_times = current_cpu_times
            current_data["CPU_Usage_Percent"] = cpu_usage

            for target in FREQ_TARGETS:
                current_data[target.name] = target.get_freq_mhz()

            current_rapl_energy: List[int] = []
            for i, (path, name) in enumerate(RAPL_TARGETS):
                current_energy = get_rapl_energy(path)
                power_w = 0.0
                if (current_energy != -1 and prev_rapl_energy[i] != -1 and delta_t_sec > 0):
                    delta_energy_uj = current_energy - prev_rapl_energy[i]
                    if delta_energy_uj < 0: delta_energy_uj = 0
                    power_w = (delta_energy_uj / 1_000_000.0) / delta_t_sec
                
                # 指数移動平均を適用
                smoothed_power = self.power_smoothers[name].update(power_w)
                current_data[name] = smoothed_power
                current_rapl_energy.append(current_energy)
            prev_rapl_energy = current_rapl_energy

            try:
                row_values = [f"{current_data[col]:.3f}" for col in self.header_cols]
                self.csv_file.write(",".join(row_values) + "\n")
            except (IOError, ValueError) as e:
                print(f"Error writing CSV: {e}", file=sys.stderr, flush=True)
                self.running = False
            
            self.data_ready.emit(current_data)

            prev_tick_time = current_tick_time
            next_tick += self.interval_sec
        
        if self.csv_file:
            self.csv_file.close()
            print(f"\nMonitoring stopped. Log saved to {self.output_file}", flush=True)

# --- GUI (レイアウトを変更) ---

class MonitorWindow(pg.GraphicsLayoutWidget):
    
    def __init__(self, max_points: int, interval_ms: int, monitor_thread: MonitorThread):
        super().__init__(show=True)
        self.setWindowTitle(f'Real-time Monitor (Interval: {interval_ms}ms)')
        # 2分割グラフ + ボタン用に高さを増やす
        self.resize(1000, 800) 
        
        self.monitor_thread = monitor_thread
        self.is_paused_gui = False
        
        # --- [変更] グラフ領域を分割 ---
        
        # 1行目: 周波数 / CPU使用率
        self.plot_item_top = self.addPlot(row=0, col=0)
        self.plot_item_top.showGrid(x=True, y=True, alpha=0.5)
        self.plot_item_top.setLabel('bottom', 'Time (s)')
        self.plot_item_top.setLabel('left', 'Freq (MHz) / Usage (%)')
        self.plot_item_top.addLegend()

        # 2行目: 電力
        self.plot_item_bottom = self.addPlot(row=1, col=0)
        self.plot_item_bottom.showGrid(x=True, y=True, alpha=0.5)
        self.plot_item_bottom.setLabel('bottom', 'Time (s)')
        self.plot_item_bottom.setLabel('left', 'Power (W)')
        self.plot_item_bottom.addLegend()

        # [重要] 上下のX軸をリンクさせる
        self.plot_item_bottom.setXLink(self.plot_item_top)
        
        # [変更] Y軸の自動スケールを無効化 (パーセンタイルで制御するため)
        self.plot_item_bottom.enableAutoRange(axis='y', enable=False)

        # 3行目: ボタンレイアウト
        button_layout = self.addLayout(row=2, col=0)
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.setFixedWidth(100)
        self.pause_button.clicked.connect(self.toggle_pause)
        proxy_widget = QtWidgets.QGraphicsProxyWidget()
        proxy_widget.setWidget(self.pause_button)
        button_layout.addItem(proxy_widget)

        # --- [変更] グラフデータ/曲線の初期化 ---
        self.max_points = max_points
        self.data: Dict[str, deque] = {}
        self.curves: Dict[str, pg.PlotDataItem] = {}
        self.time_deque = deque(maxlen=self.max_points)
        
        # [変更] プロット対象を分割 (CPU使用率は除外)
        self.plot_targets_top = []
        # FREQ_TARGETS から動的に追加 (十分な数の色を用意)
        freq_colors = [
            'g',              # green (iGPU)
            (255, 100, 100),  # light red
            (100, 100, 255),  # light blue
            (255, 165, 0),    # orange
            (180, 0, 180),    # purple
            (0, 200, 200),    # cyan
            (150, 150, 0),    # olive
            (255, 100, 200),  # pink
        ]
        for i, target in enumerate(FREQ_TARGETS):
            color = freq_colors[i % len(freq_colors)]
            self.plot_targets_top.append((target.name, color))
        self.plot_targets_bottom = [
            ("Power_Package_W", 'b'),         # Blue (Cyan 'c' is too bright)
            ("Power_Core_W", 'm'),            # Magenta (Standard magenta is okay, but could be darker. 'm' is (255,0,255))
            ("Power_Uncore_W", (0, 0, 150)),  # Dark Blue
        ]
        self.all_plot_targets = self.plot_targets_top + self.plot_targets_bottom

        # 上部グラフの曲線を初期化
        for name, color in self.plot_targets_top:
            self.data[name] = deque(maxlen=self.max_points)
            self.curves[name] = self.plot_item_top.plot(
                pen=pg.mkPen(color, width=2), name=name
            )
        # 下部グラフの曲線を初期化
        for name, color in self.plot_targets_bottom:
            self.data[name] = deque(maxlen=self.max_points)
            self.curves[name] = self.plot_item_bottom.plot(
                pen=pg.mkPen(color, width=2), name=name
            )

    def toggle_pause(self):
        if self.is_paused_gui:
            self.monitor_thread.resume()
            self.pause_button.setText("Pause")
            self.is_paused_gui = False
        else:
            self.monitor_thread.pause()
            self.pause_button.setText("Resume")
            self.is_paused_gui = True

    @QtCore.pyqtSlot(dict)
    def update_plot(self, data: dict):
        """シグナル受信時にグラフを更新するスロット"""
        
        self.time_deque.append(data["Time_ms"] / 1000.0)
        
        # [変更] すべての対象データをデキューに追加
        for name, _ in self.all_plot_targets:
            if name in data:
                self.data[name].append(data[name])

        x_data = np.array(self.time_deque, dtype=float)
        
        # [変更] すべての曲線のデータを更新
        for name, _ in self.all_plot_targets:
            y_data = np.array(self.data[name], dtype=float)
            if name in self.curves:
                self.curves[name].setData(x=x_data, y=y_data)

        # [追加] 電力グラフ (bottom) のY軸スケールをパーセンタイルで調整
        # すべての電力データを結合して計算
        all_power_values = []
        for name, _ in self.plot_targets_bottom:
            if name in self.data:
                all_power_values.extend(self.data[name])
        
        if all_power_values:
            # 98パーセンタイルを計算
            percent = 98
            percented = np.percentile(all_power_values, percent)
            # 最小範囲を確保 (例: 10W) して、極端な拡大を防ぐ
            y_max = max(percented * 1.1, 10.0)
            self.plot_item_bottom.setYRange(0, y_max)


def main():
    # --- 引数解析 (変更なし) ---
    parser = argparse.ArgumentParser(
        description="CPU/RAPL/Freq Monitor with GUI (2-Panel)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-int", "--interval", type=int, default=20,
        help="計測間隔 (ミリ秒) (デフォルト: 20ms)"
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
    args = parser.parse_args()

    if args.interval < 10:
        print(f"Warning: Interval {args.interval}ms is very fast and may "
              "impact performance or stability.", file=sys.stderr, flush=True)

    # --- 権限設定スクリプトの実行 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    init_script = os.path.join(script_dir, "monitorinit.bash")
    if os.path.exists(init_script):
        print(f"Running permission setup script: {init_script}")
        print("You may be asked for your password (sudo).")
        try:
            subprocess.run(["bash", init_script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running setup script: {e}", file=sys.stderr)
    else:
        print(f"Warning: {init_script} not found. Skipping permission setup.")

    # --- アプリケーション起動 (変更なし) ---
    app = QtWidgets.QApplication.instance() # 既存のインスタンスを取得
    if app is None:
        app = QtWidgets.QApplication(sys.argv) # 存在しない場合は作成
    
    monitor_thread = MonitorThread(
        interval_ms=args.interval,
        duration_sec=args.duration,
        output_file=args.output
    )

    window = MonitorWindow(
        max_points=args.points, 
        interval_ms=args.interval,
        monitor_thread=monitor_thread
    )

    monitor_thread.data_ready.connect(window.update_plot)
    
    def quit_app():
        monitor_thread.stop()
        
    app.aboutToQuit.connect(quit_app)

    def sigint_handler(*_):
        print("\nCtrl+C detected. Stopping...", flush=True)
        quit_app()
        app.quit()
        
    signal.signal(signal.SIGINT, sigint_handler)
    
    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None) 

    monitor_thread.start()

    print(f"Monitoring started. (Interval: {args.interval}ms, Duration: {args.duration}s)")
    print(f"Saving log to: {args.output}")
    print("Press Ctrl+C in terminal or close window (or Pause button) to stop.")

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app.exec()
    
    monitor_thread.wait(2000)
    print("Application finished.", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())