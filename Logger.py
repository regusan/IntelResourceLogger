#!/usr/bin/env python3

"""
monitor.py

CPU使用率 (/proc/stat) と RAPL消費電力 (/sys/class/powercap/) を
高い頻度で監視し、CSVファイルに出力するプログラム。
(C++コードのPython移植版)

実行方法:
python3 monitor.py

(事前に /sys/class/powercap/.../energy_uj のパーミッションを変更しておくこと)

オプション:
  -int, --interval  計測間隔 (ミリ秒) (デフォルト: 10)
  -dur, --duration  計測時間 (秒) (デフォルト: 600)
  -o, --output      出力CSVファイル名 (デフォルト: monitor_log.csv)
  -c, --console     コンソールにも結果を出力する
  -h, --help        ヘルプを表示
"""

import sys
import time
import signal
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

RAPL_TARGETS: List[Tuple[str, str]] = [
    ("/sys/class/powercap/intel-rapl:0/energy_uj", "Power_Package_W"),
    ("/sys/class/powercap/intel-rapl:0:0/energy_uj", "Power_Core_W"),
    ("/sys/class/powercap/intel-rapl:0:1/energy_uj", "Power_Uncore_W"),
]

PROC_STAT_PATH = "/proc/stat"

#グローバル変数
g_running = True

def signal_handler(sig, frame):
    """Ctrl+C (SIGINT) を受信したときに呼び出される"""
    global g_running
    if g_running:
        print("\nStopping monitor...", flush=True)
        g_running = False

@dataclass
class CpuTimes:
    """/proc/stat から読み取ったCPU時間 (Jiffies)"""
    user: int = 0
    nice: int = 0
    system: int = 0
    idle: int = 0
    iowait: int = 0
    irq: int = 0
    softirq: int = 0
    steal: int = 0

    def total(self) -> int:
        """CPU時間の合計"""
        return (self.user + self.nice + self.system + self.idle +
                self.iowait + self.irq + self.softirq + self.steal)

    def idle_total(self) -> int:
        """アイドル時間の合計"""
        return self.idle + self.iowait

def get_cpu_times(path: str) -> Optional[CpuTimes]:
    """/proc/stat を読み取る関数"""
    try:
        with open(path, 'r') as f:
            line = f.readline() # 1行目を読む
            parts = line.split()
            
            if not parts or parts[0] != 'cpu':
                print(f"Error: Invalid format in {path}", file=sys.stderr)
                return None
            
            # parts[0]は 'cpu' なので スキップ
            times = [int(p) for p in parts[1:9]]
            return CpuTimes(*times)
            
    except FileNotFoundError:
        print(f"Error: Cannot open {path}", file=sys.stderr)
        return None
    except (IOError, ValueError) as e:
        print(f"Error: Failed to read or parse {path}: {e}", file=sys.stderr)
        return None

def get_rapl_energy(path: str) -> int:
    """RAPLのenergy_ujファイルを読み取る (失敗時は -1)"""
    try:
        with open(path, 'r') as f:
            return int(f.read())
    except (FileNotFoundError, IOError, ValueError):
        # ログが溢れるのを防ぐため、ここではエラー出力しない
        return -1


def main():
    global g_running

    parser = argparse.ArgumentParser(
        description="CPU Usage and RAPL Power Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-int", "--interval", type=int, default=10,
        help="計測間隔 (ミリ秒) (デフォルト: 10)"
    )
    parser.add_argument(
        "-dur", "--duration", type=int, default=600,
        help="計測時間 (秒) (デフォルト: 600)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="monitor_log.csv",
        help="出力CSVファイル名 (デフォルト: monitor_log.csv)"
    )
    parser.add_argument(
        "-c", "--console", action="store_true",
        help="コンソールにも結果を出力する"
    )
    
    args = parser.parse_args()

    # Ctrl+Cハンドラを設定
    signal.signal(signal.SIGINT, signal_handler)

    try:
        csv_file = open(args.output, 'w', encoding='utf-8')
    except IOError as e:
        print(f"Error: Cannot open output file: {args.output} ({e})", file=sys.stderr)
        return 1

    header_cols = ["Time_ms", "DeltaT_ms", "CPU_Usage_Percent"]
    header_cols.extend([name for _, name in RAPL_TARGETS])
    csv_file.write(",".join(header_cols) + "\n")

    interval_sec = args.interval / 1000.0
    duration_sec = args.duration

    prev_cpu_times = get_cpu_times(PROC_STAT_PATH)
    if prev_cpu_times is None:
        csv_file.close()
        return 1
    
    prev_rapl_energy: List[int] = []
    for path, _ in RAPL_TARGETS:
        energy = get_rapl_energy(path)
        if energy == -1:#!/usr/bin/env python3
