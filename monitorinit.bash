echo "[Monitor Setup] Setting read permissions for power/frequency files..."

# 1. CPU Power (RAPL)
sudo chmod 444 /sys/class/powercap/intel-rapl:0/energy_uj
sudo chmod 444 /sys/class/powercap/intel-rapl:0:0/energy_uj
sudo chmod 444 /sys/class/powercap/intel-rapl:0:1/energy_uj

# 2. CPU Frequency (All Cores)
sudo chmod 444 /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

# 3. iGPU Frequency (DRM)
sudo chmod 444 /sys/class/drm/card1/gt_cur_freq_mhz

# 4. NPU Frequency
sudo chmod 444 /sys/devices/pci0000:00/0000:00:0b.0/npu_current_frequency_mhz

echo "[Monitor Setup] Permissions set."