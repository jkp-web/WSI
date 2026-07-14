#!/usr/bin/env bash
#
# 1_unoq_probe.sh  -- run ON the Arduino UNO Q
# ---------------------------------------------------------------
# Gathers everything the PC cross-compile script needs:
#   - kernel release / version / arch
#   - whether ch341 module already exists
#   - attached USB devices (to confirm the CH34x CNC controller)
#   - the running kernel config (/proc/config.gz), base64-encoded
#
# Output goes to screen AND to  ~/unoq_probe.txt
#
# Transfer to your PC either way:
#   scp arduino@<unoq-ip>:~/unoq_probe.txt .
#   ...or just copy-paste the whole block between the BEGIN/END
#   markers into a file called unoq_probe.txt on the PC.
# ---------------------------------------------------------------
set -euo pipefail

OUT="${HOME}/unoq_probe.txt"

MODSTATE="absent"
if modinfo ch341 >/dev/null 2>&1; then MODSTATE="present"; fi

USBLINE="$(lsusb | grep -i '1a86:' || echo 'no-1a86-device-found')"

CONFIG_B64=""
if [[ -r /proc/config.gz ]]; then
    CONFIG_B64="$(base64 -w0 /proc/config.gz)"
else
    echo "[!] /proc/config.gz not readable - PC script will fall back to defconfig (riskier)." >&2
fi

{
echo "-----BEGIN UNOQ PROBE-----"
echo "KREL=$(uname -r)"
echo "KVER_FULL=$(uname -v | tr ' ' '_')"
echo "ARCH=$(uname -m)"
echo "CH341_MODULE=${MODSTATE}"
echo "USB=${USBLINE}"
echo "CONFIG_GZ_B64=${CONFIG_B64}"
echo "-----END UNOQ PROBE-----"
} | tee "${OUT}"

echo ""
echo "[+] Probe saved to ${OUT}"
if [[ "${MODSTATE}" == "present" ]]; then
    echo "[+] NOTE: ch341 module already exists on this system!"
    echo "    You may not need to compile anything. Try:"
    echo "      sudo modprobe ch341 && dmesg | tail"
fi
echo "[+] Copy unoq_probe.txt to your PC (scp, USB stick, or copy-paste"
echo "    everything between the BEGIN/END markers), then run:"
echo "      bash 2_pc_build.sh unoq_probe.txt"
