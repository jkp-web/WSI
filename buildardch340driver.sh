#!/usr/bin/env bash
#
# 2_pc_build.sh  -- run on your PC (Linux or WSL2, Debian/Ubuntu-based)
# ---------------------------------------------------------------
# Usage:
#   bash 2_pc_build.sh unoq_probe.txt
#
# Reads the probe file produced by 1_unoq_probe.sh on the UNO Q,
# cross-compiles the mainline in-tree ch341 driver against the
# matching arduino/linux-qcom kernel tag with the UNO Q's exact
# running config (so vermagic matches), and produces:
#
#   ch341-unoq-<kernelrelease>.tar.gz
#     |- ch341.ko
#     |- usbserial.ko        (if built)
#     |- install.sh          (run this on the UNO Q with sudo)
#
# Copy the tarball to the UNO Q and run:
#   tar xzf ch341-unoq-*.tar.gz && cd ch341-unoq && sudo bash install.sh
# ---------------------------------------------------------------
set -euo pipefail

PROBE="${1:-unoq_probe.txt}"
[[ -r "${PROBE}" ]] || { echo "[x] Probe file '${PROBE}' not found. Usage: bash $0 unoq_probe.txt" >&2; exit 1; }

log()  { echo -e "\033[1;32m[+]\033[0m $*"; }
warn() { echo -e "\033[1;33m[!]\033[0m $*"; }
die()  { echo -e "\033[1;31m[x]\033[0m $*" >&2; exit 1; }

# ----------------------------------------------------------------
# Parse the probe block
# ----------------------------------------------------------------
getval() { grep -m1 "^$1=" "${PROBE}" | cut -d= -f2-; }

KREL="$(getval KREL)"
ARCH_M="$(getval ARCH)"
MODSTATE="$(getval CH341_MODULE)"
USBLINE="$(getval USB)"
CONFIG_B64="$(getval CONFIG_GZ_B64)"

[[ -n "${KREL}" ]] || die "Probe file missing KREL - is this the right file?"
[[ "${ARCH_M}" == "aarch64" ]] || warn "Unexpected arch '${ARCH_M}' (expected aarch64) - continuing anyway."
[[ "${MODSTATE}" == "present" ]] && warn "Probe says ch341 already exists on the UNO Q - you may not need this build."
[[ "${USBLINE}" == *"1a86:"* ]] && log "CH34x device confirmed on the UNO Q: ${USBLINE}" \
                                || warn "Probe did not see a 1a86:* USB device (controller unplugged at probe time?)."

KVER="${KREL%%-*}"                 # 6.16.7
LOCALVER="${KREL#${KVER}}"         # -g0dd6551ae96b  (may be empty)
KSRC_TAG="qcom-v${KVER}-unoq"
KSRC_REPO="https://github.com/arduino/linux-qcom.git"
WORKDIR="$(pwd)/unoq_kbuild"
BUNDLE="ch341-unoq"

log "Target kernel : ${KREL}"
log "Source tag    : ${KSRC_TAG}"

# ----------------------------------------------------------------
# Dependencies + cross toolchain (Arch/CachyOS or Debian/Ubuntu)
# ----------------------------------------------------------------
if command -v pacman >/dev/null 2>&1; then
    log "Installing build dependencies (pacman)..."
    sudo pacman -S --needed --noconfirm base-devel bc bison flex openssl libelf git \
        aarch64-linux-gnu-gcc aarch64-linux-gnu-binutils
elif command -v apt-get >/dev/null 2>&1; then
    log "Installing build dependencies (apt)..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential bc bison flex libssl-dev libelf-dev \
        git gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu
else
    warn "No pacman or apt-get found - install manually: gcc cross toolchain for"
    warn "aarch64 (aarch64-linux-gnu-gcc), bc, bison, flex, openssl/libssl dev,"
    warn "libelf, git - then re-run."
    die  "Unsupported package manager."
fi

export ARCH=arm64
export CROSS_COMPILE=aarch64-linux-gnu-

# ----------------------------------------------------------------
# Kernel source at the matching tag
# ----------------------------------------------------------------
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
if [[ ! -d linux-qcom ]]; then
    log "Cloning ${KSRC_TAG} (shallow)..."
    git clone --depth 1 -b "${KSRC_TAG}" "${KSRC_REPO}" linux-qcom \
        || die "Tag ${KSRC_TAG} not found in ${KSRC_REPO}. Check the repo's tags and edit KSRC_TAG."
fi
cd linux-qcom

# ----------------------------------------------------------------
# Config: exact running config from the probe, vermagic matched
# ----------------------------------------------------------------
if [[ -n "${CONFIG_B64}" ]]; then
    log "Restoring UNO Q's running kernel config from probe..."
    echo "${CONFIG_B64}" | base64 -d | zcat > .config
else
    warn "Probe had no config - falling back to defconfig (vermagic mismatch risk!)."
    make defconfig
fi

scripts/config --set-str LOCALVERSION "${LOCALVER}"
scripts/config --disable LOCALVERSION_AUTO
scripts/config --module USB_SERIAL
scripts/config --module USB_SERIAL_CH341
export LOCALVERSION=""   # suppress the dirty-tree '+' suffix in vermagic
make olddefconfig
log "Preparing tree (modules_prepare)..."
make modules_prepare -j"$(nproc)"

# Force-correct the cached release string if setlocalversion still added '+'
if grep -q '+' include/config/kernel.release 2>/dev/null; then
    warn "kernel.release contained '+' - forcing exact release ${KREL}"
    echo "${KREL}" > include/config/kernel.release
    sed -i "s/${KREL}+/${KREL}/" include/generated/utsrelease.h
fi

# ----------------------------------------------------------------
# Build the usb/serial modules
# In-tree M= builds lack Module.symvers (needs full vmlinux build).
# If MODVERSIONS is off, unresolved-symbol modpost errors are benign
# (symbols resolve at load time) -> demote to warnings.
# If MODVERSIONS is on, CRCs are required -> full modules build once.
# ----------------------------------------------------------------
if grep -q '^CONFIG_MODVERSIONS=y' .config; then
    log "CONFIG_MODVERSIONS=y - doing full modules build (one-time, slower)..."
    make LOCALVERSION="" modules -j"$(nproc)"
else
    log "Cross-compiling ch341 (modpost symbol errors demoted to warnings)..."
    make LOCALVERSION="" M=drivers/usb/serial modules KBUILD_MODPOST_WARN=1 -j"$(nproc)"
fi
[[ -f drivers/usb/serial/ch341.ko ]] || die "ch341.ko not produced - inspect the build output."

# Sanity check vermagic
VERMAGIC="$(modinfo -F vermagic drivers/usb/serial/ch341.ko 2>/dev/null || strings drivers/usb/serial/ch341.ko | grep -m1 vermagic)"
log "Built module vermagic: ${VERMAGIC}"
[[ "${VERMAGIC}" == *"${KREL}"* ]] || warn "vermagic does not contain ${KREL} - module may refuse to load!"

# ----------------------------------------------------------------
# Assemble install bundle
# ----------------------------------------------------------------
cd "${WORKDIR}"
rm -rf "${BUNDLE}"; mkdir "${BUNDLE}"
cp linux-qcom/drivers/usb/serial/ch341.ko "${BUNDLE}/"
[[ -f linux-qcom/drivers/usb/serial/usbserial.ko ]] && cp linux-qcom/drivers/usb/serial/usbserial.ko "${BUNDLE}/"

cat > "${BUNDLE}/install.sh" <<INSTALLER
#!/usr/bin/env bash
# Run ON the UNO Q:  sudo bash install.sh
set -euo pipefail
[[ \$EUID -eq 0 ]] || { echo "Run with sudo"; exit 1; }
EXPECTED="${KREL}"
[[ "\$(uname -r)" == "\${EXPECTED}" ]] || {
    echo "[x] Kernel mismatch: running \$(uname -r), module built for \${EXPECTED}."
    echo "    Re-run the probe + PC build against the current kernel."
    exit 1
}
MODDIR="/lib/modules/\$(uname -r)/kernel/drivers/usb/serial"
mkdir -p "\${MODDIR}"
[[ -f usbserial.ko ]] && cp -f usbserial.ko "\${MODDIR}/"
cp -f ch341.ko "\${MODDIR}/"
depmod -a
echo ch341 > /etc/modules-load.d/ch341.conf
usermod -aG dialout "\${SUDO_USER:-\$USER}" || true
modprobe ch341
sleep 1
echo "[+] Loaded. Serial devices:"
ls -l /dev/ttyUSB* 2>/dev/null || echo "    (none yet - replug the CNC controller, then: dmesg | tail)"
echo "[+] Log out/in for dialout group. GRBL is usually 115200 baud."
INSTALLER
chmod +x "${BUNDLE}/install.sh"

TARBALL="ch341-unoq-${KREL}.tar.gz"
tar czf "${TARBALL}" "${BUNDLE}"
log "Bundle ready: ${WORKDIR}/${TARBALL}"
echo ""
echo "Next steps:"
echo "  scp ${WORKDIR}/${TARBALL} arduino@<unoq-ip>:~/"
echo "  # then on the UNO Q:"
echo "  tar xzf ${TARBALL} && cd ${BUNDLE} && sudo bash install.sh"
