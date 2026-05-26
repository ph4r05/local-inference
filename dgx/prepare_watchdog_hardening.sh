#!/usr/bin/env bash
set -euo pipefail

SYSCTL_FILE="/etc/sysctl.d/99-watchdog-reboot.conf"
WATCHDOG_TEST_DIR="/etc/watchdog.d"
WATCHDOG_TEST_FILE="${WATCHDOG_TEST_DIR}/root-fs-liveness.sh"
WATCHDOG_DROPIN_DIR="/etc/systemd/system/watchdog.service.d"
WATCHDOG_DROPIN_FILE="${WATCHDOG_DROPIN_DIR}/override.conf"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 1
fi

mkdir -p "${WATCHDOG_TEST_DIR}" "${WATCHDOG_DROPIN_DIR}"

cat > "${SYSCTL_FILE}" <<'EOF'
# Reboot after a panic so the hardware watchdog is not the only path back.
kernel.panic = 10
kernel.panic_on_oops = 1

# Escalate kernel lockups into panics where supported.
kernel.softlockup_panic = 1
kernel.hung_task_panic = 1
EOF

cat > "${WATCHDOG_TEST_FILE}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Simple scheduler + filesystem liveness check for watchdog.
# If the box is badly wedged, this should fail or time out.
timeout 10s /bin/bash -lc 'ls -1 / >/dev/null'
EOF
chmod 0755 "${WATCHDOG_TEST_FILE}"

cat > "${WATCHDOG_DROPIN_FILE}" <<'EOF'
[Service]
Restart=always
RestartSec=2
EOF

sysctl --system
systemctl daemon-reload
systemctl restart watchdog

cat <<EOF
Wrote:
  ${SYSCTL_FILE}
  ${WATCHDOG_TEST_FILE}
  ${WATCHDOG_DROPIN_FILE}

Suggested watchdog.conf updates:

  watchdog-device = /dev/watchdog
  watchdog-timeout = 20
  interval = 5
  realtime = yes
  priority = 1
  retry-timeout = 30
  repair-maximum = 1
  sigterm-delay = 5
  interface = enP7s7
  pidfile = /run/sshd.pid
  test-directory = /etc/watchdog.d

Notes:
  - The new test script is: ${WATCHDOG_TEST_FILE}
  - It runs: timeout 10s /bin/bash -lc 'ls -1 / >/dev/null'
  - This adds a cheap scheduler + root filesystem liveness check.
  - Review /etc/watchdog.conf manually before the next restart.

Quick verification:
  systemctl status watchdog --no-pager
  journalctl -u watchdog -n 50 --no-pager
  sysctl kernel.panic kernel.panic_on_oops kernel.softlockup_panic kernel.hung_task_panic
EOF
