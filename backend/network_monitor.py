"""Tracks outbound network connections and exposes a live count.

This is the "offline-first" watchdog: it samples active TCP connections via
`psutil` and reports how many are outbound to non-local hosts. The FastAPI
app exposes this via `/network/stats` so you can visually confirm the
workbench never phones home.

NOTE: psutil only sees connections of the *current user* by default on some
platforms; on Linux it requires no special privileges for its own processes.
Set ALLOW_LOOPBACK to False to treat loopback connections as "safe".
"""

from __future__ import annotations

import socket
from datetime import datetime

import psutil

# Treat 127.0.0.1 / localhost / RFC1918 as "safe" local traffic by default.
LOOPBACK_IPS = {"127.0.0.1", "::1", "0.0.0.0"}
LOCAL_PREFIXES = ("10.", "192.168.", "172.16.", "172.17.", "172.18.",
                  "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
                  "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                  "172.29.", "172.30.", "172.31.", "169.254.")


def _is_local(ip: str) -> bool:
    if not ip:
        return True
    ip = ip.strip()
    if ip in LOOPBACK_IPS:
        return True
    return any(ip.startswith(p) for p in LOCAL_PREFIXES)


def sample_outbound_connections() -> list[dict]:
    """Return a snapshot of outbound (client) TCP connections."""
    conns = []
    for conn in psutil.net_connections(kind="tcp"):
        # status ESTABLISHED/SYN_SENT etc.; we care about outbound links.
        if conn.status in ("ESTABLISHED", "SYN_SENT") and conn.raddr:
            try:
                remote_ip = socket.gethostbyaddr(conn.raddr.ip)[0]
            except Exception:  # noqa: BLE001
                remote_ip = conn.raddr.ip
            conns.append({
                "local": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "",
                "remote": f"{conn.raddr.ip}:{conn.raddr.port}",
                "remote_host": remote_ip,
                "status": conn.status,
                "pid": conn.pid,
                "is_local": _is_local(conn.raddr.ip),
            })
    return conns


def outbound_nonlocal_count() -> int:
    """Count outbound connections to non-local hosts (the "phoning home" number)."""
    return sum(1 for c in sample_outbound_connections() if not c["is_local"])


def network_stats() -> dict:
    """Full stats dict for the /network/stats endpoint."""
    conns = sample_outbound_connections()
    return {
        "timestamp": datetime.now().isoformat(),
        "total_tcp_connections": len(conns),
        "outbound_local": sum(1 for c in conns if c["is_local"]),
        "outbound_nonlocal": sum(1 for c in conns if not c["is_local"]),
        "offline_clean": all(c["is_local"] for c in conns),
        "connections": conns[:200],  # cap to keep responses small
    }


if __name__ == "__main__":
    import json

    print(json.dumps(network_stats(), indent=2, default=str))
