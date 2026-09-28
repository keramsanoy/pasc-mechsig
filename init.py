"""Database connection for the OMOP CDM.

The thesis ran against AIR·MS, the Mount Sinai Health System implementation of
the OMOP CDM v5.3 on SAP HANA, reached through an SSH tunnel from an HPC node.
Nothing site-specific is hard-coded here: every host, port, schema and
credential comes from environment variables (or a local ``.env`` file read by
python-dotenv; see ``.env.example``). The scripts ``exec`` this file so that
``hana_conn`` (a DB-API connection) and ``cur`` become available in their
namespace.

Required variables
    OMOP_DB_HOST        database host (the HANA server, or the SSH tunnel target)
    OMOP_DB_PORT        database port
    OMOP_DB_NAME        database name (``databaseName`` in hdbcli)
    OMOP_DB_USER        user name (prompted if absent)
    OMOP_DB_PASSWORD    password (prompted if absent; never store it in the repo)

Optional variables
    OMOP_DB_SCHEMA               OMOP CDM schema the queries read (default: the
                                 user's default schema; see mech_signals_common)
    OMOP_USE_SSH_TUNNEL          1/0. Default: 1 unless the host name starts with
                                 OMOP_LOGIN_NODE_PREFIX (i.e. we are already on a
                                 login node that can reach the database directly)
    OMOP_SSH_LOGIN_HOSTS         comma-separated SSH hosts to try for the tunnel
    OMOP_SSH_LOGIN_HOST          preferred host, tried first
    OMOP_LOGIN_NODE_PREFIX       hostname prefix that identifies a login node
    OMOP_SSL_HOSTNAME_IN_CERT    hdbcli sslHostNameInCertificate
    OMOP_SSL_TRUSTSTORE          hdbcli sslTrustStore (default "None")
    OMOP_SSL_VALIDATE_CERTIFICATE  TRUE/FALSE (default FALSE)
    OMOP_ENCRYPT                 TRUE/FALSE (default TRUE)
    OMOP_CONNECT_TIMEOUT         seconds, 0 = no timeout (default 0)

Porting to another OMOP site: the queries in this repository are HANA-flavoured
SQL (temp tables ``#antony_cohort``, ``TOP n``, ``ADD_DAYS``). Replace the
``hdbcli`` connection below with a DB-API connection to your own database and
adapt the SQL helpers in mech_signals_common.py.
"""
import os
import sys
import socket
import logging
import shlex
import subprocess
import time
import getpass
import traceback

from dotenv import load_dotenv

# Make the repository importable when this file is exec'd from a notebook or a
# script started in another working directory.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hdbcli import dbapi

load_dotenv(os.path.join(_REPO_ROOT, ".env"))
load_dotenv()  # also honour a .env in the current working directory


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _require(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill in your site's values."
        )
    return value


def find_open_port(start=4000, end=8000):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("localhost", port)) != 0:
                return port
    return None


def _running_on_login_node():
    prefix = os.getenv("OMOP_LOGIN_NODE_PREFIX", "").strip().lower()
    if not prefix:
        return False
    return socket.gethostname().lower().startswith(prefix)


def _local_port_is_listening(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_local_tunnel(port, timeout_seconds=3.0):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _local_port_is_listening(port):
            return True
        time.sleep(0.1)
    return False


def _start_ssh_tunnel(local_port, remote_host, remote_port, login_host):
    ssh_command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
        "-g", "-N", "-f", "-L", f"{local_port}:{remote_host}:{remote_port}", login_host,
    ]
    result = subprocess.run(ssh_command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"ssh exited with code {result.returncode}").strip()
        raise RuntimeError(f"Failed to establish SSH tunnel via {login_host}: {detail}")
    if not _wait_for_local_tunnel(local_port):
        raise RuntimeError(
            f"Failed to establish SSH tunnel via {login_host}: local port {local_port} never became ready"
        )
    print(f"SSH tunnel established via {login_host} on local port {local_port}.")
    print("  Command: " + " ".join(shlex.quote(part) for part in ssh_command))


def _start_ssh_tunnel_with_fallback(local_port, remote_host, remote_port, login_hosts):
    failures = []
    for login_host in login_hosts:
        try:
            _start_ssh_tunnel(local_port, remote_host, remote_port, login_host)
            return login_host
        except RuntimeError as exc:
            failures.append(f"{login_host}: {exc}")
            print(f"SSH tunnel attempt failed via {login_host}; trying next host if available.")
    raise RuntimeError("Failed to establish SSH tunnel via any configured login host. " + " | ".join(failures))


# --- Connection parameters -------------------------------------------------
db_host = _require("OMOP_DB_HOST")
db_port = _require("OMOP_DB_PORT")
db_name = _require("OMOP_DB_NAME")

ssl_hostname_in_cert = os.getenv("OMOP_SSL_HOSTNAME_IN_CERT", db_host)
ssl_trust_store = os.getenv("OMOP_SSL_TRUSTSTORE", "None")
connect_timeout = os.getenv("OMOP_CONNECT_TIMEOUT", "0")
encrypt = os.getenv("OMOP_ENCRYPT", "TRUE")
validate_cert = os.getenv("OMOP_SSL_VALIDATE_CERTIFICATE", "FALSE")

use_ssh_tunnel = _env_flag("OMOP_USE_SSH_TUNNEL", not _running_on_login_node())

connect_host, connect_port = db_host, db_port
if use_ssh_tunnel:
    login_hosts = [h.strip() for h in os.getenv("OMOP_SSH_LOGIN_HOSTS", "").split(",") if h.strip()]
    preferred = os.getenv("OMOP_SSH_LOGIN_HOST", "").strip()
    if preferred:
        login_hosts = [preferred] + [h for h in login_hosts if h != preferred]
    if not login_hosts:
        raise RuntimeError(
            "OMOP_USE_SSH_TUNNEL is on but no OMOP_SSH_LOGIN_HOSTS are configured. "
            "Set OMOP_USE_SSH_TUNNEL=0 to connect directly."
        )
    local_port = find_open_port()
    if not local_port:
        raise RuntimeError("No open local port found for the SSH tunnel.")
    print(f"Found open port on the current node: {local_port}")
    _start_ssh_tunnel_with_fallback(local_port, db_host, db_port, login_hosts)
    connect_host, connect_port = "localhost", local_port
else:
    print(f"Connecting to the database without an SSH tunnel: {connect_host}:{connect_port}")

db_user = os.getenv("OMOP_DB_USER") or input("OMOP_DB_USER: ")
db_password = os.getenv("OMOP_DB_PASSWORD") or getpass.getpass("Enter your database password: ")

logging.basicConfig(format="%(asctime)s | %(levelname)s : %(message)s", level=logging.ERROR)
LOG = logging.getLogger(__name__)

try:
    hana_conn = dbapi.connect(
        address=connect_host,
        port=connect_port,
        user=db_user,
        databaseName=db_name,
        password=db_password,
        encrypt=encrypt,
        sslValidateCertificate=validate_cert,
        sslHostNameInCertificate=ssl_hostname_in_cert,
        sslTrustStore=ssl_trust_store,
        connectTimeout=connect_timeout,
    )
except Exception:
    logging.error(traceback.format_exc())
    raise


# --- CDM schema name ---------------------------------------------------------
# The SQL in this repository names the OMOP schema literally as ``CDMPHI`` (the
# AIR·MS convention, thesis Appendix A.4). On another site set OMOP_CDM_SCHEMA
# and every statement is rewritten on execution; the SQL text itself is left
# untouched so the MSHS run stays byte-identical.
from omop_config import CDM_SCHEMA, DEFAULT_CDM_SCHEMA  # noqa: E402

if CDM_SCHEMA != DEFAULT_CDM_SCHEMA:
    class _SchemaRewritingCursor:
        """Thin proxy that rewrites the literal schema prefix before executing."""

        def __init__(self, cursor):
            self._cursor = cursor

        @staticmethod
        def _rewrite(sql):
            return sql.replace(f"{DEFAULT_CDM_SCHEMA}.", f"{CDM_SCHEMA}.") if isinstance(sql, str) else sql

        def execute(self, sql, *args, **kwargs):
            return self._cursor.execute(self._rewrite(sql), *args, **kwargs)

        def executemany(self, sql, *args, **kwargs):
            return self._cursor.executemany(self._rewrite(sql), *args, **kwargs)

        def __iter__(self):
            return iter(self._cursor)

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    class _SchemaRewritingConnection:
        def __init__(self, conn):
            self._conn = conn

        def cursor(self, *args, **kwargs):
            return _SchemaRewritingCursor(self._conn.cursor(*args, **kwargs))

        def __getattr__(self, name):
            return getattr(self._conn, name)

    hana_conn = _SchemaRewritingConnection(hana_conn)
    print(f"CDM schema: SQL rewritten from {DEFAULT_CDM_SCHEMA}. to {CDM_SCHEMA}. at execution time")

LOG.info("Initialized OMOP database connection: %s", hana_conn.isconnected())
