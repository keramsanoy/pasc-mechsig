"""Database connection to the OMOP CDM.

The thesis ran against AIR·MS, the Mount Sinai Health System implementation of
the OMOP CDM v5.3 on SAP HANA, reached through an SSH tunnel from an HPC node.
Nothing site-specific is hard-coded: every host, port, schema and credential
comes from environment variables, or from a ``.env`` file in the repository root
or the working directory (see ``.env.example``).

    from pasc.db import connect
    conn = connect()          # prompts for user / password if not in the environment
    cur = conn.cursor()

Required variables
    OMOP_DB_HOST        database host (the HANA server, or the SSH tunnel target)
    OMOP_DB_PORT        database port
    OMOP_DB_NAME        database name (``databaseName`` in hdbcli)
    OMOP_DB_USER        user name (prompted if absent)
    OMOP_DB_PASSWORD    password (prompted if absent; never store it in the repo)

Optional variables
    OMOP_CDM_SCHEMA              schema holding the CDM tables (default CDMPHI, see
                                 pasc.config.omop); any other value is substituted
                                 into every statement at execution time
    OMOP_USE_SSH_TUNNEL          1/0. Default: 1 unless the host name starts with
                                 OMOP_LOGIN_NODE_PREFIX (a login node that reaches
                                 the database directly)
    OMOP_SSH_LOGIN_HOSTS         comma-separated SSH hosts to try for the tunnel
    OMOP_SSH_LOGIN_HOST          preferred host, tried first
    OMOP_LOGIN_NODE_PREFIX       hostname prefix that identifies a login node
    OMOP_SSL_HOSTNAME_IN_CERT    hdbcli sslHostNameInCertificate
    OMOP_SSL_TRUSTSTORE          hdbcli sslTrustStore (default "None")
    OMOP_SSL_VALIDATE_CERTIFICATE  TRUE/FALSE (default FALSE)
    OMOP_ENCRYPT                 TRUE/FALSE (default TRUE)
    OMOP_CONNECT_TIMEOUT         seconds, 0 = no timeout (default 0)

Porting to another OMOP site: the queries in this package are HANA-flavoured SQL
(temp tables ``#antony_cohort``, ``TOP n``, ``ADD_DAYS``). Replace the ``hdbcli``
connection in :func:`connect` with a DB-API connection to your own database and
adapt the SQL helpers in pasc.features.signals.
"""
import os
import socket
import logging
import shlex
import subprocess
import time
import getpass
import traceback

from dotenv import load_dotenv

from pasc.config.omop import CDM_SCHEMA, DEFAULT_CDM_SCHEMA
from pasc.config.paths import REPO_ROOT

LOG = logging.getLogger(__name__)


def _load_env():
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
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


class _SchemaRewritingCursor:
    """Thin proxy that rewrites the literal ``CDMPHI.`` schema prefix before executing."""

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


def connect():
    """Open a DB-API connection to the OMOP CDM database.

    Reads the connection settings from the environment (``.env``), opens an SSH
    tunnel when configured, prompts for missing credentials, and returns the
    connection. When ``OMOP_CDM_SCHEMA`` differs from the literal schema used in
    the SQL, the returned connection rewrites every statement on execution.
    """
    from hdbcli import dbapi  # imported lazily so the package imports without the driver

    _load_env()

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

    try:
        conn = dbapi.connect(
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
        LOG.error(traceback.format_exc())
        raise

    if CDM_SCHEMA != DEFAULT_CDM_SCHEMA:
        conn = _SchemaRewritingConnection(conn)
        print(f"CDM schema: SQL rewritten from {DEFAULT_CDM_SCHEMA}. to {CDM_SCHEMA}. at execution time")

    LOG.info("Initialized OMOP database connection: %s", conn.isconnected())
    return conn
