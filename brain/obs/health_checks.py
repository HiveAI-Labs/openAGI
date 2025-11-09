# Health Check Functions for HiveNet AI Brain
# Provides comprehensive health monitoring for all system components

from __future__ import annotations

import importlib
import os
import time
from types import ModuleType
from typing import Any, NamedTuple, TYPE_CHECKING, Protocol, Optional, Callable, Dict, cast

try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False

redis_module: ModuleType | None = None
try:
    import redis as _redis
    redis_module = _redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis_module = None

psycopg2_module: ModuleType | None = None
try:
    import psycopg2 as _psycopg2
    psycopg2_module = _psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2_module = None

import threading

if TYPE_CHECKING:
    from brain.obs import enhanced_observability as eo
    HealthStatus = eo.HealthStatus
    HealthCheck = eo.HealthCheck


# Protocols used to provide narrow typing for the optional observability
# objects. This avoids leaking the concrete implementation into this module
# while giving mypy enough information to type-check usages.
class MetricsRegistryProtocol(Protocol):
    def record_cpu_usage(self, *args: Any, **kwargs: Any) -> None: ...
    def record_memory_usage(self, *args: Any, **kwargs: Any) -> None: ...


class HealthCheckerProtocol(Protocol):
    def add_check(self, name: str, fn: Callable[[], Any]) -> None: ...
    def run_checks(self) -> None: ...

# Provide default/placeholder objects so the module has well-typed
# names regardless of whether enhanced observability is installed.
health_checker: Optional[HealthCheckerProtocol]
metrics_registry: MetricsRegistryProtocol

try:
    from brain.obs import enhanced_observability as eo
    HealthStatus = eo.HealthStatus
    HealthCheck = eo.HealthCheck
    # Use cast to narrow the types for mypy while keeping runtime behavior.
    metrics_registry = cast(MetricsRegistryProtocol, getattr(eo, "metrics_registry", None))
    health_checker = cast(Optional[HealthCheckerProtocol], getattr(eo, "health_checker", None))
except (ImportError, AttributeError):
    class HealthStatus:  # type: ignore[no-redef]
        HEALTHY = "HEALTHY"
        DEGRADED = "DEGRADED"
        UNHEALTHY = "UNHEALTHY"

    class HealthCheck(NamedTuple):  # type: ignore[no-redef]
        name: str
        status: Any
        message: str
        timestamp: float
        details: dict | None = None

    health_checker = None

    class _NoopMetrics:
        def record_cpu_usage(self, *args: Any, **kwargs: Any) -> None:
            return None

        def record_memory_usage(self, *args: Any, **kwargs: Any) -> None:
            return None

    metrics_registry = _NoopMetrics()


def check_system_resources() -> Any:
    """Check system resource usage."""
    if not PSUTIL_AVAILABLE:
        return HealthCheck(
            name="system_resources",
            status=HealthStatus.DEGRADED,
            message="psutil not installed; system resource metrics unavailable",
            timestamp=time.time(),
            details=None,
        )

    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        status = HealthStatus.HEALTHY
        message = "System resources OK"

        # Check CPU usage
        if cpu_percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"High CPU usage: {cpu_percent}%"
        elif cpu_percent > 75:
            status = HealthStatus.DEGRADED
            message = f"Elevated CPU usage: {cpu_percent}%"

        # Check memory usage
        if memory.percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"High memory usage: {memory.percent}%"
        elif memory.percent > 80:
            status = HealthStatus.DEGRADED
            message = f"Elevated memory usage: {memory.percent}%"

        # Check disk usage
        if disk.percent > 95:
            status = HealthStatus.UNHEALTHY
            message = f"Critical disk usage: {disk.percent}%"
        elif disk.percent > 85:
            status = HealthStatus.DEGRADED
            message = f"High disk usage: {disk.percent}%"

        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
        }

        # Record metrics
        metrics_registry.record_cpu_usage(cpu_percent)
        metrics_registry.record_memory_usage(
            int(memory.used),
            int(memory.total),
        )

        return HealthCheck(
            name="system_resources",
            status=status,
            message=message,
            timestamp=time.time(),
            details=details,
        )

    except Exception as e:
        return HealthCheck(
            name="system_resources",
            status=HealthStatus.UNHEALTHY,
            message=f"Failed to check system resources: {e}",
            timestamp=time.time(),
        )

def check_brain_components() -> Any:
    """Check brain component availability."""
    components = {
        "model_client": "brain.core.model_client",
        "strategy_selector": "brain.meta.strategy_selector",
        "world_model": "brain.world_model.simple_model",
        "observability": "brain.obs.enhanced_observability",
    }

    missing_components: list[str] = []
    for name, module_path in components.items():
        try:
            importlib.import_module(module_path)
        except Exception:
            missing_components.append(name)

    components_ok = not missing_components
    message = (
        "Brain components OK"
        if components_ok
        else f"Missing optional components: {missing_components}"
    )

    return HealthCheck(
        name="brain_components",
        status=HealthStatus.HEALTHY if components_ok else HealthStatus.DEGRADED,
        message=message,
        timestamp=time.time(),
        details={
            "components_available": components_ok,
            "missing_components": missing_components,
        },
    )

def check_database_connections() -> Any:
    """Check database connection health."""
    results = []
    status = HealthStatus.HEALTHY
    message = "Database connections OK"

    # Use the new database manager if available
    try:
        from brain.database.integration import get_database_manager
        db_manager = get_database_manager()
        health = db_manager.health_check()

        # Check PostgreSQL
        if health.get("postgres") is True:
            results.append({"postgresql": "connected"})
        elif health.get("postgres") is False:
            results.append({"postgresql": "failed"})
            status = HealthStatus.DEGRADED
            message = "PostgreSQL connection issues"
        else:
            results.append({"postgresql": "not configured"})

        # Check Redis
        if health.get("redis") is True:
            results.append({"redis": "connected"})
        elif health.get("redis") is False:
            results.append({"redis": "failed"})
            status = HealthStatus.DEGRADED
            message = "Redis connection issues"
        else:
            results.append({"redis": "not configured"})

        # Check vector store
        if health.get("vector") is True:
            results.append({"vector_store": f"connected ({db_manager.config.vector_backend})"})
        elif health.get("vector") is False:
            results.append({"vector_store": "failed"})
            status = HealthStatus.DEGRADED
            message = "Vector store connection issues"
        else:
            results.append({"vector_store": "not configured"})

    except ImportError:
        # Fall back to legacy checks if database manager not available
        message = "Database manager not available, using legacy checks"

        # Check Redis if configured
        redis_url = os.getenv("REDIS_URL")
        if redis_url and REDIS_AVAILABLE:
            try:
                r = redis_module.from_url(redis_url)
                r.ping()
                results.append({"redis": "connected"})
            except Exception as e:
                results.append({"redis": f"failed: {e}"})
                status = HealthStatus.DEGRADED
                message = "Redis connection issues"

        # Check PostgreSQL if configured
        postgres_url = os.getenv("DATABASE_URL")
        if postgres_url and POSTGRES_AVAILABLE:
            try:
                conn = psycopg2_module.connect(postgres_url)
                conn.close()
                results.append({"postgresql": "connected"})
            except Exception as e:
                results.append({"postgresql": f"failed: {e}"})
                status = HealthStatus.DEGRADED
                message = "PostgreSQL connection issues"

        # Check if vector stores are available
        try:
            from brain.memory.vector_faiss import FaissVectorStore
            results.append({"faiss": "available"})
        except ImportError:
            results.append({"faiss": "not available"})

    except Exception as e:
        status = HealthStatus.UNHEALTHY
        message = f"Database health check failed: {e}"
        results.append({"error": str(e)})

    return HealthCheck(
        name="database_connections",
        status=status,
        message=message,
        timestamp=time.time(),
        details={"connections": results},
    )

def check_external_services() -> Any:
    """Check external service availability."""
    results = []
    status = HealthStatus.HEALTHY
    message = "External services OK"

    # Check Ollama if configured
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        import requests
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            results.append({"ollama": "available"})
        else:
            results.append({"ollama": f"status: {response.status_code}"})
            status = HealthStatus.DEGRADED
            message = "Ollama service issues"
    except Exception as e:
        results.append({"ollama": f"unavailable: {e}"})
        status = HealthStatus.DEGRADED
        message = "Ollama service unavailable"

    # Check HuggingFace API if token is configured
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if hf_token:
        try:
            import requests
            response = requests.get(
                "https://huggingface.co/api/models",
                headers={"Authorization": f"Bearer {hf_token}"},
                timeout=5,
            )
            if response.status_code == 200:
                results.append({"huggingface": "available"})
            else:
                results.append({"huggingface": f"status: {response.status_code}"})
                status = HealthStatus.DEGRADED
                message = "HuggingFace API issues"
        except Exception as e:
            results.append({"huggingface": f"unavailable: {e}"})
            status = HealthStatus.DEGRADED
            message = "HuggingFace API unavailable"

    return HealthCheck(
        name="external_services",
        status=status,
        message=message,
        timestamp=time.time(),
        details={"services": results},
    )

def check_file_system() -> Any:
    """Check file system and artifact storage."""
    try:
        # Check artifacts directory
        artifacts_dir = os.getenv("ARTIFACTS_DIR", "./artifacts")
        artifacts_path = os.path.abspath(artifacts_dir)

        if not os.path.exists(artifacts_path):
            try:
                os.makedirs(artifacts_path, exist_ok=True)
            except Exception as e:
                return HealthCheck(
                    name="file_system",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Cannot create artifacts directory: {e}",
                    timestamp=time.time(),
                )

        # Check write permissions
        test_file = os.path.join(artifacts_path, ".health_check")
        try:
            with open(test_file, "w") as f:
                f.write("health_check")
            os.remove(test_file)
        except Exception as e:
            return HealthCheck(
                name="file_system",
                status=HealthStatus.UNHEALTHY,
                message=f"No write permission to artifacts directory: {e}",
                timestamp=time.time(),
            )

        # Check disk space for artifacts
        stat = os.statvfs(artifacts_path) if hasattr(os, "statvfs") else None
        if stat:
            free_bytes = stat.f_bavail * stat.f_frsize
            free_gb = free_bytes / (1024**3)
            if free_gb < 1.0:  # Less than 1GB free
                return HealthCheck(
                    name="file_system",
                    status=HealthStatus.DEGRADED,
                    message=".1f",
                    timestamp=time.time(),
                    details={"free_gb": free_gb},
                )

        return HealthCheck(
            name="file_system",
            status=HealthStatus.HEALTHY,
            message="File system OK",
            timestamp=time.time(),
            details={
                "artifacts_dir": artifacts_path,
                "writable": True,
            },
        )

    except Exception as e:
        return HealthCheck(
            name="file_system",
            status=HealthStatus.UNHEALTHY,
            message=f"File system check failed: {e}",
            timestamp=time.time(),
        )

def check_api_endpoints() -> Any:
    """Check internal API endpoint availability."""
    try:
        import requests
        base_url = os.getenv("BRAIN_BASE_URL", "http://localhost:8000")

        endpoints_to_check = [
            "/avix/ping",
            "/healthz",
            "/metrics",
        ]

        failed_endpoints = []
        for endpoint in endpoints_to_check:
            try:
                # Use a short connect timeout to avoid long hangs when services
                # (e.g. localhost:8000) are not accepting connections in test
                # environments. A tuple (connect, read) keeps connect attempts
                # bounded.
                response = requests.get(f"{base_url}{endpoint}", timeout=(0.5, 2))
                if response.status_code not in [200, 201]:
                    failed_endpoints.append(f"{endpoint} ({response.status_code})")
            except Exception as e:
                failed_endpoints.append(f"{endpoint} ({e})")

        if failed_endpoints:
            return HealthCheck(
                name="api_endpoints",
                status=HealthStatus.DEGRADED,
                message=f"Some endpoints failing: {failed_endpoints}",
                timestamp=time.time(),
                details={"failed_endpoints": failed_endpoints},
            )

        return HealthCheck(
            name="api_endpoints",
            status=HealthStatus.HEALTHY,
            message="API endpoints OK",
            timestamp=time.time(),
            details={"checked_endpoints": endpoints_to_check},
        )

    except Exception as e:
        return HealthCheck(
            name="api_endpoints",
            status=HealthStatus.UNHEALTHY,
            message=f"API endpoint check failed: {e}",
            timestamp=time.time(),
        )

def initialize_health_checks() -> None:
    """Initialize all health checks.

    This function is safe to call even when an enhanced observability
    implementation is not available; the module-level `health_checker` may
    therefore be None in some environments (tests, minimal installs).
    """
    if health_checker is None:
        # Nothing to register against in a minimal environment.
        return

    health_checker.add_check("system_resources", check_system_resources)
    health_checker.add_check("brain_components", check_brain_components)
    health_checker.add_check("database_connections", check_database_connections)
    health_checker.add_check("external_services", check_external_services)
    health_checker.add_check("file_system", check_file_system)
    health_checker.add_check("api_endpoints", check_api_endpoints)

def run_health_checks():
    """Run all health checks.

    Run checks asynchronously to avoid blocking request handlers or the
    monitoring worker when individual checks perform network I/O which may
    be slow or unreliable in test environments. The actual check results are
    stored in the shared metrics registry asynchronously.
    """
    # Execute checks in a daemon thread so callers are not blocked. If there is
    # no health_checker available (minimal install / tests) just return.
    if health_checker is None:
        return

    threading.Thread(target=health_checker.run_checks, daemon=True).start()

# Initialize health checks on import
initialize_health_checks()
