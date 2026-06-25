from langchain_core.tools import tool
from kubernetes import client, config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception
from kubernetes.client.exceptions import ApiException
import chromadb
import os
from chromadb.utils import embedding_functions

_doc_client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), "doc_index"))
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
_doc_collection = _doc_client.get_collection("runbooks", embedding_function=_embedding_fn)

try:
    config.load_kube_config()
    v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()
except Exception:
    v1 = None
    apps_v1 = None

API_TIMEOUT_SECONDS = 10


def _is_transient_api_error(exception):
    if isinstance(exception, ApiException):
        return exception.status in (429, 500, 502, 503, 504)
    return False


# ── READ TOOLS ────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(ApiException),
)
def _list_pods_with_retry(namespace):
    if v1 is None:
        raise RuntimeError("No Kubernetes cluster available in this environment")
    return v1.list_namespaced_pod(namespace=namespace, _request_timeout=API_TIMEOUT_SECONDS)


@tool
def get_pod_status(namespace: str = "default") -> str:
    """Get the status of all pods in a given namespace, including restart counts and ready state."""
    try:
        pods = _list_pods_with_retry(namespace)
    except (ApiException, RuntimeError) as e:
        return f"Failed to fetch pod status: {str(e)}"
    results = []
    for pod in pods.items:
        name = pod.metadata.name
        phase = pod.status.phase
        restarts = 0
        last_state = "N/A"
        if pod.status.container_statuses:
            restarts = pod.status.container_statuses[0].restart_count
            last_term = pod.status.container_statuses[0].last_state.terminated
            if last_term:
                last_state = f"Terminated, Reason: {last_term.reason}, ExitCode: {last_term.exit_code}"
        results.append(
            f"Pod: {name} | Phase: {phase} | Restarts: {restarts} | LastState: {last_state}"
        )
    return "\n".join(results) if results else "No pods found."


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(ApiException),
)
def _list_pods_for_limits_with_retry(namespace):
    if v1 is None:
        raise RuntimeError("No Kubernetes cluster available in this environment")
    return v1.list_namespaced_pod(namespace=namespace, _request_timeout=API_TIMEOUT_SECONDS)


@tool
def get_pod_resource_limits(namespace: str = "default") -> str:
    """Get the configured memory and CPU limits for all pods in a namespace."""
    try:
        pods = _list_pods_for_limits_with_retry(namespace)
    except (ApiException, RuntimeError) as e:
        return f"Failed to fetch resource limits: {str(e)}"
    results = []
    for pod in pods.items:
        name = pod.metadata.name
        for container in pod.spec.containers:
            limits = container.resources.limits or {}
            requests = container.resources.requests or {}
            results.append(
                f"Pod: {name} | Container: {container.name} | "
                f"Limits: {limits} | Requests: {requests}"
            )
    return "\n".join(results) if results else "No pods found."


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(ApiException),
)
def _list_deployments_with_retry(namespace):
    if apps_v1 is None:
        raise RuntimeError("No Kubernetes cluster available in this environment")
    return apps_v1.list_namespaced_deployment(namespace=namespace, _request_timeout=API_TIMEOUT_SECONDS)


@tool
def get_deployment_info(namespace: str = "default") -> str:
    """Get deployment info including image version and replica counts."""
    try:
        deployments = _list_deployments_with_retry(namespace)
    except (ApiException, RuntimeError) as e:
        return f"Failed to fetch deployment info: {str(e)}"
    results = []
    for d in deployments.items:
        name = d.metadata.name
        image = d.spec.template.spec.containers[0].image
        replicas = d.spec.replicas
        available = d.status.available_replicas
        results.append(
            f"Deployment: {name} | Image: {image} | "
            f"DesiredReplicas: {replicas} | AvailableReplicas: {available}"
        )
    return "\n".join(results) if results else "No deployments found."


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_transient_api_error),
)
def _read_pod_log_with_retry(pod_name, namespace, tail_lines, previous):
    if v1 is None:
        raise RuntimeError("No Kubernetes cluster available in this environment")
    return v1.read_namespaced_pod_log(
        name=pod_name,
        namespace=namespace,
        tail_lines=tail_lines,
        timestamps=True,
        previous=previous,
        _request_timeout=API_TIMEOUT_SECONDS,
    )


@tool
def get_pod_logs(pod_name: str, namespace: str = "default", tail_lines: int = 50, previous: bool = False) -> str:
    """Get the most recent logs from a specific pod. Use get_pod_status first if you don't know the
    exact pod name. Set previous=True to get logs from the PREVIOUS container instance if the pod
    has restarted — this is essential for finding the cause of a crash, since the current container's
    logs only show what happened after the restart, not before it."""
    try:
        logs = _read_pod_log_with_retry(pod_name, namespace, tail_lines, previous)
        return logs if logs.strip() else "No log output found for this pod."
    except (ApiException, RuntimeError) as e:
        if isinstance(e, ApiException) and e.status == 400 and previous:
            return (
                f"No previous container logs available for {pod_name} — this pod likely hasn't "
                f"restarted, or the previous container's logs have been garbage collected."
            )
        return f"Error fetching logs: {str(e)}"


# ── WRITE TOOLS ───────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_transient_api_error),
)
def _delete_pod_with_retry(pod_name, namespace):
    if v1 is None:
        raise RuntimeError("No Kubernetes cluster available in this environment")
    return v1.delete_namespaced_pod(
        name=pod_name,
        namespace=namespace,
        _request_timeout=API_TIMEOUT_SECONDS,
    )


@tool
def restart_pod(pod_name: str, namespace: str = "default") -> str:
    """Force a restart of a specific pod by deleting it (Kubernetes will recreate it via the
    Deployment controller). Use this when a pod is in a bad state and a fresh restart may help.
    This is a WRITE operation that requires human approval before execution."""
    try:
        _delete_pod_with_retry(pod_name, namespace)
        return f"Pod {pod_name} deleted successfully. The Deployment controller will recreate it shortly."
    except (ApiException, RuntimeError) as e:
        if isinstance(e, ApiException) and e.status == 404:
            return f"Pod {pod_name} was already gone (404) — it may have already restarted or been deleted."
        return f"Error restarting pod: {str(e)}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_transient_api_error),
)
def _patch_deployment_with_retry(deployment_name, namespace, body):
    if apps_v1 is None:
        raise RuntimeError("No Kubernetes cluster available in this environment")
    return apps_v1.patch_namespaced_deployment(
        name=deployment_name,
        namespace=namespace,
        body=body,
        _request_timeout=API_TIMEOUT_SECONDS,
    )


@tool
def patch_memory_limit(deployment_name: str, new_limit_mi: int, namespace: str = "default") -> str:
    """Patch a deployment's container memory limit (in MiB) to a new value. Use this when a pod is
    repeatedly OOMKilled and the current memory limit appears insufficient for the workload.
    This is a WRITE operation that requires human approval before execution."""
    try:
        if apps_v1 is None:
            return "Kubernetes client not available — cannot patch deployment."
        deployment = apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            _request_timeout=API_TIMEOUT_SECONDS,
        )
        container = deployment.spec.template.spec.containers[0]
        if container.resources.limits is None:
            container.resources.limits = {}
        container.resources.limits["memory"] = f"{new_limit_mi}Mi"
        _patch_deployment_with_retry(deployment_name, namespace, deployment)
        return f"Deployment {deployment_name} memory limit patched to {new_limit_mi}Mi. Pods will roll out with the new limit."
    except (ApiException, RuntimeError) as e:
        return f"Error patching deployment: {str(e)}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_transient_api_error),
)
def _patch_scale_with_retry(deployment_name, namespace, replicas):
    if apps_v1 is None:
        raise RuntimeError("No Kubernetes cluster available in this environment")
    return apps_v1.patch_namespaced_deployment_scale(
        name=deployment_name,
        namespace=namespace,
        body={"spec": {"replicas": replicas}},
        _request_timeout=API_TIMEOUT_SECONDS,
    )


@tool
def scale_deployment(deployment_name: str, replicas: int, namespace: str = "default") -> str:
    """Scale a deployment to a new replica count. Use this to add redundancy or reduce load per pod.
    This is a WRITE operation that requires human approval before execution."""
    try:
        _patch_scale_with_retry(deployment_name, namespace, replicas)
        return f"Deployment {deployment_name} scaled to {replicas} replicas."
    except (ApiException, RuntimeError) as e:
        return f"Error scaling deployment: {str(e)}"


# ── RAG TOOL ──────────────────────────────────────────────────────────────────

@tool
def search_runbooks(query: str, n_results: int = 2) -> str:
    """Search internal runbooks and past incident postmortems for relevant guidance.
    Use this to find established procedures, approved remediation actions, or past similar
    incidents before proposing a fix or drawing a conclusion. Always check this before assuming
    a remediation action is appropriate."""
    try:
        results = _doc_collection.query(query_texts=[query], n_results=n_results)
    except Exception as e:
        return f"Runbook search failed unexpectedly: {str(e)}. Proceed without runbook guidance, but note this in your response."

    if not results["documents"] or not results["documents"][0]:
        return "No relevant runbooks found."

    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output.append(f"--- Source: {meta['source']} ---\n{doc}")
    return "\n\n".join(output)