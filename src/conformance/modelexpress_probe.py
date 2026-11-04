"""One ModelExpress ListSources call, run inside an engine pod by ``conformance.modelexpress.run_probe``.

The engine image carries the ``modelexpress`` client, so the probe uses the
same client code the engine does. The request arrives as JSON on stdin (it can
hold a token, which must not show up in the pod's process list):

  {"address": "https://host:port", "caFile": "/path" | null,
   "tokenPath": "/path" | null, "token": "<jwt>" | null}

``token`` is written to a private file and used as the token path. With
neither, the client sends no token. The result is one JSON line on stdout:

  {"ok": true, "sources": [{"workerId": ..., "workerRank": ..., "modelName": ...}]}
  {"ok": false, "code": "UNAUTHENTICATED", "detail": "..."}
"""

import json
import os
import sys
import tempfile

request = json.load(sys.stdin)
for name in ("MODEL_EXPRESS_URL", "MX_SERVER_ADDRESS", "MODEL_EXPRESS_TLS_CA_FILE", "MX_AUTH_TOKEN_PATH"):
    os.environ.pop(name, None)
os.environ["MX_SERVER_ADDRESS"] = request["address"]
if request.get("caFile"):
    os.environ["MODEL_EXPRESS_TLS_CA_FILE"] = request["caFile"]
token_path = request.get("tokenPath") or "/nonexistent/modelexpress-token"
if request.get("token"):
    fd, token_path = tempfile.mkstemp()
    with os.fdopen(fd, "w") as f:
        f.write(request["token"])
os.environ["MX_AUTH_TOKEN_PATH"] = token_path

import grpc  # noqa: E402
from modelexpress import p2p_pb2  # noqa: E402
from modelexpress.client import MxClient  # noqa: E402

client = MxClient()
try:
    response = client.stub.ListSources(
        p2p_pb2.ListSourcesRequest(status_filter=p2p_pb2.SOURCE_STATUS_READY), timeout=30
    )
    sources = [
        {"workerId": ref.worker_id, "workerRank": ref.worker_rank, "modelName": ref.model_name}
        for ref in response.instances
    ]
    print(json.dumps({"ok": True, "sources": sources}))
except grpc.RpcError as err:
    print(json.dumps({"ok": False, "code": err.code().name, "detail": (err.details() or "")[:300]}))
finally:
    client.close()
    if request.get("token"):
        os.unlink(token_path)
