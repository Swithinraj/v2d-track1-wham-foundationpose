"""Insert mask stages into the open VS Code notebook via its MCP bridge."""

import json
import urllib.request
from pathlib import Path

from colab_mask_cells import cells


ROOT = Path(__file__).resolve().parents[2]
windows = [json.loads(path.read_text(encoding="utf-8"))
           for path in (Path.home() / ".vscode-notebook-mcp").glob("*.json")]
target = str(ROOT.parent).lower()
matches = [info for info in windows
           if any(target == str(Path(folder)).lower() for folder in info["workspaceFolders"])]
if len(matches) != 1:
    raise SystemExit(f"Expected one VS Code bridge for {ROOT.parent}; found {len(matches)}")
info = matches[0]
headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
           "Authorization": f"Bearer {info['token']}"}


def send(message):
    request = urllib.request.Request(info["url"], json.dumps(message).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        session = response.headers.get("mcp-session-id")
        if session:
            headers["mcp-session-id"] = session
        body = response.read().decode()
        if not body:
            return None
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            return json.loads(next(line[5:].strip() for line in body.splitlines() if line.startswith("data:")))
        return json.loads(body)


def call(name, **arguments):
    response = send({"jsonrpc": "2.0", "id": name, "method": "tools/call",
                     "params": {"name": name, "arguments": arguments}})
    result = response.get("result", {})
    if response.get("error") or result.get("isError"):
        raise RuntimeError(json.dumps(response, ensure_ascii=True)[:1500])
    return result


send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
      "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                 "clientInfo": {"name": "v2d-mask-notebook-editor", "version": "1"}}})
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
notebooks = call("notebook_list_open")["content"][0]["text"]
uri = "file:///d%3A/track1_v2d/Untitled-1.ipynb"
if uri not in notebooks:
    raise SystemExit("Expected Untitled-1.ipynb is not open in VS Code")
summary = call("notebook_list_cells", notebook_uri=uri)["content"][0]["text"]
if "v2d-vitpose-6" not in summary:
    raise SystemExit("Actor-mask gate cell is missing")
if "5a. Fetch only the three pinned source trees" in summary:
    raise SystemExit("Mask cells already inserted; leaving notebook untouched")

for offset, (kind, source) in enumerate(cells(ROOT)):
    call("notebook_insert_cell", notebook_uri=uri, index=7 + offset,
         type=kind, content=source, execute=False)
    print("Inserted", offset + 1, kind)
call("notebook_save", notebook_uri=uri)
print("Saved mask stages in", uri)
