#!/usr/bin/env python3
"""JSONL daemon wrapping analyzer.py for persistent process reuse.

Protocol: One JSON object per line over stdin/stdout. stderr for debug logs only.
Startup: sends {"id":"__ready__","success":true,"result":{"status":"ready"}}
Actions: analyze, cross_file_analyze, health, shutdown
"""

import sys
import os

# CRITICAL: Redirect stdout to stderr BEFORE any imports can print to stdout.
# This prevents any imported library from corrupting the JSONL protocol channel.
_protocol_stdout = sys.stdout
sys.stdout = sys.stderr

# Now safe to import everything
import json
import time
from collections import OrderedDict

# Add script directory to path so analyzer imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer import analyze_file, analyze_file_ast, analyze_file_regex

try:
    from cross_file_analyzer import cross_file_analyze
    HAS_CROSS_FILE = True
except ImportError:
    HAS_CROSS_FILE = False


class LRUCache:
    """Simple LRU cache keyed by (file_path, mtime), capped at max_size entries."""

    def __init__(self, max_size=200):
        self._cache = OrderedDict()
        self._max_size = max_size

    def get(self, file_path):
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            return None
        key = (file_path, mtime)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, file_path, result):
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            return
        key = (file_path, mtime)
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def size(self):
        return len(self._cache)


_cache = LRUCache(max_size=200)


def send_response(obj):
    """Write a JSON line to the protocol channel (original stdout)."""
    line = json.dumps(obj, separators=(',', ':'))
    _protocol_stdout.write(line + '\n')
    _protocol_stdout.flush()


def handle_analyze(req):
    file_path = req.get('file_path')
    engine = req.get('engine', 'auto')

    if not file_path or not os.path.exists(file_path):
        return {'success': False, 'error': f'File not found: {file_path}'}

    # Check cache (only for engine=auto)
    if engine == 'auto':
        cached = _cache.get(file_path)
        if cached is not None:
            return {'success': True, 'result': cached, 'cached': True, 'cache_size': _cache.size}

    try:
        if engine == 'regex':
            result = analyze_file_regex(file_path)
        elif engine == 'ast':
            result = analyze_file_ast(file_path)
        else:
            result = analyze_file(file_path)
    except Exception as e:
        return {'success': False, 'error': str(e)}

    if isinstance(result, dict) and 'error' in result:
        return {'success': False, 'error': result['error']}

    # Cache result (only for engine=auto)
    if engine == 'auto':
        _cache.put(file_path, result)

    return {'success': True, 'result': result, 'cached': False, 'cache_size': _cache.size}


def handle_cross_file_analyze(req):
    file_paths = req.get('file_paths', [])
    if not file_paths:
        return {'success': False, 'error': 'No file_paths provided'}
    if not HAS_CROSS_FILE:
        return {'success': False, 'error': 'cross_file_analyzer not available'}

    try:
        result = cross_file_analyze(file_paths)
        return {'success': True, 'result': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def handle_scan_prompt_injection(req):
    """Stub for semantic prompt-injection scanning (Phase 4: ONNX embeddings).

    Currently returns empty findings. When the embedding engine is built,
    this action will run semantic similarity checks in-process.
    """
    return {
        'success': True,
        'result': {
            'findings': [],
            'engine': 'none',
            'note': 'Semantic engine not yet available',
        }
    }


def handle_health():
    return {
        'success': True,
        'result': {
            'status': 'healthy',
            'cache_size': _cache.size,
            'pid': os.getpid(),
            'uptime': time.time() - _start_time,
        }
    }


def main():
    global _start_time
    _start_time = time.time()

    # Signal readiness
    send_response({
        'id': '__ready__',
        'success': True,
        'result': {'status': 'ready'}
    })

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            send_response({'id': None, 'success': False, 'error': f'Invalid JSON: {e}'})
            continue

        req_id = req.get('id')
        action = req.get('action')

        if action == 'shutdown':
            send_response({'id': req_id, 'success': True, 'result': {'status': 'shutdown'}})
            break
        elif action == 'health':
            resp = handle_health()
        elif action == 'analyze':
            resp = handle_analyze(req)
        elif action == 'cross_file_analyze':
            resp = handle_cross_file_analyze(req)
        elif action == 'scan_prompt_injection':
            resp = handle_scan_prompt_injection(req)
        else:
            resp = {'success': False, 'error': f'Unknown action: {action}'}

        resp['id'] = req_id
        send_response(resp)


if __name__ == '__main__':
    main()
