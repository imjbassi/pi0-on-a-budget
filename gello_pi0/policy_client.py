"""
policy_client.py — minimal client for openpi's websocket policy server.

openpi ships openpi-client for this, but it pins numpy<2, which conflicts with
the Windows recording environment (Python 3.14, numpy 2.x). This reimplements
the same wire protocol — msgpack with numpy arrays encoded as
{__ndarray__, data, dtype, shape} — following openpi's
packages/openpi-client/src/openpi_client/{msgpack_numpy,websocket_client_policy}.py
(Apache-2.0).
"""

import functools
import time

import msgpack
import numpy as np


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class PolicyClient:
    def __init__(self, uri="ws://localhost:8000", connect_timeout_s=300.0):
        import websockets.sync.client

        deadline = time.time() + connect_timeout_s
        while True:
            try:
                self.ws = websockets.sync.client.connect(uri, compression=None, max_size=None)
                break
            except (ConnectionRefusedError, OSError):
                if time.time() > deadline:
                    raise
                print(f"[client] waiting for policy server at {uri} ...", flush=True)
                time.sleep(3)
        self.metadata = unpackb(self.ws.recv())

    def infer(self, obs):
        self.ws.send(packb(obs))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"policy server error:\n{response}")
        return unpackb(response)

    def close(self):
        self.ws.close()
