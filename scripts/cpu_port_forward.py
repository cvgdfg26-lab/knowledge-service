"""Expose the CPU-host gateway through a localhost-only SSH connection."""

import select
import socketserver
import threading

from remote_control import ROOT, connect

lock = threading.Lock()
client = None


def transport():
    global client
    with lock:
        if client is None or not client.get_transport().is_active():
            if client:
                client.close()
            client = connect()
            client.get_transport().set_keepalive(30)
        return client.get_transport()


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        channel = None
        try:
            channel = transport().open_channel(
                "direct-tcpip", ("127.0.0.1", 18080), self.client_address, timeout=15
            )
            while True:
                readable, _, _ = select.select([self.request, channel], [], [], 30)
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    destination = channel if source is self.request else self.request
                    destination.sendall(data)
        except Exception as error:
            print("forward_error=" + type(error).__name__, flush=True)
        finally:
            if channel:
                channel.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    import os

    transport()
    with Server(("127.0.0.1", 18080), Handler) as server:
        (ROOT / ".cache/remote/forward.pid").write_text(str(os.getpid()))
        print("gateway=http://127.0.0.1:18080", flush=True)
        server.serve_forever()
