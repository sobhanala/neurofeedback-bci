import socket
import json


class UnityTCPServer:

    def __init__(
        self,
        host="0.0.0.0",
        port=6000
    ):

        self.host = host
        self.port = port

        self.server_socket = None
        self.client_socket = None

        self.client_file = None


    def start(self):

        self.server_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        self.server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        self.server_socket.bind(
            (self.host, self.port)
        )

        self.server_socket.listen(1)

        print(
            f"Waiting for Unity on "
            f"{self.host}:{self.port}"
        )

        self.client_socket, addr = (
            self.server_socket.accept()
        )

        #self.client_socket.settimeout(5)

        print(
            f"Unity connected: {addr}"
        )

        self.client_file = (
            self.client_socket.makefile("r")
        )


    def send_json(self, message):

        encoded = (
            json.dumps(message)
            + "\n"
        )

        self.client_socket.sendall(
            encoded.encode("utf-8")
        )


    def receive_json(self):

        try:

            line = (
                self.client_file.readline()
            )

            if not line:

                return None

            return json.loads(line)

        except socket.timeout:

            return "TIMEOUT"

        except Exception as e:

            print(
                f"Receive error: {e}"
            )

            return None
        
    def close(self):

        try:

            if self.client_file:
                self.client_file.close()

        except Exception:
            pass

        try:

            if self.client_socket:
                self.client_socket.close()

        except Exception:
            pass

        try:

            if self.server_socket:
                self.server_socket.close()

        except Exception:
            pass