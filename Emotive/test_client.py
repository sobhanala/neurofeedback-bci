import socket
import json


client = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)

client.connect(
    ("127.0.0.1", 6000)
)

print("Connected to Fusion")


client_file = client.makefile("r")


while True:

    line = client_file.readline()

    if not line:
        break

    message = json.loads(line)

    print()
    print("Received:")
    print(message)

    if message["msg_type"] == "COMMAND":

        ack = {

            "msg_type": "ACK",

            "command_id":
                message["command_id"]
        }

        encoded = (
            json.dumps(ack)
            + "\n"
        )

        client.sendall(
            encoded.encode("utf-8")
        )

        print(
            f"ACK sent "
            f"for command "
            f"{message['command_id']}"
        )