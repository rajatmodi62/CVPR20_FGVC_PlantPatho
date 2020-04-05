import zmq
import time
import json
import socket
from utils.print_util import cprint


def publish(results):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind("tcp://127.0.0.1:1234")

    cprint("[ Publishing results on Telegram ]", type="info3")

    time.sleep(1)
    sock.send_multipart(
        [
            bytes("experiment", 'utf-8'),
            bytes(json.dumps(results), 'utf-8')
        ]
    )

    sock.close()
    ctx.term()
