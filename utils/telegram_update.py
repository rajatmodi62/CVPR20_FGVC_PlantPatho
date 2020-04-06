import zmq
import time
import json
import socket
from utils.print_util import cprint


def publish(results):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect("tcp://192.168.100.21:1234")

    cprint("[ Publishing results on Telegram ]", type="info3")

    time.sleep(1)
    sock.send(
        bytes(json.dumps(results), 'utf-8')
    )

    sock.close()
    ctx.term()
