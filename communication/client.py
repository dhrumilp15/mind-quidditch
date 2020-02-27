#!/usr/bin/env python
import socket

# TCP_IP = "192.168.0.27"
TCP_IP = "DESKTOP-T40RD8H"
# print(TCP_IP)
TCP_PORT = 5005
BUFFER_SIZE = 1024
MESSAGE = b"Damn that drone team is pretty cool \
wowowow trailing spaces            \n"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
s.send(MESSAGE)
data = s.recv(BUFFER_SIZE)
s.close()

print("received data:", data)