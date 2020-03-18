import cv2
import threading
import http
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import time
import sys

class CamHandler(BaseHTTPRequestHandler):
    
    def __init__(self, request, client_address, server):
        img_src = 'cam.mjpg'
        self.html_page = """
            <html>
                <head></head>
                <body>
                    <img src="{}"/>

                </body>
            </html>""".format(img_src)
        self.html_404_page = """
            <html>
                <head></head>
                <body>
                    <h1>NOT FOUND</h1>
                </body>
            </html>"""
        BaseHTTPRequestHandler.__init__(self, request, client_address, server)

    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(http.HTTPStatus.OK)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    img = self.server.read_frame()
                    retval, jpg = cv2.imencode('.jpg', img)
                    if not retval:
                        raise RuntimeError('Could not encode img to JPEG')
                    jpg_bytes = jpg.tobytes()
                    self.wfile.write("--jpgboundary\r\n".encode())
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', len(jpg_bytes))
                    self.end_headers()
                    self.wfile.write(jpg_bytes)
                    time.sleep(self.server.read_delay)
                except (IOError, ConnectionError):
                    break
        elif self.path.endswith('.html'):
            self.send_response(http.HTTPStatus.OK)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.html_page.encode())
        else:
            self.send_response(http.HTTPStatus.NOT_FOUND)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.html_404_page.encode())


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    def __init__(self, capture_path, server_address, RequestHandlerClass, bind_and_activate=True):
        HTTPServer.__init__(self, server_address, RequestHandlerClass, bind_and_activate)
        ThreadingMixIn.__init__(self)
        try:
            # verifies whether is a webcam
            capture_path = int(capture_path)
        except TypeError:
            pass
        self._capture_path = capture_path
        fps = 30
        self.read_delay = 1. / fps
        self._lock = threading.Lock()
        self._camera = cv2.VideoCapture(0)
        self._camera.set(cv2.CAP_PROP_FPS, 30)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        # print(self._camera.get(cv2.CAP_PROP_FPS)) --> 30.0

    def open_video(self):
        if not self._camera.open(self._capture_path):
            raise IOError('Could not open Camera {}'.format(self._capture_path))

    def read_frame(self):
        with self._lock:
            retval, img = self._camera.read()
            if not retval:
                self.open_video()
        return img

    def serve_forever(self, poll_interval=0.5):
        self.open_video()
        try:
            super().serve_forever(poll_interval)
        except KeyboardInterrupt:
            self._camera.release()


def main():
    device = 0 if len(sys.argv) == 1 else sys.argv[1]        
    server = ThreadedHTTPServer(device, ('', 8080), CamHandler)
    print("server started at http://192.168.0.34:8080/cam.mjpg")
    server.serve_forever()


if __name__ == '__main__':
    main()