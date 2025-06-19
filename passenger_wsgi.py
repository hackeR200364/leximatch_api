from main import app
from fastapi.middleware.wsgi import WSGIMiddleware
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def application(environ, start_response):
    asgi_app = WSGIMiddleware(app)
    return asgi_app(environ, start_response)
