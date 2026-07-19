import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse
from server import app

KEY = os.environ.get("EARS_KEY", "")

class Auth(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if not KEY:
            return await call_next(request)
        if request.query_params.get("key") == KEY:
            resp = await call_next(request)
            resp.set_cookie("ears_key", KEY, httponly=True, max_age=31536000)
            return resp
        if request.cookies.get("ears_key") == KEY:
            return await call_next(request)
        if request.headers.get("x-ears-key") == KEY:
            return await call_next(request)
        return PlainTextResponse("unauthorized", status_code=401)

app.add_middleware(Auth)
