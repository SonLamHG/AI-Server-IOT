import cv2
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse, FileResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from pathlib import Path
import time, json

import numpy as np

from .camera import FrameGrabber
from .saver import FireImageSaver
from .sensor_store import store, SensorEntry


@require_GET
def index(request):
    return render(request, 'video_stream/index.html', context={})


@require_GET
def video_feed(request):
    grabber = FrameGrabber.get_instance()
    boundary = 'frame'
    return StreamingHttpResponse(
        grabber.mjpeg_generator(),
        content_type=f'multipart/x-mixed-replace; boundary={boundary}'
    )


@require_GET
def snapshot(request):
    saver = FireImageSaver()
    latest = saver.latest()
    if not latest or not latest.exists():
        return HttpResponse('No fire image', status=503)
    return FileResponse(open(latest, 'rb'), content_type='image/jpeg')


@require_GET
def fire_list(request):
    save_dir = Path(getattr(settings, 'FIRE_SAVE_DIR', Path(settings.BASE_DIR) / 'media' / 'fire'))
    files = []
    latest_mtime = 0
    if save_dir.exists():
        pics = sorted(save_dir.glob('*.jpg'), key=lambda p: p.stat().st_mtime, reverse=True)
        limit = max(1, int(request.GET.get('limit', 5)))
        files = [p.name for p in pics[:limit]]
        if pics:
            latest_mtime = int(pics[0].stat().st_mtime)
    # ETag để client có thể If-None-Match
    etag = f'W/"{latest_mtime}-{len(files)}-{files[0] if files else "none"}"'
    if request.headers.get('If-None-Match') == etag:
        return HttpResponse(status=304)
    resp = JsonResponse({'files': files})
    resp['ETag'] = etag
    return resp


@require_GET
def fire_events(request):
    # SSE: phát khi có ảnh mới trong thư mục
    save_dir = Path(getattr(settings, 'FIRE_SAVE_DIR', Path(settings.BASE_DIR) / 'media' / 'fire'))

    def stream():
        last_mtime = 0
        while True:
            try:
                pics = sorted(save_dir.glob('*.jpg'), key=lambda p: p.stat().st_mtime, reverse=True)
                cur_mtime = int(pics[0].stat().st_mtime) if pics else 0
                if cur_mtime != last_mtime:
                    last_mtime = cur_mtime
                    names = [p.name for p in pics[:5]]
                    yield f"data: {json.dumps({'files': names})}\n\n"
                else:
                    # keep-alive
                    yield "event: ping\ndata: {}\n\n"
            except Exception:
                yield "event: ping\ndata: {}\n\n"
            time.sleep(1)

    resp = StreamingHttpResponse(stream(), content_type='text/event-stream')
    resp['Cache-Control'] = 'no-cache'
    return resp


@require_GET
def simple_analysis(request):
    grabber = FrameGrabber.get_instance()
    jpg = grabber.get_frame()
    if not jpg:
        return JsonResponse({'error': 'no_frame'}, status=503)
    arr = np.frombuffer(jpg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JsonResponse({'error': 'decode_failed'}, status=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    return JsonResponse({'mean_intensity': mean_val})


@csrf_exempt
@require_http_methods(["GET", "POST"])
def sensor_api(request):
    if request.method == "POST":
        try:
            payload = json.loads(request.body.decode('utf-8') or '{}')
        except (UnicodeDecodeError, json.JSONDecodeError):
            return JsonResponse({'error': 'invalid_json'}, status=400)

        entry = SensorEntry.from_payload(payload)
        if entry is None:
            return JsonResponse({'error': 'invalid_payload'}, status=400)

        store.add(entry)
        return JsonResponse({'status': 'ok', 'data': entry.to_dict()})

    limit = request.GET.get('limit', '20')
    try:
        limit_int = max(1, min(500, int(limit)))
    except ValueError:
        limit_int = 20

    latest = store.latest()
    data = store.snapshot(limit_int)
    return JsonResponse({
        'latest': latest.to_dict() if latest else None,
        'history': data,
    })
