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
    """Stream video feed as MJPEG."""
    grabber = FrameGrabber.get_instance()
    return StreamingHttpResponse(
        grabber.generate_mjpeg_stream(),
        content_type='multipart/x-mixed-replace; boundary=frame'
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
    """List fire detection images with pagination."""
    save_dir = Path(getattr(settings, 'FIRE_SAVE_DIR', Path(settings.BASE_DIR) / 'media' / 'fire'))
    
    # Parse pagination parameters
    try:
        offset = max(0, int(request.GET.get('offset', 0)))
        limit = max(1, min(100, int(request.GET.get('limit', 20))))
    except ValueError:
        offset, limit = 0, 20
    
    files = []
    total = 0
    latest_mtime = 0
    
    if save_dir.exists():
        all_pics = sorted(save_dir.glob('*.jpg'), key=lambda p: p.stat().st_mtime, reverse=True)
        total = len(all_pics)
        
        # Apply pagination
        pics = all_pics[offset:offset + limit]
        files = [{'name': p.name, 'iso': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))} for p in pics]
        
        if all_pics:
            latest_mtime = int(all_pics[0].stat().st_mtime)
    
    # ETag for caching
    etag = f'W/"{latest_mtime}-{total}"'
    if request.headers.get('If-None-Match') == etag:
        return HttpResponse(status=304)
    
    response = JsonResponse({
        'files': files,
        'total': total,
        'offset': offset,
        'limit': limit
    })
    response['ETag'] = etag
    return response


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
