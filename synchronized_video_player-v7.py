#!/usr/bin/env python3
"""
Synchronized Dual Video Player

PyQt6 application for loading two videos, aligning them from embedded
creation/recording timestamps when available, and playing them on one shared
timeline.  File 2 is shown picture-in-picture at 15% of the video canvas width.

Install:
    py -3.13 -m pip install --upgrade PyQt6 PyOpenGL numpy

Optional but strongly recommended for timestamp and frame-rate detection:
    Install FFmpeg and make sure ffprobe.exe is in PATH.

Run:
    py -3.13 synchronized_video_player.py

If the hardware decoder is unstable:
    py -3.13 synchronized_video_player.py --software-decode



Notes:
    * Smooth Playback 7 defaults to QVideoWidget with no per-frame Python
      conversion or enhancement renderer. Keep speed at 1x for normal playback.
      Uncheck Smooth playback for image adjustments and zoom/pan, preferably
      while paused. Smooth mode retains settings but bypasses them for display.
    * Export MP4 renders the entire shared overlap forward at the selected
      speed, without audio, using saved settings (even in smooth mode). Output
      is H.264 CRF 18, 30 fps, 1920 pixels wide with the canvas aspect ratio.
      The current PiP layout, view swap, zoom and pan are applied. Spatial
      denoise/sharpen are FFmpeg approximations of the OpenGL preview. CPU
      encoding runs in a background process with progress and cancellation.
      Playback pauses to avoid competition for resources. Requires ffmpeg
      and ffprobe on PATH. A partial export never replaces the chosen target.
    * Frame Safety Update 6 removes QVideoFrame.toImage() entirely. Incoming
      frames are explicitly retained as Qt values; mapped pixel buffers are
      length-checked and copied before being released. Packed RGB and common
      8-bit planar/semi-planar YUV formats are supported. Unsupported formats
      stop conversion with a diagnostic instead of calling the old converter.
      YUV uses declared color range/matrix, with limited-range and BT.709 for
      HD / BT.601 for SD as defaults when metadata is unspecified. Chroma
      upsampling uses nearest samples. HDR/log grading is not implemented.
    * Stability update: background metadata probing ignores attached cover
      pictures. Forward playback follows the master decoder, correction
      seeks are rate-limited, and reverse preview seeks at most ~7 times/sec.
      Display conversions are capped at ~30/sec; source frames may be skipped
      for preview at faster rates. Export remains at decoded source resolution.
    * Speed buttons and slider apply 0.25x to 4x to both videos.
    * Reverse uses silent repeated seeks, not a native reverse decoder, and
      can be choppy with long-GOP video. Forward audio remains selectable.
    * Image review: brightness, exposure, contrast, gamma, shadows/highlights,
      saturation, 3x3 spatial bilateral denoise, and unsharp edge enhancement.
      Original bypasses all filters. These are SDR display operations, not
      reconstruction of missing detail or camera-specific log/HDR grading.
    * OpenGL 3.3 shaders render adjustments; Windows selects the adapter.
      Qt chooses available hardware decoders automatically, but decoding is
      not verified by this UI. Frame conversion/upload still use CPU work.
      The renderer name identifies software OpenGL if the driver falls back.
    * Wheel zooms; drag pans. In PiP use Shift+drag to pan. Zoom is digital.
      Link image adjustments copies File 1 to File 2 on enabling. View
      position/zoom stay independent. Reset image restores neutral filters.
    * Save both frames + PiP creates a new subfolder with file1_frame.png,
      file2_frame.png (unadjusted, source-resolution decoded RGB frames), and
      pip_view.png (display-resolution composition with current filters/zoom).
      Capture is the pair currently displayed, not a certified exact-time
      frame match. Originals on disk are never altered.
    * Playback codec support depends on the Qt Multimedia backend.
      FFprobe is used only for metadata inspection.
    * A video's embedded creation_time is not guaranteed to be the true camera
      recording time. The manual offset controls are provided for verification
      and correction.
"""

from __future__ import annotations

import json
import ctypes
import os
import time
import logging
import faulthandler
import tempfile
import uuid
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Optional

# Diagnostic fallback, applied before Qt loads the multimedia backend.
if '--software-decode' in sys.argv:
    os.environ['QT_FFMPEG_DECODING_HW_DEVICE_TYPES'] = ','

from PyQt6.QtCore import QElapsedTimer, QEvent, QPoint, QRect, QRectF, QSize, QSizeF, Qt, QTimer, QUrl, QObject, QRunnable, QThreadPool, pyqtSignal, QProcess
from PyQt6.QtGui import QCloseEvent, QMouseEvent, QColor, QImage, QSurfaceFormat, QPainter, QTransform
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink, QVideoFrame, QVideoFrameFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtMultimediaWidgets import QVideoWidget
import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QDockWidget, QScrollArea,
    QProgressDialog,
)


VIDEO_FILTER = (
    "Video files (*.mp4 *.mov *.m4v *.avi *.mkv *.wmv *.webm *.mts *.m2ts);;"
    "All files (*.*)"
)
MAX_AUTOSYNC_GAP_MS = 24 * 60 * 60 * 1000
DRIFT_CORRECTION_MS = 250


@dataclass
class VideoInfo:
    path: str = ""
    creation_time: Optional[datetime] = None
    creation_source: str = "Not available"
    fps: float = 30.0
    duration_ms: int = 0
    stream_index: int = -1


def format_clock(milliseconds: int, show_sign: bool = False) -> str:
    sign = ""
    if milliseconds < 0:
        sign = "-"
    elif show_sign and milliseconds > 0:
        sign = "+"

    value = abs(int(milliseconds))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{sign}{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def parse_datetime(value: object) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    # Common FFmpeg/QuickTime forms include Z, an ISO offset, or a naive time.
    try:
        normalized = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            # Only differences are used. Treat two naive camera times alike.
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        pass

    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def rational_to_float(value: object) -> float:
    try:
        result = float(Fraction(str(value)))
        return result if 0.1 <= result <= 1000 else 30.0
    except (ValueError, ZeroDivisionError):
        return 30.0


def select_video_stream(streams):
    """Ignore cover artwork and prefer the default moving-video stream."""
    candidates = [s for s in streams if s.get('codec_type') == 'video'
                  and not s.get('disposition', {}).get('attached_pic', 0)]
    return next((s for s in candidates if s.get('disposition', {}).get('default', 0)),
                candidates[0] if candidates else {})


def probe_video(path: str) -> VideoInfo:
    """Read useful video metadata with ffprobe; fail gracefully if unavailable."""
    info = VideoInfo(path=path)
    executable = shutil.which("ffprobe")
    if not executable:
        info.creation_source = "ffprobe not found; manual sync available"
        return info

    command = [
        executable,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        path,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        data = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        info.creation_source = f"Metadata unavailable: {error}"
        return info

    streams = data.get("streams", [])
    video_stream = select_video_stream(streams)
    info.stream_index = int(video_stream.get('index', -1))
    rate = video_stream.get('avg_frame_rate')
    if not rate or rate == '0/0':
        rate = video_stream.get('r_frame_rate') or '30/1'
    info.fps = rational_to_float(rate)

    duration = data.get("format", {}).get("duration") or video_stream.get("duration")
    try:
        info.duration_ms = max(0, round(float(duration) * 1000))
    except (TypeError, ValueError):
        pass

    candidates: list[tuple[str, object]] = []
    format_tags = data.get("format", {}).get("tags", {}) or {}
    stream_tags = video_stream.get("tags", {}) or {}
    tag_names = (
        "creation_time",
        "com.apple.quicktime.creationdate",
        "date",
        "encoded_date",
    )
    for name in tag_names:
        candidates.append((f"container tag: {name}", format_tags.get(name)))
    for name in tag_names:
        candidates.append((f"video-stream tag: {name}", stream_tags.get(name)))

    # Tags can differ only by capitalization depending on the encoder.
    for group_name, tags in (("container", format_tags), ("video stream", stream_tags)):
        for name, value in tags.items():
            if name.lower() in tag_names:
                candidates.append((f"{group_name} tag: {name}", value))

    for source, raw_value in candidates:
        parsed = parse_datetime(raw_value)
        if parsed is not None:
            info.creation_time = parsed
            info.creation_source = source
            break
    else:
        info.creation_source = "No embedded recording/creation timestamp"

    return info


# Values are slider units; contrast/gamma/saturation use percentages.
ADJUSTMENTS = {
    'brightness': ('Brightness', -100, 100, 0),
    'exposure': ('Exposure (0.01 EV)', -300, 300, 0),
    'contrast': ('Contrast (%)', 10, 300, 100),
    'gamma': ('Gamma (%)', 20, 300, 100),
    'shadows': ('Lift shadows', 0, 100, 0),
    'highlights': ('Reduce highlights', 0, 100, 0),
    'saturation': ('Saturation (%)', 0, 200, 100),
    'denoise': ('Spatial noise reduction', 0, 100, 0),
    'sharpen': ('Sharpen', 0, 100, 0),
}
DEFAULT_ADJUSTMENTS = {key: spec[3] for key, spec in ADJUSTMENTS.items()}

VERTEX_SHADER = '''#version 330 core
layout(location=0) in vec2 position;
layout(location=1) in vec2 texcoord;
out vec2 uv;
void main() { uv=texcoord; gl_Position=vec4(position,0,1); }
'''

FRAGMENT_SHADER = '''#version 330 core
in vec2 uv;
out vec4 frag;
uniform sampler2D video;
uniform vec2 pixel;
uniform vec2 center;
uniform float zoom;
uniform float brightness, exposure, contrast, gamma, shadows, highlights;
uniform float saturation, denoise, sharpen;
uniform int original;
void main() {
    vec2 p = (uv-0.5)/zoom + center;
    vec3 c = texture(video,p).rgb;
    if(original==0) {
        if(denoise>0.0 || sharpen>0.0) {
            vec3 sum=vec3(0); float weights=0.0;
            vec3 blur=vec3(0);
            for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++) {
                vec3 n=texture(video,p+vec2(x,y)*pixel).rgb;
                // Small bilateral neighborhood preserves strong edges.
                float w=exp(-dot(n-c,n-c)*40.0);
                sum+=n*w; weights+=w; blur+=n/9.0;
            }
            vec3 base=mix(c,sum/weights,denoise);
            c=base+sharpen*(c-blur);
        }
        c=max(c,vec3(0))*exp2(exposure);
        float l=clamp(dot(c,vec3(0.2126,0.7152,0.0722)),0.0,1.0);
        c+=shadows*0.45*(1.0-l)*(1.0-l);
        c*=1.0-highlights*0.65*l*l;
        c=(c-0.5)*contrast+0.5+brightness;
        c=pow(clamp(c,0.0,1.0),vec3(1.0/gamma));
        c=mix(vec3(dot(c,vec3(0.2126,0.7152,0.0722))),c,saturation);
    }
    frag=vec4(clamp(c,0.0,1.0),1);
}
'''


def yuv_to_rgba(y, u, v, color_space, full_range=False):
    """8-bit YCbCr -> owned RGBA. Inputs have equal, full-resolution shapes."""
    y = y.astype(np.float32)
    u = u.astype(np.float32)-128.0
    v = v.astype(np.float32)-128.0
    if full_range:
        y /= 255.0
        u /= 255.0
        v /= 255.0
    else:
        y = (y-16.0)/219.0
        u /= 224.0
        v /= 224.0
    kr,kb = {'BT709':(0.2126,0.0722),'BT2020':(0.2627,0.0593)}.get(color_space,(0.299,0.114))
    kg = 1-kr-kb
    rgba = np.empty((*y.shape,4),dtype=np.uint8)
    rgba[:,:,0] = np.clip(np.rint((y+2*(1-kr)*v)*255),0,255).astype(np.uint8)
    rgba[:,:,1] = np.clip(np.rint((y-2*kb*(1-kb)/kg*u-2*kr*(1-kr)/kg*v)*255),0,255).astype(np.uint8)
    rgba[:,:,2] = np.clip(np.rint((y+2*(1-kb)*u)*255),0,255).astype(np.uint8)
    rgba[:,:,3] = 255
    return rgba


def mapped_frame_to_image(frame):
    """Never call QVideoFrame.toImage(). Copy mapped bytes before unmapping.

    The explicit frame copy keeps the decoder buffer alive. Packed RGB uses
    Qt's documented pixel-format mapping; common 8-bit YUV uses NumPy.
    Unsupported formats raise a visible error rather than invoking the
    conversion path implicated by the user's native crash.
    """
    owned = QVideoFrame(frame)
    if not owned.isValid() or not owned.map(QVideoFrame.MapMode.ReadOnly):
        raise ValueError('Cannot map video frame. Try --software-decode.')
    try:
        width,height = owned.width(),owned.height()
        if width <= 0 or height <= 0:
            raise ValueError('Invalid video frame dimensions')
        pixel_format = owned.pixelFormat()
        image_format = QVideoFrameFormat.imageFormatFromPixelFormat(pixel_format)
        if image_format != QImage.Format.Format_Invalid:
            stride = owned.bytesPerLine(0)
            count = stride*height
            if count <= 0 or count > owned.mappedBytes(0):
                raise ValueError('Invalid packed frame buffer size')
            data = owned.bits(0).asstring(count)
            return QImage(data,width,height,stride,image_format).convertToFormat(QImage.Format.Format_RGBA8888).copy()

        def plane(index,rows,columns):
            if index >= owned.planeCount():
                raise ValueError('Video plane is missing')
            stride = owned.bytesPerLine(index)
            required = (rows-1)*stride+columns
            if stride < columns or required <= 0 or required > owned.mappedBytes(index):
                raise ValueError('Invalid video plane stride or length')
            data = owned.bits(index).asstring(required)
            return np.ndarray((rows,columns),dtype=np.uint8,buffer=data,strides=(stride,1)).copy()

        name = pixel_format.name
        supported = ('Format_YUV420P','Format_YV12','Format_NV12','Format_NV21','Format_YUV422P','Format_Y8')
        if name not in supported:
            raise ValueError(f'Unsupported mapped format: {name}. Use an 8-bit H.264 review copy.')
        y = plane(0,height,width)
        if name == 'Format_Y8':
            rgba = np.empty((height,width,4),dtype=np.uint8)
            rgba[:,:,:3] = y[:,:,None]
            rgba[:,:,3] = 255
        else:
            cw,ch = (width+1)//2,(height+1)//2
            if name == 'Format_YUV422P':
                ch = height
            if name in ('Format_NV12','Format_NV21'):
                uv = plane(1,ch,cw*2)
                u,v = uv[:,0::2],uv[:,1::2]
                if name == 'Format_NV21':
                    u,v = v,u
            else:
                u,v = plane(1,ch,cw),plane(2,ch,cw)
                if name == 'Format_YV12':
                    u,v = v,u
            vertical = 1 if name == 'Format_YUV422P' else 2
            u = u.repeat(vertical,axis=0).repeat(2,axis=1)[:height,:width]
            v = v.repeat(vertical,axis=0).repeat(2,axis=1)[:height,:width]
            fmt = owned.surfaceFormat()
            space = fmt.colorSpace().name.replace('ColorSpace_','')
            if space not in ('BT601','BT709','BT2020'):
                space = 'BT709' if height >= 720 else 'BT601'
            full = fmt.colorRange().name == 'ColorRange_Full'
            rgba = yuv_to_rgba(y,u,v,space,full)
        image = QImage(rgba.data,width,height,rgba.strides[0],QImage.Format.Format_RGBA8888)
        return image.copy()
    finally:
        owned.unmap()


def export_image_filters(settings,original,zoom,center):
    """FFmpeg review filters. Spatial filters approximate the GL preview."""
    filters = []
    if not original:
        s = settings
        if s['denoise']:
            filters.append(f"bilateral=sigmaS=1:sigmaR={0.01+s['denoise']/1000:.5f}")
        if s['sharpen']:
            filters.append(f"unsharp=3:3:{s['sharpen']/100:.5f}:3:3:0")
        if any(s[k] != DEFAULT_ADJUSTMENTS[k] for k in ('brightness','contrast','gamma','exposure','shadows','highlights','saturation')):
            filters.append('format=gbrp')
            luma = f"clip((0.2126*r(X,Y)+0.7152*g(X,Y)+0.0722*b(X,Y))/255*pow(2,{s['exposure']/100}),0,1)"
            expressions=[]
            for channel in ('r','g','b'):
                base=f"({channel}(X,Y)/255*pow(2,{s['exposure']/100})+{s['shadows']/100}*0.45*pow(1-({luma}),2))*(1-{s['highlights']/100}*0.65*pow({luma},2))"
                expressions.append(f"{channel}='255*pow(clip((({base})-0.5)*{s['contrast']/100}+0.5+{s['brightness']/200},0,1),{100/s['gamma']})'")
            filters.append('geq='+':'.join(expressions))
            if s['saturation'] != 100:
                lum='(0.2126*r(X,Y)+0.7152*g(X,Y)+0.0722*b(X,Y))'
                filters.append('geq='+':'.join(f"{c}='clip({lum}+({c}(X,Y)-{lum})*{s['saturation']/100},0,255)'" for c in ('r','g','b')))
    zoom=max(1.0,min(16.0,float(zoom)))
    if zoom > 1:
        cx,cy=[max(.5/zoom,min(1-.5/zoom,float(v))) for v in center]
        filters.append(f'crop=w=iw/{zoom}:h=ih/{zoom}:x=iw*({cx}-0.5/{zoom}):y=ih*({cy}-0.5/{zoom})')
    return filters


class ProbeSignals(QObject):
    finished = pyqtSignal(int, int, object)


class ProbeJob(QRunnable):
    def __init__(self, index, generation, path):
        super().__init__()
        self.index, self.generation, self.path = index, generation, path
        self.signals = ProbeSignals()

    def run(self):
        try:
            info = probe_video(self.path)
        except Exception:
            logging.exception('Metadata probe failed')
            info = VideoInfo(path=self.path, creation_source='Probe failed; manual sync available')
        self.signals.finished.emit(self.index, self.generation, info)


class EnhancedSurface(QOpenGLWidget):
    """Qt decode -> latest RGBA frame -> OpenGL texture and review shader.

    Color conversion and texture upload still involve the CPU. This is not a
    zero-copy decoder pipeline. Repaints reuse the uploaded texture.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sink = QVideoSink(self)
        self.sink.videoFrameChanged.connect(self._receive_frame)
        self.item = self  # Compatibility with the playback view-routing code.
        self.settings = DEFAULT_ADJUSTMENTS.copy()
        self.original = False
        self.zoom = 1.0
        self.center = [0.5, 0.5]
        self.drag_point = None
        self.in_pip = False
        self.frame = None
        self.dirty_frame = False
        self.image_size = (0, 0)
        self.display_image = QImage()
        self.display_timestamp_us = -1
        self.program = None
        self.renderer = 'OpenGL initializing; GPU decoding is managed automatically by Qt.'
        self.render_error = ''
        self.on_zoom = None
        self.view_rect = QRect()
        self.pending_image = QImage()
        self.pending_timestamp_us = -1
        # Conversion runs in a normal GUI callback, outside paintGL. Keeping
        # only the latest frame prevents a conversion queue during slow draws.
        self.frame_timer = QTimer(self)
        self.frame_timer.setInterval(33)
        self.frame_timer.timeout.connect(self._prepare_image)
        self.frame_timer.start()

    def videoSink(self):
        return self.sink

    def setAspectRatioMode(self, mode):
        pass  # All views preserve the source aspect ratio.

    def set_brightness(self, level):
        self.settings['brightness'] = level
        self.update()

    def _receive_frame(self, frame):
        if frame.isValid():
            # Retain a Qt-owned value, not just the signal's Python wrapper.
            self.frame = QVideoFrame(frame)
            self.dirty_frame = True

    def _prepare_image(self):
        if not self.dirty_frame or self.frame is None:
            return
        frame = QVideoFrame(self.frame)
        self.dirty_frame = False
        try:
            if not getattr(self,'logged_frame_format',False):
                fmt = frame.surfaceFormat()
                logging.info('Mapped conversion: %s %sx%s colorspace=%s range=%s',
                             frame.pixelFormat().name,frame.width(),frame.height(),
                             fmt.colorSpace().name,fmt.colorRange().name)
                self.logged_frame_format = True
            img = mapped_frame_to_image(frame)
            if img.isNull():
                return
            rotation = frame.rotationAngle().value if hasattr(frame,'rotationAngle') else 0
            if rotation:
                img = img.transformed(QTransform().rotate(rotation))
            if hasattr(frame,'mirrored') and frame.mirrored():
                img = img.mirrored(True,False)
            self.pending_image = img.convertToFormat(QImage.Format.Format_RGBA8888)
            self.pending_timestamp_us = frame.startTime()
            self.update()
        except Exception as error:
            logging.exception('Frame conversion failed')
            self.render_error = str(error)
            self.renderer = 'Frame conversion stopped: '+str(error)
            self.frame_timer.stop()

    def clear_frame(self):
        self.frame = None
        self.image_size = (0, 0)
        self.display_image = QImage()
        self.pending_image = QImage()
        self.dirty_frame = False
        self.logged_frame_format = False
        self.frame_timer.start()
        self.update()

    def initializeGL(self):
        try:
            self.dirty_frame = self.frame is not None
            self.image_size = (0,0)
            self.program = compileProgram(
                compileShader(VERTEX_SHADER, GL.GL_VERTEX_SHADER),
                compileShader(FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER))
            self.vao = GL.glGenVertexArrays(1)
            self.vbo = GL.glGenBuffers(1)
            GL.glBindVertexArray(self.vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
            vertices = np.array([-1,-1,0,1, 1,-1,1,1, -1,1,0,0, 1,1,1,0],dtype=np.float32)
            GL.glBufferData(GL.GL_ARRAY_BUFFER,vertices.nbytes,vertices,GL.GL_STATIC_DRAW)
            for index in (0,1):
                GL.glEnableVertexAttribArray(index)
                GL.glVertexAttribPointer(index,2,GL.GL_FLOAT,False,16,ctypes.c_void_p(index*8))
            self.texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D,self.texture)
            for param in (GL.GL_TEXTURE_MIN_FILTER,GL.GL_TEXTURE_MAG_FILTER):
                GL.glTexParameteri(GL.GL_TEXTURE_2D,param,GL.GL_LINEAR)
            for param in (GL.GL_TEXTURE_WRAP_S,GL.GL_TEXTURE_WRAP_T):
                GL.glTexParameteri(GL.GL_TEXTURE_2D,param,GL.GL_CLAMP_TO_EDGE)
            name = GL.glGetString(GL.GL_RENDERER).decode(errors='replace')
            self.renderer = f'OpenGL renderer: {name}. Decoder: Qt automatic (not verified).'
            self.context().aboutToBeDestroyed.connect(self._cleanup_gl)
        except Exception as error:
            self.program = None
            self.render_error = str(error)
            self.renderer = 'OpenGL initialization failed: ' + str(error)

    def _cleanup_gl(self):
        self.makeCurrent()
        if self.program:
            GL.glDeleteProgram(self.program)
            GL.glDeleteTextures([self.texture])
            GL.glDeleteBuffers(1,[self.vbo])
            GL.glDeleteVertexArrays(1,[self.vao])
            self.program = None
        self.doneCurrent()

    def paintGL(self):
        try:
            self._paint_frame()
        except Exception as error:
            logging.exception('OpenGL drawing failed')
            self.render_error = str(error)
            self.renderer = 'Rendering stopped: ' + str(error)
            self.frame_timer.stop()
            self.program = None

    def _paint_frame(self):
        GL.glClearColor(0,0,0,1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if not self.program:
            painter = QPainter(self)
            painter.setPen(QColor('white'))
            painter.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,
                             'OpenGL 3.3 required. Update your graphics driver.\n' + self.render_error)
            painter.end()
            return
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D,self.texture)
        if not self.pending_image.isNull():
            img = self.pending_image
            self.pending_image = QImage()
            self.display_image = img
            self.display_timestamp_us = self.pending_timestamp_us
            data = img.constBits().asstring(img.sizeInBytes())
            size = (img.width(),img.height())
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1)
            if self.image_size != size:
                GL.glTexImage2D(GL.GL_TEXTURE_2D,0,GL.GL_RGBA8,*size,0,GL.GL_RGBA,GL.GL_UNSIGNED_BYTE,data)
                self.image_size = size
            else:
                GL.glTexSubImage2D(GL.GL_TEXTURE_2D,0,0,0,*size,GL.GL_RGBA,GL.GL_UNSIGNED_BYTE,data)
        iw,ih = self.image_size
        if not iw or not ih:
            return
        scale = min(self.width()/iw,self.height()/ih)
        vw,vh = max(1,round(iw*scale)),max(1,round(ih*scale))
        x,y = (self.width()-vw)//2,(self.height()-vh)//2
        self.view_rect = QRect(x,y,vw,vh)
        dpr = self.devicePixelRatioF()
        GL.glViewport(round(x*dpr),round(y*dpr),round(vw*dpr),round(vh*dpr))
        GL.glUseProgram(self.program)
        def loc(key):
            return GL.glGetUniformLocation(self.program,key)
        GL.glUniform1i(loc('video'),0)
        GL.glUniform1i(loc('original'),int(self.original))
        GL.glUniform2f(loc('pixel'),1/iw,1/ih)
        GL.glUniform2f(loc('center'),*self.center)
        GL.glUniform1f(loc('zoom'),self.zoom)
        for key,value in self.settings.items():
            divisor = 200 if key=='brightness' else 100
            GL.glUniform1f(loc(key),value/divisor)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP,0,4)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def set_zoom(self, value):
        self.zoom = max(1,min(16,value/100))
        self._clamp_pan()
        self.update()

    def _clamp_pan(self):
        margin = 0.5/self.zoom
        self.center = [max(margin,min(1-margin,p)) for p in self.center]

    def fit_view(self):
        self.zoom = 1.0
        self.center = [0.5,0.5]
        if self.on_zoom:
            self.on_zoom(100)
        self.update()

    def wheelEvent(self,event):
        percent = round(self.zoom*100*(1.15 if event.angleDelta().y()>0 else 1/1.15))
        if self.on_zoom:
            self.on_zoom(max(100,min(1600,percent)))
        else:
            self.set_zoom(percent)
        event.accept()

    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton and (not self.in_pip or event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.drag_point = event.position()
            event.accept()
        else:
            event.ignore()  # Bubble to the PiP frame for moving/resizing.

    def mouseMoveEvent(self,event):
        if self.drag_point is not None:
            delta = event.position()-self.drag_point
            self.drag_point = event.position()
            self.center[0] -= delta.x()/max(1,self.view_rect.width())/self.zoom
            self.center[1] -= delta.y()/max(1,self.view_rect.height())/self.zoom
            self._clamp_pan()
            self.update()
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self,event):
        if self.drag_point is not None:
            self.drag_point = None
            event.accept()
        else:
            event.ignore()

    def mouseDoubleClickEvent(self,event):
        self.fit_view()
        event.accept()


class VideoSurface(QWidget):
    """Native playback by default; instantiate the costly review view on demand.

    No frame callbacks or conversion timer run in native mode. Pixel mapping
    happens only when a still image is explicitly requested.
    """
    def __init__(self,parent=None):
        super().__init__(parent)
        self.native = QVideoWidget(self)
        self.native.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.gpu = None
        self.smooth = True
        self.item = self
        self.settings = DEFAULT_ADJUSTMENTS.copy()
        self.original = False
        self.zoom = 1.0
        self.center = [0.5,0.5]
        self.on_zoom = None
        self.in_pip = False
        self.display_image = QImage()
        self.display_timestamp_us = -1
        self.idle_timer = QTimer(self)  # Never started in smooth mode.

    @property
    def center(self):
        return self.gpu.center if self.gpu else self._center

    @center.setter
    def center(self,value):
        self._center = list(value)
        if self.gpu:
            self.gpu.center = list(value)

    @property
    def frame_timer(self):
        return self.gpu.frame_timer if self.gpu else self.idle_timer

    @property
    def frame(self):
        return QVideoFrame(self.videoSink().videoFrame())

    @property
    def image_size(self):
        if self.smooth:
            size = self.native.videoSink().videoSize()
            return (size.width(),size.height())
        return self.gpu.image_size

    @property
    def renderer(self):
        if self.smooth:
            return 'Smooth playback: native Qt video display. Image filters and zoom bypassed; no per-frame Python conversion. Hardware decoder: automatic unless --software-decode was used.'
        return self.gpu.renderer

    def isValid(self):
        return True if self.smooth else self.gpu.isValid()

    def videoSink(self):
        return self.native.videoSink() if self.smooth else self.gpu.videoSink()

    def setAspectRatioMode(self,mode):
        self.native.setAspectRatioMode(mode)

    def set_smooth(self,enabled):
        self.smooth = enabled
        if not enabled and self.gpu is None:
            self.gpu = EnhancedSurface(self)
            self.gpu.center = self._center.copy()
        self.native.setVisible(enabled)
        self.native.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents,self.in_pip)
        if self.gpu:
            self.gpu.setVisible(not enabled)
            self.gpu.in_pip = self.in_pip
            self.gpu.setGeometry(self.rect())
            if enabled:
                self.gpu.frame_timer.stop()
                self.gpu.clear_frame()
                self.gpu.frame_timer.stop()
            else:
                self.gpu.frame_timer.start()
        self.native.setGeometry(self.rect())
        self.update()

    def resizeEvent(self,event):
        self.native.setGeometry(self.rect())
        if self.gpu:
            self.gpu.setGeometry(self.rect())
        super().resizeEvent(event)

    def update(self,*args):
        if getattr(self,'gpu',None) and not self.smooth:
            self.gpu.settings = self.settings.copy()
            self.gpu.original = self.original
            self.gpu.on_zoom = self.on_zoom
            self.gpu.update()
        super().update(*args)

    def set_zoom(self,value):
        self.zoom = value/100
        if self.gpu:
            self.gpu.set_zoom(value)

    def fit_view(self):
        self.zoom = 1.0
        if self.gpu:
            self.gpu.fit_view()
        if self.on_zoom:
            self.on_zoom(100)

    def clear_frame(self):
        self.display_image = QImage()
        if self.gpu:
            self.gpu.clear_frame()
            if self.smooth:
                self.gpu.frame_timer.stop()

    def _receive_frame(self,frame):
        if frame.isValid():
            self.videoSink().setVideoFrame(QVideoFrame(frame))

    def _prepare_image(self):
        if not self.smooth:
            self.gpu._prepare_image()
            return
        frame = self.frame
        if not frame.isValid():
            self.display_image = QImage()
            return
        try:
            img = mapped_frame_to_image(frame)
            rotation = frame.rotationAngle().value if hasattr(frame,'rotationAngle') else 0
            if rotation:
                img = img.transformed(QTransform().rotate(rotation))
            if hasattr(frame,'mirrored') and frame.mirrored():
                img = img.mirrored(True,False)
            self.display_image = img
            self.display_timestamp_us = frame.startTime()
        except Exception as error:
            logging.exception('Snapshot conversion failed')
            self.display_image = QImage()
            QMessageBox.warning(self,'Snapshot unavailable',str(error))

    def grabFramebuffer(self):
        if not self.smooth:
            image = self.gpu.grabFramebuffer()
            self.display_image = self.gpu.display_image.copy()
            return image
        # Build the neutral, letterboxed preview from the decoded frame;
        # grabbing the native video widget itself can give a black rectangle.
        if self.display_image.isNull():
            self._prepare_image()
        if self.display_image.isNull():
            return QImage()
        ratio = self.devicePixelRatioF()
        image = QImage(max(1,round(self.width()*ratio)),max(1,round(self.height()*ratio)),QImage.Format.Format_RGBA8888)
        image.fill(QColor('black'))
        scaled = self.display_image.scaled(image.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(image)
        painter.drawImage((image.width()-scaled.width())//2,(image.height()-scaled.height())//2,scaled)
        painter.end()
        return image


class ResizablePip(QFrame):
    """A video frame that can be dragged and resized from any edge/corner."""

    EDGE = 9
    MINIMUM = QSize(160, 90)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("pipFrame")
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(2)
        self.setMouseTracking(True)

        self.video = VideoSurface(self)
        self.video.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.video.in_pip = True
        self.video.set_smooth(True)

        self.title = QLabel("FILE 2 — PiP", self)
        self.title.setObjectName("pipTitle")
        self.title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.video)

        self._press_global = QPoint()
        self._start_geometry = QRect()
        self._resize_edges: set[str] = set()
        self._dragging = False
        self._manual_resize_callback = None

    def set_resize_callback(self, callback) -> None:
        self._manual_resize_callback = callback

    def resizeEvent(self, event: QEvent) -> None:
        super().resizeEvent(event)
        self.title.adjustSize()
        self.title.move(8, 7)

    def _edges_at(self, point: QPoint) -> set[str]:
        edges: set[str] = set()
        if point.x() <= self.EDGE:
            edges.add("left")
        if point.x() >= self.width() - self.EDGE:
            edges.add("right")
        if point.y() <= self.EDGE:
            edges.add("top")
        if point.y() >= self.height() - self.EDGE:
            edges.add("bottom")
        return edges

    @staticmethod
    def _cursor_for(edges: set[str]) -> Qt.CursorShape:
        if edges in ({"left", "top"}, {"right", "bottom"}):
            return Qt.CursorShape.SizeFDiagCursor
        if edges in ({"right", "top"}, {"left", "bottom"}):
            return Qt.CursorShape.SizeBDiagCursor
        if "left" in edges or "right" in edges:
            return Qt.CursorShape.SizeHorCursor
        if "top" in edges or "bottom" in edges:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.SizeAllCursor

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._press_global = event.globalPosition().toPoint()
        self._start_geometry = self.geometry()
        self._resize_edges = self._edges_at(event.position().toPoint())
        self._dragging = not bool(self._resize_edges)
        self.raise_()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            self.setCursor(self._cursor_for(self._edges_at(event.position().toPoint())))
            return

        delta = event.globalPosition().toPoint() - self._press_global
        parent_rect = self.parentWidget().rect()
        rect = QRect(self._start_geometry)

        if self._dragging:
            rect.moveTopLeft(rect.topLeft() + delta)
            rect.moveLeft(max(0, min(rect.left(), parent_rect.width() - rect.width())))
            rect.moveTop(max(0, min(rect.top(), parent_rect.height() - rect.height())))
        else:
            if "left" in self._resize_edges:
                rect.setLeft(rect.left() + delta.x())
            if "right" in self._resize_edges:
                rect.setRight(rect.right() + delta.x())
            if "top" in self._resize_edges:
                rect.setTop(rect.top() + delta.y())
            if "bottom" in self._resize_edges:
                rect.setBottom(rect.bottom() + delta.y())

            if rect.width() < self.MINIMUM.width():
                if "left" in self._resize_edges:
                    rect.setLeft(rect.right() - self.MINIMUM.width() + 1)
                else:
                    rect.setRight(rect.left() + self.MINIMUM.width() - 1)
            if rect.height() < self.MINIMUM.height():
                if "top" in self._resize_edges:
                    rect.setTop(rect.bottom() - self.MINIMUM.height() + 1)
                else:
                    rect.setBottom(rect.top() + self.MINIMUM.height() - 1)
            rect = rect.intersected(parent_rect)

        self.setGeometry(rect)
        if not self._dragging and self._manual_resize_callback:
            self._manual_resize_callback(self.width())
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False
        self._resize_edges.clear()
        event.accept()


class VideoCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background: #080b10;")

        self.main_video = VideoSurface(self)
        self.main_video.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.pip = ResizablePip(self)
        self.pip_percent = 15
        self._pip_initialized = False

    def set_pip_percentage(self, percent: int) -> None:
        self.pip_percent = max(5, min(80, percent))
        if self.width() <= 0:
            return
        width = max(ResizablePip.MINIMUM.width(), round(self.width() * percent / 100))
        height = max(ResizablePip.MINIMUM.height(), round(width * 9 / 16))
        width = min(width, self.width())
        height = min(height, self.height())
        margin = 18
        x = max(0, self.width() - width - margin)
        y = max(0, margin)
        self.pip.setGeometry(x, y, width, height)
        self.pip.raise_()
        self._pip_initialized = True

    def resizeEvent(self, event: QEvent) -> None:
        old_size = event.oldSize()
        self.main_video.setGeometry(self.rect())

        if not self._pip_initialized or old_size.width() <= 0 or old_size.height() <= 0:
            self.set_pip_percentage(self.pip_percent)
        else:
            # Preserve the PiP's relative location and size as the main window changes.
            old = self.pip.geometry()
            sx = self.width() / old_size.width()
            sy = self.height() / old_size.height()
            new_rect = QRect(
                round(old.x() * sx),
                round(old.y() * sy),
                max(ResizablePip.MINIMUM.width(), round(old.width() * sx)),
                max(ResizablePip.MINIMUM.height(), round(old.height() * sy)),
            )
            new_rect.setWidth(min(new_rect.width(), self.width()))
            new_rect.setHeight(min(new_rect.height(), self.height()))
            if new_rect.right() >= self.width():
                new_rect.moveRight(self.width() - 1)
            if new_rect.bottom() >= self.height():
                new_rect.moveBottom(self.height() - 1)
            self.pip.setGeometry(new_rect)
            self.pip.raise_()
        super().resizeEvent(event)


class SynchronizedVideoPlayer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Synchronized Dual Video Player — Smooth Playback 7")
        self.resize(1400, 900)
        self.setMinimumSize(850, 600)

        self.canvas = VideoCanvas()
        self.players = [QMediaPlayer(self), QMediaPlayer(self)]
        self.audio_outputs = [QAudioOutput(self), QAudioOutput(self)]
        for player, output in zip(self.players, self.audio_outputs):
            player.setAudioOutput(output)
        self.players[0].setVideoSink(self.canvas.main_video.item.videoSink())
        self.players[1].setVideoSink(self.canvas.pip.video.item.videoSink())

        self.info = [VideoInfo(), VideoInfo()]
        self.probe_generations = [0,0]
        self.probe_jobs = {}
        self.last_seek_at = [0.0,0.0]
        self.reverse_seek_at = 0.0
        self.path_edits: list[QLineEdit] = []
        self.metadata_labels: list[QLabel] = []
        self.view_swapped = False

        self.global_position_ms = 0
        self.global_duration_ms = 0
        self.metadata_relative_ms = 0
        self.delays_ms = [0, 0]
        self.playing = False
        self.direction = 1
        self.playback_speed = 1.0
        self.seeking = False
        self.clock = QElapsedTimer()
        self.clock_origin_ms = 0
        self.pending_play = False
        self.prepare_started = QElapsedTimer()
        self.prepare_timer = QTimer(self)
        self.prepare_timer.setInterval(25)
        self.prepare_timer.timeout.connect(self._finish_prepared_play)

        self.sync_timer = QTimer(self)
        self.sync_timer.setInterval(40)
        self.sync_timer.timeout.connect(self._synchronization_tick)

        self._build_interface()
        self._connect_players()
        self._apply_styles()
        self._update_audio_source()
        self._enable_review_controls(False)

    def _build_interface(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        files_group = QGroupBox("Video files")
        files_grid = QGridLayout(files_group)
        files_grid.setColumnStretch(1, 1)
        for index in range(2):
            name = QLabel(f"File {index + 1}")
            path_edit = QLineEdit()
            path_edit.setReadOnly(True)
            path_edit.setPlaceholderText(f"Select video file {index + 1}…")
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _checked=False, i=index: self._browse(i))
            metadata = QLabel("No file loaded")
            metadata.setObjectName("metadataLabel")
            metadata.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.path_edits.append(path_edit)
            self.metadata_labels.append(metadata)
            files_grid.addWidget(name, index * 2, 0)
            files_grid.addWidget(path_edit, index * 2, 1)
            files_grid.addWidget(browse, index * 2, 2)
            files_grid.addWidget(metadata, index * 2 + 1, 1, 1, 2)

        outer.addWidget(files_group)
        outer.addWidget(self.canvas, 1)

        timeline_row = QHBoxLayout()
        self.current_time_label = QLabel("00:00:00.000")
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.sliderPressed.connect(self._begin_seek)
        self.timeline.sliderMoved.connect(self._preview_seek)
        self.timeline.sliderReleased.connect(self._finish_seek)
        self.total_time_label = QLabel("00:00:00.000")
        timeline_row.addWidget(self.current_time_label)
        timeline_row.addWidget(self.timeline, 1)
        timeline_row.addWidget(self.total_time_label)
        outer.addLayout(timeline_row)

        controls = QHBoxLayout()
        self.reverse_button = QPushButton("◀ Reverse")
        self.reverse_button.clicked.connect(lambda: self._play_direction(-1))
        self.play_button = QPushButton("▶  Play")
        self.play_button.clicked.connect(lambda: self._play_direction(1))
        pause_button = QPushButton("⏸ Pause")
        pause_button.clicked.connect(self._pause)
        stop_button = QPushButton("■  Stop")
        stop_button.clicked.connect(self._stop)
        back_button = QPushButton("◀ Frame")
        back_button.clicked.connect(lambda: self._step_global_frame(-1))
        forward_button = QPushButton("Frame ▶")
        forward_button.clicked.connect(lambda: self._step_global_frame(1))
        swap_button = QPushButton("Swap views")
        swap_button.clicked.connect(self._swap_views)
        save_pair = QPushButton('Save both frames + PiP')
        save_pair.setToolTip('Two full-resolution unadjusted decoded frames, plus the adjusted on-screen PiP composition.')
        save_pair.clicked.connect(self._save_frame_set)

        controls.addWidget(self.reverse_button)
        controls.addWidget(self.play_button)
        controls.addWidget(pause_button)
        controls.addWidget(stop_button)
        controls.addWidget(back_button)
        controls.addWidget(forward_button)
        controls.addStretch(1)
        controls.addWidget(swap_button)
        controls.addWidget(save_pair)
        export_button = QPushButton('Export MP4')
        export_button.clicked.connect(self._export_video)
        controls.addWidget(export_button)
        outer.addLayout(controls)

        speed_row = QHBoxLayout()
        slower = QPushButton("− Slower")
        faster = QPushButton("+ Faster")
        normal = QPushButton("1× Normal")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(25, 400)
        self.speed_slider.setSingleStep(25)
        self.speed_slider.setValue(100)
        self.speed_label = QLabel("1.00×")
        self.speed_slider.valueChanged.connect(self._set_speed)
        slower.clicked.connect(lambda: self.speed_slider.setValue(self.speed_slider.value() - 25))
        faster.clicked.connect(lambda: self.speed_slider.setValue(self.speed_slider.value() + 25))
        normal.clicked.connect(lambda: self.speed_slider.setValue(100))
        for widget in (QLabel('Speed'), slower, self.speed_slider, faster, self.speed_label, normal):
            speed_row.addWidget(widget)
        outer.addLayout(speed_row)

        self._build_review_panel()

        settings_row = QHBoxLayout()
        sync_group = QGroupBox("Synchronization")
        sync_layout = QFormLayout(sync_group)
        self.use_metadata = QCheckBox("Use embedded timestamps when both are available")
        self.use_metadata.setChecked(False)
        self.use_metadata.toggled.connect(self._recalculate_alignment)

        offset_row = QHBoxLayout()
        self.manual_offset = QSpinBox()
        self.manual_offset.setRange(-86_400_000, 86_400_000)
        self.manual_offset.setSingleStep(10)
        self.manual_offset.setSuffix(" ms")
        self.manual_offset.setToolTip(
            "Positive values make File 2 occur later; negative values make it occur earlier."
        )
        self.manual_offset.valueChanged.connect(self._recalculate_alignment)
        minus_frame = QPushButton("− 1 frame")
        plus_frame = QPushButton("+ 1 frame")
        minus_frame.clicked.connect(lambda: self._nudge_file2(-1))
        plus_frame.clicked.connect(lambda: self._nudge_file2(1))
        offset_row.addWidget(self.manual_offset)
        offset_row.addWidget(minus_frame)
        offset_row.addWidget(plus_frame)

        self.alignment_label = QLabel("Load two files to calculate alignment")
        self.alignment_label.setWordWrap(True)
        sync_layout.addRow(self.use_metadata)
        sync_layout.addRow("File 2 adjustment:", offset_row)
        sync_layout.addRow("Effective alignment:", self.alignment_label)

        display_group = QGroupBox("Playback and display")
        display_layout = QFormLayout(display_group)
        self.audio_choice = QComboBox()
        self.audio_choice.addItems(["File 1", "File 2", "Muted"])
        self.audio_choice.setCurrentIndex(2)
        self.audio_choice.currentIndexChanged.connect(self._update_audio_source)
        self.pip_size = QSpinBox()
        self.pip_size.setRange(5, 80)
        self.pip_size.setValue(15)
        self.pip_size.setSuffix("%")
        self.pip_size.valueChanged.connect(self.canvas.set_pip_percentage)
        self.canvas.pip.set_resize_callback(self._pip_resized_manually)
        display_layout.addRow("Audio source:", self.audio_choice)
        display_layout.addRow("PiP width:", self.pip_size)
        display_layout.addRow(QLabel("Drag the PiP to move it; drag any edge or corner to resize it."))

        settings_row.addWidget(sync_group, 3)
        settings_row.addWidget(display_group, 2)
        outer.addLayout(settings_row)

        self.status_label = QLabel("Ready. Load two videos to begin.")
        self.status_label.setObjectName("statusLabel")
        outer.addWidget(self.status_label)
        self.setCentralWidget(central)

    def _build_review_panel(self):
        self.review_dock = QDockWidget('Image review controls',self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.smooth_choice = QCheckBox('Smooth playback (bypass image filters and zoom)')
        self.smooth_choice.setChecked(True)
        self.smooth_choice.toggled.connect(self._set_smooth_mode)
        layout.addWidget(self.smooth_choice)
        self.link_adjustments = QCheckBox('Link image adjustments')
        self.link_adjustments.setToolTip('Enabling copies File 1 settings to File 2. Zoom and pan stay independent.')
        layout.addWidget(self.link_adjustments)
        self.adjustment_controls = [{},{}]
        self.original_checks = []
        self.zoom_sliders = []
        self.value_labels = [{},{}]
        for index in range(2):
            group = QGroupBox(f'File {index+1}')
            form = QFormLayout(group)
            original = QCheckBox('Show original (bypass adjustments)')
            self.original_checks.append(original)
            original.toggled.connect(lambda value,i=index:self._original_changed(i,value))
            form.addRow(original)
            for key,(label,low,high,default) in ADJUSTMENTS.items():
                row = QHBoxLayout()
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(low,high)
                slider.setValue(default)
                slider.setMinimumWidth(100)
                value_label = QLabel(str(default))
                value_label.setMinimumWidth(32)
                self.value_labels[index][key] = value_label
                self.adjustment_controls[index][key] = slider
                slider.valueChanged.connect(lambda value,i=index,k=key:self._adjustment_changed(i,k,value))
                row.addWidget(slider)
                row.addWidget(value_label)
                form.addRow(label,row)
            zoom = QSlider(Qt.Orientation.Horizontal)
            zoom.setRange(100,1600)
            zoom.setValue(100)
            zoom.setToolTip('100 = fit; 1600 = 16× digital magnification (no added source detail).')
            self.zoom_sliders.append(zoom)
            zoom.valueChanged.connect(lambda value,i=index:self._surface_for_file(i).set_zoom(value))
            form.addRow('Zoom (%)',zoom)
            row = QHBoxLayout()
            reset = QPushButton('Reset image')
            reset.clicked.connect(lambda _=False,i=index:self._reset_image(i))
            fit = QPushButton('Fit view')
            fit.clicked.connect(lambda _=False,i=index:self._surface_for_file(i).fit_view())
            snapshot = QPushButton('Save preview PNG')
            snapshot.clicked.connect(lambda _=False,i=index:self._save_preview(i))
            for button in (reset,fit,snapshot):
                row.addWidget(button)
            form.addRow(row)
            layout.addWidget(group)
        self.link_adjustments.toggled.connect(self._link_changed)
        help_text = QLabel('Wheel: zoom. Drag: pan. In PiP, Shift+drag pans; plain drag moves the inset. Double-click: fit.\nFilters are viewing aids; strong denoise/sharpen can hide or distort detail. Preview PNGs are display-resolution captures, not original frames.')
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.renderer_label = QLabel('OpenGL initializing…')
        self.renderer_label.setWordWrap(True)
        layout.addWidget(self.renderer_label)
        layout.addStretch()
        scroll.setWidget(panel)
        self.review_dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,self.review_dock)
        self.menuBar().addMenu('View').addAction(self.review_dock.toggleViewAction())
        self.renderer_timer = QTimer(self)
        self.renderer_timer.timeout.connect(self._update_renderer_info)
        self.renderer_timer.start(2000)
        self._apply_image_settings()

    def _enable_review_controls(self,enabled):
        self.link_adjustments.setEnabled(enabled)
        for index in range(2):
            self.original_checks[index].setEnabled(enabled)
            self.zoom_sliders[index].setEnabled(enabled)
            for slider in self.adjustment_controls[index].values():
                slider.setEnabled(enabled)

    def _set_smooth_mode(self,enabled):
        resume = self.playing or self.pending_play
        self._pause()
        frames = [QVideoFrame(player.videoSink().videoFrame()) for player in self.players]
        for player in self.players:
            player.setVideoSink(None)
        for surface in (self.canvas.main_video,self.canvas.pip.video):
            surface.set_smooth(enabled)
        for index,player in enumerate(self.players):
            surface = self._surface_for_file(index)
            player.setVideoSink(surface.videoSink())
            surface._receive_frame(frames[index])
        self._enable_review_controls(not enabled)
        self._apply_image_settings()
        self.status_label.setText('Smooth mode: filters/zoom bypassed; saved settings still apply to export.' if enabled else 'Enhancement mode: adjustments active; pause for detailed inspection.')
        if resume:
            self._play()

    def _update_renderer_info(self):
        surface = self.canvas.main_video
        if surface.isVisible() and not surface.isValid():
            self.renderer_label.setText('OpenGL context unavailable. This version requires an OpenGL 3.3-capable graphics driver.')
        else:
            self.renderer_label.setText(surface.renderer)

    def _surface_for_file(self,index):
        surfaces = [self.canvas.main_video,self.canvas.pip.video]
        return surfaces[1-index if self.view_swapped else index]

    def _apply_image_settings(self):
        for index in range(2):
            surface = self._surface_for_file(index)
            surface.settings = {key:slider.value() for key,slider in self.adjustment_controls[index].items()}
            surface.original = self.original_checks[index].isChecked()
            surface.set_zoom(self.zoom_sliders[index].value())
            surface.on_zoom = self.zoom_sliders[index].setValue
            surface.update()

    def _adjustment_changed(self,index,key,value):
        self.value_labels[index][key].setText(str(value))
        if self.link_adjustments.isChecked():
            other = self.adjustment_controls[1-index][key]
            other.blockSignals(True)
            other.setValue(value)
            other.blockSignals(False)
            self.value_labels[1-index][key].setText(str(value))
        self._apply_image_settings()

    def _original_changed(self,index,value):
        if self.link_adjustments.isChecked():
            other = self.original_checks[1-index]
            other.blockSignals(True)
            other.setChecked(value)
            other.blockSignals(False)
        self._apply_image_settings()

    def _link_changed(self,enabled):
        if enabled:
            for key,slider in self.adjustment_controls[0].items():
                self._adjustment_changed(0,key,slider.value())
            self._original_changed(0,self.original_checks[0].isChecked())

    def _reset_image(self,index):
        for key,default in DEFAULT_ADJUSTMENTS.items():
            self.adjustment_controls[index][key].setValue(default)
        self.original_checks[index].setChecked(False)

    def _save_preview(self,index):
        surface = self._surface_for_file(index)
        if not self.info[index].path or not surface.image_size[0]:
            QMessageBox.information(self,'No frame','Load and display a video first.')
            return
        was_playing = self.playing or self.pending_play
        self._pause()
        path,_ = QFileDialog.getSaveFileName(self,'Save display-resolution preview',
                f'file{index+1}_preview_{self.global_position_ms}ms.png','PNG image (*.png)')
        if path:
            surface._prepare_image()
            surface.repaint()
            if not surface.grabFramebuffer().save(path,'PNG'):
                QMessageBox.warning(self,'Save failed','The preview could not be saved.')
        if was_playing:
            self._play()

    def _save_frame_set(self):
        if not all(info.path for info in self.info):
            QMessageBox.information(self,'Load both videos','Load and display both videos before saving a frame set.')
            return
        # Freeze without a fresh seek: export the decoded images on screen.
        was_playing = self.playing or self.pending_play
        if self.playing:
            self._update_global_from_clock()
        self.playing = False
        self.pending_play = False
        self.prepare_timer.stop()
        self.sync_timer.stop()
        for player in self.players:
            player.pause()
        self.play_button.setText('▶ Play')
        self._show_position()
        main = self.canvas.main_video
        inset = self.canvas.pip.video
        # Flush pending repaint on each surface, then retain immutable copies.
        main._prepare_image()
        inset._prepare_image()
        main_view = main.grabFramebuffer()
        inset_view = inset.grabFramebuffer()
        raw_images = [self._surface_for_file(i).display_image.copy() for i in range(2)]
        if any(image.isNull() for image in raw_images) or main_view.isNull() or inset_view.isNull():
            QMessageBox.warning(self,'No decoded frames','Play or step both videos until a frame is visible, then try again.')
            if was_playing:
                self._play()
            return
        ratio = self.canvas.devicePixelRatioF()
        composite = QImage(round(self.canvas.width()*ratio),round(self.canvas.height()*ratio),QImage.Format.Format_RGBA8888)
        composite.setDevicePixelRatio(ratio)
        composite.fill(QColor('black'))
        painter = QPainter(composite)
        painter.drawImage(QRectF(main.geometry()),main_view)
        pip_rect = QRectF(self.canvas.pip.geometry())
        painter.fillRect(pip_rect,QColor('#67a8e8'))
        point = inset.mapTo(self.canvas,QPoint(0,0))
        painter.drawImage(QRectF(point.x(),point.y(),inset.width(),inset.height()),inset_view)
        painter.setPen(QColor('white'))
        painter.fillRect(QRectF(pip_rect.x()+5,pip_rect.y()+5,100,22),QColor(0,0,0,170))
        painter.drawText(QPoint(round(pip_rect.x()+9),round(pip_rect.y()+21)),self.canvas.pip.title.text())
        painter.end()
        directory = QFileDialog.getExistingDirectory(self,'Choose destination for the three PNG images')
        if directory:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            folder = Path(directory) / f'frame_set_{stamp}'
            try:
                folder.mkdir(exist_ok=False)
                for index,image in enumerate(raw_images):
                    if not image.save(str(folder/f'file{index+1}_frame.png'),'PNG'):
                        raise OSError(f'Could not save File {index+1} frame')
                if not composite.save(str(folder/'pip_view.png'),'PNG'):
                    raise OSError('Could not save PiP image')
                self.status_label.setText(f'Saved 3 PNGs to {folder}')
            except OSError as error:
                QMessageBox.warning(self,'Save incomplete',f'{error}\nSome files may have been saved in {folder}.')
        if was_playing:
            self._play()

    def _export_video(self):
        if getattr(self,'export_process',None) is not None:
            QMessageBox.information(self,'Export running','Finish or cancel the current export first.')
            return
        executable=shutil.which('ffmpeg')
        if not executable:
            QMessageBox.warning(self,'FFmpeg required','Install FFmpeg and put ffmpeg.exe on PATH to export.')
            return
        if not all(i.path and i.stream_index>=0 for i in self.info) or self.global_duration_ms<=0:
            QMessageBox.information(self,'Videos not ready','Load both videos and wait for their metadata. Both need a shared time range.')
            return
        path,_=QFileDialog.getSaveFileName(self,'Export full shared range, forward, silent — saved settings apply',
                                         'review_render.mp4','MP4 video (*.mp4)')
        if not path:
            return
        destination=Path(path).with_suffix('.mp4')
        if str(destination)!=path and destination.exists():
            QMessageBox.warning(self,'Choose another name','An MP4 with that name already exists. Select its exact .mp4 name in the save dialog to confirm replacement, or choose a new name.')
            return
        if any(str(destination.resolve()).casefold()==str(Path(i.path).resolve()).casefold() for i in self.info):
            QMessageBox.warning(self,'Choose a new filename','The export must not overwrite either input video.')
            return
        self._pause()
        temporary=destination.with_name(destination.stem+'.rendering-'+uuid.uuid4().hex+'.mp4')
        duration=self.global_duration_ms/1000
        speed=self.playback_speed
        width=1920
        height=max(2,round(width*self.canvas.height()/max(1,self.canvas.width())/2)*2)
        inset=self.canvas.pip.geometry()
        pw=max(2,round(inset.width()/self.canvas.width()*width/2)*2)
        ph=max(2,round(inset.height()/self.canvas.height()*height/2)*2)
        px=max(0,min(width-pw,round(inset.x()/self.canvas.width()*width)))
        py=max(0,min(height-ph,round(inset.y()/self.canvas.height()*height)))
        main_index=1 if self.view_swapped else 0
        chains=[]
        args=['-nostdin','-hide_banner','-v','error','-n']
        for index,info in enumerate(self.info):
            args += ['-ss',f'{max(0,-self.delays_ms[index])/1000:.6f}','-i',info.path]
            target_w,target_h=(width,height) if index==main_index else (pw,ph)
            surface=self._surface_for_file(index)
            settings={k:control.value() for k,control in self.adjustment_controls[index].items()}
            filters=[f'trim=duration={duration:.6f}',f'setpts=(PTS-STARTPTS)/{speed:.6f}']
            filters+=export_image_filters(settings,self.original_checks[index].isChecked(),
                                          self.zoom_sliders[index].value()/100,surface.center)
            filters += [f'scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:force_divisible_by=2',
                        f'pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black','setsar=1','fps=30','format=yuv420p']
            chains.append(f'[{index}:{info.stream_index}]'+','.join(filters)+f'[v{index}]')
        chains.append(f'[v{main_index}][v{1-main_index}]overlay=x={px}:y={py}:shortest=1[out]')
        args += ['-filter_complex',';'.join(chains),'-map','[out]','-an','-c:v','libx264',
                 '-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
                 '-t',f'{duration/speed:.6f}','-progress','pipe:1',str(temporary)]
        self.export_target=destination
        self.export_temp=temporary
        self.export_seconds=duration/speed
        self.export_errors=''
        self.export_progress_buffer=''
        self.export_cancelled=False
        self.export_dialog=QProgressDialog('Rendering full shared range. Playback is paused.\nSaved adjustments apply even in smooth mode. Spatial filters may differ slightly from preview.','Cancel',0,100,self)
        self.export_dialog.setWindowTitle('Export MP4 — CPU encoding')
        self.export_dialog.setAutoClose(False)
        self.export_dialog.setValue(0)
        process=QProcess(self)
        self.export_process=process
        process.readyReadStandardOutput.connect(self._export_progress)
        process.readyReadStandardError.connect(self._export_error_text)
        process.finished.connect(self._export_finished)
        process.errorOccurred.connect(self._export_process_error)
        self.export_dialog.canceled.connect(self._cancel_export)
        process.start(executable,args)
        self.export_dialog.show()

    def _export_progress(self):
        if self.export_process is None:
            return
        self.export_progress_buffer+=bytes(self.export_process.readAllStandardOutput()).decode(errors='replace')
        lines=self.export_progress_buffer.split('\n')
        self.export_progress_buffer=lines.pop()
        for line in lines:
            if line.startswith('out_time_us='):
                try:
                    elapsed=int(line.split('=',1)[1])/1e6
                    self.export_dialog.setValue(min(99,max(0,int(100*elapsed/self.export_seconds))))
                except ValueError:
                    pass

    def _export_error_text(self):
        if self.export_process is not None:
            self.export_errors=(self.export_errors+bytes(self.export_process.readAllStandardError()).decode(errors='replace'))[-8000:]

    def _cancel_export(self):
        if getattr(self,'export_process',None) is not None:
            self.export_cancelled=True
            self.export_process.kill()

    def _export_process_error(self,error):
        if error==QProcess.ProcessError.FailedToStart:
            self.export_errors='FFmpeg could not start.'
            self._export_finished(-1,QProcess.ExitStatus.CrashExit)

    def _export_finished(self,code,status):
        if self.export_process is None:
            return
        self._export_error_text()
        process=self.export_process
        self.export_process=None
        self.export_dialog.close()
        try:
            if not self.export_cancelled and code==0 and status==QProcess.ExitStatus.NormalExit and self.export_temp.exists() and self.export_temp.stat().st_size>0:
                os.replace(self.export_temp,self.export_target)
                self.status_label.setText(f'Export saved: {self.export_target}')
                QMessageBox.information(self,'Export complete',str(self.export_target))
            else:
                self.export_temp.unlink(missing_ok=True)
                if not self.export_cancelled:
                    QMessageBox.warning(self,'Export failed',self.export_errors or 'FFmpeg did not complete.')
        except OSError as error:
            QMessageBox.warning(self,'Could not finalize export',f'{error}\nTemporary file: {self.export_temp}')
        process.deleteLater()

    def _connect_players(self) -> None:
        for index, player in enumerate(self.players):
            player.durationChanged.connect(
                lambda duration, i=index: self._duration_changed(i, duration)
            )
            player.errorOccurred.connect(
                lambda error, message, i=index: self._player_error(i, message)
            )

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #151a22; color: #e7edf6; }
            QGroupBox { border: 1px solid #364253; border-radius: 5px;
                        margin-top: 9px; padding-top: 8px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; }
            QLineEdit, QComboBox, QSpinBox { background: #0f141b; border: 1px solid #435168;
                                            border-radius: 4px; padding: 5px; }
            QPushButton { background: #26364a; border: 1px solid #4a6482;
                          border-radius: 4px; padding: 6px 12px; }
            QPushButton:hover { background: #314863; }
            QPushButton:pressed { background: #1c2b3c; }
            #metadataLabel { color: #aebbd0; font-size: 11px; }
            #statusLabel { color: #b8c7dc; padding: 3px; }
            #pipFrame { background: #05070a; border: 2px solid #67a8e8; }
            #pipTitle { background: rgba(0, 0, 0, 150); color: white;
                        font-size: 10px; font-weight: bold; padding: 3px 5px; }
            """
        )

    def _browse(self, index: int) -> None:
        start_dir = str(Path(self.info[index].path).parent) if self.info[index].path else ""
        path, _selected_filter = QFileDialog.getOpenFileName(
            self, f"Select video file {index + 1}", start_dir, VIDEO_FILTER
        )
        if path:
            self._load_file(index, path)

    def _load_file(self, index: int, path: str) -> None:
        self._pause()
        logging.info('Loading file %s: %s',index+1,path)
        self.probe_generations[index] += 1
        generation = self.probe_generations[index]
        self.info[index] = VideoInfo(path=path)
        self._surface_for_file(index).clear_frame()
        self.path_edits[index].setText(path)
        self.path_edits[index].setToolTip(path)
        self.players[index].setSource(QUrl.fromLocalFile(path))
        self.metadata_labels[index].setText('Reading metadata in background…')
        self.status_label.setText(f'Opening File {index+1}; metadata is loading in background.')
        job = ProbeJob(index,generation,path)
        job.signals.finished.connect(self._probe_finished)
        self.probe_jobs[(index,generation)] = job
        QThreadPool.globalInstance().start(job)
        self.global_position_ms = 0
        self._recalculate_alignment()

    def _probe_finished(self,index,generation,info):
        self.probe_jobs.pop((index,generation),None)
        if generation != self.probe_generations[index]:
            return  # Ignore completion for a file that was replaced.
        if self.players[index].duration() > 0:
            info.duration_ms = self.players[index].duration()
        self.info[index] = info
        logging.info('Metadata ready file %s: fps=%s duration_ms=%s',index+1,info.fps,info.duration_ms)
        if info.creation_time:
            timestamp = info.creation_time.astimezone().isoformat(timespec="milliseconds")
            metadata_text = (
                f"Timestamp: {timestamp}  |  FPS: {info.fps:.3f}  |  Source: {info.creation_source}"
            )
        else:
            metadata_text = f"Timestamp: unavailable  |  FPS: {info.fps:.3f}  |  {info.creation_source}"
        self.metadata_labels[index].setText(metadata_text)
        if self.use_metadata.isChecked():
            self._recalculate_alignment()
        else:
            self._update_global_duration()

    def _duration_changed(self, index: int, duration: int) -> None:
        if duration > 0:
            self.info[index].duration_ms = duration
            self._update_global_duration()

    def _player_error(self, index: int, message: str) -> None:
        if not message:
            message = "Unknown multimedia error"
        self.status_label.setText(f"File {index + 1} playback error: {message}")
        logging.error('File %s playback error: %s',index+1,message)

    def _metadata_delta_ms(self) -> Optional[int]:
        first = self.info[0].creation_time
        second = self.info[1].creation_time
        if first is None or second is None:
            return None
        return round((second - first).total_seconds() * 1000)

    def _recalculate_alignment(self) -> None:
        metadata_delta = self._metadata_delta_ms()
        use_delta = 0
        note = "Manual alignment"

        if self.use_metadata.isChecked() and metadata_delta is not None:
            if abs(metadata_delta) <= MAX_AUTOSYNC_GAP_MS:
                use_delta = metadata_delta
                note = f"Metadata difference {format_clock(metadata_delta, True)}"
            else:
                note = (
                    f"Metadata gap {format_clock(metadata_delta, True)} exceeds 24 hours; "
                    "ignored as a likely export timestamp"
                )
        elif self.use_metadata.isChecked():
            note = "A timestamp is required in both files; using manual alignment"

        self.metadata_relative_ms = use_delta
        relative = use_delta + self.manual_offset.value()

        # Relative > 0: File 2 starts later. Relative < 0: File 1 starts later.
        # Start at the common overlap, so neither clip waits motionless.
        # Negative delays are source offsets into the earlier recording.
        self.delays_ms = [-max(0, relative), -max(0, -relative)]
        self.alignment_label.setText(
            f"{note}. File 2 relative to File 1: {format_clock(relative, True)}"
        )
        self._update_global_duration()
        self._seek_global(self.global_position_ms)

    def _update_global_duration(self) -> None:
        ends = [
            self.delays_ms[i] + self.info[i].duration_ms
            for i in range(2)
            if self.info[i].path and self.info[i].duration_ms > 0
        ]
        self.global_duration_ms = max(0, min(ends, default=0))
        self.timeline.setRange(0, min(self.global_duration_ms, 2_147_483_647))
        self.total_time_label.setText(format_clock(self.global_duration_ms))

    def _toggle_play(self) -> None:
        if self.playing or self.pending_play:
            self._pause()
        else:
            self._play()

    def _play_direction(self, direction):
        if (self.playing or self.pending_play) and self.direction == direction:
            self._pause()
            return
        self._pause()
        self.direction = direction
        self._play()

    def _set_speed(self, percent):
        if self.playing:
            self._update_global_from_clock()
        self.playback_speed = percent / 100.0
        self.clock_origin_ms = self.global_position_ms
        self.clock.restart()
        for player in self.players:
            player.setPlaybackRate(self.playback_speed)
        self.speed_label.setText(f'{self.playback_speed:.2f}×')

    def _update_brightness(self, *_args):
        self._apply_image_settings()

    def _play(self) -> None:
        if not any(item.path for item in self.info):
            QMessageBox.information(self, "No videos", "Load at least one video first.")
            return
        self._pause()
        if self.direction > 0 and self.global_duration_ms and self.global_position_ms >= self.global_duration_ms:
            self.global_position_ms = 0
        if self.direction < 0 and self.global_position_ms <= 0:
            self.status_label.setText('Already at the beginning. Seek forward before playing backwards.')
            return
        if self.direction < 0 and self.global_duration_ms:
            self.global_position_ms = min(self.global_position_ms, max(0, self.global_duration_ms - 1))
        self.pending_play = True
        self.prepare_started.restart()
        self.play_button.setText("Cancel start")
        self.status_label.setText("Preparing both videos at the shared timeline position…")
        self._synchronize_players(force=True)
        self.prepare_timer.start()

    def _finish_prepared_play(self) -> None:
        if not self.pending_play:
            return
        if self.prepare_started.elapsed() > 10000:
            self._pause()
            self.status_label.setText("Could not prepare both videos. Check codecs, seek support, and alignment overlap.")
            return
        ready_statuses = (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.BufferingMedia,
            QMediaPlayer.MediaStatus.EndOfMedia,
        )
        loaded = [i for i in range(2) if self.info[i].path]
        if any(self.players[i].mediaStatus() not in ready_statuses
               or not self.players[i].isSeekable()
               or self.players[i].duration() <= 0 for i in loaded):
            return
        for i in loaded:
            target = self.global_position_ms - self.delays_ms[i]
            if target >= self.players[i].duration():
                self._pause()
                self.status_label.setText("No shared time at this alignment. Disable timestamps or adjust the offset.")
                return
            if abs(self.players[i].position() - target) > 50:
                now = time.monotonic()
                if now - self.last_seek_at[i] > 1.0:
                    self.players[i].setPosition(target)
                    self.last_seek_at[i] = now
                return
        self.prepare_timer.stop()
        self.pending_play = False
        # All seeks are submitted before either play command or clock starts.
        for i in loaded:
            self.players[i].setPlaybackRate(self.playback_speed)
            if self.direction > 0:
                self.players[i].play()
        self.playing = True
        self.clock_origin_ms = self.global_position_ms
        self.clock.restart()
        self.sync_timer.start()
        self._update_audio_source()
        self.play_button.setText("▶ Forward")
        self.status_label.setText('Reverse playback — audio muted; seek-based preview.' if self.direction < 0 else 'Playing forward together.')

    def _pause(self) -> None:
        if self.playing:
            self._update_global_from_clock()
        self.playing = False
        self.pending_play = False
        self.prepare_timer.stop()
        self.sync_timer.stop()
        for player in self.players:
            player.pause()
        self._synchronize_players(force=True)
        self.play_button.setText("▶  Play")
        self._update_audio_source()
        self._show_position()

    def _stop(self) -> None:
        self._pause()
        self.global_position_ms = 0
        self._seek_global(0)

    def _update_global_from_clock(self) -> None:
        # Forward playback follows an actual decoder, so loading/buffering
        # does not let a wall clock race ahead and trigger repeated seeks.
        if self.playing and self.direction > 0:
            index = 0 if self.info[0].path else 1
            self.global_position_ms = max(0,self.players[index].position()+self.delays_ms[index])
            if self.global_duration_ms:
                self.global_position_ms = min(self.global_position_ms,self.global_duration_ms)
            return
        if self.clock.isValid():
            self.global_position_ms = max(0, self.clock_origin_ms + round(self.clock.elapsed() * self.playback_speed * self.direction))
            if self.global_duration_ms:
                self.global_position_ms = min(
                    self.global_position_ms, self.global_duration_ms
                )

    def _synchronization_tick(self) -> None:
        if not self.playing:
            return
        self._update_global_from_clock()
        if (self.direction < 0 and self.global_position_ms <= 0) or (self.direction > 0 and self.global_duration_ms and self.global_position_ms >= self.global_duration_ms):
            self._pause()
            return
        if self.direction < 0:
            now = time.monotonic()
            if now-self.reverse_seek_at >= 0.15:
                self.reverse_seek_at = now
                self._synchronize_players(force=True)
        else:
            self._synchronize_players()
        self._show_position()

    def _synchronize_players(self, force: bool = False) -> None:
        for index, player in enumerate(self.players):
            if not self.info[index].path:
                continue
            target = self.global_position_ms - self.delays_ms[index]
            duration = self.info[index].duration_ms or player.duration()
            active = target >= 0 and (duration <= 0 or target < duration)

            if not active:
                boundary = 0 if target < 0 else max(0, duration)
                if abs(player.position() - boundary) > DRIFT_CORRECTION_MS or force:
                    player.setPosition(boundary)
                player.pause()
                continue

            now = time.monotonic()
            error = abs(player.position()-target)
            if (force and error > 1) or (not force and error > DRIFT_CORRECTION_MS and now-self.last_seek_at[index] > 1.0):
                player.setPosition(max(0, target))
                self.last_seek_at[index] = now
            if self.playing and self.direction > 0 and player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                player.play()

    def _seek_global(self, position: int) -> None:
        resume = self.playing or self.pending_play
        self._pause()
        self.global_position_ms = max(0, min(int(position), self.global_duration_ms or int(position)))
        self._synchronize_players(force=True)
        self._show_position()
        if resume:
            self._play()

    def _begin_seek(self) -> None:
        self.seeking = True

    def _preview_seek(self, position: int) -> None:
        self.current_time_label.setText(format_clock(position))

    def _finish_seek(self) -> None:
        self.seeking = False
        self._seek_global(self.timeline.value())

    def _show_position(self) -> None:
        if not self.seeking:
            self.timeline.setValue(min(self.global_position_ms, self.timeline.maximum()))
        self.current_time_label.setText(format_clock(self.global_position_ms))

    def _step_global_frame(self, direction: int) -> None:
        self._pause()
        fps = self.info[0].fps if self.info[0].path else self.info[1].fps
        frame_ms = max(1, round(1000 / max(0.1, fps)))
        self._seek_global(self.global_position_ms + direction * frame_ms)

    def _nudge_file2(self, direction: int) -> None:
        fps = self.info[1].fps if self.info[1].path else 30.0
        frame_ms = max(1, round(1000 / max(0.1, fps)))
        self.manual_offset.setValue(self.manual_offset.value() + direction * frame_ms)

    def _update_audio_source(self) -> None:
        selection = self.audio_choice.currentIndex() if hasattr(self, "audio_choice") else 0
        if self.direction < 0 and (self.playing or self.pending_play):
            selection = 2
        self.audio_outputs[0].setVolume(1.0 if selection == 0 else 0.0)
        self.audio_outputs[1].setVolume(1.0 if selection == 1 else 0.0)

    def _swap_views(self) -> None:
        old_main_frame = self.canvas.main_video.frame
        old_pip_frame = self.canvas.pip.video.frame
        self.canvas.main_video.center, self.canvas.pip.video.center = self.canvas.pip.video.center, self.canvas.main_video.center
        self.canvas.main_video.clear_frame()
        self.canvas.pip.video.clear_frame()
        self.view_swapped = not self.view_swapped
        for player in self.players:
            player.setVideoSink(None)
        if self.view_swapped:
            self.players[0].setVideoSink(self.canvas.pip.video.item.videoSink())
            self.players[1].setVideoSink(self.canvas.main_video.item.videoSink())
            self.canvas.pip.title.setText("FILE 1 — PiP")
        else:
            self.players[0].setVideoSink(self.canvas.main_video.item.videoSink())
            self.players[1].setVideoSink(self.canvas.pip.video.item.videoSink())
            self.canvas.pip.title.setText("FILE 2 — PiP")
        self.canvas.pip.title.adjustSize()
        self._update_brightness()
        if old_pip_frame is not None:
            self.canvas.main_video._receive_frame(old_pip_frame)
        if old_main_frame is not None:
            self.canvas.pip.video._receive_frame(old_main_frame)

    def _pip_resized_manually(self, width: int) -> None:
        if self.canvas.width() <= 0:
            return
        percent = max(5, min(80, round(width * 100 / self.canvas.width())))
        self.pip_size.blockSignals(True)
        self.pip_size.setValue(percent)
        self.pip_size.blockSignals(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        if getattr(self,'export_process',None) is not None:
            QMessageBox.information(self,'Export running','Finish or cancel the export before closing the player.')
            event.ignore()
            return
        logging.info('Window closing normally')
        for surface in (self.canvas.main_video,self.canvas.pip.video):
            surface.frame_timer.stop()
        self.renderer_timer.stop()
        self.prepare_timer.stop()
        self.sync_timer.stop()
        for player in self.players:
            player.stop()
            player.setSource(QUrl())
        event.accept()


def main() -> int:
    log_dir = Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir())) / 'DualVideoReview'
    log_dir.mkdir(parents=True,exist_ok=True)
    log_path = log_dir/'player.log'
    logging.basicConfig(filename=log_path,level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    # Retained for the life of the event loop; native fatal errors may write here.
    crash_log = open(log_dir/'crash.log','a',encoding='utf-8')
    faulthandler.enable(file=crash_log,all_threads=True)
    logging.info('Starting player; software decode=%s', '--software-decode' in sys.argv)
    print(f'Diagnostic logs: {log_dir}',flush=True)
    def exception_hook(kind,value,tb):
        logging.error('Unhandled Python exception',exc_info=(kind,value,tb))
        sys.__excepthook__(kind,value,tb)
    sys.excepthook = exception_hook
    # Windows chooses the adapter; Qt negotiates video decode independently.
    fmt = QSurfaceFormat()
    fmt.setVersion(3,3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(0)
    fmt.setStencilBufferSize(0)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    app.setApplicationName("Synchronized Dual Video Player")
    window = SynchronizedVideoPlayer()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
