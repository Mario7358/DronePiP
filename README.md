Synchronized Video Player

A Python/PyQt6 application for reviewing two videos on a shared timeline, with a resizable 
picture-in-picture (PiP) view, image adjustments, frame capture, and MP4 export.

When embedded creation or recording timestamps are available, the application uses them to help 
align the videos. Manual offset controls let you verify and correct the alignment.

## Screenshot
<img width="1355" height="1032" alt="image" src="https://github.com/user-attachments/assets/b88bbcbc-8c50-4678-9f7b-efa38844d642" />


Why I Built This

I developed this application to review and synchronize multiple videos, primarily drone footage. 
I wanted to compare recordings in a picture-in-picture view, adjust the image for easier viewing, 
and save frames directly without relying on the Windows Snipping Tool.  This was created to help 
with getting video ready to review for some of the criminal cases I work with.

The application also exports a combined video with the selected image adjustments and PiP layout.

This is my first GitHub repository.

Features

* Two videos played on a shared timeline.
* Timestamp-assisted alignment with manual offset controls.
* Resizable PiP view, initially set to 15% of the video canvas width.
* Playback speeds from 0.25× to 4×.
* Reverse preview.
* Brightness, exposure, contrast, gamma, shadows/highlights, and saturation adjustments.
* Spatial denoise and sharpening.
* Independent digital zoom and pan for each video.
* Frame capture from both videos and the combined PiP view.
* MP4 export with the current layout and saved adjustments.
* Software decoding option for troubleshooting.

Installation

The commands below use Python 3.13 on Windows.

Install the Python dependencies:


py -3.13 -m pip install --upgrade PyQt6 PyOpenGL numpy


For timestamp and frame-rate detection, install FFmpeg and make sure ffprobe.exe 
is available on your PATH.

MP4 export requires both ffmpeg.exe and ffprobe.exe on your PATH.



Run

From the folder containing the script:

py -3.13 synchronized_video_player.py


If hardware decoding is unstable, try:

py -3.13 synchronized_video_player.py --software-decode


Basic Use

1. Load the first and second video.
2. Check the timestamp-based alignment and correct it with the manual offset controls
   if needed.
3. Keep Smooth playback enabled and playback speed at 1× for normal viewing.
4. Disable Smooth playback to preview image adjustments and zoom/pan, preferably
   while paused.
5. Adjust the PiP size and position as needed.
6. Save the current frames or export the combined video.

Playback and Image Adjustments

Smooth Playback

Smooth playback uses QVideoWidget without per-frame Python conversion or the enhancement
renderer.

Image adjustments and zoom/pan settings are retained but bypassed for display while smooth
playback is enabled. Saved settings still apply to MP4 export.

Image Review

Available adjustments include:

* Brightness and exposure
* Contrast and gamma
* Shadows and highlights
* Saturation
* Spatial bilateral denoise
* Unsharp edge enhancement

Original bypasses the filters. Reset image restores neutral filter settings.

Enabling Link image adjustments copies File 1’s adjustments to File 2. View position and zoom remain independent.

These adjustments are SDR display operations. They do not reconstruct missing detail or provide camera-specific log or HDR grading.

Zoom and Pan

Use the mouse wheel to zoom.
Drag to pan.
Inside the PiP view, use Shift + drag to pan.

Zoom is digital.

Reverse Preview

Reverse preview uses repeated seeks rather than a native reverse decoder. It can appear choppy, especially with long-GOP video.

Reverse preview is silent. Audio remains selectable during forward playback.

Save Frames

Save both frames + PiP creates a new subfolder containing:

| File            | Contents                                                         |
| --------------- | ---------------------------------------------------------------- |
| file1_frame.png | Unadjusted, source-resolution decoded RGB frame from File 1      |
| file2_frame.png | Unadjusted, source-resolution decoded RGB frame from File 2      |
| pip_view.png    | Display-resolution composition with the current filters and zoom |

The capture saves the pair currently displayed. It does not guarantee an exact-time frame match.

Original video files are never altered.

Export MP4

Export MP4 renders the entire overlapping portion of the shared timeline forward at the selected playback speed.

Export settings:

* H.264 video, CRF 18
* 30 fps
* 1920 pixels wide, preserving the canvas aspect ratio
* No audio
* Current PiP layout, view swap, image adjustments, zoom, and pan applied

Denoise and sharpening use FFmpeg approximations of the OpenGL preview, so the exported appearance may 
differ slightly.

CPU encoding runs in a background process with progress reporting and cancellation. Playback pauses 
during export to reduce competition for system resources.

A partial export never replaces the selected destination file.

Technical Notes

* Playback codec support depends on the Qt Multimedia backend.
* FFprobe inspects metadata; background probing ignores attached cover images.
* Qt automatically selects available hardware decoders, but the interface does not verify whether
* hardware decoding is active.
* Image adjustments use OpenGL 3.3 shaders. Windows selects the graphics adapter, and the renderer
* name identifies software OpenGL fallback.
* Frame conversion and upload still require CPU processing.
* Forward synchronization follows the master decoder, with rate-limited correction seeks.
* Reverse preview performs at most approximately seven seeks per second.
* Display conversions are capped at approximately 30 per second. Preview frames may be skipped at
* higher playback speeds.

Frame Conversion

The application avoids `QVideoFrame.toImage()`. Incoming Qt video frames are retained while mapped 
pixel buffers are length-checked and copied before release.

Supported conversion formats include packed RGB and common 8-bit planar and semi-planar YUV formats. 
Unsupported formats stop conversion and produce a diagnostic.

YUV conversion uses the declared color range and matrix when available. When unspecified, defaults 
are limited range with BT.709 for HD and BT.601 for SD. Chroma upsampling uses nearest samples.

Timestamp Limitations

An embedded `creation_time` value is not guaranteed to represent the true camera recording time.

Check the alignment against visible events and use the manual offset controls to correct discrepancies.

Development

This project was developed with assistance of AI for coding and troubleshooting. 
The idea and requirements came from my hands-on experience reviewing drone videos and working in 
a digital forensics unit.
