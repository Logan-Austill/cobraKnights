# LaneVision — JSON image URL playback

1. Put your downloaded JSON metadata in the data folder beside Start Playback.app.
2. Stop any previous app, then double-click Start Playback.app.
3. The player reads the JSON automatically, downloads images from each record's
   url, and displays them in timestamp order. Local JPG files are not required.
4. Double-click Stop Playback.app when finished.

This uses local JSON metadata and direct Blob image downloads, not the company
metadata API. No username/password/token setup is used. Internet access is required
for uncached images, and the image URLs must permit downloading. If a URL requires
credentials or has expired, the image is shown as unavailable, not silently replaced
by a local JPG. Signed query strings are preserved and are not logged or displayed.

Your exported JSON array is supported: siteId, cameraId, name, url, lastModified.
Only HTTPS URLs under siteproicebox.blob.core.windows.net/cameraimages/ are accepted.
Redirects are not followed. The url takes priority even when a matching JPG exists
locally. Records without a url may still reference a local file using image_path.
lastModified is storage modification time, not confirmed capture time. Actual
captured_at/capturedAt/timestamp fields take precedence when provided with a timezone.

The dashboard uses the Mac browser's current date and time. It shows the selected
camera's latest image timestamped at or before now, including when switching sites
or cameras. Future-dated records are withheld until their time arrives. Timestamps
are compared as absolute instants, including their date and timezone; old recordings
are not shifted to today's date. The actual image timestamp remains visible.

Metadata is checked every five seconds. Automatic changes wait at least ten seconds
after the current image loads and then select the newest eligible image, skipping
intermediate frames when necessary. Refresh images immediately fetches the newest
eligible image again, allowing retries after download failures. There is no metadata
API connection: newly available images must be listed in the local JSON files.

Duplicate image/site/camera/time records are
collapsed. Downloaded images are cached in the local runtime folder, capped at
256 MiB; individual images are limited to 10 MiB. No files are uploaded anywhere.

The original site dashboard design is retained. JSON setup details are not shown
in the presentation UI. No false external-API connection indicator is displayed.
Missing lane geometry leaves occupancy and waiting estimates unknown.

Your supplied best.pt is included. Detection loads in the background if compatible
PyTorch/Ultralytics/Pillow libraries are already installed, including the earlier
LaneVision environment. Playback does not install packages or wait for inference.
If those libraries are unavailable, images play and truck counts remain unknown.
Python 3.9+ is needed. The Mac apps support Intel and Apple Silicon and are unsigned;
macOS may require approval. If the adjacent data folder is not visible, a folder
picker asks you to select it. Logs/cache/state are in:

    ~/Library/Application Support/LaneVision Playback

Keep private metadata, downloaded images and weights out of public GitHub repos.
The source ZIP excludes your data and weights. Automated tests mock image downloads;
actual Blob access and model inference still require end-to-end verification.

HTTPS downloads supplement Python system trust with the bundled Mozilla CA roots
from certifi. Certificate and hostname verification remain enabled. Refresh the
CA bundle when maintaining or rebuilding the distribution.

## Camera lane totals and manual wait timer
Edit lane_config.json beside Start Playback.app (outside data). The outer keys are
site IDs; inner keys are camera IDs; values are total lane counts. Replace null
with a whole number, for example: {"37406": {"1493": 4, "1485": 3}}.
Null means not configured. Save the file; changes appear within five seconds.
Lane occupancy uses saved lane polygons and truck detections.

Add 30 min adds thirty minutes and starts the countdown. Stop freezes it; Resume
continues it. Adding while stopped preserves the pause. Reset clears it to zero.
Timers are independent per site/camera and survive image/camera changes, but reset
when the browser page is reloaded or closed. These are manual estimates, not model
predictions. No timer changes when a new truck detection arrives.

## Draw lane polygons
Select a site/camera and wait for its image to load. Choose Configure lanes.
Click corners around the unloading area, enter a unique lane name, and select
Finish lane. Repeat for each lane, then Save lanes. Undo corner removes the last
point; Clear drawing discards the unfinished shape. Redraw replaces a saved lane
only after Finish lane; Delete removes it when you save. Cancel discards edits.
The editor freezes the current image while the dashboard continues running.

Polygons are saved in lane_regions.json beside the app, keyed by site/camera.
Coordinates are normalized to image width/height so resizing the display does not
change their locations. Redraw if the camera moves, zooms, or changes crop.
The existing lane_config.json controls lane totals when set. Otherwise the number
of saved lane polygons supplies the total. These totals must match to classify occupancy.

## Automatic occupancy
Each truck is assigned to at most one lane. Overlap is divided by the area of
that lane within the truck's horizontal span, reducing bias toward wide foreground
lanes. Lower-quarter overlap contributes a 0.4 weighting, and the horizontal span
of lane/box overlap penalizes tiny endpoint intersections. Matches within 10% of
the highest score remain amber/Unknown. This camera-image heuristic supports partial
overlap; it does not measure wheels or recover true ground geometry. Review results
on representative images. Polygons stay unchanged and are reused per site/camera.
Available means at least one free lane; Full means all lanes are occupied.
Missing detection results/configuration or inconsistent
lane totals show Unknown. Red overlays indicate occupied, green free, amber ambiguous.
Occupancy describes the displayed image, not guaranteed present-day conditions if
that image is old. Occlusion and detector misses can cause incorrect free estimates;
check lane placement and results on representative images. Site status is Available if any camera’s latest eligible image is Available. It is Full
only when all cameras report Full; otherwise Unknown. Sidebar numeric counts
come from the camera with the most total lanes, then most occupied lanes to break
a tie. Truck Count and occupied/total lanes use that same camera’s latest eligible
image. Its camera ID is displayed; counts are not summed across overlapping cameras. Updating polygons
refreshes occupancy without rerunning YOLO, within five seconds. Manual wait timers
remain independent of occupancy.

Previous and Next browse only the selected camera's images at or before the Mac's
current date/time. Browsing pauses automatic replacement until Refresh images or a
camera/site change returns to latest mode. Future images are excluded from navigation.
Images from another date display that date to distinguish yesterday from today.
Saved lane polygons are reused for every image from the same site/camera, including
historical images. Camera/site changes never erase configurations.

Sidebar Status text is green for Available and red for Full. Latest images from
all cameras are evaluated in the background; historical browsing does not replace
the latest-image site status. Additional cameras can increase inference work.

## Demo sign-in
The app opens with the SitePro heading, logo, username and password fields.
Click Sign in to enter. Entries are not checked against an account service,
transmitted, or saved by the app; both fields are cleared on entry. This is a
presentation screen, not access control. The dashboard begins loading after
Sign in. Existing local API-key checks remain unchanged.
