Keybinds

Click the Measurement Dashboard window first so it has keyboard focus.

Key	What it does
s	Save a measurement, but only if the save gate passes
n	Switch to the next camera (laptop → USB → ...). Recalibrates after switching
r	Reset the tracker and smoothing (use it when you swap to a new object)
m	Show or hide the binary mask window (for diagnosing detection problems)
q or Esc	Quit

The dashboard window also has four sliders:

Canny Low / Canny High: edge detection sensitivity. Adjust them if the object’s outline isn’t picked up cleanly.
Min Area (px): ignores anything smaller than this, which helps filter noise.
Smoothing x10: higher means the numbers react faster but jitter more. Lower means steadier but slower.
Steps to take a measurement
Print the marker once. Run python -m measure_engine.make_marker, then print marker_id0_5cm.png at 100% scale with no “fit to page”. Measure the black square with a ruler to confirm it’s exactly 5.0 cm. If it isn’t, every measurement will be off by the same ratio.
Start the app with python -m measure_engine. Press n if you want the external webcam.
Place the marker flat in view, in the same plane as the object (for example, both lying on the same table). A cyan outline appears around the marker when it’s detected, and the status changes from uncalibrated to calibrated.
Place the object near the marker, with both fully inside the GUIDE BOX (70%) rectangle and not touching the frame edges. A green box with red corner dots appears around the object.
Hold still for a second so the smoothed width and height settle. Plain, contrasting backgrounds work best.
Wait for LOCKED [S TO SAVE], then press s. You’ll see the banner SNAPSHOT SAVED.
For the next object, swap it in and press r if the numbers look stale or wrong. Repeat from step 4.
If s is refused

The banner will say why:

NO CALIBRATION ANCHOR: the marker isn’t visible and no recent scale is cached. Show the marker.
NO TARGET DETECTED: no object outline was found. Adjust the sliders or the background.
OBJECT CLIPPED: the object touches the frame edge. Move it inward.
OUTSIDE GUIDE BOX: the object or marker is outside the guide rectangle. Move them inside.
Where results go
measurements.csv gets one row per save, with width and height in cm, area, pixel sizes, scale and calibration status.
The captures/ folder gets the annotated screenshot for each save.

The README’s Controls list only mentions s, m and q, so you may want to add n and r there too.
