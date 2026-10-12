# Domain Glossary

Canonical vocabulary for the Monocular Object Measurement Engine Prototype. Definitions only; formulas and implementation details live in [PRD.md](PRD.md).

---

### Reference Marker
A printed black-and-white square pattern of exactly known size (5.0 cm per side) that the engine can recognise in the camera image. It is the only source of real-world scale. Also called the *ArUco marker*.

### Marker Card
The piece of paper or card the Reference Marker is printed on, when it stands on its own beside the object rather than being taped to it. The engine must never mistake the Marker Card for the object being measured.

### Measurement Plane
The flat front face of the object that faces the camera. Width and Height are measured on this plane only.

### Co-Planar Rule
The Reference Marker must lie on the Measurement Plane: the same distance from the camera as the object's front face. A marker nearer or farther than the object gives a wrong scale.

### Perpendicular Alignment
The camera looks straight at the Measurement Plane, at a right angle to it, so the face appears undistorted rather than as a trapezoid.

### Scale Factor
How many real-world centimetres one image pixel represents on the Measurement Plane. Derived from the Reference Marker every frame.

### Calibration Status
Whether the engine currently trusts its Scale Factor:
- **Calibrated**: the Reference Marker is visible in this frame.
- **Cached**: the marker was briefly hidden; the last known Scale Factor is still being used, for at most about one second.
- **Uncalibrated**: the marker has been missing too long; no measurement is shown.

### Target Object
The rigid object being measured: the largest valid outline in view that is not the Marker Card.

### Width
The real-world extent of the Target Object's front face along its most horizontal side.

### Height
The real-world extent of the Target Object's front face along the side perpendicular to its Width.

### Smoothing
Blending each new reading with recent ones so the displayed numbers stay steady instead of flickering.

### Object Swap
Replacing the Target Object with a different one. The engine recognises an Object Swap and restarts Smoothing so the old object's size does not bleed into the new reading.

### Guide Box
The rectangle covering the central 70% of the camera view. Measurements are most trustworthy inside it, because lens distortion grows towards the edges.

### Clipped
The Target Object runs off the edge of the camera frame, so its true size cannot be seen.

### Measurement Dashboard
The main window: the live colour view with the object outline, marker highlight, Guide Box and the Width × Height readout.

### Binary Tracking Mask
The diagnostic window: a black-and-white view of the outlines the engine detected, used to check that labels and background clutter are being ignored.

### Snapshot
A saved record of one accepted measurement: a row in the measurements log plus an image of the dashboard at that moment.
