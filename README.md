# Camera-Calibration


• [Calibration] Estimate a plane-to-plane projectivity for each of the two planes of the
calibration grid using the 3D/2D point pairs picked by the user.
• [Calibration] Use the estimated plane-to-plane projectivities to assign 3D coordinates
to all the detected corners on the calibration grid.
• [Calibration] Estimate a 3×4 camera projection matrix P from all the detected corners
on the calibration grid using linear least squares.
• [Decomposition] Use QR decomposition to decompose the camera projection matrix P
into the product of a 3×3 camera calibration matrix K and a 3×4 matrix [R T] composed
of the rigid body motion of the camera.
• [Decomposition] Normalize the resulting camera calibration matrix K into the standard
form (refer to slide 19 of lecture notes on camera calibration).
• [Epiplar geometry] Estimate an essential matrix from the camera projection matrices of
a pair of camera.