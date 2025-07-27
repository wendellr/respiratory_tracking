import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import scipy.fftpack

# Inicia webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")
    exit()

# Captura frame inicial
ret, old_frame = cap.read()
if not ret:
    print("Error capturing initial frame.")
    cap.release()
    exit()

# === BRIGHTNESS ADJUSTMENT ON INITIAL FRAME ONLY ===
bright_frame = cv2.convertScaleAbs(old_frame, alpha=1.5, beta=60)

old_gray = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2GRAY)

# Manual ROI (abdomen area)
print("Select the region over the abdomen and press ENTER or SPACE.")
r = cv2.selectROI("Select abdomen", bright_frame, showCrosshair=True)
cv2.destroyWindow("Select abdomen")
x, y, w, h = r
roi = old_gray[y:y+h, x:x+w]

# Detect good points
p0 = cv2.goodFeaturesToTrack(roi, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 += np.array([[x, y]])

# Optical Flow params
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

dy_history = deque(maxlen=300)

# Realtime plot
plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot([], [], 'g')
ax.set_ylim(-10, 10)
ax.set_title("Vertical motion detected (breathing)")
ax.set_xlabel("Frame")
ax.set_ylabel("Delta Y")

# Loop
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None and st.sum() > 0:
        dy = np.mean(p1[st == 1][:, 1] - p0[st == 1][:, 1])
        dy_history.append(dy)

        for i, (new, old) in enumerate(zip(p1, p0)):
            if st[i]:
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 1)

    old_gray = frame_gray.copy()
    p0 = p1

    cv2.imshow("Abdomen Tracking", frame)

    if len(dy_history) > 2:
        line.set_ydata(list(dy_history))
        line.set_xdata(range(len(dy_history)))
        ax.set_xlim(0, len(dy_history))
        ax.set_ylim(-max(abs(np.min(dy_history)), abs(np.max(dy_history))) - 1,
                    max(abs(np.min(dy_history)), abs(np.max(dy_history))) + 1)
        fig.canvas.draw()
        fig.canvas.flush_events()


    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Ending
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()

# FFT and respiratory rate estimation
fs = 30  # fps from camera features
dy_array = np.array(dy_history)
dy_centered = dy_array - np.mean(dy_array)

n = len(dy_centered)
frequencies = np.fft.rfftfreq(n, d=1/fs)
fft_magnitude = np.abs(np.fft.rfft(dy_centered))

freq_mask = (frequencies > 0.1) & (frequencies < 1.0)
frequencies_filtered = frequencies[freq_mask]
fft_filtered = fft_magnitude[freq_mask]

if len(fft_filtered) > 0:
    peak_index = np.argmax(fft_filtered)
    peak_freq_hz = frequencies_filtered[peak_index]
    breathing_rate_bpm = peak_freq_hz * 60

 
    print(f"Estimated respiratory rate: {breathing_rate_bpm:.1f} breaths per minute")

    plt.figure(figsize=(10, 4))
    plt.plot(frequencies_filtered * 60, fft_filtered, color='purple')
    plt.xlabel("Frequency (rpm)")
    plt.ylabel("Magnitude")
    plt.title(f"Frequency Spectrum - Breathing ({breathing_rate_bpm:.1f} rpm)")
    plt.grid(True)

    peak_rpm = peak_freq_hz * 60
    peak_mag = fft_filtered[peak_index]
    plt.annotate(f'{peak_rpm:.1f} rpm',
                 xy=(peak_rpm, peak_mag),
                 xytext=(peak_rpm + 2, peak_mag),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12,
                 color='blue')
    plt.show()
else:
    print("Could not estimate respiratory rate.")
