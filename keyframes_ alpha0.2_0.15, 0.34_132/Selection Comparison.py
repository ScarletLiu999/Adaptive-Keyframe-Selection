import matplotlib.pyplot as plt

# === Step 1: Read your keyframes from keyframes.txt ===
with open("keyframes.txt", "r") as f:
    lines = f.readlines()

# Clean and extract frame indices
your_keyframes = [
    int(line.strip().replace("frame", "").replace(".jpg", ""))
    for line in lines if line.strip() != ""
]

# === Step 2: Define total frame count and baseline selection ===
total_frames = 1500  # known total number of frames
baseline_keyframes = list(range(0, total_frames, 50))  # fixed interval

# === Step 3: Plot ===
plt.figure(figsize=(14, 3))
plt.title("Keyframe Selection Comparison", fontsize=16)

# All frames as background gray lines
for i in range(total_frames):
    plt.axvline(i, ymin=0.45, ymax=0.55, color='lightgray', linewidth=0.3)

# Plot baseline (fixed-interval) keyframes
plt.scatter(baseline_keyframes, [0.6] * len(baseline_keyframes),
            color='red', label='Fixed-interval baseline', zorder=3)

# Plot your keyframes
plt.scatter(your_keyframes, [0.4] * len(your_keyframes),
            color='blue', label='Submodular selection (ours)', zorder=3)

# === Style ===
plt.yticks([])
plt.xlabel("Frame Index")
plt.ylim(0.2, 0.8)
plt.xlim(-10, total_frames + 10)
plt.legend(loc='upper right')
plt.tight_layout()
plt.grid(False)
plt.show()
