# PIN Entry Motion Analysis - Data Collection System

Complete system for collecting synchronized accelerometer and keypad data for PIN entry motion analysis.

## Project Structure

```
project/
├── acquisition/
│   ├── arduino/
│   │   └── firmware_gy61.ino          # Arduino firmware
│   ├── server/
│   │   ├── app.py                      # Flask server
│   │   ├── templates/
│   │   │   └── index.html              # Web keypad UI
│   │   └── requirements.txt
│   └── collector/
│       └── serial_join.py              # Serial data collector
├── data/
│   └── raw/                            # Collected data
├── docs/
│   └── protocol.md                     # Protocol documentation
└── configs/
    └── default.yaml                    # Configuration
```

## Hardware Setup

### GY-61 (ADXL335) + LPY503AL Gyroscope Connections

**Accelerometer (ADXL335 - 3-axis analog, calibrated)**:
```
GY-61 Pin    →    Arduino Pin
─────────────────────────────
VCC (3.3V)   →    3.3V
GND          →    GND
X            →    A0
Y            →    A1
Z            →    A2
```

**Gyroscope (LPY503AL - 2-axis analog, calibrated)**:
```
LPY503AL Pin       →    Arduino Pin
──────────────────────────────────
VCC (3.3V)         →    3.3V
GND                →    GND
Pitch (X output)   →    A4
Yaw (Y output)     →    A5
```

**Note**: The firmware uses **pre-calibrated measurements** from your characterization to provide accurate physical units (g, deg/s, angles).

### ADC Reference Setup

For best accuracy, use external 3.3V reference:
1. Connect Arduino **AREF** pin to **3.3V** through a 100nF capacitor to GND
2. The firmware uses `analogReference(EXTERNAL)`
3. Verify your Arduino board supports EXTERNAL reference before connecting

**Warning**: Some boards may be damaged if AREF is connected incorrectly. Check your board's documentation.

### Anti-Aliasing (Recommended)

Add RC low-pass filters on each sensor output:

**For accelerometer (all 3 axes)**:
- **Resistor**: 10kΩ between sensor output and Arduino analog input
- **Capacitor**: 470nF from Arduino analog input to GND
- **Cutoff frequency**: ~34 Hz

**For gyroscope (X and Z axes)**:
- **Resistor**: 10kΩ between sensor output and Arduino analog input
- **Capacitor**: 470nF from Arduino analog input to GND
- **Cutoff frequency**: ~34 Hz

This suppresses aliasing from noise above 100 Hz.

## Software Setup

### 1. Install Arduino Firmware

```bash
# Open Arduino IDE
# Load: acquisition/arduino/firmware_gy61.ino
# Select your board (e.g., Arduino Uno)
# Select port (e.g., /dev/ttyUSB0 or COM3)
# Upload
```

**Verify upload**:
```bash
# Linux/Mac
screen /dev/ttyUSB0 115200

# Windows - use Arduino Serial Monitor or PuTTY
# You should see: "# GY-61 acquisition ready"
```

### 2. Install Python Dependencies

```bash
cd acquisition/server
pip install -r requirements.txt
```

### 3. Test Serial Connection

```bash
cd acquisition/collector
python serial_join.py --port /dev/ttyUSB0 --subject test --output ../../data/raw
```

You should see:
```
Connected to /dev/ttyUSB0
Arduino: # GY-61 + LY503AL acquisition ready
Arduino: # Sensors: ADXL335 (3-axis accel) + LY503AL (2-axis gyro X/Z, 4x amp)
Collection started: test/session_20241103_143052
Writing to: data/raw/imu_test_session_20241103_143052_143052.parquet
Wrote 1000 frames (seq 0-999)
```

Press `Ctrl+C` to stop.

### 4. Start Web Server

```bash
cd acquisition/server
python app.py
```

Open browser: `http://localhost:5000`

## Usage

### Complete Collection Session

**Terminal 1** - Start serial collector:
```bash
cd acquisition/collector
python serial_join.py --port /dev/ttyUSB0 --subject subject001 --session session001
```

**Terminal 2** - Start web server:
```bash
cd acquisition/server
python app.py
```

**Browser** - Open `http://localhost:5000`:
1. Select **Train Mode** or **Test Mode**
2. Enter **Subject ID**: `subject001`
3. Enter **Session ID**: `session001`
4. Click **Start Session**
5. Enter PINs as prompted

### Data Collection Modes

#### Train Mode
- Shows target PIN to enter
- User enters the displayed PIN
- PIN is stored with sequence for training
- Collect ~20 sequences per session

#### Test Mode
- No target PIN shown
- User enters any PIN
- Shows "Predicted: (feature needs to be added)" placeholder
- True PIN not stored (for later blind testing)

## Data Output

### File Structure

```
data/raw/
├── subject001/
│   └── session001/
│       ├── metadata.yaml              # Session metadata
│       ├── sequences.json             # Sequence info
│       └── events.json                # Keypad events
└── imu_subject001_session001_*.parquet  # IMU data
```

### Parquet Schema

**IMU Data** (`imu_*.parquet`):
```
t_ns              (int64)    - Server-aligned timestamp (nanoseconds)
seq               (int32)    - Frame sequence number
ax_raw            (int16)    - X-axis raw ADC count
ay_raw            (int16)    - Y-axis raw ADC count
az_raw            (int16)    - Z-axis raw ADC count
gp_raw            (int16)    - Gyro pitch raw ADC count
gy_raw            (int16)    - Gyro yaw raw ADC count
ax_g              (float32)  - Calibrated acceleration X (g)
ay_g              (float32)  - Calibrated acceleration Y (g)
az_g              (float32)  - Calibrated acceleration Z (g)
pitch_rate        (float32)  - Angular rate pitch (deg/s)
yaw_rate          (float32)  - Angular rate yaw (deg/s)
pitch_filtered    (float32)  - Complementary filtered pitch angle (deg)
roll_filtered     (float32)  - Complementary filtered roll angle (deg)
tick_us           (int64)    - Arduino microsecond timestamp
subject_id        (string)   - Subject identifier
session_id        (string)   - Session identifier
```

**Key Features**:
- Raw ADC values preserved for debugging
- Calibrated physical units using measured sensor characteristics
- Complementary filter (98% gyro, 2% accel) reduces drift
- Ready for immediate feature extraction

**Events** (`events.json`):
```json
{
  "t_ns": 1699024800000000000,
  "sequence_id": "seq_1699024800_abc123",
  "digit": 5,
  "edge": "down",
  "subject_id": "subject001",
  "session_id": "session001",
  "mode": "train",
  "rtt_ms": 12.3
}
```

**Sequences** (`sequences.json`):
```json
{
  "sequence_id": "seq_1699024800_abc123",
  "mode": "train",
  "pin_string": "1234",
  "subject_id": "subject001",
  "session_id": "session001",
  "trial_idx": 0,
  "keypad_layout": "3x4",
  "t_start_ns": 1699024800000000000,
  "t_end_ns": 1699024805000000000
}
```

## Calibration (Optional but Recommended)

Perform 6-pose calibration at session start:

1. **+X face up**: Hold sensor flat with X-axis pointing up (2-3 seconds)
2. **-X face up**: Flip to opposite side
3. **+Y face up**: Rotate so Y-axis points up
4. **-Y face up**: Flip to opposite side
5. **+Z face up**: Rotate so Z-axis points up
6. **-Z face up**: Flip to opposite side

Mark these periods in your notes for later bias/scale calibration.

## Time Synchronization

The system maintains sub-millisecond time alignment:

1. **Server sends SYNC** every 1-2 seconds
2. **Arduino echoes** with tagged frame
3. **Linear regression** maps Arduino `tick_us` → server `t_ns`
4. **Drift compensation** adjusts for clock rate differences

Check sync quality:
```python
import pyarrow.parquet as pq

# Load IMU data
df = pq.read_table('data/raw/imu_*.parquet').to_pandas()

# Check frame timing
df['dt_ms'] = df['t_ns'].diff() / 1e6
print(f"Mean frame interval: {df['dt_ms'].mean():.2f} ms")
print(f"Std frame interval: {df['dt_ms'].std():.2f} ms")
# Expected: ~5.0 ms ± 0.1 ms for 200 Hz
```

## Troubleshooting

### No data from Arduino
- Check serial port: `ls /dev/tty*` (Linux/Mac) or Device Manager (Windows)
- Verify baud rate: 115200
- Check USB cable (use data cable, not charge-only)

### High timing jitter
- Warning appears: "High sync residual: XX ms"
- Check for USB hub issues (connect directly)
- Reduce system load
- Consider different USB port

### Keypad not responding
- Check browser console for errors (F12)
- Verify Flask server is running
- Check WebSocket connection: look for "Connected to server" message

### Missing frames
- Check serial_join.py output for "Warning: Discarded X bytes"
- Reduce serial baud rate if problems persist
- Check for EMI near USB cable

## Converting ADC to Physical Units

The system now includes **factory-calibrated conversions** based on your measured sensor characteristics.

### Accelerometer Calibration (Pre-configured)

```python
# Your measured calibration values (already in firmware)
ACCEL_OFFSET_X_V = 1.096  # Zero-g voltage offset
ACCEL_OFFSET_Y_V = 1.093
ACCEL_OFFSET_Z_V = 1.106
ACCEL_SENS_X_V_PER_G = 0.2207  # Sensitivity (V/g)
ACCEL_SENS_Y_V_PER_G = 0.2236
ACCEL_SENS_Z_V_PER_G = 0.2227

# Conversion (already done in firmware)
voltage_v = adc_raw * (3.3 / 1023.0)
accel_g = (voltage_v - offset_v) / sensitivity_v_per_g
```

### Gyroscope Calibration (Pre-configured)

```python
# Your measured calibration values (already in firmware)
GYRO_SENS_V_PER_DPS = 0.0133  # 13.3 mV/(deg/s)

# Zero-rate calibration (performed at startup)
# - Samples 1000 readings while stationary
# - Computes average ADC value as zero-rate offset
# - Continuously re-calibrates during stationary periods

# Conversion (already done in firmware)
voltage_v = adc_raw * (3.3 / 1023.0)
rate_dps = (voltage_v - zero_rate_voltage) / GYRO_SENS_V_PER_DPS
```

### Complementary Filter

The system combines gyroscope and accelerometer data to estimate orientation:

```python
# Filter parameters (in firmware)
ALPHA = 0.98  # 98% trust gyroscope, 2% trust accelerometer

# Pitch angle estimation
pitch_filtered = alpha * (pitch_prev + pitch_rate * dt) + (1 - alpha) * pitch_accel

# Roll angle estimation  
roll_filtered = alpha * (roll_prev + yaw_rate * dt) + (1 - alpha) * roll_accel
```

**Benefits**:
- Gyroscope provides smooth, responsive angle tracking
- Accelerometer prevents long-term drift
- Result: Accurate orientation even during dynamic motion

**Note**: All conversions are done on the Arduino and stored in the Parquet files. You can use the calibrated values directly for feature extraction.

## Next Steps

After collecting data:

1. **Analyze alignment**: Check digit events align with IMU peaks
2. **Extract features**: Compute acceleration magnitude, jerk, frequency content
3. **Build windows**: Extract [-300ms, +400ms] around each keypress
4. **Train model**: Classify digit from motion patterns

See planned analysis pipeline in separate documentation.

## Known Limitations

- **2-axis gyro only**: LPY503AL provides pitch and yaw angular rates
  - Roll rate not directly measured (can be estimated from filtered angles)
  - Sufficient for most PIN entry motion patterns
- **Analog sensor noise**: Higher than digital IMUs but mitigated by:
  - Factory calibration with measured offsets and sensitivities
  - RC low-pass filtering
  - Complementary filter for orientation
  - 200 Hz sampling for signal quality
- **ADC resolution**: 10-bit provides ~0.003V resolution
  - With calibrated sensitivity: ~0.015g resolution for accelerometer
  - With calibrated sensitivity: ~0.25 deg/s resolution for gyroscope
  - Sufficient for gross motion pattern recognition

## License

MIT License - feel free to use and modify for your research.

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review protocol documentation in `docs/protocol.md`
3. Verify hardware connections match diagram
