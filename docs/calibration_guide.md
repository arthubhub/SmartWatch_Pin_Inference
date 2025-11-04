# Sensor Calibration Guide

## Overview

This system uses **factory-calibrated** sensors with your measured characteristics already programmed into the firmware. This document explains the calibration values and how to verify or update them if needed.

## Current Calibration Values

### Accelerometer (GY-61 / ADXL335)

These values are based on your measurements and are programmed in the Arduino firmware:

```cpp
// Zero-g voltage offsets (volts)
const float ACCEL_OFFSET_X_V = 1.096;
const float ACCEL_OFFSET_Y_V = 1.093;
const float ACCEL_OFFSET_Z_V = 1.106;

// Sensitivities (volts per g)
const float ACCEL_SENS_X_V_PER_G = 0.2207;
const float ACCEL_SENS_Y_V_PER_G = 0.2236;
const float ACCEL_SENS_Z_V_PER_G = 0.2227;
```

**Interpretation**:
- **Offset**: Voltage output when the axis experiences 0g (perpendicular to gravity)
- **Sensitivity**: Change in voltage per 1g of acceleration
- Your sensor outputs ~1.1V at rest and changes by ~0.22V per g

### Gyroscope (LPY503AL)

```cpp
// Sensitivity (volts per degree/second)
const float GYRO_SENS_V_PER_DPS = 0.0133;  // 13.3 mV/(deg/s)

// Zero-rate offsets (determined at each startup)
// - Measured during 1000-sample calibration
// - Typically around 512 counts (1.65V)
// - Varies with temperature and power supply
```

**Interpretation**:
- **Sensitivity**: Output voltage changes by 13.3mV per deg/s of rotation
- **Zero-rate**: Output voltage when sensor is not rotating (auto-calibrated)

## Automatic Calibration at Startup

### Gyroscope Zero-Rate Calibration

The system automatically calibrates gyroscope zero-rate offsets at every startup:

**Procedure**:
1. Arduino powers up
2. System prints: `"# Calibrating gyroscope... keep sensor stationary"`
3. Takes 1000 samples over ~0.5 seconds
4. Computes average ADC value for each axis
5. Uses these as zero-rate baselines

**User action required**: Keep sensor completely still during this 1-second period.

### Continuous Drift Compensation

The firmware includes **slow recalibration** during operation:

```cpp
const float RECALIB_ALPHA = 0.001;  // Very slow adaptation
const float STATIONARY_THRESHOLD = 0.3;  // deg/s

// If angular rate is very small, slowly adjust zero-rate
if (abs(angular_rate) < STATIONARY_THRESHOLD) {
    zero_rate = (1 - RECALIB_ALPHA) * zero_rate + RECALIB_ALPHA * current_reading;
}
```

This compensates for:
- Temperature drift during session
- Slow power supply variations
- Long-term sensor drift

## When to Re-Calibrate Sensors

### Accelerometer (Rarely Needed)

Your measured calibration values should remain valid unless:

**Physical damage**:
- Sensor is dropped or experiences high impact
- Visible damage to PCB or component

**Manufacturing variation**:
- You replace the GY-61 module with a different one
- Different production batch may have different characteristics

**Temperature extremes**:
- Operating outside 0-70°C range
- ADXL335 has low temperature coefficient (~0.01%/°C)

### Gyroscope (Automatic)

Zero-rate calibration happens automatically at:
- Every system startup
- Continuously during stationary periods

**Manual recalibration needed if**:
- Large temperature change during session (>10°C)
- Suspect electrical noise affecting baseline
- Angular rate readings show constant non-zero bias

## How to Verify Calibration

### Check Accelerometer Calibration

```python
import pyarrow.parquet as pq
import numpy as np

# Load IMU data
df = pq.read_table('data/raw/imu_*.parquet').to_pandas()

# Find stationary period (low acceleration variance)
window = 200  # 1 second at 200 Hz
variances = []
for i in range(len(df) - window):
    chunk = df.iloc[i:i+window]
    var = chunk['ax_g'].var() + chunk['ay_g'].var() + chunk['az_g'].var()
    variances.append(var)

stationary_idx = np.argmin(variances)
stationary_data = df.iloc[stationary_idx:stationary_idx+window]

# Compute acceleration magnitude
ax = stationary_data['ax_g'].mean()
ay = stationary_data['ay_g'].mean()
az = stationary_data['az_g'].mean()
mag = np.sqrt(ax**2 + ay**2 + az**2)

print(f"Stationary acceleration magnitude: {mag:.3f} g")
print(f"Expected: 1.000 g (Earth's gravity)")
print(f"Error: {abs(mag - 1.0) * 100:.1f}%")

# Good calibration: error < 5%
# Acceptable: error < 10%
# Poor: error > 10% - consider recalibration
```

### Check Gyroscope Calibration

```python
# Find stationary period
stationary_data = df.iloc[stationary_idx:stationary_idx+window]

# Check zero-rate bias
pitch_bias = stationary_data['pitch_rate'].mean()
yaw_bias = stationary_data['yaw_rate'].mean()

print(f"Pitch rate bias: {pitch_bias:.3f} deg/s")
print(f"Yaw rate bias: {yaw_bias:.3f} deg/s")

# Good calibration: bias < 0.5 deg/s
# Acceptable: bias < 1.0 deg/s
# Poor: bias > 1.0 deg/s - check for temperature drift or noise
```

### Check Complementary Filter

```python
# During rotation, check if filtered angles are reasonable
rotating_data = df.iloc[5000:5200]  # 1 second of data

print(f"Pitch range: {rotating_data['pitch_filtered'].min():.1f} to {rotating_data['pitch_filtered'].max():.1f} deg")
print(f"Roll range: {rotating_data['roll_filtered'].min():.1f} to {rotating_data['roll_filtered'].max():.1f} deg")

# Typical wrist motion during PIN entry:
# - Pitch: ±45 degrees
# - Roll: ±60 degrees
# If ranges are much larger or show drift, check calibration
```

## How to Update Calibration Values

### Re-Calibrate Accelerometer (Advanced)

If you need to recalibrate your specific sensor:

**Required equipment**:
- Level surface
- Arduino serial monitor
- ~10 minutes

**Procedure**:

1. **Collect 6-pose data** (hold each pose for 3 seconds):
   - +X up: X-axis pointing to sky
   - -X up: X-axis pointing to ground
   - +Y up: Y-axis pointing to sky
   - -Y up: Y-axis pointing to ground
   - +Z up: Z-axis pointing to sky
   - -Z up: Z-axis pointing to ground

2. **Record raw ADC values** for each pose:
   ```python
   # Example: +Z up pose
   pose_data = df[df['seq'].between(1000, 1600)]  # 3 seconds
   ax_mean = pose_data['ax_raw'].mean()
   ay_mean = pose_data['ay_raw'].mean()
   az_mean = pose_data['az_raw'].mean()
   ```

3. **Calculate calibration** (see formulas below)

4. **Update firmware**:
   ```cpp
   // In firmware_gy61.ino, update these values:
   const float ACCEL_OFFSET_X_V = <your_new_value>;
   const float ACCEL_SENS_X_V_PER_G = <your_new_value>;
   // ... repeat for Y and Z
   ```

5. **Re-upload** firmware to Arduino

### Calibration Formulas

**For each axis** (example for X-axis):

```python
# From +X and -X poses
adc_plus_x = 710   # Example: raw ADC when X-axis up
adc_minus_x = 310  # Example: raw ADC when X-axis down

# Convert to voltage
v_plus_x = adc_plus_x * (3.3 / 1023)   # ~2.29V
v_minus_x = adc_minus_x * (3.3 / 1023) # ~1.00V

# Calculate sensitivity (V/g)
# Difference between +1g and -1g is 2g
sensitivity_v_per_g = (v_plus_x - v_minus_x) / 2.0

# Calculate zero-g offset (V)
offset_v = (v_plus_x + v_minus_x) / 2.0

print(f"X-axis sensitivity: {sensitivity_v_per_g:.4f} V/g")
print(f"X-axis offset: {offset_v:.4f} V")
```

Repeat for Y and Z axes.

## Complementary Filter Tuning

The filter weight can be adjusted if needed:

```cpp
const float ALPHA = 0.98;  // Default: 98% gyro, 2% accel
```

**Higher ALPHA (e.g., 0.99)**:
- More responsive to rapid motion
- Less correction from accelerometer
- May drift more during long sessions

**Lower ALPHA (e.g., 0.95)**:
- More correction from accelerometer
- Better long-term stability
- May be sluggish during rapid motion

**For PIN entry** (quick, discrete motions): default 0.98 is optimal.

## Factory Reset

To return to your original measured calibration values:

1. Re-download the original `firmware_gy61.ino` from the artifacts
2. Upload to Arduino
3. Calibration will use the values you provided in your script

## Troubleshooting

### Accelerometer reads wrong magnitude

**Symptom**: Stationary magnitude is 0.8g or 1.2g instead of 1.0g

**Causes**:
- Incorrect calibration values
- Sensor on non-level surface during measurement
- Temperature extreme

**Solution**:
1. Verify AREF is connected to 3.3V
2. Check offset and sensitivity values
3. Re-calibrate if error > 10%

### Gyroscope shows constant drift

**Symptom**: Angular rate shows non-zero value when stationary

**Causes**:
- Insufficient warm-up time
- Temperature change during session
- Electrical noise

**Solution**:
1. Restart Arduino (triggers recalibration)
2. Wait 2-3 minutes for sensor to reach stable temperature
3. Keep sensor very still during startup calibration
4. Check power supply quality (add decoupling capacitors)

### Filtered angles drift over time

**Symptom**: Pitch/roll angles slowly change even when stationary

**Causes**:
- Gyroscope bias not properly calibrated
- Complementary filter weight too high
- Accelerometer calibration incorrect

**Solution**:
1. Verify gyroscope zero-rate calibration
2. Try lower ALPHA value (e.g., 0.96)
3. Check accelerometer magnitude is ~1.0g
4. Increase RECALIB_ALPHA for faster drift compensation

## References

- Your calibration script values (source of truth)
- ADXL335 datasheet: Section on sensitivity and zero-g offset
- LPY503AL datasheet: Section on zero-rate level and sensitivity
- Arduino `analogReference()` documentation for your board
