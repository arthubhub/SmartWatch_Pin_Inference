# LY503AL Gyroscope Hardware Guide

## Overview

The **LY503AL** is a 2-axis MEMS gyroscope that measures angular rate (rotation speed) around the X and Z axes. It provides analog voltage outputs proportional to rotation rate.

## Key Specifications

- **Axes**: X (roll) and Z (yaw) only - **no Y-axis (pitch)**
- **Supply voltage**: 3.3V
- **Output type**: Analog voltage
- **Range**: ±300 deg/s (typical, for 4x amplified output)
- **Sensitivity (4x amp)**: 13.2 mV/(deg/s) typical at 3.3V
- **Zero-rate output**: ~1.65V (VCC/2)
- **Bandwidth**: ~140 Hz typical

## Pin Configuration

```
LY503AL Pin      Function                Connection
────────────────────────────────────────────────────
VCC              Power supply            → Arduino 3.3V
GND              Ground                  → Arduino GND
X_OUT            1x amplified X output   → (not used)
Z_OUT            1x amplified Z output   → (not used)
X_OUT_4X         4x amplified X output   → Arduino A3
Z_OUT_4X         4x amplified Z output   → Arduino A4
ST               Self-test               → (leave floating)
PD               Power down              → (leave floating or VCC)
```

## Why Use 4x Amplified Outputs?

The LY503AL provides both **1x** and **4x** amplified outputs for each axis:

### 1x Outputs
- **Range**: ±1200 deg/s
- **Sensitivity**: 3.3 mV/(deg/s)
- **Resolution**: Lower (0.32 deg/s per ADC count with 10-bit ADC)

### 4x Outputs (Recommended)
- **Range**: ±300 deg/s
- **Sensitivity**: 13.2 mV/(deg/s)
- **Resolution**: Higher (0.08 deg/s per ADC count with 10-bit ADC)

**For PIN entry wrist motion**, angular rates rarely exceed ±200 deg/s, making the 4x outputs ideal for better resolution without range concerns.

## Physical Installation

### Orientation Conventions

When mounting the sensor:

```
      ┌──────────┐
      │  LY503AL │
      │          │
   X  │    ●     │  (● = dot/marking on chip)
   ↑  │          │
   │  │          │
   │  └──────────┘
   └─────→ Z
```

- **X-axis (roll)**: Rotation around the X-axis (pointing forward from wrist)
- **Z-axis (yaw)**: Rotation around the Z-axis (pointing up from wrist)
- **Missing Y-axis (pitch)**: Would be rotation around Y-axis (left-right across wrist)

### Recommended Mounting

**For wrist-worn sensing**:
1. Mount sensor on top of wrist (watch position)
2. Align X-axis pointing toward fingers
3. Z-axis points upward (perpendicular to wrist surface)
4. Secure with velcro strap or elastic band

**Wiring tips**:
- Use short wires (< 15 cm) to minimize noise pickup
- Twist signal wires together with ground wire
- Keep away from power supply lines

## Anti-Aliasing Filters

Add RC low-pass filters on both gyro outputs:

```
LY503AL X_OUT_4X ──┬──[ 10kΩ ]──┬── Arduino A3
                   │             │
                   ·           [470nF]
                               │
                               └── GND

LY503AL Z_OUT_4X ──┬──[ 10kΩ ]──┬── Arduino A4
                   │             │
                   ·           [470nF]
                               │
                               └── GND
```

- **Cutoff frequency**: 34 Hz
- **Purpose**: Remove high-frequency noise above 100 Hz that would alias at 200 Hz sampling

## Power Supply Considerations

1. **Use clean 3.3V source**: Connect to Arduino 3.3V regulator output
2. **Add decoupling capacitors**:
   - 100nF ceramic capacitor between VCC and GND (close to sensor)
   - Optional: 10µF electrolytic in parallel for additional filtering
3. **Common ground**: Ensure Arduino GND and sensor GND are connected

## Calibration Requirements

### Zero-Rate Offset Calibration

The gyroscope outputs ~1.65V when stationary, but actual zero-rate voltage varies with:
- Temperature
- Individual sensor variation
- Power supply noise

**Procedure** (at start of each session):
1. Place sensor on stable surface
2. Keep completely stationary for 10-30 seconds
3. Record average ADC values for both axes
4. These are your zero-rate offsets (biases)

```python
# Example calibration
stationary_data = record_stationary_period(duration=30)  # seconds
gx_bias = np.mean(stationary_data['gx_raw'])
gz_bias = np.mean(stationary_data['gz_raw'])

# Apply to all subsequent readings
gx_corrected = gx_raw - gx_bias
gz_corrected = gz_raw - gz_bias
```

### Temperature Drift

The LY503AL exhibits temperature drift:
- **Typical**: ±0.03 deg/s/°C

**Mitigation**:
- Re-calibrate at start of each session
- Allow sensor to warm up (2-3 minutes) before calibration
- Maintain consistent room temperature

## Converting ADC to Physical Units

```python
import numpy as np

# Constants
V_REF = 3300  # mV (ADC reference voltage)
ADC_RESOLUTION = 1024  # 10-bit ADC
SENSITIVITY_4X = 13.2  # mV/(deg/s) for 4x amplified output

def adc_to_angular_rate(adc_raw, zero_rate_bias):
    """
    Convert raw ADC count to angular rate (deg/s)
    
    Parameters:
    - adc_raw: Raw ADC reading (0-1023)
    - zero_rate_bias: Zero-rate ADC offset from calibration
    
    Returns:
    - Angular rate in degrees per second
    """
    # Subtract zero-rate offset
    adc_corrected = adc_raw - zero_rate_bias
    
    # Convert to voltage (mV)
    voltage_mv = (adc_corrected / ADC_RESOLUTION) * V_REF
    
    # Convert to angular rate (deg/s)
    rate_dps = voltage_mv / SENSITIVITY_4X
    
    return rate_dps

# Example usage
gx_raw = 520  # Example ADC reading
gx_bias = 512  # From calibration
gx_rate = adc_to_angular_rate(gx_raw, gx_bias)
print(f"Gyro X rate: {gx_rate:.2f} deg/s")

# Convert to rad/s if needed
gx_rate_rad = gx_rate * (np.pi / 180.0)
print(f"Gyro X rate: {gx_rate_rad:.4f} rad/s")
```

## Expected Motion Ranges

**During typical PIN entry**:

| Motion Type           | X-axis (Roll)    | Z-axis (Yaw)     |
|----------------------|------------------|------------------|
| Resting              | ±5 deg/s         | ±5 deg/s         |
| Reaching for key     | ±50 deg/s        | ±100 deg/s       |
| Fast digit entry     | ±150 deg/s       | ±200 deg/s       |
| Maximum expected     | ±250 deg/s       | ±250 deg/s       |

The 4x amplified output (±300 deg/s range) comfortably covers these motions while providing good resolution.

## Noise Characteristics

**Typical noise levels** (4x output):
- **White noise**: ~0.05 deg/s/√Hz
- **At 200 Hz sampling**: ~0.7 deg/s RMS

**Noise reduction strategies**:
1. RC low-pass filtering (34 Hz cutoff)
2. Digital filtering in post-processing
3. Averaging over multiple samples

## Troubleshooting

### High Noise Levels
- Check power supply quality (add decoupling capacitors)
- Verify RC filter components are correct values
- Keep wires away from switching power supplies
- Ensure good ground connection

### Drift During Session
- Re-calibrate zero-rate offset
- Allow longer warm-up period
- Check for temperature changes in environment
- Verify sensor is securely mounted (vibration causes drift)

### Saturated Outputs
- If readings consistently at ADC limits (0 or 1023):
  - Motion may exceed ±300 deg/s range
  - Consider using 1x outputs instead (wider range)
  - Check for electrical noise coupling

### Inverted Readings
- If rotation direction is opposite of expected:
  - This is normal - rotation direction depends on mounting
  - Simply negate the values in software
  - Or swap sensor orientation

## Integration with Accelerometer

**Combined 5-DOF IMU** (3-axis accel + 2-axis gyro):

**What you measure**:
- Linear acceleration (3 axes): ax, ay, az
- Angular rate (2 axes): ωx (roll), ωz (yaw)

**Missing measurement**:
- Angular rate around Y-axis (pitch): ωy

**Why this still works for PIN entry**:
- Most significant wrist rotations during typing are roll (X) and yaw (Z)
- Pitch (Y) contributes less to distinguishing digit patterns
- Combined accel + 2-axis gyro provides rich motion signature

**Feature extraction** (for later ML):
```python
# Magnitude features
accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
gyro_mag = np.sqrt(gx**2 + gz**2)

# Combined features
total_motion = accel_mag + k * gyro_mag  # k = weighting factor

# Jerk (rate of acceleration change)
jerk_mag = np.gradient(accel_mag, dt)

# Angular acceleration
alpha_x = np.gradient(gx, dt)
alpha_z = np.gradient(gz, dt)
```

## Safety Notes

1. **AREF Connection**: When using EXTERNAL reference on Arduino:
   - Connect AREF to 3.3V through 100nF capacitor to GND
   - **Never** connect AREF while using internal reference
   - Check your specific Arduino board documentation

2. **ESD Protection**: MEMS gyroscopes are sensitive to static discharge:
   - Handle by edges only
   - Use ESD wrist strap when handling bare sensors
   - Mount in protective enclosure for field use

3. **Mechanical Shock**: Avoid dropping or striking the sensor:
   - Can cause permanent calibration shift
   - May damage internal MEMS structure
   - Use shock-absorbing mounting if needed

## References

- **Datasheet**: ST LY503AL/ALH Dual-axis Yaw-rate Gyroscope
- **Application Note**: ST AN3393 - Understanding Gyroscope Sensitivity
- **Calibration Guide**: ST AN3192 - Using LSM303DLH for a tilt compensated electronic compass

## Next Steps

After hardware setup:
1. Upload Arduino firmware with gyro support
2. Verify gyro readings are reasonable (near zero when stationary)
3. Perform zero-rate calibration
4. Test with hand rotations to verify response
5. Proceed with data collection
