// GY-61 (ADXL335) + LPY503AL Gyroscope Data Acquisition Firmware
// 200 Hz sampling with calibrated sensors and complementary filter
// Pure binary protocol (no ASCII during streaming)

#include <Arduino.h>
#include <avr/interrupt.h>
#include <math.h>

// -------------------- Pins --------------------
const uint8_t PIN_ACCEL_X = A0;
const uint8_t PIN_ACCEL_Y = A1;
const uint8_t PIN_ACCEL_Z = A2;
const uint8_t PIN_GYRO_PITCH = A4;  // Gyro X (pitch)
const uint8_t PIN_GYRO_YAW   = A5;  // Gyro Y (yaw)

// -------------------- ADC --------------------
const float VREF = 3.3;  // FIXED: Was 5.0V, should be 3.3V
const float ADC_RES = 1023.0;
const float LSB_V = VREF / ADC_RES;  // 0.00322 V/LSB

// -------------------- Gyro calibration --------------------
const float GYRO_SENS_V_PER_DPS = 0.0133;  // 13.3 mV/(°/s)
const float GYRO_SENS_COUNTS_PER_DPS = GYRO_SENS_V_PER_DPS / LSB_V;  // ≈ 4.13

// -------------------- Accel calibration (measured) --------------------
const float ACCEL_OFFSET_X_V = 1.096;
const float ACCEL_OFFSET_Y_V = 1.093;
const float ACCEL_OFFSET_Z_V = 1.106;
const float ACCEL_SENS_X_V_PER_G = 0.2207;
const float ACCEL_SENS_Y_V_PER_G = 0.2236;
const float ACCEL_SENS_Z_V_PER_G = 0.2227;

// -------------------- Protocol --------------------
const uint32_t MAGIC_DATA = 0xA1B2C3D4; // 54B frame
const uint32_t MAGIC_SYNC = 0xA5B6C7D8; // 16B sync ack
const uint8_t  SYNC_REQ   = 0x55;       // host->device (1B + 8B timestamp)

const uint16_t SAMPLE_RATE_HZ = 200;
const float    SAMPLE_PERIOD_S = 0.005f;  // 1/200 Hz = 5ms

// Complementary filter
const float ALPHA = 0.98f;
const float RECALIB_ALPHA = 0.001f;
const float STATIONARY_THRESHOLD = 0.3f;  // deg/s

// -------------------- State --------------------
volatile uint32_t g_seq = 0;
volatile bool     g_new_sample = false;

volatile int16_t g_ax_raw = 0, g_ay_raw = 0, g_az_raw = 0;
volatile int16_t g_gp_raw = 0, g_gy_raw = 0;

volatile float g_ax_g = 0.0f, g_ay_g = 0.0f, g_az_g = 0.0f;
volatile float g_pitch_rate = 0.0f, g_yaw_rate = 0.0f;
volatile float g_pitch_filtered = 0.0f, g_roll_filtered = 0.0f;

volatile float g_zero_pitch = 0.0f, g_zero_yaw = 0.0f;

// FIXED: Need to preserve gyro integration state
volatile float g_pitch_gyro = 0.0f;
volatile float g_roll_gyro  = 0.0f;

volatile uint64_t g_tick_us = 0;

// -------------------- Frames --------------------
struct __attribute__((packed)) Frame {
  uint32_t magic;           // 4
  uint32_t seq;             // 4
  uint64_t tick_us;         // 8
  int16_t ax_raw;           // 2
  int16_t ay_raw;           // 2
  int16_t az_raw;           // 2
  int16_t gp_raw;           // 2
  int16_t gy_raw;           // 2
  float   ax_g;             // 4
  float   ay_g;             // 4
  float   az_g;             // 4
  float   pitch_rate;       // 4
  float   yaw_rate;         // 4
  float   pitch_filtered;   // 4
  float   roll_filtered;    // 4
}; // 54 bytes


// -------------------- Gyro calibration --------------------
void calibrateGyro(uint16_t samples = 1000) {
  Serial.println(F("# Calibrating gyro... keep sensor stationary"));

  double sum_pitch = 0, sum_yaw = 0;
  for (uint16_t i = 0; i < samples; i++) {
    sum_pitch += analogRead(PIN_GYRO_PITCH);
    sum_yaw   += analogRead(PIN_GYRO_YAW);
    delayMicroseconds(500);
  }
  g_zero_pitch = sum_pitch / samples;
  g_zero_yaw   = sum_yaw / samples;
}

// -------------------- Timer ISR: 200 Hz --------------------
ISR(TIMER1_COMPA_vect) {
  // Read sensors FIRST for consistent timing
  int raw_pitch = analogRead(PIN_GYRO_PITCH);
  int raw_yaw   = analogRead(PIN_GYRO_YAW);
  int raw_x     = analogRead(PIN_ACCEL_X);
  int raw_y     = analogRead(PIN_ACCEL_Y);
  int raw_z     = analogRead(PIN_ACCEL_Z);

  uint64_t tick = micros();


  // Gyro in deg/s
  float local_pitch_rate = (raw_pitch - g_zero_pitch) / GYRO_SENS_COUNTS_PER_DPS;
  float local_yaw_rate   = (raw_yaw   - g_zero_yaw)   / GYRO_SENS_COUNTS_PER_DPS;

  // Slow zero-rate recalibration if stationary
  if (fabs(local_pitch_rate) < STATIONARY_THRESHOLD) {
    g_zero_pitch = (1.0f - RECALIB_ALPHA) * g_zero_pitch + RECALIB_ALPHA * raw_pitch;
  }
  if (fabs(local_yaw_rate) < STATIONARY_THRESHOLD) {
    g_zero_yaw = (1.0f - RECALIB_ALPHA) * g_zero_yaw + RECALIB_ALPHA * raw_yaw;
  }

  // Integrate gyro angles
  g_pitch_gyro += local_pitch_rate * SAMPLE_PERIOD_S;
  g_roll_gyro  += local_yaw_rate   * SAMPLE_PERIOD_S;

  // Accelerometer volts
  float vx = raw_x * LSB_V;
  float vy = raw_y * LSB_V;
  float vz = raw_z * LSB_V;

  // Convert to g
  float local_ax_g = (vx - ACCEL_OFFSET_X_V) / ACCEL_SENS_X_V_PER_G;
  float local_ay_g = (vy - ACCEL_OFFSET_Y_V) / ACCEL_SENS_Y_V_PER_G;
  float local_az_g = (vz - ACCEL_OFFSET_Z_V) / ACCEL_SENS_Z_V_PER_G;

  // Angles from accel (deg)
  float acc_pitch = atan2(local_ay_g, sqrtf(local_ax_g*local_ax_g + local_az_g*local_az_g)) * 180.0f / PI;
  float acc_roll  = atan2(-local_ax_g, local_az_g) * 180.0f / PI;

  // FIXED: Complementary filter - use existing filtered values, not zero-initialized ones
  float local_pitch_filtered = ALPHA * (g_pitch_filtered + local_pitch_rate * SAMPLE_PERIOD_S)
                             + (1.0f - ALPHA) * acc_pitch;
  float local_roll_filtered  = ALPHA * (g_roll_filtered  + local_yaw_rate   * SAMPLE_PERIOD_S)
                             + (1.0f - ALPHA) * acc_roll;

  // Store atomically
  g_ax_raw = raw_x; g_ay_raw = raw_y; g_az_raw = raw_z;
  g_gp_raw = raw_pitch; g_gy_raw = raw_yaw;

  g_ax_g = local_ax_g; g_ay_g = local_ay_g; g_az_g = local_az_g;
  g_pitch_rate = local_pitch_rate; g_yaw_rate = local_yaw_rate;
  g_pitch_filtered = local_pitch_filtered; g_roll_filtered = local_roll_filtered;

  g_tick_us = tick;
  g_new_sample = true;
  g_seq++;
}

// -------------------- Timer setup --------------------
void setupTimer1() {
  noInterrupts();
  TCCR1A = 0; TCCR1B = 0; TCNT1 = 0;

  // 16MHz / (64 * 1250) = 200 Hz
  OCR1A = 1249;
  TCCR1B |= (1 << WGM12);
  TCCR1B |= (1 << CS11) | (1 << CS10);  // prescaler 64
  TIMSK1 |= (1 << OCIE1A);
  interrupts();
}

// -------------------- Setup --------------------
void setup() {
  Serial.begin(460800);
  while (!Serial && millis() < 3000) {}

  Serial.println(F("# GY-61 (ADXL335) + LPY503AL Acquisition System"));
  Serial.println(F("# Calibrated sensors with complementary filter"));
  Serial.println(F("# Sample rate: 200 Hz"));
  Serial.println(F("# VREF: 3.3V"));

  // If using internal 3.3V regulator on Arduino, use DEFAULT
  analogReference(DEFAULT);  // Change to EXTERNAL if 3.3V connected to AREF

  // ADC call
  for (int i = 0; i < 10; i++) {
    analogRead(PIN_ACCEL_X); analogRead(PIN_ACCEL_Y); analogRead(PIN_ACCEL_Z);
    analogRead(PIN_GYRO_PITCH); analogRead(PIN_GYRO_YAW);
    delay(10);
  }

  calibrateGyro(1000);

  setupTimer1();

  Serial.println(F("# Streaming data"));
  Serial.flush();
}

// -------------------- Loop --------------------
void loop() {


  if (g_new_sample) {
    Frame frame;
    noInterrupts();
    frame.magic = MAGIC_DATA;
    frame.seq   = g_seq - 1;
    frame.tick_us = g_tick_us;
    frame.ax_raw = g_ax_raw; frame.ay_raw = g_ay_raw; frame.az_raw = g_az_raw;
    frame.gp_raw = g_gp_raw; frame.gy_raw = g_gy_raw;
    frame.ax_g = g_ax_g; frame.ay_g = g_ay_g; frame.az_g = g_az_g;
    frame.pitch_rate = g_pitch_rate; frame.yaw_rate = g_yaw_rate;
    frame.pitch_filtered = g_pitch_filtered; frame.roll_filtered = g_roll_filtered;
    g_new_sample = false;
    interrupts();

    Serial.write((uint8_t*)&frame, sizeof(frame));
  }
}