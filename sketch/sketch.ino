#include <Adafruit_NeoPixel.h>
#include <SPI.h>

#define PIN        6 // The pin your DIN is connected to
#define NUMPIXELS 64 // 8x8 matrix has 64 LEDs

const int ROW_START = 0;
const int ROW_END   = 5;
const int COL_START = 0;
const int COL_END   = 5;


// --- Pin Definitions ---
// WARNING: Using a digital pin for VDD is risky. Connect to Arduino's 5V pin for safety.
const int logicVDD_Pin = 10;

// Hardware SPI pins on Arduino Uno/Nano: MOSI is 11, SCK is 13

// --- SPI Configuration ---
const long LENS_SPEED = 77000; // or 500000 for newer lenses
SPISettings lensSettings(LENS_SPEED, MSBFIRST, SPI_MODE3);


// Create the NeoPixel object
Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  Bridge.begin();
  Bridge.provide_safe("centers", lightCenterSquare);
  Bridge.provide_safe("movefocuspos",moveFocusByStepsPos);
  Bridge.provide_safe("movefocusneg",moveFocusByStepsNeg);
  pixels.begin();           // Initialize the NeoPixel library
  pixels.setBrightness(150); // Set brightness (0-255). Keep it low to save power.
  pixels.clear();           // Turn off all LEDs initially  
  pinMode(logicVDD_Pin, OUTPUT);
  digitalWrite(logicVDD_Pin, HIGH);
  delay(100);
  SPI.begin();
  initializeLens();
}

void loop() {
    // Keep loop non-blocking so the bridge can handle messages
}

void lightCenterSquare(int urrentMode) {
  // We want rows 2, 3, 4, and 5
  // We want columns 2, 3, 4, and 5
  
  // Define the specific color (Red, Green, Blue)
  uint32_t colorg = pixels.Color(0, 255, 0); // Cyan-ish blue
  uint32_t colorr = pixels.Color(255, 0, 0);
  //uint32_t color = pixels.Color(255, 180, 255);
  uint32_t color = pixels.Color(255, 180, 255);
  

  if(urrentMode==2) {
    pixels.clear();
    pixels.setPixelColor(31, colorg);
    pixels.setPixelColor(24, colorr);


  }
 
  if(urrentMode==1){
    pixels.clear();
    pixels.setPixelColor(27, color);
    pixels.setPixelColor(28, color);
    pixels.setPixelColor(26, color);
    pixels.setPixelColor(20, color);
    pixels.setPixelColor(18, color);
    pixels.setPixelColor(19, color);
    pixels.setPixelColor(34, color);
    pixels.setPixelColor(35, color);
    pixels.setPixelColor(36, color);
    
    
  }
  
  
  
  pixels.show(); // Send the updated color data to the hardware
}

void initializeLens() {
  
  SPI.beginTransaction(lensSettings);
  byte response1 = SPI.transfer(0xF2);
  delayMicroseconds(10);
  byte response2 = SPI.transfer(0x0A);
  delayMicroseconds(10);
  SPI.endTransaction();
  
}

/**
 * @brief Moves the focus using the correct Two's Complement math for negative steps.
 * @param steps The number of steps to move (must be a positive integer).
 * @param direction The direction: '+' for minimum, '-' for infinity.
 */
void moveFocusByStepsPos(int steps) {
  uint16_t commandValue;

  
  commandValue = steps;
  
  commitsteps(commandValue);

  

    
}

void moveFocusByStepsNeg(int steps) {
  uint16_t commandValue;

 
    
    
  int16_t signedSteps = -steps;
  commandValue = (uint16_t)signedSteps;
  commitsteps(commandValue);

  
     
}  
void commitsteps(uint16_t commandValue){
  byte highByte = (commandValue >> 8) & 0xFF;
  byte lowByte = commandValue & 0xFF;

  SPI.beginTransaction(lensSettings);
  delayMicroseconds(1000);
  byte r1 = SPI.transfer(0x44);
  delayMicroseconds(1000);

  byte r2 = SPI.transfer(highByte);
  delayMicroseconds(1000);

  byte r3 = SPI.transfer(lowByte);
  delayMicroseconds(1000);
  
  SPI.endTransaction();
}