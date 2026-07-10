#include <Adafruit_NeoPixel.h>

#define PIN        6 // The pin your DIN is connected to
#define NUMPIXELS 64 // 8x8 matrix has 64 LEDs

const int ROW_START = 0;
const int ROW_END   = 5;
const int COL_START = 0;
const int COL_END   = 5;


// Create the NeoPixel object
Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  Bridge.begin();
  Bridge.provide_safe("centers", lightCenterSquare);
  pixels.begin();           // Initialize the NeoPixel library
  pixels.setBrightness(150); // Set brightness (0-255). Keep it low to save power.
  pixels.clear();           // Turn off all LEDs initially
  
       // Call our custom function
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