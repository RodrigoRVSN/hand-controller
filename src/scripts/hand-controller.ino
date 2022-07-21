//parte do shield
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Firebase Connection
#include <ESP8266WiFi.h>
#include <FirebaseArduino.h>

#define WIFI_SSID "WIFI ALUNOS"
#define WIFI_PASSWORD "200897"
#define FIREBASE_HOST "https://myhand-ff333-default-rtdb.firebaseio.com/"
#define FIREBASE_AUTH "uOuUtvTjq9B8YfWdX3ZhCoWoMguUIPfm9wZWYKVu"
// Firebase Connection

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define mindinho 7
#define anelar 3
#define meio 2
#define indicador 1
#define polegar 0

//Defines para funcionamento do shield
#define SERVOMIN  100
#define SERVOMAX  555
#define Frequency 50

//protótipos das funções de controle dos dedos
// void writeServo(int position, int tempo);
// void beginServos();

//parte do shield
// #include<Servo.h>

#define numOfValuesReceived 5
#define digitsPerValuesReceived 1

int valuesReceived[numOfValuesReceived];
//$00000
int stringLength = numOfValuesReceived * digitsPerValuesReceived + 1;
int counter = 0;
bool counterStart = false;
String receivedString;

void setup() {
  Serial.begin(9600);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Conectando ao wifi");

  while (WiFi.status() != WL_CONNECTED)
  {
    Serial.print(".");
    delay(300);
  }

  Serial.println();
  Firebase.begin(FIREBASE_HOST, FIREBASE_AUTH);

  // beginServos();
}

void receiveData(){
    // TODO: Use this consts to manipulate the robot hand
    float thumb_finger = Firebase.getFloat("/mao/polegar");
    float indicator_finger = Firebase.getFloat("/mao/indicador");
    float mid_finger = Firebase.getFloat("/mao/meio");
    float ring_finger = Firebase.getFloat("/mao/anelar");
    float pinky_finger = Firebase.getFloat("/mao/mindinho");

    // change serial to data via firebase
    while(Serial.available()){
        char serialContent = Serial.read();
        if(serialContent == '$') {
          counterStart = true;
        }
        if(counterStart){
          if(counter < stringLength){
            receivedString = String(receivedString + serialContent);
            counter++;
          }
          if(counter >= stringLength){
            for(int i = 0; i < numOfValuesReceived; i++) {  
              int num = (i * digitsPerValuesReceived) + 1;  
              valuesReceived[i] = receivedString.substring(num, num + digitsPerValuesReceived).toInt();
            }
            receivedString="";
            counter = 0;
            counterStart = false;
          }
        }
    }
}

void loop() {
  //receiveData();
  //Serial.println(' ');
  /*
  if(valuesReceived[0] == 0){writeServo(polegar, 120);}else{writeServo(polegar,0);} ok
  if(valuesReceived[1] == 0){writeServo(indicador, 140);}else{writeServo(indicador,0);} ~off
  if(valuesReceived[2] == 0){writeServo(meio, 140);}else{writeServo(meio,0);} ok
  if(valuesReceived[3] == 0){writeServo(anelar, 120);}else{writeServo(anelar,0);} ok
  if(valuesReceived[4] == 0){writeServo(mindinho, 120);}else{writeServo(mindinho,0);} ~off
  delay(400);
  */
  for(int position = 0; position < 130; position++){
    // writeServo(mindinho, position);
    delay(10);  
  }
}

void writeServo(int servoPort, int position){
  int servoPosition = map(position, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(servoPort, 0, servoPosition);
  delay(50);
}

void beginServos() {
  pwm.begin();
  pwm.setPWMFreq(Frequency);
}