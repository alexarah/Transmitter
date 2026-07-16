#include <SPI.h>
#include <RF24.h>

#define PIN_CE    4
#define PIN_CSN   5
#define PIN_MOSI 23
#define PIN_MISO 19
#define PIN_SCK  18

RF24 radio(PIN_CE, PIN_CSN);

const byte ADDRESS[6] = "WBAN3";

struct PPGPacket {
  int32_t  ppg_raw;
  float    ppg_val;
  uint16_t seq;
};

uint16_t seqTerakhir   = 0;
bool     pertama       = true;
uint32_t totalDiterima = 0;
uint32_t totalHilang   = 0;

#define LOG_INTERVAL 125

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("=============================================");
  Serial.println("  ESP32 PPG WBAN Receiver");
  Serial.println("  nRF24L01 → Serial USB → gabungan_PPG.py");
  Serial.println("=============================================");

  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CSN);

  if (!radio.begin()) {
    Serial.println("[ERROR] nRF24L01 tidak terdeteksi!");
    while (true) delay(1000);
  }

  radio.setPALevel(RF24_PA_LOW);
  radio.setDataRate(RF24_250KBPS);
  radio.setChannel(100);
  radio.setPayloadSize(sizeof(PPGPacket));
  radio.openReadingPipe(0, ADDRESS);
  radio.startListening();

  Serial.print("[OK] nRF24L01 siap sebagai RECEIVER! Payload size = ");
  Serial.print(sizeof(PPGPacket));
  Serial.println(" bytes");
  Serial.println("[..] Menunggu paket PPG dari TX...\n");
}

void loop() {
  if (!radio.available()) return;

  PPGPacket pkt;
  radio.read(&pkt, sizeof(pkt));
  totalDiterima++;

  // Hitung packet loss dari lompatan SEQ
  if (!pertama) {
    uint16_t expected = seqTerakhir + 1;
    if (pkt.seq != expected) {
      totalHilang += (uint16_t)(pkt.seq - expected);
    }
  }
  pertama     = false;
  seqTerakhir = pkt.seq;

  unsigned long ts_rx = millis();

  // Kirim ke Python — format identik dengan thread_rx() di gabungan_PPG.py
  Serial.print("START|PPG:");
  Serial.print(pkt.ppg_raw);
  Serial.print("|SEQ:");
  Serial.print(pkt.seq);
  Serial.print("|TS:");
  Serial.print(ts_rx);
  Serial.println("|END");

}
