CropDoc + StreetCare

I built this because two problems bothered me and no one had solved them properly together.

India loses ₹90,000 crore every year to crop diseases. Most of that is preventable — if farmers catch the problem early enough. But 70 million smallholder farmers have no agronomist to call, no expert nearby, and often no internet connection either. On the same streets where these farmers sell their produce, there are 62 million stray animals — dogs, cats, cows — that are sick, injured, or malnourished, and the people who want to help them have no idea what to do.

Same problem. Different species. No accessible solution.

So I built one.

---

 What it actually does

You open the app. You point your camera at a diseased leaf or a sick street animal. You get a diagnosis in 30 seconds — in Hindi, or Marathi, or Tamil, or whatever language you speak. With treatment steps. With a feeding guide. With a map of nearby vets. Spoken out loud if you can't read the screen.

No internet required for the core diagnosis. No app store needed for farmers with basic phones — they can just text a photo via SMS and get a reply.

That last part is the piece I'm most proud of. A farmer with a ₹2,000 keypad phone, standing in a field with no WiFi, can text a leaf photo to a number and get a Hindi diagnosis back in under 10 seconds. No one else has built that for Indian agriculture.

---

 The technical reality

**Crop model:** MobileNetV3-Small trained on 54,306 PlantVillage images across 38 disease classes. Two-phase training — head-only first, then fine-tune the top 40 layers. INT8 quantized to 3.5MB so it runs offline on a ₹4,000 Android phone. 87%+ validation accuracy.

One trick that made a real difference: during training I deliberately downscale images to 112×112 and back up to 224×224 to simulate cheap phone camera blur. Without this, the model trained on clean PlantVillage images failed on actual field photos. With it, real-world performance jumped significantly.
*Animal model:** EfficientNetV2B0 across 20 health conditions — dogs, cats, cows, goats. The dataset problem was harder here. No clean labelled animal disease dataset exists for Indian street animals. I built one by combining Oxford Pets + Animals-10 as healthy baselines, then wrote PIL augmentation functions to simulate mange (desaturated patches + red skin tone), injury (red channel wound regions), distemper (green tint around nose), tick infestation (small dark ellipses). 60% accuracy — honest about this. With real field photos it improves dramatically, which is why the community reporting feature exists.

Backend:** FastAPI on Render. Seven endpoints. TFLite inference, gTTS voice output in 5 languages, OpenWeatherMap disease outbreak risk, Google Maps vet locator, Supabase community sightings, Twilio SMS webhook, Claude AI chat for follow-up questions.

App:React Native + Expo SDK 54. Bottom-tab navigation, camera/gallery scanning, Grad-CAM heatmap overlay, confidence bars, AI chat, community map with filters.

---

 Live

Backend API: **https://cropdoc-streetcare.onrender.com**

API docs: **https://cropdoc-streetcare.onrender.com/docs**

> First request takes 30-60 seconds — Render free tier sleeps after inactivity. Subsequent requests are fast.

---

 Stack

| What | How |
|---|---|
| Crop ML | TensorFlow → MobileNetV3Small → TFLite INT8 |
| Animal ML | TensorFlow → EfficientNetV2B0 → TFLite float16 |
| Training | Google Colab T4 GPU |
| Backend | FastAPI + Uvicorn + Python 3.11 |
| Database | Supabase (PostgreSQL) |
| Voice | gTTS — Hindi, Marathi, Tamil, Telugu, English |
| SMS fallback | Twilio MMS webhook |
| Weather | OpenWeatherMap API |
| Maps | Google Places API |
| AI Chat | Claude (claude-sonnet-4-6) |
| Mobile | React Native + Expo SDK 54 |
| Hosting | Render (backend) |
| Tested on | iPhone 14 |

---

 Endpoints

```
GET  /health              → server status
GET  /docs                → interactive API docs
POST /predict/crop        → leaf image → disease diagnosis
POST /predict/animal      → animal image → health assessment
POST /voice               → text + language → MP3 audio
POST /chat                → question + context → AI answer
GET  /sightings           → lat/lon → nearby animal reports
POST /report/animal       → submit animal sighting to community map
POST /webhook/sms         → Twilio calls this with MMS → returns SMS reply
```

---

 Model performance

| Model | Architecture | Accuracy | Size |
|---|---|---|---|
| Crop disease | MobileNetV3-Small INT8 | 87.4% | 3.5 MB |
| Animal health | EfficientNetV2B0 float16 | ~60% (synthetic data) | ~6 MB |

The 60% on animals is real and I don't hide it. The knowledge base in `animal_db.py` gives detailed, accurate first aid regardless of model confidence — so the app is useful even when the classifier is uncertain.

---

Run it locally

```bash
git clone https://github.com/aditiraj0129-netizen/CropDoc-StreetCare
cd CropDoc-StreetCare/backend

# Create env file
cp .env.example .env
# Fill in your API keys

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

For the mobile app:

```bash
cd app
npm install --legacy-peer-deps
npx expo start --clear
# Scan QR with Expo Go (SDK 54)
```

---<img width="384" height="400" alt="cropdoc1" src="https://github.com/user-attachments/assets/070b2737-7d81-4950-b447-bad3a9c54a1c" />

<img width="282" height="400" alt="c2" src="https://github.com/user-attachments/assets/eb55e106-b5fe-4080-82bc-d4d068f44988" />
<img width="272" height="460" alt="c3" src="https://github.com/user-attachments/assets/776ddd3d-79e7-4a34-9f9e-84ab08e13626" />
<img width="274" height="400" alt="c4" src="https://github.com/user-attachments/assets/62f35268-90ef-4032-820f-433a0840bd99" />
<img width="274" height="400" alt="c6" src="https://github.com/user-attachments/assets/c3a051ad-51a3-44e3-9a09-7bd2a426582e" />
<img width="282" height="300" alt="c7" src="https://github.com/user-attachments/assets/5e82501b-b1c4-45a0-a4b6-449a446fc68a" />
<img width="282" height="300" alt="c8" src="https://github.com/user-attachments/assets/555c6396-20f8-4c9b-a94a-b3cd2162af9b" />
<img width="274" height="300" alt="c9" src="https://github.com/user-attachments/assets/242b1433-4cf2-452d-800f-616c4627becf" />













## What I'd build next

Real-world data collection is the most important thing. The community reporting feature exists specifically to gather photos of actual sick animals — once I have 2,000+ real disease images per class, the animal model goes from 60% to 85%+. That's the priority.

After that: YOLOv8 object detection to identify crop/animal species before the disease classifier runs. Right now the user has to select mode manually. With YOLO it becomes fully automatic — point camera, get diagnosis, no tapping required.

Third: ONDC agri layer integration so farmers can order the recommended fungicide directly from the app after diagnosis.

