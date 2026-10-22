# FaceGate — loyihani tushunib olish uchun qo'llanma

Bu faylni o'qib chiqsangiz, loyihaning har bir qismini tushunasiz va intervyuda
har bir savolga javob bera olasiz.

---

## 1. Loyiha nima qiladi?

FaceGate — **yuz orqali davomat tizimi**, masalan maktab uchun.

O'quvchi kirish joyidagi kamera oldiga keladi. Tizim:
1. uning yuzini **topadi** (detection),
2. qaysi o'quvchi ekanini **taniydi** (recognition),
3. tirik odam ekanini **tekshiradi**, ya'ni boshini burishni so'raydi (liveness),
4. **bazaga yozadi**: "Ali Valiyev, 7-B, 08:02 da keldi".

O'qituvchi esa bitta buyruq bilan sinfida kim kelgani va kim kelmaganini ko'radi.

Oddiy tizimlardan farqi: **uni aldab bo'lmaydi.** Birovning rasmini telefonda
kameraga ko'rsatsangiz, rasm tekshiruvdan o'tmaydi.

---

## 2. Maktabda qanday ishlatiladi

### 1-bosqich: Ro'yxatga olish (har bir o'quvchi uchun bir marta)

**A) Bitta o'quvchini kamera orqali:**
```bash
python register.py --name "Ali Valiyev" --class 7-B
```
Kamera ochiladi va 15 ta namuna oladi. Shu vaqt ichida boshni biroz qimirlatib turing.

**B) Bitta o'quvchini rasmlardan:**
```bash
python register.py --name "Ali Valiyev" --class 7-B --images rasmlar/ali
```

**C) Butun maktabni bir yo'la (eng qulay yo'l):**

Rasmlarni shunday papkalarga joylang:
```
students/
  7-A/
    Ali Valiyev/        1.jpg  2.jpg  3.jpg
    Madina Karimova/    1.jpg  2.jpg
  7-B/
    Jasur Toshmatov/    rasm.png
```
Keyin bitta buyruq:
```bash
python enroll_folder.py students
```
Har bir o'quvchiga 2-3 ta rasm (turli burchak va yorug'likda) qo'yish bitta rasmdan
yaxshiroq. Rasmi yaroqsiz o'quvchilar ro'yxati oxirida chiqadi. Buyruqni qayta
ishga tushirsangiz, mavjud o'quvchilar yangilanadi, ikki marta qo'shilmaydi.

**O'quvchini o'chirish:**
```bash
python register.py --name "Ali Valiyev" --class 7-B --delete
```

### 2-bosqich: Har kuni ertalab

Kirish joyidagi kompyuterda:
```bash
python app.py
```
Kamera ishlab turadi, o'quvchilar birin-ketin keladi va bosh burish topshirig'ini
bajaradi. Har bir o'quvchi kuniga **bir marta** yoziladi. Qayta kelsa,
"already checked in" chiqadi.

### 3-bosqich: Hisobot

```bash
python report.py                       # bugun, barcha sinflar
python report.py --class 7-B           # faqat 7-B sinfi
python report.py --date 2026-10-21     # boshqa kun
python report.py --class 7-B --csv     # Excel'da ochiladigan fayl ham saqlaydi
python report.py --students            # ro'yxatdagi barcha o'quvchilar
```

Natija shunday ko'rinadi:
```
=== 7-B:  24/26 present ===
   [+] Ali Valiyev               07:58:12
   [+] Jasur Toshmatov           08:01:45
   [-] Madina Karimova           ABSENT
```

---

## 3. "AI o'quvchini bazadan qanday qidiradi?"

Bu eng ko'p beriladigan savol, shuning uchun batafsil:

1. **Ro'yxatga olishda** o'quvchining yuzi **128 ta raqamga** aylantiriladi.
   Bu uning yuzining "barmoq izi", **embedding** deb ataladi. Bazaga rasm emas,
   faqat shu raqamlar saqlanadi.

2. **Har kuni** kameradagi yuz ham 128 ta raqamga aylantiriladi.

3. Bu raqamlar bazadagi **barcha** o'quvchilarning raqamlari bilan **bir vaqtda**
   solishtiriladi. Kodda bu bitta qator (`recognizer.py`):
   ```python
   sims = self.embeddings @ embedding
   ```
   `@` bu matritsa ko'paytmasi. 1000 ta o'quvchi bo'lsa ham millisekundlarda bajariladi.

4. Eng o'xshash o'quvchi tanlanadi. Agar o'xshashlik **0.363** dan yuqori bo'lsa,
   "bu Ali" deydi. Pastroq bo'lsa, "Unknown".

Ya'ni alohida "qidiruv" yo'q: har bir yuz avtomatik ravishda butun baza bilan solishtiriladi.

---

## 4. Fayllar va ularning vazifasi

| Fayl | Nima qiladi |
|---|---|
| `app.py` | Bosh dastur: kamera, tanish, liveness, davomat |
| `register.py` | Bitta o'quvchini ro'yxatga olish yoki o'chirish |
| `enroll_folder.py` | Butun maktabni papkalardan ro'yxatga olish |
| `report.py` | Kim keldi / kim kelmadi hisoboti |
| `database.py` | SQLite baza: o'quvchilar va davomat jadvallari |
| `detector.py` | **YuNet**: yuzni topadi, 5 ta nuqta beradi, yaw'ni hisoblaydi |
| `recognizer.py` | **SFace**: yuzni 128 raqamga aylantiradi va solishtiradi |
| `liveness.py` | "Boshingizni burang" topshirig'i |
| `ui.py` | Ekranga chizish: ramka, blur, yuqoridagi panel |
| `config.py` | Barcha sozlamalar bir joyda |
| `benchmark.py` | Eski Haar va yangi YuNet'ni solishtiradi |
| `download_models.py` | Ikkita tayyor modelni yuklab oladi |

---

## 5. Baza (SQLite)

**SQLite** — bitta fayldan iborat to'liq ma'lumotlar bazasi (`data/facegate.db`).
U Python bilan birga keladi, hech narsa o'rnatish yoki server ishga tushirish shart emas.

Ikkita jadval bor:

**`students`** — o'quvchilar
| id | name | class_name | embedding |
|---|---|---|---|
| 1 | Ali Valiyev | 7-B | (128 ta raqam) |

**`attendance`** — davomat
| student_id | date | time | similarity |
|---|---|---|---|
| 1 | 2026-10-22 | 07:58:12 | 0.84 |

Muhim qoidalar:
- `UNIQUE (name, class_name)`: bitta sinfda bir xil ismli ikki yozuv bo'lmaydi.
  Turli sinflarda bir xil ism bo'lishi mumkin.
- `UNIQUE (student_id, date)`: bir o'quvchi kuniga faqat bir marta yoziladi.
- `ON DELETE CASCADE`: o'quvchi o'chirilsa, uning davomati ham o'chadi.
- "Kim kelmadi" hisoboti `LEFT JOIN` bilan olinadi: barcha o'quvchilar olinadi,
  davomat yozuvi yo'qlari esa "ABSENT" bo'ladi.

Bazani ko'rmoqchi bo'lsangiz, bepul **DB Browser for SQLite** dasturida `data/facegate.db` ni oching.

---

## 6. Asosiy tushunchalar (oddiy tilda)

**Detection va Recognition**
- *Detection*: "rasmda yuz bormi va qayerda?" (YuNet)
- *Recognition*: "bu yuz kimniki?" (SFace)

Eski slaydingizda "facial recognition" deb yozilgan edi, lekin u faqat detection edi.
Endi ikkalasi ham bor.

**Haar cascade va YuNet**
Haar (2001-yil) qo'lda yasalgan oddiy naqshlarni qidiradi.
YuNet esa minglab rasmlarda o'qitilgan neyron tarmoq (CNN). U aniqroq va
**haqiqiy confidence** beradi (0.91 = 91% ishonch). Haar confidence bermaydi.

**Cosine similarity**
Ikki embedding qanchalik o'xshashligini ko'rsatadi: 1.0 = bir xil, 0 = umuman boshqa.

**Nega 15 ta namuna?**
Turli burchak va ifodadagi embedding'larning **o'rtachasi** bitta rasmdan ko'ra ishonchliroq.

**Liveness va yaw**
`yaw = (burun_x − ko'zlar_o'rtasi_x) / ko'zlar_orasidagi_masofa`
- To'g'ri qarasangiz: yaw ≈ 0.
- Boshni bursangiz: burun yonga siljiydi, yaw kattalashadi.
- **Rasm yassi**: uni qanday qiyshaytirmang, burun ko'zlarga nisbatan siljimaydi.
- Yo'nalish (LEFT/RIGHT) tasodifiy, oldindan tayyorlanib bo'lmaydi.
- Har kim boshini biroz qiyshiq tutadi, shuning uchun boshlang'ich yaw'dan farq o'lchanadi.

**State machine (`app.py`)**
Har bir kadrda: yuz yo'q → kutamiz; notanish → "Unknown" va blur;
bugun kelgan → "already checked in"; aks holda → topshiriq beramiz →
bajarsa bazaga yozamiz, bajarmasa 2 soniya kutib qaytadan.

---

## 7. Sozlash

Ekrandagi `yaw` qiymatini kuzating.
- Boshni burganda topshiriq o'tmasa: `config.py` dagi `YAW_TURNED` ni kamaytiring (masalan, 0.25).
- LEFT va RIGHT teskari ishlasa: `liveness.py` dagi `turned = ...` shartida `LEFT` va `RIGHT` ni almashtiring.
- Begona odamni "tanib" qo'ysa: `MATCH_THRESHOLD` ni oshiring (masalan, 0.45).
- O'z o'quvchisini tanimasa: yaxshiroq yorug'likda qayta ro'yxatga oling yoki ko'proq rasm qo'shing.

---

## 8. Haqiqiy maktabda ishlatishdan oldin

- **Ota-onalarning roziligi.** Yuz ma'lumoti biometrik shaxsiy ma'lumot hisoblanadi,
  ayniqsa voyaga yetmaganlar uchun. Ota-onalarning yozma roziligi va maktab
  ma'muriyatining ruxsati kerak. Shaxsiy ma'lumotlar haqidagi qonun talablarini aniqlang.
- **Har yili qayta ro'yxatga olish.** Bolalarning yuzi tez o'zgaradi.
- **Kamerani to'g'ri joylash.** Yuz balandligida, yorug' joyda, orqada derazasiz.
- **Bitta navbat.** Tizim kameraga eng yaqin (eng katta) yuzni tekshiradi,
  shuning uchun o'quvchilar birin-ketin kelishi kerak.
- **Egizaklar.** Tizim ularni adashtirishi mumkin. Bu barcha yuz tanish tizimlarining kamchiligi.

---

## 9. Intervyuda so'rashi mumkin bo'lgan savollar

1. **Detection va recognition farqi?** → 6-bo'lim.
2. **Nega Haar emas, YuNet?** → aniqroq, confidence beradi, burchakka chidamli; benchmark jadvalini ko'rsating.
3. **Bazadan qanday qidiradi?** → embedding'lar matritsasi bilan bitta ko'paytma (3-bo'lim).
4. **Threshold'ni qanday tanladingiz?** → OpenCV SFace uchun 0.363 ni tavsiya qiladi; balandroq = qat'iyroq.
5. **Nega SQLite?** → bitta fayl, server kerak emas, bitta maktab uchun yetarli. Kattalashsa PostgreSQL'ga o'tish oson.
6. **Liveness'ni qanday aldash mumkin?** → odamning boshini burayotgan **videosi** bilan.
   Yechim: chuqurlik (depth) kamerasi yoki o'qitilgan anti-spoofing modeli.
   Kamchiliklarni o'zingiz aytsangiz, bu katta plyus!
7. **Privacy?** → rasmlar saqlanmaydi, faqat embedding'lar; notanish yuzlar blur qilinadi.

---

## 10. Keyingi qadamlar (xohlasangiz)

- Benchmark natijalarini README'ga qo'shish va demo GIF yozish
- Streamlit bilan oddiy web dashboard (o'qituvchi brauzerda hisobotni ko'radi)
- Ko'z yumishni (blink) ham tekshirish (MediaPipe Face Mesh bilan)
