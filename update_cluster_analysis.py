# -*- coding: utf-8 -*-
import json

# Učitaj notebook
with open('prezentacija.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Pronađi ćeliju sa opisom klastera
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and '**Otkriveni šabloni:**' in ''.join(cell['source']):
        # Zameni sadržaj sa detaljnim opisom
        new_content = """**Otkriveni šabloni - Prirodni klasteri pesama:**

K-means algoritam identifikovao je **4 distinktne grupe pesama** na osnovu kombinacije audio karakteristika:

### **Klaster 0 - Metal/Energični (30.2% pesama, popularnost: 34.1)**
**Ključne kombinacije:**
- ↑↑ **Visoka energy (+27%)** + **instrumentalness (+40%)**
- ↓↓ Niska acousticness (-79%), valence (-33%), loudness (-28%)
- ↑ Iznad proseka: tempo (+12%)

**Profil:** Hard/metal žanrovi (grindcore, death-metal, black-metal, metalcore)  
**Karakteristike:** Energične, instrumentalne pesme sa teškim zvukom ali tihijim glasnoćom od očekivanja

---

### **Klaster 1 - Akustični/Tihi (23.4% pesama, popularnost: 34.2 - NAJVIŠA)**
**Ključne kombinacije:**
- ↑↑ **Visoka acousticness (+113%)** + **loudness (+28%)**
- ↓↓ Niska energy (-38%), instrumentalness (-72%)
- ↓ Ispod proseka: valence (-15%), danceability (-5%)

**Profil:** Akustični žanrovi (tango, romance, honky-tonk, jazz, comedy)  
**Karakteristike:** Tihe, akustične pesme sa vokalima  
**Interesantno:** Uprkos niskoj energiji, ovaj klaster ima NAJVEĆU prosečnu popularnost! To pokazuje da akustična kombinacija ima svoj jaki market.

---

### **Klaster 2 - Ambijent/Instrumentalni (7.5% pesama, popularnost: 28.6 - NAJNIŽA)**
**Ključne kombinacije:**
- ↑↑ **Ekstremno visoka instrumentalness (+405%)** + **acousticness (+158%)** + **loudness (+146%)**
- ↓↓ Sve ostalo nisko: energy (-68%), danceability (-35%), valence (-60%)

**Profil:** Ambijentalni žanrovi (sleep, new-age, classical, ambient, piano)  
**Karakteristike:** Veoma tihe, instrumentalne, mirne kompozicije  
**Interesantno:** Najmanji klaster sa najnižom popularnošću - niša tržište

---

### **Klaster 3 - Plesni/Veseli (38.9% pesama, popularnost: 32.9)**
**Ključne kombinacije:**
- ↑↑ **Visoka danceability (+22%)** + **valence (+46%)**
- ↓↓ Niska acousticness (-37%), instrumentalness (-66%), loudness (-23%)
- ≈ Energy iznad proseka (+15%), tempo prosečan

**Profil:** Plesni žanrovi (reggaeton, reggae, latino, dancehall, forró)  
**Karakteristike:** Vesele, plesne pesme sa vokalima  
**Interesantno:** Najveći klaster (39% svih pesama) - mainstream popularna muzika

---

### **Ključni zaključci o kombinacijama:**

1. **Akustične + glasne** pesme (Klaster 1) postižu NAJVIŠU prosečnu popularnost uprkos niskoj energiji
2. **Danceability + valence** (Klaster 3) kombinacija dominira tržištem - 39% svih pesama
3. **Instrumentalne** pesme (Klaster 2) imaju najnižu popularnost - niša
4. **Energy ≠ glasnoća**: Klaster 0 ima visoku energy ali nižu glasnoću (metal žanrovi)

**Zaključak:** Popularnost zavisi od **šablona** (specifične kombinacije karakteristika), ne pojedinačnih atributa. Različiti šabloni mogu biti uspešni - akustični tihi zvuk (Klaster 1) i energični plesni zvuk (Klaster 3) oba dostižu visoku popularnost, ali kroz potpuno različite kombinacije karakteristika!
"""
        
        # Podeli na linije za Jupyter format
        cell['source'] = [line + '\n' for line in new_content.split('\n')]
        if cell['source'] and cell['source'][-1].endswith('\n\n'):
            cell['source'][-1] = cell['source'][-1][:-1]
        
        print(f"Ažurirana ćelija {i}: Detaljan opis klastera")
        break

# Sačuvaj notebook
with open('prezentacija.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Notebook uspešno ažuriran sa detaljnim analizama klastera!")
