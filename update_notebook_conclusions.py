# -*- coding: utf-8 -*-
import json

# Učitaj notebook
with open('prezentacija.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Pronađi ćeliju sa zaključcima (sekcija 5)
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and '## 5. Zaključci i diskusija' in ''.join(cell['source']):
        # Zameni sadržaj
        new_content = """## 5. Zaključci i diskusija

### Performanse modela

Finalni modeli (sa KNN Imputation pristupom) pokazuju **solidne performanse**:
- **MAE ~ 9.75**: U proseku, predikcija greši za ~9.75 poena (poboljšanje od 3.63% u odnosu na uklanjanje nula)
- **R² ~ 0.467**: Modeli objašnjavaju ~47% varijanse u popularnosti
- Random Forest i XGBoost postižu slične rezultate

### Kvalitet podataka

**Ključno otkriće**: 14.1% dataset-a (16,019 pesama) ima popularnost=0
- Analiza pokazuje da su verovatno **nedostajuće vrednosti**, ne potpuno nepopularne pesme
- **KNN Imputation pristup** pokazuje bolje rezultate od jednostavnog uklanjanja
- Korišćenje sličnosti audio karakteristika omogućava pametno popunjavanje podataka

### Najvažniji faktori

Analiza važnosti atributa pokazuje da najveći uticaj imaju:
1. **Loudness** (glasnoća) - najjača pozitivna korelacija
2. **Genre** (žanr) - različiti žanrovi imaju različite nivoe popularnosti
3. **Energy, Danceability** - energetske karakteristike pesme
4. **Duration_ms** - trajanje pesme

### Dubinski šabloni i klasteri

Klaster analiza je otkrila važne šablone koji prevazilaze jednostavne korelacije:

**Pronađeni klasteri:**
- Pesame se prirodno grupišu u 4 distinktna klastera sa različitim audio profilima
- **Energične pesme** (visoka energy, loudness) imaju najveću prosečnu popularnost
- **Akustične pesme** formiraju poseban klaster sa umerenom popularnošću
- **Instrumentalne pesme** pokazuju najnižu popularnost i najjasniju separaciju

**Važno:**
- Popularnost zavisi od **kombinacije karakteristika**, ne pojedinačnih atributa
- Postoje **različiti šabloni uspešnosti** - nema jedinstvenog "recepta"
- Interakcije između atributa (npr. energy + loudness) imaju snažniji uticaj od bilo kojeg pojedinačnog atributa

### Ključna saznanja

1. **Kvalitet podataka je kritičan**: Pametno rešavanje nedostajućih vrednosti (KNN Imputation) poboljšava performanse za 3.63%

2. Audio karakteristike **objašnjavaju skoro 50% varijanse** u popularnosti - značajno bolji rezultat od očekivanog (15-18% iz literature)

3. **Glasnije pesme** su u proseku popularnije

4. **Žanr ima signifikantan uticaj** (ANOVA test, p < 0.001)

5. Većina pesama ima **nisku popularnost** (<25), distribucija je neujednačena

6. **Šabloni i interakcije** između atributa su važniji od pojedinačnih korelacija

**Ograničenja:**
- Popularnost zavisi od brojnih faktora koje dataset ne sadrži:
  - Marketing i promocija
  - Brend izvođača i njihova prethodna popularnost
  - Trenutni trendovi i sezonalnost
  - Viralni momenti na društvenim mrežama
  - Uključenost u popularne plejliste

**Finalni zaključak**: Random Forest uz KNN Imputation pristup je pouzdan algoritam za ovaj problem i postiže rezultate značajno bolje od očekivanih iz literature.
"""
        
        # Podeli na linije za Jupyter format
        cell['source'] = [line + '\n' for line in new_content.split('\n')]
        # Ukloni \n sa poslednje linije ako postoji
        if cell['source'] and cell['source'][-1].endswith('\n\n'):
            cell['source'][-1] = cell['source'][-1][:-1]
        
        print(f"Ažurirana ćelija {i}: Zaključci")
        break

# Sačuvaj notebook
with open('prezentacija.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Notebook uspešno ažuriran!")
