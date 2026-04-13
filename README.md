# Przewidywanie odpływu klientów banku — sieć neuronowa od zera

Projekt klasyfikacji binarnej przewidujący, czy klient banku zrezygnuje z usług. Cała implementacja oparta wyłącznie na **NumPy** — bez scikit-learn ani frameworków deep learning.

## Co wyróżnia projekt

- Własna implementacja sieci neuronowej z normalizacją wsadową, dropoutem i optymalizatorem Adam — napisana od podstaw
- AUC-ROC **0.85** na zbiorze testowym
- Wyjaśnialność modelu oparta na analizie czułości dla pojedynczych predykcji
- Kompletny pipeline: surowe CSV → inżynieria cech → trening → ewaluacja → wizualizacje

## Architektura

```
Wejście (16) → FC(64) → BatchNorm → ReLU → Dropout(0.3)
             → FC(32) → BatchNorm → ReLU → Dropout(0.3)
             → FC(1)  → Sigmoid
```

Wszystkie komponenty (propagacja w przód i wstecz, normalizacja wsadowa z bieżącymi statystykami, optymalizator Adam) zaimplementowane ręcznie w NumPy.

## Dane

[Bank Customer Churn](https://www.kaggle.com/datasets/abbas829/bank-customer-churn) — 10 000 rekordów, 13 cech, 20,4% wskaźnik odpływu.

**Inżynieria cech:**
| Cecha | Opis |
|---|---|
| `BalanceSalaryRatio` | Saldo / (Wynagrodzenie + 1) |
| `AgeGroup` | Przedziały: <30, 30–45, 45–60, 60+ |
| `HasBalance` | Flaga binarna: saldo > 0 |
| `TenurePerAge` | Staż / (Wiek + 1) |
| `ProductsPerTenure` | Liczba produktów / (Staż + 1) |

## Wyniki

| Metryka | Wartość |
|---|---|
| Dokładność | 84,15% |
| AUC-ROC | 0,8501 |
| F1-Score | 0,6204 |
| Precyzja | 0,6574 |
| Czułość | 0,5873 |

Optymalny próg decyzyjny wyznaczony na 0,637 w celu zrównoważenia precyzji i czułości na niezbalansowanym zbiorze (zastosowanie pos_weight podczas treningu).

## Wizualizacje

Wygenerowane wykresy zapisane w katalogu `wyniki/`:

| Plik | Zawartość |
|---|---|
| `01_rozkład_klas.png` | Rozkład klas |
| `02_cechy_numeryczne.png` | Rozkłady cech numerycznych |
| `03_cechy_kategoryczne.png` | Rozkłady cech kategorycznych |
| `04_korelacja.png` | Mapa ciepła korelacji |
| `05_krzywa_uczenia.png` | Krzywe uczenia (strata treningowa/walidacyjna) |
| `06_macierz_roc.png` | Macierz pomyłek + krzywa ROC |
| `07_rozkład_ryzyka.png` | Rozkład przewidywanych wyników ryzyka |

## Parametry treningu

| Parametr | Wartość |
|---|---|
| Epoki | 300 (wczesne zatrzymanie) |
| Rozmiar batcha | 256 |
| Spadek współczynnika uczenia | 0,985 / epokę |
| Przycinanie gradientów | norma = 5,0 |
| Inicjalizacja wag | Inicjalizacja He |
| Podział trening / test | 80% / 20% |

## Uruchomienie

```bash
pip install -r requirements.txt
jupyter notebook sample.ipynb
```

Notebook pobiera zbiór danych automatycznie przez `kagglehub`. Wymagana konfiguracja [danych uwierzytelniających Kaggle API](https://www.kaggle.com/docs/api).

## Technologie

- Python 3.x
- NumPy
- Pandas
- Matplotlib / Seaborn
- Kagglehub
- Jupyter Notebook
