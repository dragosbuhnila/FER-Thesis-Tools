import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

# Simulazione dati (ogni distribuzione è la distribuzione delle medie-roi di 70 individui)
angry_righteye = np.random.normal(300, 50, size=70)  
angry_lefteye = np.random.normal(320, 60, size=70)

# Test di normalità
statA, pA = shapiro(angry_righteye)
statB, pB = shapiro(angry_lefteye)
print("Shapiro A p‑value:", pA, "Shapiro B p‑value:", pB)

if pA > 0.05 and pB > 0.05:
    # entrambe normali → t-test indipendenti
    stat, p = ttest_ind(angry_righteye, angry_lefteye, equal_var=False)
    print("t-test p‑value:", p)
else:
    # non normale → Mann‑Whitney U test
    U, p = mannwhitneyu(angry_righteye, angry_lefteye, alternative='two-sided')
    print("Mann‑Whitney U p‑value:", p)

# notare che i test usati sono solo per esempio