1) In primo luogo devi avere la cartella saliency_maps, contenente la cartella basefaces (https://drive.google.com/drive/folders/1e4qJjoih7mxK2a2-lLRWg0EQs9cv78_m?usp=drive_link)

2) In secondo luogo dovresti anche installare le dependencies necessarie. Per farlo fai pip add poetry (se non funziona cerca come si installa poetry), e poi fare poetry install.

3) Siccome gli script funzionano che metti i file su cui vuoi agire in una cartella di input, devi prima creare questa cartella (github non me la fa inviare) con lo stesso nome specificato all'interno dello script (ad esempio per zzz_extract_roi-means.py trovi INPUT_FOLDER = "zzz_input_saliency_maps"). Per l'output invece la cartella viene generata in automatico.

4) Nota che non genero propriamente vettori ma dei dizionari. Basta runnare zzz_example... per vedere il formato dopo averne generato uno.

5) Se vuoi avere le ROI del tipo "merged" [Occhi, Sopracciglio, Naso, ...] invece che "separate" [Occhio Dx, Occhio Sx, Sopracciglio Dx, ...] basta che metti separate_lr=False quando chiami compute_heatmap_statistics(). La prof ha detto che li preferisce "separate" ma aveva anche detto che potevamo tentare entrambi i modi per capire che succede. Secondo me ha senso tentare anche l'approccio "merged" perchè sicuramente da piu possibilità di correlazione ma finche non abbiamo dati alla mano ovviamente non so bene.


# Quali script abbiamo:
- extract_roi-means.py => Data una cartella con dei npy che devono essere le mappe di salienza, genera in una cartella target altrettanti file di vettori. Fondamentalmente trasforma delle mappe di salienza in vettori di medie, con una media per ogni roi individuata.
- example_of_reading_roi-means-vector.py => Runnalo per vedere il formato
