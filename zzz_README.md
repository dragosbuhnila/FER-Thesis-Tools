1) In primo luogo devi avere la cartella saliency_maps, contenente la cartella basefaces (https://drive.google.com/drive/folders/1e4qJjoih7mxK2a2-lLRWg0EQs9cv78_m?usp=drive_link)

2) In secondo luogo dovresti anche installare le dependencies necessarie. Per farlo fai "pip add poetry" (se non funziona cerca come si installa poetry), e poi fare "poetry install" mentre sei con la console nella cartella base (contenente TODO, README, ...). Credo sia necessaria una versione di python superiore al 3.10 ma non ne sono sicuro. Io ho usato il 3.13.

3) Ho scritto gli script in modo che tu metta i file sui quali vuoi agire dentro una cartella. Le ho caricate con dei file già dentro, così da poter testare gli script. (ricorda che volendo puoi prendere e usare le funzioni che uso negli script, usando gli stessi parametri, e usarle direttamente in un tuo script personalizzato, magari nel quale prima estrai i percorsi e nomi dei file sui quali vuoi agire per automatizzare il tutto. Se non riesci comunque puoi usare la "metodologia cartelle")

4) Nota che per lo script di estrazione dei vettori di medie, non genero propriamente vettori ma dei dizionari. Basta runnare zzz_example_of_reasing_... per vedere il formato dopo averne generato uno.

5) Nel computare i test statistici, se vuoi avere le ROI del tipo "merged" [Occhi, Sopracciglio, Naso, ...] invece che "separate" [Occhio Dx, Occhio Sx, Sopracciglio Dx, ...] basta che metti separate_lr=False quando chiami compute_heatmap_statistics(). La prof ha detto che li preferisce "separate" ma aveva anche detto che potevamo tentare entrambi i modi per capire che succede. Secondo me ha senso tentare anche l'approccio "merged" perchè sicuramente da piu possibilità di correlazione ma finche non abbiamo dati alla mano ovviamente non so bene. (nota che invece per la comparazione senza test statistico, quindi nello script "compare_two_saliencies.py" usiamo le ROI delle AU invece che Occhi, ecc., e in quel caso ho sempre usato la versione merged.)


# Quali script abbiamo:
- extract_roi-means.py => Data una cartella con dei npy che devono essere le mappe di salienza, genera in una cartella target altrettanti file di vettori. Fondamentalmente trasforma delle mappe di salienza in vettori di medie, con una media per ogni roi individuata.
- example_of_reading_roi-means-vector.py => Runnalo per vedere il formato dei vettori, e usalo come reference per saper leggere i npy.
- compare_two_saliencies.py => Runnalo dopo aver messo 2 saliencies nella cartella input adatta. Ritornerà in console il valore di differenza tra le 2.
