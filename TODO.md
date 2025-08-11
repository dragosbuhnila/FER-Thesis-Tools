Canonicals: use the higher quality ones

ROIs:
    - Define Corrugator Supercilii Area
    - Make the ROIs manually

Scripts:
    - Aggregation {cm:2025-08-11}
    - ROIs means extraction
        - e.g. dati tutti in npy in una cartella crea file con gli stessi nomi ma che contengono le medie delle 9 ROI invece che le immagini (chiamiamo l'output salienze-roimean)
        - l'utente potra usarli per il proprio codice
    - Comparazione di 2 salienze-roimean all'interno di una cartella (max 2 salienze quindi)

Comparison Task: (note that the order of subtraction and meaning has changed)
    - FEDMAR vs MATVIN
    - MARFRO vs MATVIN
    - maschi vs femmine;
    - upper tail vs lower tail;
    - Bubbles ConvNext vs FEDMAR
    - Bubble ConvNext vs MARFRO
    - External Perturbation ConvNext vs FEDMAR
    - External Perturbation ConvNext vs MARFRO
    - GradCam LAYER30 ConvNext vs FEDMAR
    - GradCam LAYER30 ConvNext vs FEDMAR

