1) Per ogni emozione (escluse le neutral), posso fare:

   a. 10 immagini con maschere **corrette**  
      (occlusione con le AU dell'emozione corretta)

   b. 40 immagini con maschere **scorrette**  
      (occlusione con le AU di una delle 5 emozioni scorrette — siccome *neutral* non ha AU — che possono essere in positivo o in negativo)
      
      - 40 / 10 = 4 immagini per ogni emozione

   c. *(vuoto nel testo originale)*

   d. Tenterei le opzioni:
      - 20/30 
      - 10/40  
      Siccome danno numeri tondi per le 10 occlusioni

   - **Nota:** In questa maniera prendiamo esattamente tutti i soggetti, ma ad ognuno applichiamo un tipo di maschera diversa (che sia questa corretta, oppure scorretta in positivo, o scorretta in negativo)

---

2) Per il **neutral**, invece:

   a. Abbiamo 50 immagini totali
      - Da dividere su 6 × 2 = **12 occlusioni**
      - Quindi 50 / 12 = **4.1667**
         1. Ovvero 4 per ogni occlusione
         2. Per gli ultimi 2 volti:
            - Solo 2 delle rimanenti 12 occlusioni verranno scelte  
              (possibilmente di emozioni diverse e RH diverso)

---

3) Renderei il tutto **randomico ma seedabile**:

   a. Dunque, per le **6 emozioni**:
      - Per ogni emozione farei uno *shuffle* dei volti, così da non avere sempre la 001 come prima e la 104 come ultima
      - Quindi basta mettere che:
         - I primi **X** sono i *corretti*
         - I seguenti **Y**, a 3 a 3 (o a 4 a 4, in base a se facciamo 20/30 o 10/40), sono gli *scorretti*, sempre nello stesso ordine

   b. Per **neutral**:
      - Facciamo uguale: shuffle dei volti
      - Assegniamo ogni 4 volti le stesse occlusioni
         1. Per il penultimo:
            - Si prende a RH+ e si sceglie una delle 6 emozioni *randomicamente*
         2. L'ultimo uguale, ma a RH-

---

### Nota finale

Questo comporta ogni volta una **distribuzione diversa delle occlusioni**, che però affligge pochissimo la distribuzione totale, che rimane **equilibrata**.  
Infatti si parla di sole 2 immagini su 350, ovvero:

> 2 / 350 × 100 = **0.5714%**  

...che è < dell’1% — quindi va bene.
