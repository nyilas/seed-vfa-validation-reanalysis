# Documento di progetto — Paper su small-data regression, validazione e active learning per il dataset Seed/VFA

**Dataset di riferimento:** `Seed_Dataset.csv`  
**Target predittivo:** `Cret total VFAs`  
**Dominio applicativo:** nanofiltrazione per recupero/trattenimento di acidi grassi volatili (VFA)  
**Obiettivo informatico del paper:** costruire una pipeline metodologica per valutare e sviluppare modelli predittivi affidabili in piccoli dataset tabulari sperimentali, distinguendo interpolazione, estrapolazione e limiti di identificabilità.

---

## 1. Visione generale del progetto

Il progetto non deve essere impostato come un semplice paper del tipo:

> "Applichiamo alcuni algoritmi di machine learning per predire Cret total VFAs."

Questa impostazione sarebbe debole, perché il dataset contiene solo 80 righe, 79 osservazioni distinte e 46 condizioni sperimentali indipendenti. Il valore scientifico più forte non è dimostrare che un certo algoritmo ottiene il miglior R², ma mostrare che, in un piccolo dataset sperimentale strutturato, la prestazione predittiva dipende radicalmente dal protocollo di validazione.

La storia centrale del paper dovrebbe essere:

> Nei piccoli dataset sperimentali tabulari, modelli apparentemente accurati possono in realtà interpolare condizioni già viste o riconoscere domini sperimentali noti. Una validazione corretta deve separare le repliche, le condizioni sperimentali e i domini. Nel dataset Seed/VFA, i modelli completi ottengono prestazioni elevate sulle stesse membrane, ma collassano quando devono generalizzare a membrane non viste. Modelli parsimoniosi basati sulle variabili di processo sono meno accurati in interpolazione ma più stabili fuori dominio. Questo rende il dataset un caso di studio utile per validazione, diagnosi di generalizzazione e progettazione sequenziale di nuovi esperimenti.

---

## 2. Posizionamento del paper

### 2.1 Tipo di paper consigliato

Il paper dovrebbe essere un lavoro **metodologico-applicativo** o **data-centric ML for science**, non un paper di pura chimica e non un paper puramente algoritmico.

La formulazione più adatta è:

> **Validation-aware small-data regression for experimental tabular data.**

Il caso VFA/nanofiltrazione serve come banco di prova reale, ma il contributo principale è informatico:

1. definire protocolli di validazione coerenti con la struttura sperimentale;
2. mostrare il divario tra interpolazione ed estrapolazione;
3. diagnosticare aliasing, leakage e limiti di identificabilità;
4. proporre una pipeline di modellazione parsimoniosa;
5. proporre, come evoluzione, una strategia di active learning/sequential design per scegliere nuovi esperimenti.

---

## 3. Possibili titoli

### Titolo principale consigliato

**Validation-Aware Regression for Small Experimental Tabular Data: Interpolation, Extrapolation, and Active Learning in VFA Nanofiltration**

### Alternative più brevi

1. **When Small-Data Regression Fails to Generalize: A Case Study on VFA Nanofiltration**
2. **Interpolation Is Not Generalization: Validation-Aware Regression for Small Experimental Data**
3. **From Prediction to Experiment Selection: Small-Data Regression for VFA Retention**
4. **Robust Small-Data Regression under Experimental Constraints: Lessons from VFA Nanofiltration**
5. **Diagnosing Generalization in Small Tabular Scientific Datasets**

### Titolo più informatico

**A Validation-Centric Pipeline for Small Tabular Regression under Experimental Design Constraints**

---

## 4. Claim centrale del paper

Il claim principale non deve essere:

> "Il modello X predice Cret con R² alto."

Il claim corretto dovrebbe essere:

> "Mostriamo che in piccoli dataset tabulari sperimentali la performance predittiva può essere fortemente sovrastimata se repliche, condizioni e domini sperimentali non vengono trattati correttamente. Sul dataset Seed/VFA, un modello completo raggiunge prestazioni elevate in interpolazione sulle membrane viste, ma fallisce in estrapolazione verso membrane nuove. Modelli parsimoniosi basati su variabili di processo generalizzano meno in interpolazione ma degradano in modo più stabile fuori dominio. Proponiamo quindi una pipeline di validazione, diagnosi e progettazione sequenziale di nuovi esperimenti."

---

## 5. Contributi attesi

### Contributo 1 — Analisi strutturale del dataset

Mostrare che il dataset non è i.i.d., ma è un disegno sperimentale a blocchi:

- 80 righe totali;
- 79 osservazioni distinte;
- 46 condizioni sperimentali indipendenti;
- 4 membrane;
- 2 regimi di feed;
- repliche sperimentali;
- target continuo `Cret total VFAs`.

### Contributo 2 — Validazione orientata all'obiettivo

Definire tre scenari:

1. validazione ingenua o random;
2. interpolazione onesta entro membrane già viste;
3. estrapolazione verso membrane nuove.

Il punto forte è mostrare che l'accuratezza cambia drasticamente a seconda dello scenario.

### Contributo 3 — Diagnosi di generalizzazione e identificabilità

Mostrare che:

- le proprietà di membrana sono fortemente aliasate perché ci sono solo 4 membrane;
- alcune feature aiutano a riconoscere la membrana, ma non necessariamente a spiegare una legge generalizzabile;
- pH e zeta potential sono correlati e rappresentano quasi lo stesso asse chimico;
- le feature importance possono essere fuorvianti in presenza di collinearità e disegno sperimentale sparso.

### Contributo 4 — Strategia evolutiva verso active learning

Usare il dataset come seed iniziale e proporre una strategia per scegliere nuovi esperimenti:

- massima incertezza;
- diversità nello spazio sperimentale;
- copertura di membrane;
- riduzione del divario interpolazione/estrapolazione;
- eventuale ottimizzazione di Cret.

---

## 6. Domande di ricerca

### RQ1 — Quanto è affidabile la predizione di Cret con soli 46 esperimenti indipendenti?

Si valuta se un modello può predire `Cret total VFAs` in modo stabile, distinguendo tra prestazione apparente e prestazione validata correttamente.

### RQ2 — La prestazione predittiva cambia tra interpolazione ed estrapolazione?

Questa è probabilmente la domanda più importante.

- Interpolazione: nuove condizioni sulle stesse membrane.
- Estrapolazione: membrana mai vista.

### RQ3 — Le feature di membrana aiutano davvero a generalizzare o solo a riconoscere membrane note?

Questa domanda introduce il tema dell'aliasing.

Con sole 4 membrane, MWCO, rugosità, rejection, contact angle e PWP non possono essere separati chiaramente. Il modello può usarle come identificatori di dominio, non come descrittori causali.

### RQ4 — Un modello parsimonioso può essere più affidabile fuori dominio?

Si confrontano:

- modello completo;
- modello parsimonioso;
- modello solo processo;
- RandomForest;
- eventuale GPR/SVR/Kernel Ridge.

### RQ5 — Quali nuovi esperimenti ridurrebbero l'incertezza e migliorerebbero la generalizzazione?

Questa domanda apre la parte di active learning/sequential design.

---

## 7. Descrizione dettagliata del dataset

### 7.1 Forma del dataset

- Numero righe originali: 80.
- Numero colonne: 15.
- Numero feature: 14.
- Target: `Cret total VFAs`.
- Valori mancanti: 0.
- Duplicati esatti: 1.
- Osservazioni distinte: 79.
- Condizioni sperimentali indipendenti: 46.
- Membrane distinte: 4.
- Regimi di feed: 2.

### 7.2 Famiglie di feature

#### A. Proprietà di membrana

Queste variabili descrivono la membrana:

- `MWCO`;
- `Average surface roughness`;
- `Zeta potential`;
- `Static contact angle`;
- `MgSO4 rejection`;
- `NaCl rejection`;
- `PWP`.

Nota critica: molte di queste proprietà sono quasi costanti per membrana. Con sole 4 membrane, non è possibile stimare in modo robusto l'effetto individuale di ciascuna proprietà.

#### B. Condizioni operative

- `pH`;
- `Temperature`;
- `Pressure`.

Sono le variabili più utili per un modello generalizzabile perché possono variare all'interno dello stesso dominio sperimentale.

#### C. Composizione del feed

- cationi monovalenti;
- cationi divalenti;
- anioni monovalenti;
- anioni divalenti.

I molti zeri non vanno trattati come rumore: rappresentano il passaggio tra soluzione modello e matrice reale.

#### D. Target

- `Cret total VFAs`.

È la risposta sperimentale da predire.

---

## 8. Analisi esplorativa da includere nel paper

Questa sezione deve dimostrare che il dataset contiene segnale, ma anche vincoli strutturali.

### 8.1 Distribuzione del target

Obiettivo:

- mostrare range, media, mediana, deviazione standard;
- verificare asimmetria;
- individuare eventuali outlier;
- mostrare se il target cambia tra feed semplice e feed complesso.

Figura consigliata:

**Figura 1 — Distribuzione di Cret total VFAs**

Possibili pannelli:

1. istogramma del target;
2. boxplot del target per tipo di feed;
3. rug plot o strip plot per osservare densità e repliche.

Messaggio da scrivere:

> Il target mostra variabilità sufficiente per la modellazione, ma la distribuzione è determinata da un numero limitato di condizioni sperimentali e da blocchi di feed/membrana.

---

### 8.2 Correlazioni con il target

Obiettivo:

- mostrare che pH è il predittore marginale più forte;
- mostrare il ruolo secondario della pressione;
- mostrare la correlazione negativa dello zeta potential;
- evidenziare che correlazione non equivale a causalità.

Figura consigliata:

**Figura 2 — Correlazioni Pearson/Spearman con Cret**

Pannelli:

1. barplot Pearson;
2. barplot Spearman;
3. evidenza grafica delle prime 5 variabili.

Messaggio:

> pH emerge come driver dominante; zeta potential mostra associazione negativa coerente con la chimica del sistema; pressure contribuisce in modo secondario. Tuttavia, alcune associazioni sono confondibili a causa del disegno sperimentale.

---

### 8.3 Effetto pH

Obiettivo:

- mostrare andamento crescente di Cret con pH;
- mostrare saturazione oltre pH 7;
- collegare il risultato a deprotonazione dei VFA ed esclusione di Donnan.

Figura consigliata:

**Figura 3 — Effetto del pH su Cret**

Pannelli:

1. media e intervallo di confidenza di Cret per livello di pH;
2. scatter Cret vs pH colorato per membrana o feed;
3. eventuale curva lowess o media per gruppi.

Messaggio:

> Il pH è il principale driver osservato: l'aumento del pH è associato a maggiore ritenzione dei VFA, probabilmente per deprotonazione e maggiore repulsione elettrostatica. L'effetto non appare lineare su tutto il range, ma tende a saturare.

---

### 8.4 Zeta potential e pH

Obiettivo:

- mostrare che zeta potential e pH sono fortemente collegati;
- chiarire che tenerli entrambi nel modello può creare instabilità;
- mostrare che lo zeta non è una feature indipendente dal pH.

Figura consigliata:

**Figura 4 — Zeta potential vs pH per membrana**

Pannelli:

1. line plot zeta-pH per ciascuna membrana;
2. scatter Cret vs zeta colorato per pH.

Messaggio:

> Zeta potential varia sistematicamente con pH e membrana. Per questo motivo, la sua importanza predittiva non può essere interpretata isolatamente.

---

### 8.5 Struttura a blocchi del dataset

Obiettivo:

- mostrare che il dataset non è i.i.d.;
- mostrare separazione per membrana e feed;
- giustificare la necessità di validazione raggruppata.

Figure consigliate:

**Figura 5 — PCA o UMAP dello spazio delle feature**

Pannelli:

1. PCA colorata per membrana;
2. PCA colorata per tipo feed;
3. PCA colorata per pH.

Messaggio:

> Le componenti principali mostrano che la struttura del dataset è dominata da membrana e feed. Questo rende inappropriata una validazione casuale standard se l'obiettivo è misurare generalizzazione reale.

---

## 9. Sezione metodologica: protocolli di validazione

Questa è la sezione più importante del paper.

### 9.1 Perché non basta una random cross-validation

Con repliche e condizioni sperimentali ripetute, una random CV può mettere una replica nel train e una replica simile nel test. Il modello sembra predire bene, ma in realtà sta interpolando quasi la stessa condizione.

Da scrivere chiaramente:

> In small experimental datasets, samples are not always independent observations. Replicates and repeated conditions must be treated as groups during validation.

### 9.2 Protocollo 1 — Random CV

Serve solo come baseline ingenua.

Scopo:

- mostrare performance apparente;
- illustrare possibile sovrastima.

Non usarlo come risultato principale.

### 9.3 Protocollo 2 — GroupKFold per condizione sperimentale

Definizione:

- ogni gruppo = stessa combinazione sperimentale;
- repliche sempre nello stesso fold;
- misura interpolazione onesta su nuove condizioni, ma entro le stesse membrane.

Obiettivo:

> Valutare se il modello predice nuove condizioni sperimentali su membrane già note.

### 9.4 Protocollo 3 — Leave-One-Membrane-Out

Definizione:

- ogni fold lascia fuori una membrana intera;
- il modello viene testato su una membrana mai vista.

Obiettivo:

> Valutare la generalizzazione fuori dominio.

È il protocollo più difficile e più informativo per il claim metodologico.

### 9.5 Tabella dei protocolli

Da inserire nel paper:

| Protocollo | Unità lasciata fuori | Domanda misurata | Rischio |
|---|---|---|---|
| Random CV | singola riga | prestazione apparente | leakage tra repliche |
| GroupKFold per condizione | condizione sperimentale | interpolazione onesta | membrane ancora viste |
| Leave-One-Membrane-Out | membrana | estrapolazione/domain generalization | alta varianza |

---

## 10. Modelli da implementare

Il paper non deve confrontare troppi modelli. Meglio pochi modelli scelti bene.

### 10.1 Baseline

#### DummyRegressor

Predice la media del target.

Serve per rispondere:

> Il modello impara qualcosa oltre la media?

Metriche attese:

- RMSE circa 0.72;
- R² circa 0.

### 10.2 Modello lineare completo

Feature:

- tutte le feature disponibili, eventualmente dopo rimozione del duplicato;
- standardizzazione;
- Ridge regression.

Scopo:

- modello ad alta capacità relativa per il dataset;
- utile in interpolazione;
- fragile in estrapolazione.

Risultato atteso:

- buono in GroupKFold;
- pessimo in leave-one-membrane-out.

### 10.3 Modello parsimonioso

Feature consigliate:

- pH;
- pressure;
- MWCO;
- feed type.

Oppure:

- pH;
- pressure;
- membrane ID/MWCO;
- feed type.

Scopo:

- ridurre il numero di variabili;
- evitare aliasing e instabilità;
- mantenere generalizzazione più stabile.

### 10.4 Modello solo processo

Feature:

- pH;
- pressure;
- temperature;
- feed type.

Scopo:

- verificare se il segnale generalizzabile risiede nelle condizioni operative più che nei descrittori della membrana.

Questo modello è cruciale per la storia del paper.

### 10.5 RandomForest regolarizzato

Feature:

- tutte o quasi tutte.

Scopo:

- confronto con modello non lineare;
- gli alberi non estrapolano linearmente fuori range;
- può essere più stabile del lineare completo in leave-one-membrane-out.

Attenzione:

- usare profondità limitata;
- numero minimo di campioni per foglia;
- tuning leggero;
- evitare overfitting.

### 10.6 Gaussian Process Regression

Da usare in una delle due modalità.

#### Modalità A — modello principale per incertezza

Feature consigliate:

- solo processo;
- parsimonioso.

Kernel:

- RBF o Matern;
- WhiteKernel;
- ARD se possibile.

Scopo:

- ottenere predizione e incertezza;
- supportare active learning.

#### Modalità B — estensione opzionale

Se il paper si concentra sulla validazione, GPR può essere messo come estensione o analisi aggiuntiva.

### 10.7 Kernel Ridge / SVR

Da usare come confronto con GPR:

- SVR RBF;
- Kernel Ridge RBF.

Sono modelli adatti a small data, ma non forniscono naturalmente incertezza calibrata.

---

## 11. Metriche da usare

### 11.1 Metriche principali

- MAE;
- RMSE;
- R².

### 11.2 Metriche secondarie

- Spearman tra y reale e y predetto;
- bias medio;
- errore per membrana;
- errore per livello di pH;
- errore per feed semplice/complesso.

### 11.3 Intervalli di confidenza

Con dataset piccolo, non basta riportare una media.

Riportare:

- media ± deviazione standard tra fold;
- bootstrap confidence intervals;
- distribuzione degli errori.

### 11.4 Rumore irriducibile

Usare le repliche per stimare il pavimento di rumore.

Messaggio:

> Se il rumore irriducibile stimato è RMSE ≈ 0.23, un modello con RMSE ≈ 0.31 in interpolazione è vicino al limite osservabile. Tuttavia, questo non implica generalizzazione a membrane nuove.

---

## 12. Esperimenti da eseguire

### Esperimento 1 — Data audit

Obiettivo:

- verificare duplicati;
- identificare gruppi di replica;
- identificare condizioni indipendenti;
- costruire variabili derivate: feed type, membrane ID, condition ID.

Output:

- tabella di audit;
- file `data_audit.csv`;
- log riproducibile.

### Esperimento 2 — EDA completa

Obiettivo:

- correlazioni;
- distribuzione target;
- pH effect;
- zeta-pH;
- PCA;
- feed effect;
- membrane effect.

Output:

- figure EDA;
- tabella statistiche;
- report descrittivo.

### Esperimento 3 — Baseline e random CV

Obiettivo:

- mostrare performance apparente;
- non usarla come risultato principale.

Output:

- tabella baseline;
- confronto con protocolli più severi.

### Esperimento 4 — GroupKFold per condizione

Obiettivo:

- stimare interpolazione onesta.

Modelli:

- Dummy;
- Ridge completo;
- Ridge parsimonioso;
- solo processo;
- RandomForest;
- GPR opzionale.

Output:

- tabella R²/RMSE/MAE;
- scatter predetto vs reale;
- errori per fold.

### Esperimento 5 — Leave-One-Membrane-Out

Obiettivo:

- stimare generalizzazione a membrana nuova.

Output:

- tabella per membrana lasciata fuori;
- scatter predetto/reale per membrana;
- grafico performance interpolazione vs estrapolazione.

### Esperimento 6 — Feature ablation

Obiettivo:

- capire quali gruppi di feature aiutano o danneggiano la generalizzazione.

Set di feature:

1. solo processo;
2. solo membrana;
3. processo + feed;
4. processo + membrana;
5. completo;
6. senza zeta;
7. senza PWP;
8. senza rejection;
9. feed simple only;
10. feed complex only.

Output:

- heatmap modello × feature set × protocollo;
- ranking stabilità.

### Esperimento 7 — Diagnosi di collinearità e aliasing

Obiettivo:

- mostrare che alcune feature non sono identificabili.

Analisi:

- matrice di correlazione;
- VIF;
- condition number;
- grouping per membrana;
- numero di valori unici per feature;
- feature costanti o quasi costanti per dominio.

Output:

- tabella di identifiability risk;
- figura network/correlation blocks.

### Esperimento 8 — Incertezza predittiva

Obiettivo:

- valutare se GPR fornisce incertezza utile.

Metriche:

- calibration curve;
- interval coverage;
- negative log likelihood;
- rapporto tra errore e incertezza.

Output:

- predizione ± intervallo;
- errore vs incertezza;
- uncertainty map nello spazio pH-pressure.

### Esperimento 9 — Active learning simulato

Obiettivo:

- simulare scelta sequenziale di nuovi esperimenti.

Setup:

- partire da un sottoinsieme iniziale;
- aggiungere una condizione alla volta;
- confrontare strategie.

Strategie:

1. random acquisition;
2. uncertainty sampling;
3. diversity sampling;
4. uncertainty + diversity;
5. membrane-balanced acquisition;
6. expected improvement se l'obiettivo è massimizzare Cret.

Output:

- learning curves;
- numero di esperimenti necessari per raggiungere una soglia RMSE;
- esperimenti suggeriti;
- regioni dello spazio poco esplorate.

---

## 13. Figure principali del paper

### Figura 1 — Overview del dataset

Contenuto:

- schema delle 4 famiglie di variabili;
- 80 righe, 46 condizioni, 4 membrane, 2 feed;
- target Cret.

Tipo:

- diagramma concettuale;
- può essere realizzato con matplotlib o draw.io.

Messaggio:

> Il dataset è piccolo, strutturato e non i.i.d.

### Figura 2 — Correlation and mechanism

Contenuto:

- barplot correlazioni con target;
- pH effect;
- zeta vs pH per membrana.

Messaggio:

> Il segnale dominante è coerente con il meccanismo chimico pH/zeta/Donnan, ma le feature sono collineari.

### Figura 3 — Experimental structure

Contenuto:

- PCA colorata per membrana;
- PCA colorata per feed;
- eventuale heatmap di feature.

Messaggio:

> Il dataset è dominato dalla struttura sperimentale; la validazione deve rispettare gruppi e domini.

### Figura 4 — Validation protocols

Contenuto:

- schema visivo dei tre split:
  1. random CV;
  2. group-by-condition CV;
  3. leave-one-membrane-out.

Messaggio:

> Lo split definisce il significato della prestazione.

Questa figura è molto importante e può essere una delle figure chiave del paper.

### Figura 5 — Performance gap

Contenuto:

- barplot R² e RMSE;
- confronto interpolazione vs estrapolazione;
- modelli: completo, parsimonioso, solo processo, RandomForest.

Messaggio:

> Il modello completo è ottimo in interpolazione ma non generalizza a membrane nuove.

### Figura 6 — Predicted vs observed

Contenuto:

- scatter y reale vs y predetto;
- pannello per interpolazione;
- pannello per membrana nuova;
- colore = pH o membrana.

Messaggio:

> Gli errori non sono casuali: cambiano per dominio e regione sperimentale.

### Figura 7 — Feature group ablation

Contenuto:

- heatmap feature set × protocollo;
- score R²/RMSE.

Messaggio:

> Le feature di membrana aiutano a interpolare ma non necessariamente a generalizzare.

### Figura 8 — Active learning proposal

Contenuto:

- learning curve RMSE vs numero di esperimenti;
- confronto random vs uncertainty vs diversity;
- mappa dei prossimi esperimenti suggeriti.

Messaggio:

> La diagnosi di generalizzazione può guidare la progettazione dei prossimi esperimenti.

---

## 14. Tabelle principali del paper

### Tabella 1 — Dataset summary

Colonne:

- numero righe;
- osservazioni distinte;
- condizioni indipendenti;
- membrane;
- feed;
- feature;
- target;
- missing;
- duplicati;
- range target.

### Tabella 2 — Feature groups

Colonne:

- nome feature;
- famiglia;
- unità;
- valori unici;
- ruolo;
- rischio interpretativo.

### Tabella 3 — Validation protocols

Colonne:

- protocollo;
- split unit;
- domanda misurata;
- possibile leakage;
- uso nel paper.

### Tabella 4 — Model comparison

Colonne:

- modello;
- feature;
- numero variabili;
- interpolazione R²/RMSE/MAE;
- nuova membrana R²/RMSE/MAE.

### Tabella 5 — Leave-one-membrane-out details

Colonne:

- membrana lasciata fuori;
- numero campioni;
- range target;
- modello migliore;
- RMSE;
- errore medio;
- commento.

### Tabella 6 — Ablation results

Colonne:

- feature set;
- protocollo;
- R²;
- RMSE;
- stabilità;
- interpretazione.

### Tabella 7 — Active learning candidates

Colonne:

- candidato;
- pH;
- pressure;
- temperature;
- feed;
- membrana;
- incertezza;
- diversità;
- motivazione.

---

## 15. Struttura consigliata del manoscritto

### Abstract

Contenuti:

1. problema generale: small experimental tabular data;
2. rischio: validazione ingenua e overestimation;
3. caso studio: VFA nanofiltration;
4. metodo: validation-aware pipeline;
5. risultati: gap interpolazione/estrapolazione;
6. estensione: active learning for next experiments.

Non enfatizzare troppo la chimica. Enfatizzare la lezione metodologica.

### 1. Introduction

Punti da includere:

- ML sempre più usato in esperimenti scientifici;
- spesso i dataset sono piccoli e costosi;
- le osservazioni non sono i.i.d.;
- repliche, domini e condizioni sperimentali possono creare leakage;
- la predizione va valutata rispetto all'obiettivo: interpolare o estrapolare;
- caso studio su VFA nanofiltration;
- contributi.

Frase chiave:

> In small experimental datasets, model performance is not an intrinsic property of the algorithm, but a property of the validation question.

### 2. Related work

Sottosezioni:

1. ML for small tabular scientific data;
2. regression and uncertainty in experimental sciences;
3. validation under grouped/structured data;
4. domain generalization in tabular data;
5. active learning/sequential experimental design.

Nota: questa sezione richiede una ricerca bibliografica mirata.

### 3. Dataset and experimental setting

Contenuti:

- descrizione del dataset;
- target;
- feature groups;
- repliche;
- condizioni indipendenti;
- membrane;
- feed;
- duplicato;
- rumore da repliche;
- implicazioni per la validazione.

Evitare di trattare le 80 righe come 80 campioni indipendenti.

### 4. Exploratory and structural analysis

Contenuti:

- correlazioni;
- pH effect;
- zeta-pH;
- feed effect;
- PCA;
- aliasing delle membrane;
- collinearità;
- numero di valori unici per feature.

Obiettivo:

> Dimostrare che il dataset ha un segnale interpretabile ma anche vincoli strutturali.

### 5. Validation-aware modeling pipeline

Contenuti:

- preprocessing;
- rimozione duplicato;
- creazione gruppi;
- feature standardization dentro fold;
- modelli;
- protocolli di validazione;
- metriche;
- stima rumore da repliche.

Questa è la sezione metodologica centrale.

### 6. Results

Sottosezioni:

1. random validation overestimates performance;
2. grouped interpolation is feasible;
3. membrane extrapolation is difficult;
4. parsimonious models degrade gracefully;
5. feature groups behave differently across protocols;
6. uncertainty identifies under-sampled regions.

### 7. Active learning for next experiments

Contenuti:

- motivazione;
- simulazione retrospettiva;
- acquisition functions;
- confronto con random;
- lista di esperimenti suggeriti;
- limiti.

Questa sezione può essere opzionale se il paper diventa troppo lungo.

### 8. Discussion

Punti da discutere:

- il dataset è utile per interpretazione e interpolazione;
- non basta per claim forti su membrane nuove;
- il limite è il disegno sperimentale, non solo il modello;
- feature importance può essere ingannevole;
- modelli complessi non risolvono aliasing e scarsità;
- la soluzione è combinare modellazione e disegno sperimentale.

### 9. Threats to validity / Limitations

Da includere obbligatoriamente.

Limiti:

1. solo 4 membrane;
2. solo 46 condizioni indipendenti;
3. temperatura a pochi livelli;
4. pressione concentrata in pochi punti;
5. active learning simulato retrospettivamente;
6. assenza di validazione esterna su nuove membrane reali;
7. feature di membrana aliasate;
8. rischio di dipendenza dal dataset.

### 10. Conclusion

Messaggio finale:

> La validazione corretta è parte del modello. Nei piccoli dataset sperimentali, un R² alto può indicare interpolazione, non generalizzazione. Una pipeline validation-aware permette di diagnosticare questo problema e di trasformarlo in una strategia per progettare nuovi esperimenti.

---

## 16. Appendici consigliate

### Appendix A — Full dataset audit

- missing values;
- duplicati;
- condizioni;
- repliche;
- valori unici;
- unità.

### Appendix B — Hyperparameters

- griglie di tuning;
- parametri finali;
- random seeds;
- fold definitions.

### Appendix C — Additional model results

- modelli esclusi dal main paper;
- risultati completi;
- sensitivity analysis.

### Appendix D — Code reproducibility

- struttura repository;
- ambiente;
- versioni librerie;
- comandi per riprodurre figure e tabelle.

### Appendix E — Active learning details

- acquisition functions;
- pseudocode;
- candidati generati;
- vincoli sperimentali.

---

## 17. Repository consigliata

Struttura suggerita:

```text
seed-vfa-small-data/
├── README.md
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── data/
│   ├── raw/
│   │   └── Seed_Dataset.csv
│   ├── processed/
│   │   ├── seed_clean.csv
│   │   ├── seed_with_groups.csv
│   │   └── condition_groups.csv
│   └── external/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling_interpolation.ipynb
│   ├── 04_modeling_extrapolation.ipynb
│   ├── 05_uncertainty.ipynb
│   └── 06_active_learning.ipynb
├── src/
│   └── seed_vfa/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── splits.py
│       ├── models.py
│       ├── evaluation.py
│       ├── diagnostics.py
│       ├── uncertainty.py
│       ├── active_learning.py
│       └── plotting.py
├── scripts/
│   ├── run_data_audit.py
│   ├── run_eda.py
│   ├── run_models.py
│   ├── run_ablation.py
│   ├── run_uncertainty.py
│   └── run_active_learning.py
├── results/
│   ├── tables/
│   ├── figures/
│   ├── models/
│   └── logs/
├── paper/
│   ├── manuscript.tex
│   ├── figures/
│   └── tables/
└── tests/
    ├── test_splits.py
    ├── test_data.py
    └── test_metrics.py
```

---

## 18. Linguaggio consigliato per lo sviluppo

### Scelta principale: Python

Consiglio fortemente **Python** come linguaggio principale.

Motivi:

1. ecosistema maturo per machine learning tabulare;
2. ottimo supporto per cross-validation raggruppata;
3. integrazione naturale con pandas, scikit-learn, scipy e matplotlib;
4. facile produzione di figure e tabelle per paper;
5. buona riproducibilità;
6. adatto sia a notebook esplorativi sia a pipeline scriptabili;
7. compatibile con GPR, active learning e Bayesian optimization.

### Versione consigliata

Usare:

```text
Python 3.11 oppure Python 3.12
```

Python 3.11 è spesso la scelta più stabile per compatibilità scientifica.

---

## 19. Librerie consigliate

### Core data science

```text
numpy
pandas
scipy
scikit-learn
statsmodels
```

### Visualizzazione

```text
matplotlib
plotly
```

Matplotlib è preferibile per figure da paper. Plotly può essere utile per esplorazione interattiva, ma le figure finali dovrebbero essere statiche e riproducibili.

### Modelli

```text
scikit-learn
xgboost          # opzionale
catboost         # opzionale
```

Per questo dataset, scikit-learn è sufficiente per la maggior parte del lavoro.

### Gaussian Process / incertezza

Prima scelta:

```text
scikit-learn GaussianProcessRegressor
```

Se la parte di active learning diventa centrale:

```text
gpytorch
botorch
ax-platform
```

Però questi strumenti aggiungono complessità. Per il primo paper, scikit-learn potrebbe bastare.

### Interpretabilità

```text
sklearn.inspection
shap              # opzionale, con cautela
```

Preferire permutation importance e ablation. SHAP può essere usato solo come analisi esplorativa, non come prova causale.

### Esperimenti e configurazione

```text
pyyaml
joblib
tqdm
hydra-core        # opzionale
```

### Testing

```text
pytest
```

### Qualità codice

```text
ruff
black
mypy             # opzionale
```

### Gestione ambiente

Opzione semplice:

```text
venv + requirements.txt
```

Opzione più robusta:

```text
conda/mamba + environment.yml
```

Opzione moderna consigliata:

```text
uv + pyproject.toml
```

Se lavori su Linux/CachyOS, `uv` è molto comodo.

---

## 20. Perché non R come linguaggio principale?

R sarebbe buono per statistica classica, modelli lineari e visualizzazione. Tuttavia, per questo progetto Python è più adatto perché:

- si integra meglio con scikit-learn;
- gestisce più facilmente pipeline ML riproducibili;
- facilita active learning e GPR;
- si collega meglio a un paper di informatica;
- è più naturale per esperimenti ripetuti e benchmark.

R può essere usato come supporto, ma non lo userei come ambiente principale.

---

## 21. Perché non deep learning?

Non usare deep learning standard come contributo centrale.

Motivi:

- 46 condizioni indipendenti sono troppo poche;
- alto rischio di overfitting;
- difficile interpretazione;
- poco credibile per un revisore;
- modelli semplici sono più appropriati;
- il problema scientifico è la validazione, non la capacità del modello.

Se vuoi includere una rete neurale, farlo solo come baseline negativa o appendice, non come protagonista.

---

## 22. Pipeline tecnica consigliata

### Step 1 — Data loading

Script:

```text
scripts/run_data_audit.py
```

Funzioni:

- caricare CSV;
- pulire nomi colonne;
- convertire numerici;
- rimuovere duplicato;
- creare ID riga originale;
- creare `membrane_id`;
- creare `feed_type`;
- creare `condition_id`.

Output:

```text
data/processed/seed_clean.csv
data/processed/seed_with_groups.csv
results/tables/data_audit.csv
```

### Step 2 — Exploratory analysis

Script:

```text
scripts/run_eda.py
```

Output figure:

```text
results/figures/target_distribution.png
results/figures/correlation_target.png
results/figures/ph_effect.png
results/figures/zeta_ph_by_membrane.png
results/figures/pca_structure.png
results/figures/feed_target_distribution.png
```

### Step 3 — Split definitions

Script/funzione:

```text
src/seed_vfa/splits.py
```

Implementare:

- random CV;
- GroupKFold by condition;
- LeaveOneGroupOut by membrane;
- eventualmente repeated group split.

Salvare gli split:

```text
results/splits/random_cv.json
results/splits/group_condition_cv.json
results/splits/leave_membrane_out.json
```

Questo è importante per la riproducibilità.

### Step 4 — Modeling

Script:

```text
scripts/run_models.py
```

Input:

- dataset pulito;
- split;
- feature set;
- modello;
- seed.

Output:

```text
results/tables/model_results.csv
results/tables/model_results_by_fold.csv
results/predictions/predictions_all.csv
```

Ogni predizione deve salvare:

- row_id;
- condition_id;
- membrane_id;
- protocol;
- fold;
- model;
- y_true;
- y_pred;
- residual;
- uncertainty, se disponibile.

### Step 5 — Ablation

Script:

```text
scripts/run_ablation.py
```

Feature set:

```text
full
process_only
parsimonious
membrane_only
process_plus_feed
without_zeta
without_pwp
without_rejections
```

Output:

```text
results/tables/ablation_results.csv
results/figures/ablation_heatmap.png
```

### Step 6 — Uncertainty

Script:

```text
scripts/run_uncertainty.py
```

Usare:

- GPR;
- bootstrap ensembles;
- conformal prediction opzionale.

Output:

```text
results/tables/uncertainty_metrics.csv
results/figures/uncertainty_calibration.png
results/figures/error_vs_uncertainty.png
```

### Step 7 — Active learning

Script:

```text
scripts/run_active_learning.py
```

Strategie:

- random;
- uncertainty;
- diversity;
- uncertainty + diversity;
- membrane-balanced.

Output:

```text
results/tables/active_learning_results.csv
results/figures/active_learning_curves.png
results/tables/suggested_next_experiments.csv
```

---

## 23. Pseudocodice della pipeline

```text
Load raw dataset
Clean column names
Remove exact duplicate
Create membrane_id
Create feed_type
Create condition_id from experimental factors
Estimate replicate noise floor

For each validation protocol:
    Create grouped splits
    For each feature set:
        For each model:
            For each fold:
                Fit preprocessing on train only
                Fit model on train
                Predict on test
                Store predictions and residuals
            Aggregate metrics

Run ablation analysis
Run uncertainty analysis
Run active learning simulation
Generate final figures and tables
Write manuscript
```

---

## 24. Criteri di successo

Il progetto può essere considerato riuscito se produce:

1. una pipeline riproducibile;
2. una chiara distinzione tra interpolazione ed estrapolazione;
3. una dimostrazione quantitativa del performance gap;
4. una spiegazione del perché il modello completo collassa fuori dominio;
5. una proposta concreta per nuovi esperimenti;
6. figure chiare e convincenti;
7. un paper che non sovravende i risultati.

---

## 25. Rischi principali

### Rischio 1 — Dataset troppo piccolo

Mitigazione:

- non vendere il paper come benchmark definitivo;
- presentarlo come case study metodologico;
- usare validazione rigorosa;
- dichiarare limiti.

### Rischio 2 — Troppi modelli

Mitigazione:

- pochi modelli ben scelti;
- focus sui protocolli di validazione;
- modelli complessi solo in appendice.

### Rischio 3 — Claim chimici troppo forti

Mitigazione:

- usare "associated with", "consistent with", "predictive of";
- evitare "causes";
- distinguere interpretazione statistica e meccanismo fisico.

### Rischio 4 — Active learning debole perché simulato

Mitigazione:

- dichiararlo come retrospettivo;
- usarlo come design recommendation;
- non promettere validazione esterna.

### Rischio 5 — Feature importance fuorviante

Mitigazione:

- usare ablation e stability;
- discutere aliasing;
- evitare SHAP come prova principale.

---

## 26. Evoluzioni possibili

### Evoluzione A — Paper breve solo su validazione

Focus:

- dataset;
- protocolli;
- performance gap;
- lesson learned.

Vantaggio:

- più pulito;
- più facile da chiudere.

Svantaggio:

- contributo metodologico limitato.

### Evoluzione B — Paper completo validazione + active learning

Focus:

- diagnosi;
- gap interpolazione/estrapolazione;
- proposta active learning.

Vantaggio:

- storia più completa;
- maggiore originalità.

Svantaggio:

- più lavoro;
- più rischi sperimentali.

### Evoluzione C — Aggiunta di altri dataset small-tabular

Focus:

- generalizzare la pipeline a più casi scientifici.

Vantaggio:

- paper più forte per venue informatiche;
- meno dipendente da un solo dataset.

Svantaggio:

- serve trovare dataset comparabili;
- aumenta molto il lavoro.

### Evoluzione D — Dataset paper / data descriptor

Focus:

- pubblicare il dataset come risorsa documentata;
- benchmark per small-data regression con repliche e domini.

Vantaggio:

- contributo data-centric;
- utile alla comunità.

Svantaggio:

- serve curare molto metadata, licenza, documentazione e baseline.

### Evoluzione E — Incertezza e conformal prediction

Focus:

- intervalli predittivi affidabili;
- calibrazione rispetto al rumore da repliche.

Vantaggio:

- molto interessante per small data;
- utile in contesti sperimentali.

Svantaggio:

- con 46 condizioni la calibrazione è difficile.

---

## 27. Roadmap operativa

### Fase 0 — Preparazione

Durata stimata: 2-3 giorni.

Task:

- definire obiettivo paper;
- fissare claim;
- creare repository;
- congelare versione dataset;
- preparare ambiente Python.

Output:

- repository;
- README iniziale;
- dataset raw archiviato.

### Fase 1 — Data audit

Durata stimata: 2-3 giorni.

Task:

- verificare duplicati;
- creare gruppi;
- creare condition_id;
- stimare repliche;
- documentare dataset.

Output:

- `data_audit.csv`;
- tabella dataset summary;
- script riproducibile.

### Fase 2 — EDA

Durata stimata: 4-6 giorni.

Task:

- distribuzioni;
- correlazioni;
- pH effect;
- zeta-pH;
- PCA;
- feed effect;
- membrane effect.

Output:

- figure 1-3;
- sezione dataset/EDA quasi pronta.

### Fase 3 — Modellazione base

Durata stimata: 5-7 giorni.

Task:

- implementare split;
- implementare modelli;
- eseguire GroupKFold;
- eseguire leave-one-membrane-out;
- salvare predizioni.

Output:

- tabella model comparison;
- figura performance gap;
- scatter predicted/observed.

### Fase 4 — Ablation e diagnostica

Durata stimata: 4-6 giorni.

Task:

- feature groups;
- ablation;
- collinearità;
- VIF;
- stability;
- error analysis.

Output:

- ablation heatmap;
- identifiability table;
- discussion ready.

### Fase 5 — Incertezza

Durata stimata: 4-7 giorni.

Task:

- GPR;
- prediction intervals;
- error vs uncertainty;
- calibration.

Output:

- uncertainty figures;
- sezione opzionale.

### Fase 6 — Active learning

Durata stimata: 7-14 giorni.

Task:

- definire pool/candidati;
- simulare acquisition;
- confrontare strategie;
- produrre learning curves.

Output:

- active learning curves;
- suggested experiments;
- sezione paper.

### Fase 7 — Scrittura paper

Durata stimata: 10-15 giorni.

Task:

- abstract;
- introduction;
- related work;
- methods;
- results;
- discussion;
- conclusion;
- appendici.

Output:

- prima bozza completa.

### Fase 8 — Revisione

Durata stimata: 7-10 giorni.

Task:

- controllare claim;
- semplificare figure;
- aggiungere limiti;
- controllare riproducibilità;
- adattare venue.

Output:

- versione sottomissibile.

---

## 28. Priorità se il tempo è poco

Se hai poco tempo, fare solo:

1. data audit;
2. EDA;
3. GroupKFold vs leave-one-membrane-out;
4. confronto full vs parsimonious vs process-only vs RF;
5. performance gap figure;
6. discussion sui limiti;
7. active learning solo come future work.

Questa versione è già un paper coerente.

---

## 29. Minimal viable paper

Il paper minimo dovrebbe contenere:

- dataset audit;
- validazione corretta;
- confronto interpolazione/estrapolazione;
- dimostrazione collasso modello completo;
- modello parsimonioso più stabile;
- interpretazione pH/zeta;
- limiti;
- future work su active learning.

Titolo adatto:

**Interpolation Is Not Generalization: A Small-Data Case Study on VFA Nanofiltration**

---

## 30. Versione estesa del paper

La versione estesa dovrebbe contenere anche:

- GPR;
- uncertainty quantification;
- active learning simulato;
- proposed next experiments;
- eventuale rilascio dataset;
- codice riproducibile.

Titolo adatto:

**From Validation to Experiment Selection: Small-Data Regression and Active Learning for VFA Nanofiltration**

---

## 31. Messaggio finale del progetto

Il progetto dovrebbe arrivare a una conclusione chiara:

> Il dataset Seed/VFA è abbastanza informativo per capire alcuni meccanismi e per interpolare entro membrane già studiate, ma non basta per generalizzare in modo affidabile a membrane nuove. Il limite non è semplicemente la scelta dell'algoritmo, ma la struttura del disegno sperimentale. Per questo motivo, nei piccoli dataset scientifici, la validazione deve essere progettata in base alla domanda predittiva, e i modelli devono essere usati anche per guidare la raccolta dei prossimi dati.

---

## 32. Checklist finale

Prima di scrivere o sottomettere il paper, verificare:

- [ ] Il duplicato è stato rimosso prima degli split.
- [ ] Le repliche non sono divise tra train e test.
- [ ] Gli split sono salvati e riproducibili.
- [ ] La standardizzazione è fatta dentro ogni fold.
- [ ] Le feature selection sono fatte dentro ogni fold.
- [ ] Random CV non è usata come claim principale.
- [ ] Interpolazione ed estrapolazione sono separate.
- [ ] I risultati sono riportati con RMSE, MAE e R².
- [ ] È presente una discussione sul rumore da repliche.
- [ ] È presente una discussione su aliasing e collinearità.
- [ ] Le figure mostrano chiaramente il performance gap.
- [ ] Non si fanno claim causali non supportati.
- [ ] L'active learning è dichiarato come simulazione o proposta.
- [ ] Il codice produce tutte le figure e tabelle da script.
- [ ] Il paper dichiara apertamente i limiti.
