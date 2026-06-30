# Documento di progetto rivisto — Validation-aware reanalysis del dataset Seed/VFA

**File principale:** `Seed_Dataset.csv`  
**Paper sorgente del dataset:** *Artificial intelligence for predicting volatile fatty acids rejection in nanofiltration membranes*  
**Target:** `Cret total VFAs` / `RVFA`  
**Dominio applicativo:** nanofiltrazione per recupero/concentrazione di acidi grassi volatili (VFA)  
**Angolazione proposta:** re-analisi informatica/metodologica centrata su validazione, generalizzazione, repliche, domini sperimentali e limiti della synthetic augmentation in small experimental tabular data.

---

## 1. Perche' questo documento e' stato rivisto

La prima versione del progetto assumeva che il dataset fosse semplicemente un piccolo dataset sperimentale disponibile per sviluppare un paper su small-data regression, uncertainty-aware modeling e active learning.

Dopo la lettura del paper sorgente, la situazione cambia: gli autori hanno gia' pubblicato un lavoro di AI sullo stesso dataset. Il loro paper non e' puramente sperimentale: contiene gia' machine learning, CatBoost, feature selection, synthetic data generation, MMD-controlled augmentation e SHAP.

Quindi il nostro paper non puo' essere formulato come:

> Applichiamo modelli ML al dataset per predire Cret/RVFA.

Quella parte e' gia' coperta dal paper sorgente.

La nuova formulazione deve essere:

> Rianalizziamo un dataset sperimentale gia' usato per AI modeling da una prospettiva informatica centrata sul significato della validazione predittiva. Mostriamo che, in un piccolo dataset tabulare sperimentale, un alto R² ottenuto con split random non implica necessariamente generalizzazione scientifica. La prestazione cambia quando si rispettano repliche, condizioni sperimentali e domini/membrane.

Questa nuova angolazione rende il nostro lavoro distinto, piu' informatico e potenzialmente piu' forte.

---

## 2. Cosa ha fatto il paper originale

Il paper originale propone un framework AI per predire la rejection dei VFA in nanofiltrazione. Gli elementi principali sono:

1. dataset sperimentale iniziale di 80 punti;
2. 14 feature iniziali;
3. target `RVFA`, equivalente alla concentrazione relativa dei VFA nel retentato rispetto al feed;
4. feature selection per ridurre le variabili da 14 a 5;
5. modello di riferimento CatBoost;
6. split training/testing 75/25;
7. generazione di dati sintetici con dieci algoritmi;
8. confronto tra diversi rapporti synthetic/actual;
9. controllo della divergenza con MMD;
10. SHAP per interpretabilita'.

Il loro claim principale e':

> Una moderata data augmentation sintetica puo' migliorare le performance di CatBoost su un dataset sperimentale scarso, mentre un eccesso di dati sintetici puo' peggiorare le prestazioni. Il controllo tramite MMD migliora ulteriormente accuratezza e interpretabilita'.

Risultati riportati dal paper sorgente:

- baseline CatBoost con soli dati reali: circa R² = 0.885;
- miglior risultato con synthetic augmentation controllata: circa R² = 0.937;
- feature piu' influenti: pH, zeta potential, pressure;
- synthetic/actual ratio moderato come zona migliore;
- MMD come metrica utile per controllare la qualita' della generazione sintetica.

---

## 3. Conseguenza per il nostro paper

Il paper sorgente riduce lo spazio per un lavoro applicativo semplice, ma apre uno spazio molto interessante per un paper informatico.

### 3.1 Cosa non dobbiamo fare

Non dobbiamo proporre un paper del tipo:

- confronto generico di algoritmi ML;
- CatBoost vs RandomForest vs SVR sullo stesso split;
- uso di SHAP per confermare pH/zeta/pressure;
- synthetic data augmentation alternativa senza una differenza metodologica chiara;
- tentativo di battere R² = 0.937.

Queste strade sarebbero troppo vicine al paper originale o rischierebbero di sembrare una ripetizione.

### 3.2 Cosa dobbiamo fare

Dobbiamo spostare il contributo su:

1. validazione corretta;
2. distinzione tra interpolazione ed estrapolazione;
3. raggruppamento delle repliche;
4. leave-one-membrane-out;
5. effective sample size invece di semplice numero di righe;
6. feature aliasing;
7. domain recognition vs mechanistic generalization;
8. limiti della synthetic augmentation quando il protocollo di validazione non misura il giusto tipo di generalizzazione;
9. active learning come evoluzione per chiudere il gap sperimentale.

---

## 4. Nuova identita' del paper

### 4.1 Versione sintetica

Il nostro paper diventa una:

> **validation-aware reanalysis** di un dataset sperimentale small-tabular gia' usato in un paper AI ambientale.

### 4.2 Formulazione lunga

> We revisit a recently published AI-assisted nanofiltration dataset from a validation-centric machine learning perspective. While the original study investigated controlled synthetic augmentation for improving CatBoost performance, we analyze how experimental grouping, replicated conditions, membrane domains, and validation protocols affect the interpretation of predictive accuracy. We show that high random-test performance does not necessarily imply domain-level generalization and that validation design is as important as model choice or augmentation strategy in small experimental tabular data.

### 4.3 Claim principale aggiornato

> In small experimental tabular datasets, predictive performance is not only a property of the model, but also of the validation question. On the Seed/VFA dataset, high performance under random or interpolation-oriented validation can coexist with weak generalization to unseen membrane domains. Therefore, before applying synthetic augmentation or complex models, one must define the unit of generalization and respect the experimental structure during validation.

---

## 5. Titoli aggiornati

### Titolo consigliato

**Interpolation Is Not Generalization: A Validation-Aware Reanalysis of Small-Data Regression in VFA Nanofiltration**

### Alternative

1. **Beyond Random Splits: Validation-Aware Regression for Small Experimental Tabular Data**
2. **What Does High R² Mean in Small Experimental Datasets? A Reanalysis of VFA Nanofiltration Data**
3. **Validation Design Matters: Interpolation, Extrapolation, and Domain Effects in Small-Data Regression**
4. **When Synthetic Augmentation Is Not Enough: Generalization Diagnostics for Small Experimental Data**
5. **From Prediction Scores to Generalization Claims: A Case Study on VFA Nanofiltration**
6. **Replicates, Domains, and Leakage: Reassessing Predictive Performance in Small Scientific Datasets**

### Titolo piu' informatico

**A Validation-Centric Framework for Small Tabular Regression under Experimental Design Constraints**

---

## 6. Differenza rispetto al paper originale

| Aspetto | Paper originale | Nostro paper |
|---|---|---|
| Dominio | Environmental/water process engineering | Machine learning methodology / data-centric ML |
| Domanda | Synthetic augmentation migliora CatBoost? | Che cosa misura davvero la performance predittiva? |
| Modello centrale | CatBoost | Modelli semplici + protocolli di validazione |
| Split | Random train/test 75/25 | Random, grouped by condition, leave-one-membrane-out |
| Dati sintetici | Contributo centrale | Oggetto di discussione critica / analisi secondaria |
| MMD | Controllo della qualita' sintetica | Possibile estensione, non centro del paper |
| SHAP | Interpretabilita' del modello | Stabilita'/fragilita' della feature importance sotto aliasing |
| Risultato chiave | R² aumenta con augmentation moderata | R² cambia significato al cambiare dell'unita' di generalizzazione |
| Claim | AI modeling puo' essere migliorato con synthetic data | La validazione definisce il significato della generalizzazione |

---

## 7. Domande di ricerca aggiornate

### RQ1 — Quanto cambia la performance passando da random split a validazione raggruppata?

Il paper originale usa uno split random 75/25. Noi dobbiamo misurare quanto questa scelta sia ottimistica rispetto a protocolli che rispettano la struttura sperimentale.

Confronti:

1. random split;
2. repeated random split;
3. GroupKFold per condizione sperimentale;
4. leave-one-membrane-out.

Output atteso:

- random split produce performance piu' alte;
- grouped split riduce la performance ma resta realistico per interpolazione;
- leave-one-membrane-out evidenzia il vero limite di generalizzazione.

---

### RQ2 — Le repliche e le quasi-repliche causano leakage predittivo?

Il dataset contiene repliche e condizioni sperimentali ripetute. Se una replica finisce nel training e una nel test, il modello puo' ottenere ottime performance senza generalizzare davvero.

Analisi da fare:

- identificare gruppi di replica;
- calcolare quante repliche sarebbero separate in random split;
- stimare la differenza tra random split e grouped split;
- quantificare il rischio di leakage sperimentale.

---

### RQ3 — Le feature di membrana aiutano a generalizzare o solo a riconoscere membrane viste?

Con sole quattro membrane, molte feature di membrana sono quasi identificatori del dominio:

- MWCO;
- roughness;
- contact angle;
- rejection;
- PWP;
- zeta potential, anche se varia con pH.

Domanda:

> Un modello usa queste feature per imparare una relazione trasferibile o per riconoscere una delle quattro membrane viste?

Esperimenti:

1. full feature set;
2. process-only;
3. membrane-only;
4. process + membrane;
5. leave-one-membrane-out.

---

### RQ4 — La feature selection basata su SFR e correlazioni e' sufficiente?

Il paper originale ragiona sul sample-to-feature ratio: 60/14 circa 4.3, poi 60/5 = 12 dopo feature selection.

Noi possiamo aggiungere un punto metodologico:

> Lo SFR calcolato sulle righe puo' essere fuorviante se le righe non sono indipendenti. Il numero effettivo di condizioni sperimentali indipendenti e' inferiore al numero di osservazioni.

Analisi:

- SFR nominale: training rows / features;
- SFR effettivo: independent conditions / features;
- domain-aware SFR: number of membranes / membrane descriptors;
- replicate-aware SFR.

---

### RQ5 — La synthetic augmentation migliora davvero la generalizzazione fuori dominio?

Questa e' una domanda opzionale ma molto interessante.

Il paper originale mostra miglioramenti con synthetic augmentation su random test. Noi possiamo chiedere:

> Se il test e' leave-one-membrane-out, la synthetic augmentation aiuta ancora?

Questa analisi va fatta con molta cautela per non trasformare il paper in una replica del loro.

Possibili livelli:

- livello minimo: discussione teorica;
- livello intermedio: esperimento con una o due semplici augmentation;
- livello completo: replicare MMD-controlled GC e testarla in grouped/LOMO.

Consiglio: inserirla come estensione, non come nucleo centrale.

---

### RQ6 — Quali nuovi esperimenti servirebbero per ridurre il gap di generalizzazione?

Questa domanda porta verso active learning/sequential experimental design.

Se il modello fallisce su membrane nuove, la risposta non e' solo cambiare algoritmo, ma raccogliere nuovi dati in modo intelligente.

Strategie:

- uncertainty sampling;
- diversity sampling;
- membrane-balanced sampling;
- pH-pressure coverage;
- design of experiments guidato dal modello.

---

## 8. Dataset: lettura aggiornata

### 8.1 Forma

- 80 righe originali;
- 15 colonne totali;
- 14 feature;
- 1 target;
- 1 duplicato esatto;
- 79 osservazioni distinte;
- 46 condizioni sperimentali indipendenti;
- 4 membrane;
- 2 regimi di feed.

### 8.2 Target

Il paper originale usa `RVFA = Cret / Cfeed`. Nel nostro dataset disponibile la colonna e' indicata come `Cret total VFAs`.

Dato che `Cfeed` nel paper originale e' quasi costante, `Cret` e `RVFA` sono sostanzialmente proporzionali. Tuttavia, nel nostro paper bisogna essere precisi:

- se usiamo direttamente la colonna disponibile, chiamiamola `Cret total VFAs`;
- se vogliamo allinearci al paper originale, possiamo indicare che `RVFA` e' proporzionale a `Cret` dato il feed quasi costante;
- evitare confusione tra rejection, concentration ratio e concentrazione assoluta.

### 8.3 Feature groups

#### Proprietà di membrana

- MWCO;
- average surface roughness;
- zeta potential;
- static contact angle;
- MgSO4 rejection;
- NaCl rejection;
- PWP.

#### Variabili di processo

- pH;
- temperature;
- pressure.

#### Feed chemistry

- monovalent cation;
- divalent cation;
- monovalent anion;
- divalent anion.

### 8.4 Variabili derivate da creare

- `membrane_id`;
- `feed_type`: simple/model solution vs complex/real matrix;
- `condition_id`: combinazione di membrana, pH, temperature, pressure, feed composition;
- `replicate_group_id`;
- `domain_id`: equivalente a membrana;
- `is_duplicate`;
- `is_replicate`.

---

## 9. Ipotesi metodologiche da verificare

### H1 — Random split sovrastima la generalizzazione

Motivo:

- repliche e condizioni simili possono finire sia in train sia in test;
- le membrane viste in training compaiono anche in test;
- il modello non deve trasferire a nuovi domini.

### H2 — Grouped split riduce ma non elimina l'ottimismo

Motivo:

- le condizioni sono separate;
- ma le membrane sono ancora condivise tra train e test.

### H3 — Leave-one-membrane-out e' il test piu' severo

Motivo:

- misura trasferimento a dominio nuovo;
- evidenzia aliasing delle feature di membrana;
- mostra se il modello ha imparato relazioni generali o solo idiosincrasie.

### H4 — I modelli parsimoniosi generalizzano meglio fuori dominio

Motivo:

- meno dipendenza dai descrittori di membrana;
- meno possibilita' di usare feature come ID di dominio;
- maggiore robustezza.

### H5 — Feature importance puo' cambiare radicalmente tra protocolli

Motivo:

- SHAP/permutation importance su random split puo' riflettere correlazioni locali;
- leave-one-domain-out puo' mostrare che alcune feature non sono trasferibili.

---

## 10. Protocolli di validazione da implementare

### 10.1 Protocollo A — Random split 75/25

Serve a replicare concettualmente il setting del paper originale.

Uso:

- baseline comparativa;
- non deve essere il claim principale.

Output:

- performance media su molti seed;
- varianza della performance;
- rischio leakage stimato.

### 10.2 Protocollo B — Repeated random split

Serve a mostrare la variabilita' dello split casuale.

Impostazione:

- 100 o 1000 ripetizioni;
- train/test 75/25;
- salvataggio della composizione dei test set.

Analisi:

- distribuzione R²/RMSE;
- quante volte test contiene repliche di condizioni train;
- quante membrane appaiono in train e test.

### 10.3 Protocollo C — GroupKFold per condition_id

Serve a misurare interpolazione onesta.

Vincolo:

- repliche della stessa condizione sempre nello stesso fold.

Domanda:

> Il modello predice nuove condizioni sperimentali entro membrane gia' viste?

### 10.4 Protocollo D — Leave-One-Membrane-Out

Serve a misurare estrapolazione/domain generalization.

Vincolo:

- una membrana intera lasciata fuori.

Domanda:

> Il modello predice una membrana mai vista?

### 10.5 Protocollo E — Leave-One-Feed-Regime-Out

Opzionale.

Domanda:

> Il modello addestrato su feed semplice generalizza a feed complesso, o viceversa?

Probabilmente difficile, ma utile per mostrare limiti del dataset.

### 10.6 Protocollo F — Nested validation

Opzionale ma metodologicamente forte.

Uso:

- se facciamo tuning;
- necessario per evitare tuning leakage.

---

## 11. Modelli da usare

Il paper non deve diventare una gara tra algoritmi. I modelli servono a dimostrare il ruolo della validazione.

### 11.1 DummyRegressor

Predice la media.

Scopo:

- baseline minima;
- definire se i modelli imparano segnale reale.

### 11.2 Ridge completo

Feature:

- tutte le feature numeriche o feature selezionate;
- standardizzazione dentro il fold.

Scopo:

- modello semplice ma sensibile all'estrapolazione;
- utile per mostrare collasso su membrana nuova.

### 11.3 Ridge parsimonioso

Feature consigliate:

- pH;
- pressure;
- feed_type;
- eventualmente MWCO o membrane_id.

Scopo:

- testare se meno feature aiutano a generalizzare.

### 11.4 Process-only model

Feature:

- pH;
- pressure;
- temperature;
- feed_type.

Scopo:

- isolare il segnale operativo;
- verificare se il trasferimento fuori dominio dipende piu' dalle condizioni di processo che dalle feature di membrana.

### 11.5 CatBoost baseline

Inserire CatBoost per collegarsi al paper originale.

Uso:

- non come protagonista;
- come riferimento diretto al paper sorgente;
- possibilmente senza synthetic augmentation nella prima fase.

Attenzione:

- tuning dentro CV;
- profondita' limitata;
- confronto su protocolli corretti.

### 11.6 RandomForest / ExtraTrees

Uso:

- modello non lineare robusto;
- non estrapola linearmente fuori range;
- utile come confronto.

### 11.7 Gaussian Process Regression

Uso:

- solo se vogliamo aprire la parte uncertainty/active learning;
- meglio su feature set parsimonioso o process-only.

Kernel:

- RBF o Matern;
- WhiteKernel;
- ARD opzionale.

### 11.8 Synthetic augmentation models

Non centrali.

Possibili usi:

1. non usarli nella prima versione;
2. citare il paper originale come lavoro su augmentation;
3. testare una augmentation semplice solo come sanity check;
4. testare se synthetic augmentation aiuta anche in leave-one-membrane-out.

Consiglio: rimandare la parte synthetic a una sezione secondaria o future work, a meno che non emergano risultati molto forti.

---

## 12. Feature set da confrontare

### FS1 — Full

Tutte le feature disponibili.

Domanda:

> Quanto si ottiene massimizzando l'informazione disponibile?

### FS2 — Original-paper selected

Feature selezionate dal paper originale:

- zeta potential;
- pH;
- pressure;
- PWP;
- monovalent anion feed.

Domanda:

> Il feature set del paper originale resta robusto sotto validazione domain-aware?

### FS3 — Process-only

- pH;
- pressure;
- temperature;
- feed_type.

Domanda:

> Quanto del segnale e' generalizzabile senza descrittori di membrana?

### FS4 — Membrane-only

- MWCO;
- roughness;
- zeta;
- contact angle;
- MgSO4 rejection;
- NaCl rejection;
- PWP.

Domanda:

> Le feature di membrana predicono o identificano il dominio?

### FS5 — Parsimonious generalization

- pH;
- pressure;
- feed_type;
- MWCO oppure membrane_id.

Domanda:

> Qual e' il compromesso tra accuratezza e robustezza?

### FS6 — No-zeta

Rimuovere zeta.

Domanda:

> Quanto zeta e' ridondante rispetto a pH e membrana?

### FS7 — No-membrane-properties

Rimuovere tutte le proprieta' statiche di membrana.

Domanda:

> Il modello puo' generalizzare usando solo condizioni di processo e feed?

---

## 13. Analisi da fare prima della modellazione

### 13.1 Data audit

Output obbligatori:

- numero righe;
- duplicati;
- osservazioni distinte;
- condizioni indipendenti;
- gruppi di replica;
- membrane;
- feed regimes;
- valori unici per feature;
- missing values;
- target distribution.

### 13.2 Effective sample size

Calcolare:

- nominal N = 80;
- distinct N = 79;
- condition N = 46;
- domain N = 4;
- training N effettivo per ogni protocollo.

Messaggio:

> Il numero di righe sovrastima l'informazione indipendente disponibile.

### 13.3 Correlazioni e collinearita'

Calcolare:

- Pearson;
- Spearman;
- VIF;
- condition number;
- hierarchical clustering delle feature.

Messaggio:

> Alcune feature sono statisticamente ridondanti e sperimentalmente aliasate.

### 13.4 Domain aliasing

Per ogni feature, calcolare:

- quanti valori unici ha per membrana;
- quanto varia dentro membrana;
- quanto varia tra membrane;
- intra-domain variance / inter-domain variance.

Output:

- tabella `feature_domain_aliasing.csv`.

---

## 14. Figure aggiornate del paper

### Figura 1 — Original vs proposed question

Tipo: schema concettuale.

Pannello A: paper originale

```text
80 real samples -> feature selection -> synthetic augmentation -> CatBoost -> random test R²
```

Pannello B: nostro paper

```text
80 samples -> audit experimental structure -> grouped validation -> interpolation/extrapolation gap -> design recommendations
```

Messaggio:

> Non stiamo replicando il paper originale: stiamo cambiando la domanda di generalizzazione.

---

### Figura 2 — Dataset structure

Contenuto:

- 80 righe;
- 79 distinte;
- 46 condizioni;
- 4 membrane;
- 2 feed;
- repliche.

Possibile forma:

- diagramma a blocchi;
- Sankey semplice;
- tile plot righe x feature groups.

Messaggio:

> Il dataset non e' i.i.d.; e' un disegno sperimentale strutturato.

---

### Figura 3 — Target and mechanism

Contenuto:

- target distribution;
- pH effect;
- zeta vs pH per membrana;
- target vs zeta colorato per pH.

Messaggio:

> Esiste un segnale chimicamente interpretabile, dominato da pH/zeta/pressure, gia' coerente con il paper originale.

Nota:

Questa figura deve essere secondaria. Non e' la novita' principale.

---

### Figura 4 — Validation protocols

Figura centrale.

Mostrare visivamente:

1. random split;
2. grouped-by-condition split;
3. leave-one-membrane-out split.

Messaggio:

> Cambiare split significa cambiare domanda scientifica.

---

### Figura 5 — Performance gap across validation protocols

Figura piu' importante.

Contenuto:

- barplot R² e RMSE;
- modelli: CatBoost, Ridge full, Ridge parsimonious, process-only, RF/GPR;
- protocolli: random, grouped condition, leave-one-membrane-out.

Messaggio:

> Performance alta in random/interpolation non implica generalizzazione a nuova membrana.

---

### Figura 6 — Predicted vs observed by protocol

Contenuto:

- scatter y_true vs y_pred;
- pannello random;
- pannello grouped;
- pannello leave-one-membrane-out;
- colore = membrana o pH.

Messaggio:

> Gli errori aumentano e diventano sistematici fuori dominio.

---

### Figura 7 — Feature set ablation

Contenuto:

- heatmap: feature set x validation protocol;
- valore = R² o RMSE;
- evidenziare full vs original-paper-selected vs process-only.

Messaggio:

> Le feature che aiutano in random split non sono necessariamente quelle che generalizzano.

---

### Figura 8 — Feature importance instability

Contenuto:

- permutation importance o SHAP ranking per protocollo;
- confronto random vs grouped vs leave-one-membrane-out.

Messaggio:

> La feature importance non e' assoluta: dipende dal protocollo e dal dominio testato.

---

### Figura 9 — Effective sample size and SFR

Contenuto:

- confronto tra N nominale, N distinto, N condizioni, N domini;
- SFR nominale vs effective SFR;
- feature count originale, selezionato, parsimonioso.

Messaggio:

> Lo SFR basato sulle righe puo' essere ottimistico in dataset con repliche e domini.

---

### Figura 10 — Active learning / next experiments

Opzionale.

Contenuto:

- mappa pH-pressure;
- regioni scoperte;
- punti suggeriti;
- learning curves simulate.

Messaggio:

> La diagnosi di generalizzazione puo' guidare la raccolta di nuovi dati.

---

## 15. Tabelle aggiornate

### Tabella 1 — Paper positioning

| Dimensione | Paper originale | Nostro paper |
|---|---|---|
| Obiettivo | Synthetic augmentation | Validation-aware generalization |
| Modello | CatBoost | Multiple simple diagnostic models |
| Split | Random 75/25 | Grouped/domain-aware |
| Output | Higher R² | Meaning of R² |
| Interpretabilita' | SHAP consistency | Stability and aliasing |

### Tabella 2 — Dataset audit

Colonne:

- N originale;
- N distinto;
- N condizioni;
- N membrane;
- N feed;
- duplicati;
- repliche;
- target range.

### Tabella 3 — Feature groups and risks

Colonne:

- feature;
- family;
- unit;
- selected by original paper?;
- unique values;
- within-membrane variability;
- aliasing risk;
- role in our models.

### Tabella 4 — Validation protocols

Colonne:

- protocol;
- split unit;
- scientific question;
- leakage risk;
- expected optimism;
- use in paper.

### Tabella 5 — Model results

Colonne:

- model;
- feature set;
- protocol;
- R²;
- RMSE;
- MAE;
- Spearman;
- notes.

### Tabella 6 — Leave-one-membrane-out details

Colonne:

- held-out membrane;
- test N;
- target range;
- best model;
- worst model;
- RMSE;
- systematic bias;
- interpretation.

### Tabella 7 — Effective SFR

Colonne:

- feature set;
- feature count;
- nominal training rows;
- independent training conditions;
- nominal SFR;
- effective SFR;
- domain count;
- warning level.

### Tabella 8 — Active learning candidates

Opzionale.

Colonne:

- candidate condition;
- pH;
- pressure;
- temperature;
- feed;
- membrane;
- uncertainty;
- diversity score;
- rationale.

---

## 16. Struttura aggiornata del manoscritto

### Abstract

Struttura consigliata:

1. piccolo dataset sperimentale e rischio di overestimated ML performance;
2. menzione del dataset VFA gia' usato in un paper AI su synthetic augmentation;
3. nostra prospettiva: validation-aware reanalysis;
4. confronto tra random split, grouped split e leave-one-membrane-out;
5. risultato: performance alta non equivale sempre a domain generalization;
6. contributo: checklist/pipeline per small experimental tabular data.

Esempio di abstract skeleton:

> Machine learning models are increasingly used to predict outcomes in data-scarce experimental sciences. However, in small structured datasets, predictive performance can be strongly affected by how training and test samples are split. We revisit a recently published VFA nanofiltration dataset, originally used to study CatBoost and controlled synthetic augmentation, from a validation-centric perspective. We compare random splits, condition-grouped cross-validation, and leave-one-membrane-out evaluation to distinguish interpolation from domain-level extrapolation. Our results show that models with high random-test accuracy can degrade substantially when tested on unseen membrane domains, while parsimonious process-based models provide more stable but less accurate predictions. These findings highlight that validation design, effective sample size, and feature-domain aliasing are critical for interpreting predictive claims in small experimental tabular data.

---

### 1. Introduction

Punti:

- ML in environmental science e experimental sciences;
- dataset piccoli, costosi, strutturati;
- rischio di usare random split come default;
- synthetic augmentation promettente ma non sufficiente se la validazione non misura la domanda giusta;
- dataset VFA come caso studio;
- differenza tra original study e nostra reanalysis;
- contributi.

Contributi da elencare:

1. audit strutturale del dataset;
2. confronto tra protocolli di validazione;
3. diagnosi di interpolation/extrapolation gap;
4. analisi di feature aliasing e effective sample size;
5. raccomandazioni per small experimental tabular ML;
6. possibile active learning extension.

---

### 2. Background and related work

Sottosezioni:

1. Small-data machine learning in scientific domains;
2. Data splitting and leakage in structured datasets;
3. Grouped cross-validation and domain generalization;
4. Feature selection and effective sample size;
5. Synthetic data augmentation for tabular scientific data;
6. Active learning for experimental design.

Nota:

Qui il paper originale va introdotto non come concorrente, ma come motivazione:

> A recent study used this dataset to investigate controlled synthetic augmentation. We ask a complementary question: how should predictive performance be interpreted when the dataset contains replicates and domain structure?

---

### 3. Dataset and original modeling context

Contenuti:

- descrivere dataset;
- descrivere brevemente il paper originale;
- spiegare perche' noi facciamo una reanalysis;
- chiarire che non stiamo contestando l'obiettivo originale;
- chiarire che il nostro obiettivo e' diverso.

Questa sezione e' fondamentale per evitare l'impressione di copia.

---

### 4. Structural audit of the dataset

Contenuti:

- righe e feature;
- duplicato;
- repliche;
- condizioni indipendenti;
- membrane;
- feed;
- domain aliasing;
- effective SFR;
- target distribution.

Figure:

- Figura 2 dataset structure;
- Figura 9 effective SFR.

---

### 5. Validation protocols

Contenuti:

- random split;
- repeated random split;
- group-by-condition;
- leave-one-membrane-out;
- eventuale leave-one-feed-regime-out;
- definizione matematica dell'unita' di generalizzazione.

Figura:

- Figura 4 validation protocols.

---

### 6. Models and feature sets

Contenuti:

- modelli;
- feature sets;
- preprocessing;
- tuning;
- metriche;
- gestione della feature selection dentro fold;
- salvataggio predizioni.

Da specificare:

- scaling dentro ogni fold;
- nessuna feature selection globale fuori CV;
- tuning nested se necessario;
- repliche mai separate nei protocolli grouped.

---

### 7. Results

Sottosezioni:

#### 7.1 Random-test performance reproduces high apparent predictability

Mostrare che, con split simile al paper originale, si ottengono performance alte.

Tono:

- non dire che il paper originale e' sbagliato;
- dire che questo protocollo misura una certa forma di generalizzazione: nuovi punti casuali dalla stessa struttura sperimentale.

#### 7.2 Grouped validation reduces optimistic bias

Mostrare che le repliche contano.

#### 7.3 Leave-one-membrane-out reveals domain generalization limits

Risultato centrale.

#### 7.4 Parsimonious and process-only models degrade more gracefully

Mostrare full vs parsimonious.

#### 7.5 Feature importance depends on validation protocol

Mostrare instabilita' di SHAP/permutation.

#### 7.6 Optional: synthetic augmentation under domain-aware validation

Solo se facciamo questa estensione.

---

### 8. Discussion

Punti:

1. un alto R² non ha significato unico;
2. random split misura interpolazione locale;
3. grouped split misura interpolazione onesta;
4. leave-one-domain-out misura trasferimento a domini nuovi;
5. synthetic augmentation puo' migliorare un protocollo ma non necessariamente risolvere domain shift;
6. feature importance puo' essere confusa da aliasing;
7. nei dataset sperimentali piccoli, il disegno sperimentale e' parte del modello.

---

### 9. Practical checklist

Questa puo' essere un contributo molto utile.

Checklist proposta:

1. identificare repliche;
2. identificare condizioni indipendenti;
3. identificare domini sperimentali;
4. calcolare effective sample size;
5. definire l'unita' di generalizzazione;
6. scegliere split coerenti;
7. confrontare random vs grouped;
8. non interpretare SHAP senza analisi di aliasing;
9. usare synthetic augmentation solo dopo avere definito lo scenario di validazione;
10. progettare nuovi esperimenti per coprire i domini deboli.

---

### 10. Active learning extension

Da includere come sezione o future work.

Possibili obiettivi:

- ridurre incertezza su membrane nuove;
- migliorare copertura pH-pressure;
- scegliere nuove membrane;
- disambiguare feature aliasate;
- aumentare effective sample size.

---

### 11. Limitations

Da dichiarare in modo esplicito:

1. dataset piccolo;
2. solo 4 membrane;
3. solo 46 condizioni indipendenti;
4. reanalysis di dataset esistente;
5. non validiamo su nuovi esperimenti reali;
6. active learning eventualmente retrospettivo;
7. non replichiamo necessariamente tutta la synthetic augmentation del paper originale;
8. conclusioni metodologiche da verificare su altri dataset.

---

### 12. Conclusion

Messaggio finale:

> The main lesson is not that one model is best, but that validation design determines what a predictive score means. In small experimental tabular datasets, random-test performance can be a poor proxy for scientific generalization. A validation-aware workflow can reveal whether a model is interpolating known experimental domains, exploiting replicated conditions, or truly transferring to unseen domains.

---

## 17. Esperimenti aggiornati da eseguire

### Esperimento 0 — Reproduction context

Obiettivo:

- documentare cosa fa il paper originale;
- non necessariamente replicare tutta la pipeline;
- usare il loro setup come baseline concettuale.

Output:

- tabella confronto paper originale vs nostro setup;
- breve sezione nel manoscritto.

### Esperimento 1 — Data audit e gruppi

Task:

- rimuovere duplicato;
- creare condition_id;
- creare membrane_id;
- creare feed_type;
- creare replicate_group_id;
- stimare rumore da repliche.

Output:

- `data/processed/seed_with_groups.csv`;
- `results/tables/data_audit.csv`;
- Figura 2.

### Esperimento 2 — Random split sensitivity

Task:

- repeated random split 75/25;
- modelli: CatBoost, Ridge, RF, process-only;
- misurare varianza performance;
- misurare leakage potenziale.

Output:

- distribuzione performance;
- tabella random split;
- confronto con risultati del paper originale.

### Esperimento 3 — GroupKFold per condition_id

Task:

- repliche nello stesso fold;
- modelli e feature set;
- metriche fold-by-fold.

Output:

- performance di interpolazione onesta;
- scatter predicted vs observed.

### Esperimento 4 — Leave-one-membrane-out

Task:

- lasciare fuori ogni membrana;
- valutare modelli;
- analizzare errori sistematici.

Output:

- performance per membrana;
- figura performance gap;
- risultato centrale.

### Esperimento 5 — Feature set ablation

Task:

- confrontare FS1-FS7;
- protocolli B/C/D;
- valutare stabilita'.

Output:

- heatmap ablation;
- tabella feature set.

### Esperimento 6 — Feature importance stability

Task:

- permutation importance dentro fold;
- SHAP opzionale per CatBoost;
- confronto ranking tra protocolli.

Output:

- ranking stability plot;
- discussione su aliasing.

### Esperimento 7 — Effective SFR analysis

Task:

- calcolare SFR nominale e effettivo;
- confrontare full, selected, process-only;
- discutere differenza tra sample count e information count.

Output:

- Figura 9;
- Tabella 7.

### Esperimento 8 — Synthetic augmentation under grouped validation

Opzionale.

Task minimo:

- implementare semplice Gaussian copula o bootstrap perturbation;
- testare solo random vs grouped vs leave-one-membrane-out;
- verificare se il miglioramento random si mantiene fuori dominio.

Task avanzato:

- replicare GC MMD-controlled;
- confrontare con baseline CatBoost;
- valutare MMD non solo rispetto alla distribuzione globale ma anche rispetto ai domini.

Nota:

Questa estensione puo' diventare molto interessante, ma aumenta il rischio di sovrapporsi al paper originale.

### Esperimento 9 — Active learning simulation

Opzionale/estensione.

Task:

- usare GPR o ensemble per incertezza;
- simulare acquisizione retrospettiva;
- confrontare random, uncertainty, diversity, domain-balanced.

Output:

- learning curves;
- next experiment candidates;
- future work forte.

---

## 18. Linguaggio e stack di sviluppo aggiornati

### 18.1 Scelta consigliata: Python

Alla luce del paper originale, Python resta la scelta principale.

Motivi:

1. il paper originale usa Python 3.11, quindi usare Python facilita confronto e riproducibilita';
2. scikit-learn offre direttamente GroupKFold, LeaveOneGroupOut, Pipeline, GridSearchCV;
3. CatBoost ha ottimo supporto Python;
4. synthetic data e MMD sono piu' facili da integrare in Python;
5. active learning, GPR e Bayesian optimization sono piu' naturali in Python;
6. una venue informatica si aspetta spesso pipeline ML in Python.

### 18.2 Pacchetti consigliati

Core:

```text
python>=3.11
numpy
pandas
scipy
scikit-learn
statsmodels
matplotlib
seaborn opzionale solo per esplorazione, non necessario per figure finali
catboost
joblib
tqdm
pyyaml
```

Nota: per le figure finali del paper usare preferibilmente `matplotlib` puro o uno stile controllato.

Per interpretabilita':

```text
shap
sklearn.inspection
```

Per Gaussian process e active learning:

```text
scikit-learn
botorch opzionale
gpytorch opzionale
```

Per synthetic data opzionale:

```text
sdv opzionale
copulas opzionale
```

Per qualita' codice:

```text
ruff
black
pytest
mypy opzionale
```

### 18.3 Ruolo possibile di R

R resta utile, ma non come linguaggio principale.

Usi sensati:

- grafici esplorativi con ggplot2;
- modelli statistici/misti;
- analisi ANOVA o regression diagnostics;
- report Quarto;
- validazione indipendente dei risultati.

Tuttavia, per questo paper, Python e' piu' coerente con:

- confronto con paper originale;
- workflow ML;
- grouped validation;
- CatBoost;
- active learning;
- reproducible ML repository.

---

## 19. Repository aggiornata

```text
seed-vfa-validation-reanalysis/
├── README.md
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── Seed_Dataset.csv
│   │   └── original_paper_metadata.md
│   ├── processed/
│   │   ├── seed_clean.csv
│   │   ├── seed_with_groups.csv
│   │   └── condition_groups.csv
│   └── synthetic/                  # opzionale
├── notebooks/
│   ├── 01_original_paper_context.ipynb
│   ├── 02_data_audit.ipynb
│   ├── 03_eda_structure.ipynb
│   ├── 04_random_split_sensitivity.ipynb
│   ├── 05_grouped_validation.ipynb
│   ├── 06_leave_membrane_out.ipynb
│   ├── 07_feature_ablation.ipynb
│   ├── 08_feature_importance_stability.ipynb
│   ├── 09_synthetic_grouped_optional.ipynb
│   └── 10_active_learning_optional.ipynb
├── src/
│   └── seed_vfa/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── groups.py
│       ├── splits.py
│       ├── features.py
│       ├── models.py
│       ├── evaluation.py
│       ├── diagnostics.py
│       ├── importance.py
│       ├── synthetic.py
│       ├── active_learning.py
│       └── plotting.py
├── scripts/
│   ├── run_data_audit.py
│   ├── run_random_split.py
│   ├── run_grouped_cv.py
│   ├── run_leave_membrane_out.py
│   ├── run_ablation.py
│   ├── run_importance_stability.py
│   ├── run_synthetic_optional.py
│   └── make_figures.py
├── results/
│   ├── tables/
│   ├── figures/
│   ├── predictions/
│   ├── splits/
│   └── logs/
├── paper/
│   ├── manuscript.tex
│   ├── references.bib
│   ├── figures/
│   └── tables/
└── tests/
    ├── test_groups.py
    ├── test_splits.py
    ├── test_no_leakage.py
    └── test_metrics.py
```

---

## 20. File di output fondamentali

### Predizioni fold-by-fold

`results/predictions/all_predictions.csv`

Colonne:

- row_id;
- original_row_id;
- condition_id;
- replicate_group_id;
- membrane_id;
- feed_type;
- protocol;
- split_id;
- fold;
- model;
- feature_set;
- y_true;
- y_pred;
- residual;
- uncertainty;
- is_duplicate_removed.

### Metriche aggregate

`results/tables/model_metrics.csv`

Colonne:

- protocol;
- model;
- feature_set;
- R2_mean;
- R2_std;
- RMSE_mean;
- RMSE_std;
- MAE_mean;
- MAE_std;
- n_train;
- n_test;
- n_conditions_train;
- n_conditions_test;
- n_domains_train;
- n_domains_test.

### Leakage audit

`results/tables/leakage_audit.csv`

Colonne:

- split_id;
- protocol;
- condition_overlap;
- replicate_overlap;
- membrane_overlap;
- feed_overlap;
- leakage_warning.

---

## 21. Punti di attenzione etica e scientifica

### 21.1 Non presentare il lavoro come se il dataset fosse nostro

Va dichiarato chiaramente che il dataset proviene dal paper originale ed e' stato reso disponibile dagli autori.

### 21.2 Non attaccare gli autori

Il tono deve essere:

> complementary perspective, validation-centric reanalysis

non:

> their work is wrong.

### 21.3 Distinguere obiettivi diversi

Il loro obiettivo:

> synthetic augmentation for CatBoost under data scarcity.

Il nostro obiettivo:

> interpretation of predictive performance under experimental structure.

### 21.4 Controllare licenza e condizioni d'uso

Prima di pubblicare:

- verificare licenza del paper;
- verificare licenza del dataset;
- citare dataset e paper originale;
- se necessario, chiedere conferma agli autori per riuso e redistribuzione.

---

## 22. Minimal viable paper aggiornato

Se vogliamo chiudere una versione snella, fare:

1. citazione e descrizione del paper originale;
2. audit strutturale del dataset;
3. random split sensitivity;
4. GroupKFold per condition_id;
5. leave-one-membrane-out;
6. full vs original-selected vs process-only;
7. performance gap;
8. effective SFR;
9. checklist finale.

Non includere ancora:

- active learning;
- synthetic augmentation replicata;
- GPR avanzato;
- SHAP esteso.

Titolo per versione minima:

**Beyond Random Splits: Validation-Aware Reanalysis of a Small VFA Nanofiltration Dataset**

---

## 23. Versione estesa aggiornata

Aggiungere:

1. feature importance stability;
2. synthetic augmentation sotto grouped/LOMO;
3. uncertainty-aware GPR;
4. active learning;
5. raccomandazioni per nuovi esperimenti;
6. confronto con MMD globale vs MMD domain-aware.

Titolo per versione estesa:

**From Synthetic Accuracy to Scientific Generalization: Validation-Aware Modeling and Experiment Selection for Small VFA Nanofiltration Data**

---

## 24. Roadmap aggiornata

### Fase 1 — Reframing e bibliografia mirata

Durata: 3-5 giorni.

Task:

- leggere con attenzione il paper originale;
- estrarre metodologia, split, feature selection, risultati;
- costruire tabella original vs ours;
- cercare letteratura su grouped CV, leakage, small data, domain generalization.

Output:

- sezione positioning;
- bib iniziale.

### Fase 2 — Audit del dataset

Durata: 2-3 giorni.

Task:

- duplicati;
- repliche;
- condition_id;
- domain_id;
- effective sample size.

Output:

- Figura 2;
- Tabella 2;
- Tabella 7.

### Fase 3 — Random split sensitivity

Durata: 3-5 giorni.

Task:

- repeated random split;
- CatBoost/Ridge/RF;
- confronto con paper originale.

Output:

- distribuzione performance;
- leakage audit.

### Fase 4 — Grouped e LOMO validation

Durata: 5-7 giorni.

Task:

- GroupKFold condition;
- leave-one-membrane-out;
- feature sets.

Output:

- Figura 5;
- Figura 6;
- Tabella 5;
- Tabella 6.

### Fase 5 — Ablation e importance stability

Durata: 5-7 giorni.

Task:

- feature set ablation;
- permutation importance;
- SHAP opzionale.

Output:

- Figura 7;
- Figura 8.

### Fase 6 — Paper draft minima

Durata: 7-10 giorni.

Task:

- abstract;
- introduction;
- dataset;
- validation protocols;
- results;
- discussion;
- checklist.

Output:

- prima bozza completa.

### Fase 7 — Estensioni opzionali

Durata: 1-3 settimane.

Possibili:

- synthetic augmentation domain-aware;
- active learning;
- GPR uncertainty;
- secondo dataset.

---

## 25. Nuova checklist finale

Prima di considerare il lavoro pronto:

- [ ] Il paper originale e' citato e posizionato correttamente.
- [ ] Il nostro contributo non si sovrappone alla synthetic augmentation.
- [ ] E' chiaro che stiamo facendo una validation-aware reanalysis.
- [ ] Il duplicato e' rimosso prima degli split.
- [ ] Le repliche sono identificate.
- [ ] Le condizioni indipendenti sono identificate.
- [ ] Le membrane sono trattate come domini.
- [ ] Random split e grouped split sono separati.
- [ ] Leave-one-membrane-out e' incluso.
- [ ] CatBoost e' usato come riferimento, non come unico protagonista.
- [ ] La feature selection non avviene fuori dalla CV.
- [ ] Scaling e tuning avvengono dentro fold.
- [ ] Le metriche sono salvate fold-by-fold.
- [ ] Si riporta RMSE/MAE oltre a R².
- [ ] Si discute effective sample size.
- [ ] Si discute feature aliasing.
- [ ] Non si afferma che SHAP dimostri causalita'.
- [ ] Non si dice che il paper originale e' sbagliato.
- [ ] Si distingue chiaramente interpolazione da estrapolazione.
- [ ] La conclusione riguarda il significato della validazione, non il modello migliore.

---

## 26. Conclusione aggiornata del progetto

Il dataset resta utile, ma non per un paper generico di predizione ML. Il paper originale ha gia' coperto la parte di AI modeling applicativo, synthetic augmentation, CatBoost, MMD e SHAP.

La nostra opportunita' e' diversa e piu' informatica:

> usare lo stesso dataset come caso di studio per mostrare che, nei piccoli dataset sperimentali, la validazione definisce il significato della performance. Un alto R² su split random puo' indicare interpolazione entro domini gia' visti, non generalizzazione a nuovi domini sperimentali. La pipeline proposta deve quindi identificare repliche, condizioni, domini, effective sample size e protocolli di split coerenti con la domanda scientifica.

Questa e' una storia solida per un paper metodologico, soprattutto se formulata come contributo di **data-centric machine learning**, **validation-aware modeling**, **small tabular regression** e **domain generalization in experimental science**.
