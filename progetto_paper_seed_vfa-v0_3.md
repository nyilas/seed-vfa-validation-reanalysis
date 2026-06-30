# Documento di progetto v0.3 — Replicate-anchored reanalysis del dataset Seed/VFA

**File principale:** `Seed_Dataset.csv`
**Paper sorgente del dataset:** Cairone et al., *Artificial intelligence for predicting volatile fatty acids rejection in nanofiltration membranes*, Journal of Water Process Engineering 82 (2026) 109412 (open access, CC-BY).
**Target:** `Cret total VFAs` (numericamente identico a `RVFA` del paper sorgente: media 2.66, range 1.35–4.44, std 0.72).
**Angolazione:** re-analisi informatica/metodologica centrata su (1) dipendenza della performance dal protocollo di validazione e (2) dipendenza della spiegazione dall'identificabilità del disegno, ancorate a un pavimento di rumore derivato dalle repliche.
**Venue primaria:** TMLR (Transactions on Machine Learning Research).
**Documento gemello:** `AGENTS.md` (specifica esecutiva per l'agente di sviluppo). Questo documento è la fonte della *strategia*; `AGENTS.md` è la fonte della *implementazione*. In caso di conflitto su un dettaglio scientifico, vince questo documento; su un dettaglio implementativo, vince `AGENTS.md`.

---

## 0. Cosa cambia da v0.2 a v0.3 (e perché)

| Cambiamento | Motivo |
|---|---|
| Il **pavimento di rumore** passa da nota di data-audit a **risultato di testa** (Figura 1). | È l'unico metro *indipendente dallo split*; trasforma il leakage da sospetto a dimostrato. Vedi §2. |
| I due pilastri (performance-come-artefatto e spiegazione-come-artefatto) diventano **co-protagonisti**. | Il pilastro "interpolazione ≠ generalizzazione" ha ora un fratello pubblicato (Li et al. 2025, §6); appoggiarsi solo su quello è rischioso. L'esperimento di identificabilità pH/zeta è la dimostrazione più pulita e meno battuta. |
| Active learning, synthetic-under-LOMO e GPR-uncertainty **escono dal nucleo** e diventano future work. | Non validabili su 4 membrane; erano una passività, non un punto di forza. Lo scope ristretto è il paper. Vedi §5. |
| Titolo consigliato **ritirato e sostituito**. | "Interpolation Is Not Generalization" collide con un titolo Nature Comm. Materials 2025. Vedi §9. |
| Aggiunto **posizionamento puntuale del prior art**. | Né il leakage né l'instabilità dell'importanza sono nuovi: la novità è il *pacchetto*. Va dichiarato per non farsi attaccare. Vedi §6. |
| Aggiunta **strategia a due paper** e tono diplomatico verso SEED. | Decisione presa: il lavoro non è bloccato dal paper sorgente; due paper distinti, venue distinte. Vedi §8. |

---

## 1. La tesi, in una frase

> Su piccoli dataset sperimentali strutturati (DOE), sia il *punteggio* di performance sia la *spiegazione* che si riportano sono artefatti di scelte raramente dichiarate — la partizione di cross-validation e la (non)identificabilità del disegno. Lo dimostriamo congiuntamente su un dataset, calibrando tutto rispetto a un pavimento di rumore irriducibile derivato dalle repliche, e impacchettiamo diagnostiche calcolabili che segnalano entrambi i fallimenti prima che ingannino.

La nanofiltrazione è il banco di prova; l'oggetto di studio è il **protocollo di valutazione** e il **metodo di attribuzione**. Questo rende il lavoro informatico, non ingegneristico.

---

## 2. L'asset centrale: il paper sorgente è il caso di studio, il pavimento di rumore è la prova

Il paper sorgente non è solo il motivo del reframing: è un campione reale, peer-reviewed (gennaio 2026), delle due patologie che vogliamo diagnosticare. Tre fatti, dal paper:

1. **Split random 75/25 con repliche presenti.** Il paper dichiara di aver evitato il leakage, ma quella dichiarazione riguarda la separazione sintetico/reale, *non* il fatto che repliche della stessa condizione sperimentale finiscano metà in train e metà in test. Con 46 condizioni indipendenti e repliche quasi sempre in coppia, uno split random le separa quasi sistematicamente → leakage da unità sperimentale.
2. **Feature selection (Pearson + RFE) apparentemente globale**, fuori dalla CV → selection leakage (seconda via, secondaria).
3. **pH e zeta tenuti entrambi** (soglia di correlazione 0.7; pH–ζ = −0.61 passa sotto), poi SHAP li classifica #1 e #2 e il risultato è presentato come "coerente con la conoscenza di dominio" = validazione. Ma pH e zeta sono fisicamente accoppiati e statisticamente aliasati: è il fenomeno "la feature importance mente sotto aliasing".

### Il gancio del pavimento di rumore (da verificare sui dati, ma già eloquente)

- Pavimento di rumore dalle repliche: **RMSE ≈ 0.23** (irriducibile: nessun modello onesto può fare meglio del rumore con cui l'esperimento riproduce se stesso).
- Baseline del paper: R² = 0.885, RMSE = 0.283 → std implicita del loro test set ≈ 0.835.
- Miglior modello aumentato del paper: R² = 0.937 → RMSE implicito ≈ 0.835·√(1−0.937) ≈ **0.21**, cioè **pari o inferiore al pavimento di rumore**.

Un modello che predice meglio del rumore sperimentale è logicamente impossibile senza leakage (o senza un test set anomalo). Questo è il perno: **il leakage non si argomenta, si dimostra**. La logica è inattaccabile perché il pavimento non dipende dallo split.

> ⚠️ **Da fare subito (vedi `AGENTS.md`, Exp-A):** ricalcolare il pavimento e gli RMSE in modo esatto sui dati reali. Sto ricostruendo gli RMSE dall'R² aggregato e da una varianza di test inferita. Se il risultato regge — e quasi certamente regge — è la Figura 1. Se per qualche ragione non regge, il paper *non crolla*: il pilastro (1) e il (2) restano in piedi sul confronto random vs grouped vs LOMO e sull'esperimento di identificabilità.

---

## 3. Le due dimostrazioni (co-protagoniste)

### Demo 1 — La performance dipende dal protocollo

Stesso modello, si fa scorrere la partizione: **random k-fold** (repliche divise → leaky) → **grouped-by-condition** (interpolazione onesta) → **leave-one-membrane-out** (estrapolazione a dominio nuovo). Si mostra lo spettro completo: gonfiato → onesto → collasso. Due linee di riferimento orizzontali ovunque: modello banale (RMSE ≈ 0.72) e pavimento di rumore (RMSE ≈ 0.23). Sotto-risultato che vale da solo: i modelli ad albero (RandomForest) **non** esplodono in estrapolazione perché non estrapolano oltre il range visto — *quale* classe di modelli fallisce e *come* è esso stesso un reperto.

- **Metrica chiave:** *leakage gap* = RMSE(grouped) − RMSE(random) e *extrapolation gap* = RMSE(LOMO) − RMSE(grouped).
- **Messaggio:** il numero alto sotto split random misura interpolazione locale (più leakage), non trasferimento a un dominio nuovo.

### Demo 2 — La spiegazione dipende dall'identificabilità

L'esperimento pulito è il **drop-one-of-an-aliased-pair** su pH e zeta (r ≈ −0.61, praticamente un asse solo). Si toglie lo zeta → l'importanza migra sul pH; si rimette lo zeta e si toglie il pH → migra sullo zeta. La conclusione "il pH conta, lo zeta no" è un artefatto del tie-breaking, non un fatto. Lo si mostra su quattro metodi che *dovrebbero* concordare e non concordano: coefficienti standardizzati, permutation importance, SHAP, e le **ARD length-scale di un GPR** (che assegnerà length-scale corta a uno dei due e lunga all'altro in modo sostanzialmente arbitrario).

- **Metrica chiave:** *importance-stability score* — quanto si muove il ranking di attribuzione tra fold/semi e tra metodi, ristretto ai gruppi aliasati.
- **Messaggio:** la coincidenza tra l'importanza stimata e la conoscenza di dominio non è validazione se le feature sono aliasate; può essere un artefatto del disegno.

> Le due facce: *il tuo punteggio dipende dallo split, la tua spiegazione dipende dall'identificabilità*. Entrambe pienamente dimostrabili su un solo dataset, senza punti deboli di validazione.

---

## 4. Il pavimento di rumore come ancora logica (riusabile)

Il pavimento di rumore è il contributo concettuale trasferibile più forte. R² e RMSE sono metriche *relative*: senza riferimento non sai se 0.31 è buono o gonfiato. Le repliche danno un riferimento che **non dipende dallo split**: nessun modello può scendere sotto il rumore con cui l'esperimento riproduce se stesso. Questo converte la diagnosi da "guarda come cambia R²" (debole) a "questo RMSE implica una precisione sotto il rumore sperimentale → leakage necessario" (logico).

Concetto non nuovo in metrologia/chemiometria (errore irriducibile, Bayes error da repliche); la novità è l'**uso come rilevatore di leakage** nel contesto small-data ML, ancorato a numeri pubblicati.

---

## 5. Disciplina di scope: cosa è dentro, cosa è fuori

**Dentro (il paper = Minimal Viable Paper):**
1. Audit strutturale del dataset (duplicato, repliche, condizioni, membrane, feed).
2. Pavimento di rumore dalle repliche (Figura 1).
3. Demo 1: random → grouped → LOMO, con leakage gap ed extrapolation gap.
4. Demo 2: identificabilità pH/zeta, importanza su 4 metodi, importance-stability.
5. Effective sample size / effective SFR.
6. Bundle diagnostico riusabile (§7) + rilascio dataset documentato.

**Fuori (future work, una riga ciascuno nel manoscritto, niente esperimenti nel nucleo):**
- Active learning / disegno sequenziale. *Non validabile su 4 membrane; passività, non punto di forza.*
- Synthetic augmentation sotto grouped/LOMO. *Rischia di sovrapporsi al paper sorgente; tienila come domanda aperta.*
- GPR uncertainty / calibrazione contro il pavimento. *Elegante ma di nicchia; il GPR entra solo come strumento nella Demo 2 (ARD ingannato dall'aliasing), non come contributo.*

> Regola operativa per l'agente e per te: **se un esperimento non serve a Demo 1, Demo 2 o al pavimento di rumore, non va nella v1.**

---

## 6. Posizionamento rispetto al prior art (onesto)

Nulla rende il lavoro obsoleto, ma né il pilastro (1) né il (2) sono nuovi come *fenomeni*. La novità è il pacchetto. Ancoraggi obbligatori:

- **Kapoor & Narayanan (Patterns, 2023)** — survey canonica del leakage, tassonomia di 8 tipi, 17 campi, ~329 paper. → Citare come fondamento. "Il leakage esiste" non è il nostro contributo; noi instanziamo il tipo *no-grouping/replicate leakage* su un claim appena pubblicato.
- **Rashomon / model multiplicity / variable importance sotto correlazione** — Fisher–Rudin–Dominici (Model Class Reliance), Donnelly et al. (Rashomon Importance Distribution), Verdinelli & Wasserman su correlazione che distorce SHAP/LOCO/permutation. → Il pilastro (2) come fenomeno è noto; il nostro valore è instanziarlo su pH/zeta e collegarlo direttamente alla validazione SHAP del paper sorgente.
- **Li et al., Communications Materials 6, 9 (2025), "Probing OOD generalization in ML for materials"** — fratello più vicino del pilastro (1): la maggior parte dei test OOD riflette interpolazione, non estrapolazione → generalizzabilità sovrastimata. *Non* ci rende obsoleti (dominio diverso, larga scala, niente repliche/pavimento, niente identificabilità, niente claim pubblicato specifico, niente small-DOE), ma ci toglie il diritto di presentare "interpolazione ≠ generalizzazione" come tesi inedita. → Citare in apertura; spostare il titolo.
- **Letteratura "split prima di aumentare / grouped CV per campioni dipendenti"** (SMOTE-before-split, blocked/grouped CV) — ormai da manuale. → Citare; non rivendicare.

**Dichiarazione di novità difendibile (da mettere quasi verbatim nell'intro):** un caso di studio riproducibile su un claim appena pubblicato (2026); il fallimento *congiunto* di performance e spiegazione sullo stesso piccolo dataset DOE; il pavimento di rumore come metro assoluto indipendente dallo split che trasforma il leakage in dimostrazione; un bundle diagnostico tagliato sui piccoli dati DOE; un dataset rilasciato documentato con struttura a domini e repliche.

---

## 7. Il contributo riusabile: il bundle diagnostico

La differenza tra "aneddoto" e "contributo" è un artefatto che gira su *altri* dataset. Il bundle, da rilasciare come codice + checklist:

1. **Rilevatore di gruppi scambiabili / aliasing:** condition number del disegno, VIF, e quali insiemi di feature sono indistinguibili (le 4 membrane → 4 punti nello spazio delle proprietà; pH≡zeta).
2. **Stimatore del pavimento di rumore** dalle repliche → RMSE target assoluto.
3. **Leakage gap:** RMSE(grouped) − RMSE(random).
4. **Extrapolation gap:** RMSE(LOMO) − RMSE(grouped).
5. **Importance-stability score:** spostamento del ranking di attribuzione tra fold/semi/metodi, ristretto ai gruppi aliasati.

Più il **rilascio del dataset** con `condition_id`, `membrane_id`, `replicate_group_id`, `feed_type` — di per sé un contributo citabile e a basso rischio in venue data-centric.

---

## 8. Strategia di venue e rapporto con SEED

### 8.1 Venue
- **Primaria: TMLR.** Accettazione basata sulla correttezza delle affermazioni, non su novelty/impatto; submission a rotazione, review brevi. Su misura per un case study rigoroso che *esplicitamente non* rivendica un modello nuovo: neutralizza l'attacco "non è abbastanza nuovo".
- **Sbocchi a valle da TMLR:** MLRC (traccia riproducibilità ufficiale a NeurIPS via TMLR — finestra stretta, da verificare se ancora praticabile per il 2026) e Journal-to-Conference (presentazione a NeurIPS/ICLR/ICML con certificazione).
- **NeurIPS Evaluations & Datasets Track:** scope perfetto (accoglie risultati negativi, analisi critiche, valutazione come oggetto di studio), ma deadline 2026 già passata → bersaglio 2027.
- **Inquadramento:** scrivere il manoscritto come *reproducibility / stress-test di un claim pubblicato*, così è MLRC-eligibile e J2C-eligibile insieme.

### 8.2 Rapporto con SEED (decisione presa)
- Il dataset proviene da SEED ma **non è il laboratorio dell'autore**; il lavoro non deve essere bloccato dal paper sorgente.
- **Strategia a due paper:** il paper informatico (questo) su venue CS; un eventuale paper ingegneristico/chimico (loro) su venue ambientali. Obiettivi e venue distinti, nessuna sovrapposizione.
- **Tono:** rigoroso nei risultati, diplomatico nel packaging. I risultati tecnici (incluso il pavimento-di-rumore-sotto-soglia) restano, ma formulati come **proprietà del protocollo e del dataset**, non come "il loro numero è sbagliato". Frasi tipo: *"We revisit ... under a different validation question"*, *"This is not a refutation of the original engineering contribution, which targeted a different objective"*, *"Under random partitioning that does not group replicates, ..."*.
- **Igiene:** citare paper e dataset sorgente; verificare licenza/condizioni di riuso e redistribuzione; dichiarare che il dataset non è nostro.

---

## 9. Titoli candidati (il vecchio è ritirato)

Ritirato: *"Interpolation Is Not Generalization"* (collisione con Li et al. 2025).

Candidati nuovi (attorno al fallimento congiunto + identificabilità + ancora di rumore):
1. **When the Score and the Explanation Both Mislead: A Replicate-Anchored Reanalysis of a Small Experimental Tabular Claim**
2. **Below the Noise Floor: Replicate-Anchored Diagnostics for Leakage and Identifiability in Small Experimental Regression**
3. **What a High R² Hides: Validation Protocol and Design Identifiability in Small-Data Scientific Regression**
4. **The Replicate Noise Floor as an Absolute Yardstick for Leakage in Small Tabular Science**

---

## 10. Struttura del manoscritto (lean)

1. **Introduction** — small-data scientific ML; il claim pubblicato come motivazione; la doppia tesi; contributi (la dichiarazione di §6).
2. **Background & related work** — Kapoor–Narayanan; grouped/blocked CV; Rashomon/MCR/RID e importanza sotto correlazione; Li et al. 2025; synthetic-before-split.
3. **Dataset and original modeling context** — descrizione; il setup del paper sorgente come baseline concettuale; perché una reanalysis (tono §8.2).
4. **The replicate noise floor** — definizione, calcolo, uso come riferimento assoluto (Figura 1). *Risultato di testa.*
5. **Demo 1: performance is an artifact of the protocol** — random/grouped/LOMO; leakage gap; extrapolation gap; comportamento per classe di modelli.
6. **Demo 2: explanation is an artifact of identifiability** — drop-one-of-aliased-pair pH/zeta; 4 metodi; importance-stability.
7. **A diagnostic bundle for small experimental tabular data** — i 5 strumenti di §7 + checklist.
8. **Discussion** — interpolazione vs estrapolazione; il disegno è parte del modello; limiti.
9. **Limitations** — 4 membrane, 46 condizioni, reanalysis, niente validazione su nuovi esperimenti reali.
10. **Conclusion** — la validazione definisce il significato del punteggio; le repliche danno il metro assoluto.

---

## 11. Figure del MVP

- **Fig. 1 — Noise floor & leakage:** RMSE per protocollo con le due linee (banale 0.72, rumore 0.23); evidenziare dove il modello scende sotto il rumore.
- **Fig. 2 — Dataset structure:** 80 righe → 79 distinte → 46 condizioni → 4 membrane → 2 feed (non i.i.d., è un DOE).
- **Fig. 3 — Performance gap:** R²/RMSE per modello × protocollo (random, grouped, LOMO).
- **Fig. 4 — Predicted vs observed by protocol:** scatter, colore = membrana, errori sistematici fuori dominio.
- **Fig. 5 — Identifiability:** migrazione dell'importanza pH↔zeta nel drop-one, su 4 metodi.
- **Fig. 6 — Effective SFR:** N nominale vs distinto vs condizioni vs domini; SFR nominale vs effettivo.

---

## 12. Roadmap

| Fase | Durata | Output |
|---|---|---|
| F1 — Audit + noise floor | 3–4 gg | `seed_with_groups.csv`, `data_audit.csv`, pavimento di rumore verificato, Fig. 1–2 |
| F2 — Demo 1 (protocolli) | 5–7 gg | metriche fold-by-fold, leakage/extrapolation gap, Fig. 3–4 |
| F3 — Demo 2 (identificabilità) | 5–7 gg | importance su 4 metodi, importance-stability, Fig. 5 |
| F4 — Bundle + effective SFR | 3–4 gg | bundle diagnostico, Fig. 6, Tabella SFR |
| F5 — Draft TMLR | 7–10 gg | bozza completa lean |
| F6 — Rilascio dataset | 2–3 gg | dataset documentato + scheda licenza |

---

## 13. Obiezioni dei revisori da pre-emptare

- *"Il leakage è noto."* → Sì; il contributo è il pacchetto + il fallimento congiunto + il pavimento come prova + il dataset. Dichiarato in intro.
- *"Un solo dataset."* → Inquadrato come case study + benchmark rilasciato, non come pipeline universale; il secondo dataset è una mossa di rafforzamento, non un prerequisito.
- *"State attaccando un paper specifico."* → No: stiamo rispondendo a una domanda di validazione diversa; il contributo ingegneristico originale aveva un altro obiettivo (tono §8.2).
- *"L'instabilità dell'importanza è nota (Rashomon)."* → Sì; la instanziamo sull'aliasing di disegno e la colleghiamo a una validazione interpretativa pubblicata, con uno score riusabile.

---

## 14. Puntatore all'implementazione

Tutta l'implementazione (invarianti anti-leakage, costruzione gruppi, protocolli, modelli, schemi di output, test di accettazione) è in **`AGENTS.md`**. Le decisioni scientifiche di questo documento sono vincolanti per quel file.
