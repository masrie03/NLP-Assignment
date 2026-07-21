# arXiv Spelling Correction System — Partie A (CT052-3-M-NLP)

Système probabiliste et contextuel de correction orthographique, entraîné sur
le corpus **arXiv Paper Abstracts** (51 774 papers, ~9,7 millions de mots
après nettoyage, 32 498 mots uniques retenus).

## Structure du projet

```
spellcheck_project/
├── data/
│   └── raw/
│       └── arxiv_data.csv        # Corpus brut (titles, summaries, terms)
├── models/
│   ├── unigrams.pkl               # dict[str, int]           mot -> fréquence
│   └── bigrams.pkl                # dict[str, dict[str,int]] w1 -> {w2: freq}
├── src/
│   ├── preprocessing.py           # Étape 1 : nettoyage + construction dictionnaires
│   ├── nlp_engine.py               # Étape 2 : Levenshtein + détection Non-word/Real-word
│   └── gui_app.py                  # Étape 3 : interface Tkinter
└── README.md
```

## Comment exécuter

### 1. Installer les dépendances
```bash
pip install pandas
```
(`tkinter` et `pickle` sont inclus dans la bibliothèque standard de Python.)

### 2. (Re)générer les dictionnaires à partir du corpus brut
```bash
cd src
python preprocessing.py
```
Cela régénère `models/unigrams.pkl` et `models/bigrams.pkl`. **Cette étape n'a
besoin d'être relancée que si tu modifies le corpus brut ou les règles de
nettoyage** — l'application GUI charge directement les fichiers `.pkl` pour
un démarrage instantané.

### 3. Lancer l'interface graphique
```bash
python gui_app.py
```

### 4. (Optionnel) Tester le moteur seul, en ligne de commande
```bash
python nlp_engine.py
```
Affiche des exemples de distance de Levenshtein, de candidats Non-word et de
détection contextuelle Real-word.

## Choix techniques (à documenter dans le rapport)

- **Nettoyage** : suppression des blocs LaTeX (`$...$`, `$$...$$`) avant
  tokenisation, pour éviter que la syntaxe mathématique ne pollue le
  vocabulaire scientifique.
- **Seuil de fréquence minimale** : `Count >= 3` pour éliminer les
  coquilles isolées des chercheurs dans le corpus brut.
- **Distance de Levenshtein** : implémentation par programmation dynamique
  (matrice complète conservée pour la traçabilité dans le rapport),
  coûts uniformes (insertion = suppression = substitution = 1).
- **Lissage Add-k (Laplace)** : `P(w2|w1) = (count(w1,w2)+k) / (count(w1)+k·V)`
  avec `k = 1.0` et `V = 32 498` (taille du vocabulaire). Le seuil de
  détection Real-word (`5e-5`) a été calibré empiriquement sur ce corpus —
  voir les commentaires dans `nlp_engine.py` pour la justification.
- **Filtrage bigrams** : les transitions ne sont comptées qu'à l'intérieur
  d'un même titre ou d'un même résumé (jamais entre deux documents distincts).

## Limites connues (à mentionner dans la section discussion du rapport)

- Un mot Non-word suivi d'un mot valide peut provoquer une détection
  Real-word "en cascade" sur le mot suivant (car le bigramme
  `[mot_corrigé_implicite] -> [mot_suivant]` n'existe pas non plus). Ce
  comportement est visible et peut être discuté comme piste d'amélioration
  (ex: ignorer le contexte immédiatement après un Non-word détecté).
- Le seuil `5e-5` est un hyperparamètre fixe ; une version plus avancée
  pourrait le normaliser par mot (ex: percentile de la distribution des
  probabilités de bigrammes pour ce mot précédent).
