# 📦 Système d'Archivage Automatique

## Vue d'ensemble

Le système d'archivage sauvegarde automatiquement tous les résultats de vos runs précédents, permettant de comparer différentes configurations et d'analyser l'évolution de votre modèle.

## Fonctionnement Automatique

### Lors de chaque exécution de `run_pipeline.py` :

1. **Archive automatique** : Les résultats précédents sont sauvegardés dans `archives/run_YYYYMMDD_HHMMSS/`
2. **Nouveau run** : Le modèle s'entraîne normalement
3. **Nouveaux résultats** : Remplacent les anciens dans `outputs/`

### Contenu d'une archive :

```
archives/run_20251209_193045/
├── archive_info.json          # Métadonnées de l'archive
├── results.json                # Résultats complets du run
├── models/
│   └── stage_a_mlp.keras     # Modèle sauvegardé
└── plots/                      # Toutes les visualisations
    ├── training_history.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── ...
```

## Utilisation

### 1. Lancer un nouveau run (avec archivage automatique)

```bash
python run_pipeline.py
```

### 2. Analyser les résultats actuels et comparer

```bash
python analyze_results.py
```

Affiche :
- Métriques du run actuel
- Comparaison automatique avec le run précédent
- Liste des 5 archives les plus récentes

### 3. Gérer les archives manuellement

**Lister toutes les archives :**
```bash
python scripts/archive_manager.py --list
```

**Comparer le run actuel avec le plus récent :**
```bash
python scripts/archive_manager.py --compare
```

**Comparer avec une archive spécifique :**
```bash
python scripts/archive_manager.py --compare archives/run_20251209_193045
```

**Comparer deux archives entre elles :**
```bash
python scripts/archive_manager.py --compare archives/run_20251209_193045 archives/run_20251209_184512
```

**Archiver manuellement les résultats actuels :**
```bash
python scripts/archive_manager.py --archive
```

## Exemples de Comparaison

### Sortie typique de comparaison :

```
================================================================================
COMPARISON ENTRE DEUX RUNS
================================================================================

Run 1: 2025-12-09T18:39:54.073717
Run 2: 2025-12-09T19:45:23.156832

--------------------------------------------------------------------------------
Métrique              Run 1           Run 2      Différence
--------------------------------------------------------------------------------
Test Accuracy        0.5773          0.5942         +0.0169
Test AUC             0.5752          0.5885         +0.0133
Test F1              0.6234          0.6401         +0.0167
Epochs                   13              8              -5
Threshold            0.500           0.430         -0.070
--------------------------------------------------------------------------------

💡 RÉSUMÉ:
   ✅ Amélioration de l'accuracy: +1.69%
   ✅ Amélioration de l'AUC: +0.0133
================================================================================
```

## Structure des Dossiers

```
NBA_Games_Predictions_ML_NN/
├── outputs/                    # Résultats du run actuel
│   ├── results.json
│   └── plots/
├── archives/                   # Tous les runs archivés
│   ├── run_20251209_183045/
│   ├── run_20251209_190512/
│   └── run_20251209_193045/
└── models/                     # Modèle actuel
    └── stage_a_mlp.keras
```

## Scénarios d'Usage

### Expérimentation avec différentes configurations

1. **Baseline** : Run avec paramètres par défaut → archivé automatiquement
2. **Test régularisation** : Modifier L2=0.005 → `python run_pipeline.py` → archivé
3. **Test dropout** : Modifier dropout=0.5 → `python run_pipeline.py` → archivé
4. **Comparer** : `python analyze_results.py` pour voir l'évolution

### Retrouver le meilleur modèle

```bash
# Lister toutes les archives avec leurs métriques
python scripts/archive_manager.py --list

# Copier le meilleur modèle
cp archives/run_20251209_193045/models/stage_a_mlp.keras models/best_model.keras
```

### Analyser une régression

Si un nouveau run est moins performant :
```bash
# Comparer avec l'archive précédente
python scripts/archive_manager.py --compare latest

# Restaurer l'ancien modèle si nécessaire
cp archives/run_20251209_193045/models/stage_a_mlp.keras models/stage_a_mlp.keras
```

## Métadonnées Sauvegardées

Chaque archive contient un fichier `archive_info.json` :

```json
{
  "archive_timestamp": "20251209_193045",
  "archive_date": "2025-12-09T19:30:45.123456",
  "archived_items": [
    "results.json",
    "plots/ (10 files)",
    "models/stage_a_mlp.keras"
  ],
  "original_results": {
    "timestamp": "2025-12-09T19:28:32.073717",
    "epochs_trained": 8,
    "test_accuracy": 0.5942,
    "test_auc": 0.5885,
    "test_f1": 0.6401,
    "optimal_threshold": 0.430
  }
}
```

## Nettoyage

Les archives peuvent s'accumuler. Pour nettoyer :

```bash
# Garder seulement les 10 dernières
python -c "from pathlib import Path; import shutil; [shutil.rmtree(p) for p in sorted(Path('archives').glob('run_*'))[:-10]]"
```

Ou manuellement :
```bash
rm -r archives/run_20251201_*  # Supprimer les archives de décembre 1
```

## Conseils

- ✅ **Ne supprimez jamais** le dossier `archives/` entier
- ✅ **Documentez vos expérimentations** : Ajoutez des notes dans un fichier `experiments.md`
- ✅ **Gardez au moins 5-10 archives** pour tracer l'évolution
- ✅ **Avant une modification majeure**, lancez d'abord `python run_pipeline.py` pour archiver l'état actuel

## Intégration avec Git

Le `.gitignore` devrait contenir :
```
archives/
outputs/
models/
```

Pour partager une archive spécifique :
```bash
# Compresser une archive
tar -czf run_20251209_193045.tar.gz archives/run_20251209_193045/

# Ou avec zip
Compress-Archive -Path archives\run_20251209_193045 -DestinationPath run_20251209_193045.zip
```
