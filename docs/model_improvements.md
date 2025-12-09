# Améliorations du Modèle MLP - NBA Game Predictions

## 📊 Analyse des Résultats Initiaux

Les résultats initiaux montraient des signes de **surapprentissage** :
- **Accuracy Train**: 68.3% vs **Test**: 57.7%
- **AUC Train**: 0.681 vs **Test**: 0.575
- Le modèle s'entraînait pendant 13 epochs avec patience=10

## ✅ Modifications Implémentées

### 1. **Réduction de la Patience (10 → 5)**
```python
patience = 5  # Au lieu de 10
```
**Objectif** : Arrêter l'entraînement plus tôt pour éviter le surapprentissage.

### 2. **Régularisation L2**
```python
kernel_regularizer=regularizers.l2(0.001)  # Sur toutes les couches Dense
```
**Objectif** : Pénaliser les poids trop importants, favorisant un modèle plus généralisable.

**Force de régularisation** : `l2_reg = 0.001` (léger)

### 3. **Dropout Layers**
```python
layers.Dropout(0.3)  # 30% de dropout après chaque couche cachée
```
**Objectif** : Désactiver aléatoirement 30% des neurones pendant l'entraînement pour améliorer la généralisation.

### 4. **ReduceLROnPlateau**
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # Divise le learning rate par 2
    patience=3,        # Après 3 epochs sans amélioration
    min_lr=1e-6
)
```
**Objectif** : Ajuster dynamiquement le taux d'apprentissage pour une convergence plus fine.

### 5. **Optimisation du Seuil de Décision**
```python
optimal_threshold = find_optimal_threshold(y_val, y_pred_proba, method='f1')
```
**Méthodes disponibles** :
- `'f1'` : Maximise le F1-score (balance précision/rappel) ✅ **Par défaut**
- `'youden'` : Maximise l'indice de Youden (sensibilité + spécificité - 1)
- `'balanced'` : Minimise la différence entre sensibilité et spécificité

**Objectif** : Trouver le seuil optimal (au lieu de 0.5) basé sur le set de validation pour maximiser la performance.

## 📈 Architecture Finale du Modèle

```
Input (n features)
    ↓
Dense(64, relu) + L2(0.001)
    ↓
Dropout(0.3)
    ↓
Dense(32, relu) + L2(0.001)
    ↓
Dropout(0.3)
    ↓
Dense(1, sigmoid)
```

## 🎯 Autres Propositions (Sans Changer de Modèle)

### A. **Augmentation de Données Temporelles**
- Créer des features de moyennes mobiles sur 10/15 jours
- Ajouter des features d'écart-type pour capturer la variance des performances

### B. **Feature Engineering Avancé**
- Ratio home/away win rates
- Streak features (nombre de victoires/défaites consécutives)
- Performance head-to-head entre équipes spécifiques
- Features de "back-to-back" games (fatigue)

### C. **Ensembling Léger**
- Entraîner 3-5 modèles avec différentes seeds
- Faire la moyenne des prédictions (bagging)

### D. **Ajuster l'Architecture**
- Tester différentes profondeurs : `[64, 64, 32]` ou `[128, 64, 32]`
- Expérimenter avec BatchNormalization au lieu de/avec Dropout
- Tester différentes fonctions d'activation (LeakyReLU, ELU)

### E. **Class Weighting**
Si déséquilibre de classes (environ 60% home wins) :
```python
class_weight = {0: 1.5, 1: 1.0}  # Pondérer les away wins
```

### F. **Cross-Validation Temporelle**
- Implémenter une validation croisée respectant l'ordre chronologique
- Entraîner sur plusieurs splits temporels pour robustesse

## 🚀 Utilisation

Le pipeline met automatiquement en œuvre toutes les améliorations :

```bash
python run_pipeline.py
```

Le modèle affichera :
1. La configuration (L2, Dropout, Patience)
2. Le seuil optimal trouvé sur le set de validation
3. Les performances avec seuil par défaut (0.5) ET seuil optimal
4. Les visualisations complètes dans `outputs/plots/`

## 📊 Métriques à Surveiller

- **F1 Score** : Balance entre précision et rappel
- **Gap Train-Test** : Indicateur de surapprentissage
- **AUC** : Performance indépendante du seuil
- **Confusion Matrix** : Distribution des erreurs

## 🔧 Paramètres Ajustables

Dans `scripts/train_model.py`, fonction `train_model()` :
```python
train_model(
    X_train, y_train, X_val, y_val,
    epochs=100,
    batch_size=64,
    patience=5,           # ← Ajustable
    l2_reg=0.001,        # ← Ajustable (0.0001 - 0.01)
    dropout_rate=0.3     # ← Ajustable (0.2 - 0.5)
)
```

## 📝 Notes

- Les visualisations incluent maintenant les courbes de learning rate si ReduceLROnPlateau est actif
- Le seuil optimal est sauvegardé dans `outputs/results.json`
- Les métriques incluent désormais le F1-score pour chaque set
