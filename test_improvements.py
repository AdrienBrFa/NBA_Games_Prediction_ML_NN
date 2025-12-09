"""
Script de test rapide pour vérifier les améliorations du modèle.
"""

import sys
from pathlib import Path
import numpy as np

# Ajout du path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.train_model import find_optimal_threshold

# Test de la fonction d'optimisation du seuil
print("="*60)
print("TEST : Fonction d'optimisation du seuil de décision")
print("="*60)

# Créer des données fictives
np.random.seed(42)
y_true = np.random.randint(0, 2, 100)
y_pred_proba = np.random.rand(100)

# Tester les 3 méthodes
for method in ['f1', 'youden', 'balanced']:
    print(f"\nMéthode : {method}")
    threshold, metric = find_optimal_threshold(y_true, y_pred_proba, method=method)
    print(f"  → Seuil optimal : {threshold:.3f}")
    print(f"  → Métrique : {metric:.4f}")

print("\n" + "="*60)
print("✅ Test réussi ! Les fonctions sont opérationnelles.")
print("="*60)
print("\nVous pouvez maintenant lancer le pipeline complet :")
print("  python run_pipeline.py")
print("="*60)
