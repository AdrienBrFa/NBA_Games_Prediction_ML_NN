"""
Script pour comparer les performances avant/après les améliorations.
Peut aussi comparer avec les runs archivés.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Ajouter le répertoire scripts au path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.archive_manager import list_archives, compare_archives, print_comparison

# Charger les résultats actuels si disponibles
results_path = Path("outputs/results.json")

if results_path.exists():
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("="*80)
    print("RÉSUMÉ DES PERFORMANCES DU MODÈLE")
    print("="*80)
    
    # Afficher le seuil optimal
    if 'optimal_threshold' in results:
        print(f"\n🎯 Seuil optimal trouvé : {results['optimal_threshold']:.3f}")
    
    # Tableau comparatif
    print("\n📊 MÉTRIQUES PAR DATASET")
    print("-" * 80)
    print(f"{'Métrique':<20} {'Train':>15} {'Validation':>15} {'Test':>15}")
    print("-" * 80)
    
    metrics_to_show = ['accuracy', 'auc', 'f1_score', 'log_loss']
    
    for metric in metrics_to_show:
        train_val = results['train_metrics'].get(metric, 'N/A')
        val_val = results['val_metrics'].get(metric, 'N/A')
        test_val = results['test_metrics'].get(metric, 'N/A')
        
        if isinstance(train_val, (int, float)):
            print(f"{metric.upper():<20} {train_val:>15.4f} {val_val:>15.4f} {test_val:>15.4f}")
        else:
            print(f"{metric.upper():<20} {str(train_val):>15} {str(val_val):>15} {str(test_val):>15}")
    
    print("-" * 80)
    
    # Analyse du surapprentissage
    overfitting_gap = results['train_metrics']['accuracy'] - results['test_metrics']['accuracy']
    print(f"\n📈 Gap Train-Test (Accuracy): {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")
    
    if overfitting_gap < 0.05:
        print("   ✅ Excellent ! Peu de surapprentissage.")
    elif overfitting_gap < 0.10:
        print("   ⚠️  Surapprentissage modéré. Considérer plus de régularisation.")
    else:
        print("   ❌ Surapprentissage important. Augmenter la régularisation.")
    
    # Matrice de confusion test
    print("\n🎲 MATRICE DE CONFUSION (Test Set)")
    print("-" * 80)
    cm = results['test_metrics']['confusion_matrix']
    print(f"                    Predicted")
    print(f"                Away Win    Home Win")
    print(f"Actual  Away Win    {cm[0][0]:<8}    {cm[0][1]:<8}")
    print(f"        Home Win    {cm[1][0]:<8}    {cm[1][1]:<8}")
    
    # Calculer précision et rappel
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n   Précision (Home Win): {precision:.4f}")
    print(f"   Rappel (Home Win): {recall:.4f}")
    
    # Historique d'entraînement
    print("\n📚 HISTORIQUE D'ENTRAÎNEMENT")
    print("-" * 80)
    print(f"Nombre d'epochs: {results['epochs_trained']}")
    print(f"Loss finale (train): {results['training_history']['loss'][-1]:.4f}")
    print(f"Loss finale (val): {results['training_history']['val_loss'][-1]:.4f}")
    print(f"Accuracy finale (train): {results['training_history']['accuracy'][-1]:.4f}")
    print(f"Accuracy finale (val): {results['training_history']['val_accuracy'][-1]:.4f}")
    
    # Recommandations
    print("\n💡 RECOMMANDATIONS")
    print("="*80)
    
    if results['test_metrics']['accuracy'] < 0.60:
        print("1. L'accuracy est relativement faible (<60%)")
        print("   → Considérer plus de features ou feature engineering")
        print("   → Essayer un modèle plus profond [128, 64, 32]")
    
    test_f1 = results['test_metrics'].get('f1_score')
    test_acc = results['test_metrics']['accuracy']
    
    if test_f1 is not None and test_f1 < test_acc - 0.05:
        print("2. F1-score significativement inférieur à l'accuracy")
        print("   → Le modèle favorise une classe (probablement Home Win)")
        print("   → Essayer class_weight pour équilibrer")
    
    if overfitting_gap > 0.08:
        print("3. Surapprentissage détecté")
        print("   → Augmenter L2 reg (ex: 0.005 ou 0.01)")
        print("   → Augmenter Dropout (ex: 0.4 ou 0.5)")
        print("   → Réduire la taille du modèle")
    
    print("\n" + "="*80)
    
    # Comparer avec la dernière archive si disponible
    print("\n")
    archives = list_archives()
    if archives:
        print("🔄 COMPARAISON AVEC LE RUN PRÉCÉDENT")
        comparison = compare_archives('latest')
        if comparison:
            print_comparison(comparison)
    
else:
    print("❌ Aucun fichier results.json trouvé.")
    print("   Veuillez d'abord exécuter : python run_pipeline.py")

# Liste des archives disponibles
print("\n" + "="*80)
print("📦 ARCHIVES DISPONIBLES")
print("="*80)

archives = list_archives()
if archives:
    print(f"\nNombre total d'archives: {len(archives)}\n")
    for i, archive in enumerate(archives[:5], 1):  # Afficher les 5 plus récentes
        print(f"{i}. {archive['archive_timestamp']}")
        if archive.get('original_results'):
            res = archive['original_results']
            acc = res.get('test_accuracy')
            auc = res.get('test_auc')
            f1 = res.get('test_f1')
            if acc is not None:
                print(f"   Accuracy: {acc:.4f}", end="")
            if auc is not None:
                print(f" | AUC: {auc:.4f}", end="")
            if f1 is not None:
                print(f" | F1: {f1:.4f}", end="")
            print()
    
    if len(archives) > 5:
        print(f"\n... et {len(archives) - 5} autre(s) archive(s)")
    
    print("\nPour comparer des runs spécifiques:")
    print("  python scripts/archive_manager.py --compare")
    print("  python scripts/archive_manager.py --list")
else:
    print("\nAucune archive trouvée. Les résultats seront archivés au prochain run.")

print("="*80)
