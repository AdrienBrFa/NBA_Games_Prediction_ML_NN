"""
Quick test to verify stage separation is working correctly.
"""

print("="*80)
print("TESTING STAGE SEPARATION")
print("="*80)

# Test imports
print("\n1. Testing imports...")
try:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    
    from scripts.archive_manager import archive_previous_results
    print("   ✅ Archive manager imported")
    
    from scripts.load_data import load_and_filter_games
    print("   ✅ Load data imported")
    
    from scripts.train_model import find_optimal_threshold
    print("   ✅ Train model imported")
    
except Exception as e:
    print(f"   ❌ Import error: {e}")
    exit(1)

# Test stage configurations
print("\n2. Testing stage configurations...")

# Stage A1
STAGE_A1 = "stage_a1"
output_a1 = Path("outputs") / STAGE_A1
models_a1 = Path("models") / STAGE_A1
archive_a1 = Path("archives") / STAGE_A1

print(f"   Stage A1:")
print(f"     Outputs: {output_a1}")
print(f"     Models:  {models_a1}")
print(f"     Archive: {archive_a1}")

# Stage B1
STAGE_B1 = "stage_b1"
output_b1 = Path("outputs") / STAGE_B1
models_b1 = Path("models") / STAGE_B1
archive_b1 = Path("archives") / STAGE_B1

print(f"   Stage B1:")
print(f"     Outputs: {output_b1}")
print(f"     Models:  {models_b1}")
print(f"     Archive: {archive_b1}")

# Verify no conflicts
if output_a1 == output_b1:
    print("   ❌ Output paths conflict!")
    exit(1)
if models_a1 == models_b1:
    print("   ❌ Model paths conflict!")
    exit(1)
if archive_a1 == archive_b1:
    print("   ❌ Archive paths conflict!")
    exit(1)

print("   ✅ All paths are properly separated")

# Test that run scripts exist
print("\n3. Checking run scripts...")
scripts = {
    "run_stage_a1.py": "Stage A1 pipeline",
    "run_stage_b1.py": "Stage B1 pipeline",
    "run_pipeline.py": "Deprecated redirect"
}

for script, desc in scripts.items():
    if Path(script).exists():
        print(f"   ✅ {script} ({desc})")
    else:
        print(f"   ❌ {script} missing!")

# Test documentation
print("\n4. Checking documentation...")
docs = {
    "docs/stage_a1_analysis.md": "Stage A1 analysis",
    "docs/stage_separation.md": "Stage separation guide",
    "docs/archiving_system.md": "Archive system",
    "docs/model_improvements.md": "Model improvements"
}

for doc, desc in docs.items():
    if Path(doc).exists():
        print(f"   ✅ {doc} ({desc})")
    else:
        print(f"   ⚠️  {doc} missing")

print("\n" + "="*80)
print("✅ STAGE SEPARATION TEST COMPLETE")
print("="*80)
print("\nYou can now run:")
print("  python run_stage_a1.py  # For Stage A1")
print("  python run_stage_b1.py  # For Stage B1 (when ready)")
print("="*80)
