"""
Fix numpy pickle loading issue by re-saving with current numpy version
"""
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("=" * 70)
print("Fixing NumPy Pickle Compatibility")
print("=" * 70)
print(f"Current NumPy version: {np.__version__}")

# Try to load and re-save pickle files
for pkl_file in ['scaler.pkl', 'label_encoder.pkl']:
    print(f"\n📁 Processing {pkl_file}...")
    
    if not os.path.exists(pkl_file):
        print(f"❌ File not found: {pkl_file}")
        continue
    
    try:
        # Try loading with different methods
        obj = None
        
        # Method 1: Try with numpy compatibility
        try:
            import pickle5  # For Python 3.7 compatibility
            with open(pkl_file, 'rb') as f:
                obj = pickle5.load(f)
            print(f"✅ Loaded with pickle5")
        except:
            pass
        
        # Method 2: Try with encoding
        if obj is None:
            try:
                with open(pkl_file, 'rb') as f:
                    obj = pickle.load(f, encoding='latin1')
                print(f"✅ Loaded with latin1 encoding")
            except:
                pass
        
        # Method 3: Try with protocol
        if obj is None:
            try:
                with open(pkl_file, 'rb') as f:
                    obj = pickle.load(f, fix_imports=True)
                print(f"✅ Loaded with fix_imports=True")
            except Exception as e:
                print(f"❌ Could not load: {str(e)[:200]}")
                continue
        
        # Re-save with current numpy version
        backup_name = pkl_file + '.backup'
        import shutil
        shutil.copy(pkl_file, backup_name)
        print(f"📦 Backup created: {backup_name}")
        
        # Save with current protocol
        with open(pkl_file, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✅ Re-saved {pkl_file} with current NumPy version")
        
        # Verify it can be loaded
        with open(pkl_file, 'rb') as f:
            test_obj = pickle.load(f)
        print(f"✅ Verified: {pkl_file} can be loaded")
        
    except Exception as e:
        print(f"❌ Error processing {pkl_file}: {str(e)}")

print("\n" + "=" * 70)
print("✅ Done! Try loading the app again.")
print("=" * 70)


