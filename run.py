import os
import sys
import pandas as pd

# 1. Add 'src' to the system path so Python can find the 'noshow_iq' package
sys.path.append(os.path.join(os.getcwd(), "src"))

# 2. Import your components from the package
# Note: Ensure the file in src/noshow_iq is named 'preprocessing.py' (with the 'ing')
from noshow_iq.preprocessing import preprocess_pipeline
from noshow_iq.model import train_and_evaluate

def main():
    # Define data path
    data_path = os.path.join("data", "KaggleV2-May-2016.csv")

    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return

    # --- PHASE 1: PREPROCESSING ---
    print("🚀 Phase 1: Preprocessing data...")
    try:
        df = preprocess_pipeline(data_path)
        print(f"   ✓ Preprocessing completed. Shape: {df.shape}")
        print(f"   ✓ Columns: {list(df.columns)}")
        print(f"   ✓ No-show distribution: {df['no_show'].value_counts().to_dict()}")
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- PHASE 2: TRAINING ---
    print("\n🚀 Phase 2: Training the model...")
    try:
        X = df.drop(columns=['no_show'])
        y = df['no_show']
        print(f"   ✓ Features shape: {X.shape}, Target shape: {y.shape}")

        # This calls the class logic we finalized earlier
        model_obj, metrics = train_and_evaluate(X, y, model_type='random_forest')
        print("   ✓ Model training completed")
        print(f"   ✓ Metrics: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"❌ Error in training: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- PHASE 3: SAVING ---
    print("\n🚀 Phase 3: Saving the model artifact...")
    try:
        # Saving to the root so api.py can find it easily
        model_obj.save_model("model.joblib")
        print("   ✓ Model saved successfully")
    except Exception as e:
        print(f"❌ Error in saving: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ PROJECT STATUS: Model trained and saved successfully!")
    print(f"Final F1-Score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()