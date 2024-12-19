import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Modul 1: Load Dataset
def load_dataset(file_path, input_columns, label_column, threshold):
    """
    Load dataset dari file CSV.
    Args:
        file_path (str): Path ke file CSV.
        input_columns (list): Kolom fitur input.
        label_column (str): Kolom target/label.
        threshold (float): Ambang batas untuk mendiskretisasi kolom target.
    Returns:
        X (DataFrame): Data fitur input.
        y (Series): Data target diskret.
    """
    df = pd.read_csv(file_path)
    print("Kolom yang tersedia:", df.columns)
    
    # Hapus baris dengan nilai NaN pada kolom input dan label
    df = df.dropna(subset=input_columns + [label_column])
    
    # Encoding kolom yang bertipe objek (string) menjadi numerik
    for col in input_columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    X = df[input_columns]
    y = (df[label_column] >= threshold).astype(int)  # Diskretisasi target menjadi 0 (rendah) atau 1 (tinggi)
    return X, y

# Modul 2: Split Data
def split_data(X, y, test_size=0.2):
    """
    Membagi data menjadi data training dan testing.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    print("Data training dan validasi telah dibuat.")
    return X_train, X_val, y_train, y_val

# Modul 3: Training Model (Random Forest)
def train_random_forest(X_train, y_train):
    """
    Melatih model Random Forest.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model Random Forest telah selesai dilatih.")
    return model

# Modul 4: Evaluasi Model
def evaluate_model(model, X_val, y_val):
    """
    Evaluasi model menggunakan data validasi.
    """
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print("\nHasil Evaluasi Model:")
    print(f"Accuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Modul 5: Plot Distribusi Data
def plot_data_distribution(y):
    """
    Menampilkan distribusi kelas pada data.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Distribusi Kelas pada Data")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah")
    plt.show()

# Path ke dataset
file_path = "D:\semester 3\praktikum KB\Project UAS KB\project numerik\soil_analysis_data.csv"

# Kolom input dan target yang disesuaikan dengan dataset
input_columns = ['District', 'Soil Type', 'Organic Matter (%)', 'Nitrogen Content (kg/ha)', 'Phosphorus Content (kg/ha)', 'Potassium Content (kg/ha)']
label_column = 'pH Level'  # Menyesuaikan dengan kolom yang ada dalam dataset
threshold = 7.0  # Ambang batas untuk mendiskretisasi pH Level menjadi tinggi dan rendah

# 1. Load dataset
print("Loading dataset...")
X, y = load_dataset(file_path, input_columns, label_column, threshold)

# 2. Plot distribusi data
plot_data_distribution(y)

# 3. Split data menjadi training dan testing
X_train, X_val, y_train, y_val = split_data(X, y)

# 4. Training model Random Forest
model = train_random_forest(X_train, y_train)

# 5. Evaluasi model
evaluate_model(model, X_val, y_val)
