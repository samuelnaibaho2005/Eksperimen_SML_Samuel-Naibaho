import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Memuat dataset dari file CSV
    Args:
        filepath (str): Path ke file CSV
        
    Returns:
        pd.DataFrame: Dataset yang sudah dimuat
    """
    print(f"Memuat dataset dari: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Shape dataset: {df.shape}")
    print(f"Kolom: {df.columns.tolist()}\n")
    return df


def check_missing_values(df):
    """
    Mengecek missing values dalam dataset
    
    Args:
        df (pd.DataFrame): Dataset input
        
    Returns:
        pd.DataFrame: Dataset tanpa perubahan
    """
    print("Cek Missing Value")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("Tidak ada missing values\n")
    else:
        print("Ada missing values :")
        print(missing[missing > 0])
        print()
        # Action: Drop missing values untuk memastikan data bersih
        df.dropna(inplace=True)
        print("  -> Missing values telah dihapus.")
    return df


def check_duplicates(df):
    """
    Mengecek data duplikat dalam dataset
    
    Args:
        df (pd.DataFrame): Dataset input
        
    Returns:
        pd.DataFrame: Dataset tanpa perubahan
    """
    print("Mengecek Data Duplikat")
    duplicates = df.duplicated().sum()
    print(f"Total duplikat: {duplicates}")
    
    # Khusus untuk kolom kategorikal
    categorical_cols = df.select_dtypes(include='object').columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            dup_in_col = df[col].duplicated().sum()
            if dup_in_col > 0:
                print(f"  • Duplikat di kolom '{col}': {dup_in_col}")
    print()
    
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"  -> {duplicates} baris duplikat telah dihapus.")
    return df


def describe_data(df):
    """
    Menampilkan statistik deskriptif dataset
    
    Args:
        df (pd.DataFrame): Dataset input
        
    Returns:
        pd.DataFrame: Dataset tanpa perubahan
    """
    print("Statistik Deskriptif")
    print(df.describe())
    print()
    return df


def drop_unnecessary_columns(df, columns_to_drop=['id', 'image']):
    """
    Menghapus kolom yang tidak diperlukan
    
    Args:
        df (pd.DataFrame): Dataset input
        columns_to_drop (list): List nama kolom yang akan dihapus
        
    Returns:
        pd.DataFrame: Dataset setelah kolom dihapus
    """
    print("Menhapus Kolom yang Penting")
    df_clean = df.copy()
    
    cols_to_remove = [col for col in columns_to_drop if col in df_clean.columns]
    
    if cols_to_remove:
        df_clean.drop(columns=cols_to_remove, inplace=True)
        print(f"Kolom dihapus: {cols_to_remove}")
        print(f"Shape sekarang: {df_clean.shape}\n")
    else:
        print("Tidak ada kolom yang perlu dihapus\n")
    
    return df_clean


def normalize_features(df, numeric_columns=None):
    """
    Melakukan normalisasi fitur numerik menggunakan MinMaxScaler
    
    Args:
        df (pd.DataFrame): Dataset input
        numeric_columns (list): List kolom numerik yang akan dinormalisasi (optional)
        
    Returns:
        tuple: (df_normalized, scaler, numeric_cols)
    """
    print("Normalisasi")
    
    # Jika tidak ditentukan, otomatis detect kolom numerik
    if numeric_columns is None:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        numeric_cols = [col for col in numeric_columns if col in df.columns]
    
    print(f"Kolom yang dinormalisasi: {numeric_cols}\n")
    
    # Inisialisasi MinMaxScaler
    min_max_scaler = MinMaxScaler()
    
    # Fit dan transform
    scaled_values = min_max_scaler.fit_transform(df[numeric_cols])
    
    # Buat salinan DataFrame dan update nilai kolom numerik
    df_normalized = df.copy()
    df_normalized[numeric_cols] = scaled_values
    
    print(f"Normalisasi selesai!")
    print(f"Range setiap kolom (harus 0-1):")
    for col in numeric_cols:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        print(f"    • {col}: [{min_val:.4f}, {max_val:.4f}]")
    print()
    
    return df_normalized, min_max_scaler, numeric_cols


def preprocessing_pipeline(input_filepath, output_filepath=None, 
                          columns_to_drop=['id', 'image'],
                          numeric_columns=None):
    """
    Pipeline lengkap untuk preprocessing data nutrisi
    
    Args:
        input_filepath (str): Path ke file CSV input
        output_filepath (str): Path untuk menyimpan hasil (optional)
        columns_to_drop (list): Kolom yang akan dihapus
        numeric_columns (list): Kolom numerik yang akan dinormalisasi (optional)
        
    Returns:
        tuple: (df_normalized, scaler, numeric_cols, metadata)
    """
    print("=" * 70)
    print("MEMULAI PIPELINE PREPROCESSING - DATASET NUTRISI INDONESIA")
    print("=" * 70)
    print()
    
    # 1. Load data
    df = load_data(input_filepath)
    
    # 2. Check missing values
    df = check_missing_values(df)
    
    # 3. Check duplicates
    df = check_duplicates(df)
    
    # 4. Describe data
    df = describe_data(df)
    
    # 5. Drop unnecessary columns
    df_clean = drop_unnecessary_columns(df, columns_to_drop)
    
    # 6. Normalize features
    df_normalized, scaler, numeric_cols = normalize_features(df_clean, numeric_columns)
    
    # 7. Save hasil preprocessing
    if output_filepath:
        print("Menyimpan Hasil")
        df_normalized.to_csv(output_filepath, index=False)
        print(f"File berhasil disimpan: {output_filepath}")
        print(f"Shape: {df_normalized.shape}\n")
    
    # 8. Metadata
    metadata = {
        'input_file': input_filepath,
        'output_file': output_filepath,
        'original_shape': df.shape,
        'final_shape': df_normalized.shape,
        'columns_dropped': columns_to_drop,
        'normalized_columns': numeric_cols,
        'scaler': scaler
    }
    
    print("=" * 70)
    print("PREPROCESSING SELESAI!")
    print("=" * 70)
    print(f"Dataset asli: {metadata['original_shape'][0]} rows × {metadata['original_shape'][1]} columns")
    print(f"Dataset akhir: {metadata['final_shape'][0]} rows × {metadata['final_shape'][1]} columns")
    print()
    
    return df_normalized, scaler, numeric_cols, metadata


def get_preprocessing_info():
    """
    Menampilkan informasi tentang preprocessing yang dilakukan
    
    Returns:
        None
    """
    info = """
    TAHAPAN PREPROCESSING:
    1. Load data dari file CSV
    2. Check missing values
    3. Check data duplikat
    4. Describe statistik data
    5. Drop kolom yang tidak diperlukan (id, image)
    6. Normalisasi fitur numerik dengan MinMaxScaler
       - Range 0-1 untuk setiap fitur
       - Outliers tetap dipertahankan (valid values)
    
    OUTPUT:
    - File CSV dengan data yang sudah dinormalisasi
    - Scaler object untuk digunakan di tahap modeling
    
    DATASET: Indonesian Food and Drink Nutrition Dataset
    SUMBER: Kaggle
    FITUR: calories, proteins, fat, carbohydrate
    """
    print(info)


#-----------------
# MAIN EXECUTION
#-----------------

if __name__ == "__main__":
    import os
    
    # Tampilkan informasi
    get_preprocessing_info()
    
    # Auto-detect path
    if os.path.exists("nutrition.csv"):
        input_file = "nutrition.csv"
    elif os.path.exists("../nutrition.csv"):
        input_file = "../nutrition.csv"
    else:
        # Default path
        input_file = "nutrition.csv"
    
    output_file = "nutrition_preprocessing.csv"
    
    print(f"\nInput file: {input_file}")
    print(f"Output file: {output_file}\n")
    
    try:
        # Jalankan preprocessing pipeline
        df_normalized, scaler, numeric_cols, metadata = preprocessing_pipeline(
            input_file, 
            output_file,
            columns_to_drop=['id', 'image', 'name'],
            numeric_columns=['calories', 'proteins', 'fat', 'carbohydrate']
        )
        
        # Tampilkan sample hasil
        print("\n=== SAMPLE HASIL PREPROCESSING ===")
        print(df_normalized.head())
        print()
        
        print("✓ Semua proses berhasil dijalankan!")
        
    except FileNotFoundError as e:
        print(f"Error : File tidak ditemukan!")
        print(f"Pastikan '{input_file}' ada di direktori yang sama dengan script ini")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Silakan periksa kembali file dan konfigurasi script.")
