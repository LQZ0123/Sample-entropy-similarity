import torch
import pandas as pd
import numpy as np
from smart import FPSearch
from typing import Optional, Tuple
from tqdm import tqdm

def process_file(df: pd.DataFrame, intensity_col: str = "2-P7", smiles_col: str = "SMILES") -> Tuple[Optional[torch.Tensor], dict]:
    """
    Process DataFrame to generate weighted fused fingerprint tensor (normalized by intensity)
    
    Args:
        df: Input DataFrame containing molecular data
        intensity_col: Column name for intensity values
        smiles_col: Column name for SMILES strings
        
    Returns:
        Normalized final fingerprint tensor (None if no valid data)
        Dictionary mapping SMILES to normalized intensity values
    """
    intensity = df[intensity_col].values
    valid_mask = ~np.isnan(intensity)
    
    if not np.any(valid_mask):
        print("Warning: No valid intensity data available")
        return None, {}

    valid_intensity = intensity[valid_mask]
    valid_intensity_log = np.log10(valid_intensity + 1)
    max_log = np.max(valid_intensity_log)
    
    if max_log < 1e-8:
        print("Warning: Maximum valid intensity is near zero, normalization skipped")
        intensity_norm = np.zeros_like(intensity)
    else:
        intensity_log = np.log10(intensity + 1)
        intensity_log[~valid_mask] = 0
        intensity_norm = intensity_log / max_log

    fp_tensor_final: Optional[torch.Tensor] = None
    fp_tensor_list: list[torch.Tensor] = []
    smiles_intensity_dict: dict[str, float] = {}

    fp_searcher = FPSearch()
    score_weight = fp_searcher.get_score()

    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        current_intensity = intensity_norm[idx]
        
        if pd.isna(smiles) or smiles == "":
            print(f"Warning: Empty SMILES at row {idx}, skipped")
            continue
        
        smiles_intensity_dict[smiles] = float(current_intensity)
        
        try:
            fp_tensor = fp_searcher.fun_smile(smiles)
            if fp_tensor is None:
                print(f"Warning: Failed to parse SMILES at row {idx}, skipped")
                continue
            
            fp_tensor = torch.mul(fp_tensor, score_weight)
            
        except Exception as e:
            print(f"Warning: Fingerprint generation failed at row {idx}: {str(e)}, skipped")
            continue
        
        fp_weighted = fp_tensor * current_intensity
        fp_tensor_list.append(fp_weighted)
        
        if fp_tensor_final is None:
            fp_tensor_final = fp_weighted
        else:
            if fp_tensor_final.shape != fp_weighted.shape:
                print(f"Warning: Fingerprint dimension mismatch at row {idx}, skipped")
                continue
            fp_tensor_final = torch.max(fp_tensor_final, fp_weighted)
    
    if fp_tensor_final is None:
        print("Warning: No valid fingerprint data generated")
        return None, smiles_intensity_dict
    
    fp_max = torch.max(fp_tensor_final)
    if fp_max > 1e-8:
        fp_tensor_final = fp_tensor_final / fp_max
    else:
        print("Warning: Final tensor max value near zero, normalization skipped")
    
    return fp_tensor_final, smiles_intensity_dict


def read_and_process_file(path: str, smiles_col: str = "SMILES") -> Optional[pd.ExcelWriter]:
    """
    Read Excel file, process fingerprints, calculate similarity/entropy, and export results
    
    Args:
        path: Path to input Excel file
        smiles_col: Column name for SMILES strings
        
    Returns:
        ExcelWriter object with saved results (None if processing failed)
    """
    target = ['Actual']
    source_list = ['Coal', 'Petroleum', 'Metal', 'Dyeing', 'Fluoro', 'Pesticides']
    
    target_intensity_norm = []
    pollution_intensity_norm = []
    target_fp_intensity_norm = []
    pollution_fp_intensity_norm = [] 

    for i in target:
        df = pd.read_excel(path, i).reset_index(drop=True)
        head = df.columns[4:] 
        for name in tqdm(head, desc="Processing intensity columns"):
            fp, smiles_dict = process_file(df, name, smiles_col)
            target_intensity_norm.append({"sample": name, "smiles_intensity": smiles_dict})
            target_fp_intensity_norm.append({"sample": name, "fp_intensity": fp})

    for i in source_list:
        try:
            df = pd.read_excel(path, i).reset_index(drop=True)
        except:
            continue
        
        head = [col for col in df.columns[4:] if 'Eff' in col]
        for name in tqdm(head, desc="Processing pollution intensity columns"):
            fp, smiles_dict = process_file(df, name, smiles_col)
            pollution_intensity_norm.append({"sample": name, "smiles_intensity": smiles_dict})
            pollution_fp_intensity_norm.append({"sample": name, "fp_intensity": fp})

    fp_searcher = FPSearch()
    target_count = len(target_intensity_norm)
    pollution_count = len(pollution_intensity_norm)
    
    cas_cosine = np.zeros((target_count, pollution_count))
    cas_entropy = np.zeros((target_count, pollution_count))   
    fp_cosine = np.zeros((target_count, pollution_count))
    fp_entropy = np.zeros((target_count, pollution_count))

    for j in tqdm(range(target_count), desc="Calculating similarity and entropy"):
        target_int_dict = target_intensity_norm[j]["smiles_intensity"]
        target_fp_dict = target_fp_intensity_norm[j]["fp_intensity"]
        
        for i in range(pollution_count):
            pollute_int_dict = pollution_intensity_norm[i]["smiles_intensity"]
            pollute_fp_dict = pollution_fp_intensity_norm[i]["fp_intensity"]
            
            cos_val, ent_val = cal_similarity(pollute_int_dict, target_int_dict)
            cas_cosine[j, i] = cos_val
            cas_entropy[j, i] = ent_val

            fp_cos = fp_searcher.cosine_similarity(pollute_fp_dict, target_fp_dict)
            fp_ent = fp_searcher.sample_entropy_between(pollute_fp_dict, target_fp_dict)
            fp_cosine[j, i] = fp_cos.item()
            fp_entropy[j, i] = fp_ent.item()

    category = 'Unknown'
    if 'NTA' in path:
        category = 'NTA'
    elif 'TA' in path:
        category = 'TA'

    output_path = path.replace(".xlsx", "_result.xlsx") if ".xlsx" in path else path + "_result_tanimoto.xlsx"
    target_samples = [d["sample"] for d in target_intensity_norm]
    pollute_samples = [d["sample"] for d in pollution_intensity_norm]
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(cas_cosine, index=target_samples, columns=pollute_samples).to_excel(writer, sheet_name=f"{category}-SMILES-cosine")
        pd.DataFrame(cas_entropy, index=target_samples, columns=pollute_samples).to_excel(writer, sheet_name=f"{category}-SMILES-entropy")
        pd.DataFrame(fp_cosine, index=target_samples, columns=pollute_samples).to_excel(writer, sheet_name=f"{category}-fp-cosine")
        pd.DataFrame(fp_entropy, index=target_samples, columns=pollute_samples).to_excel(writer, sheet_name=f"{category}-fp-entropy")

    print(f"Processing completed! Results saved to: {output_path}")
    return writer

def cal_similarity(int_dict1: dict, int_dict2: dict) -> Tuple[float, float]:
    """
    Calculate cosine similarity and entropy between two intensity dictionaries
    
    Args:
        int_dict1: First intensity dictionary (pollution source)
        int_dict2: Second intensity dictionary (target sample)
        
    Returns:
        Cosine similarity value
        Entropy similarity value
    """
    all_smiles = set(int_dict1.keys()) | set(int_dict2.keys())
    vec1 = [int_dict1.get(s, 0) for s in all_smiles]
    vec2 = [int_dict2.get(s, 0) for s in all_smiles]
    
    if len(vec1) == 0 or len(vec2) == 0:
        return -1.0, -1.0
    
    fp_searcher = FPSearch()
    cosine = fp_searcher.cosine_similarity(torch.tensor(vec1), torch.tensor(vec2))
    entropy_sim = fp_searcher.sample_entropy_between(torch.tensor(vec1), torch.tensor(vec2))
    
    return cosine.item(), entropy_sim.item()