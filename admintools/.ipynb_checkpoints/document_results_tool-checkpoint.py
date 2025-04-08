import numpy as np
import pandas as pd

def write(save_name, file_name, commit_hash, V, data, l_optional = None):
    commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash
    
    if isinstance(data, pd.DataFrame):
        data.to_csv("../../"+save_name+f"_V{V}.csv")
        
    elif isinstance(data, np.ndarray):
        np.save("../../"+save_name+f"_V{V}.npy", data)
    
    with open("../../"+save_name + f"_V{V}.txt", 'w') as f:
        f.write(commit_link + "\n")
        f.write(file_name+"\n")
        if l_optional is not None:
            for opt in l_optional:
                f.write(opt+"\n")
                

                