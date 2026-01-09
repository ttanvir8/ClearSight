import os
import shutil

def setup_toy_data():
    base_dir = "/home/tanvir/Work/754/project/other_code/ClearSight/data"
    toy_dir = os.path.join(base_dir, "toy_data")
    
    # Source paths
    src_train_dir = os.path.join(base_dir, "allweather")
    src_val_dir = os.path.join(base_dir, "allweather_val")
    
    # Destination paths
    dst_train_dir = os.path.join(toy_dir, "train")
    dst_val_dir = os.path.join(toy_dir, "val")
    
    # Create directories
    os.makedirs(os.path.join(dst_train_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(dst_train_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(dst_val_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(dst_val_dir, "gt"), exist_ok=True)
    
    # Process Train Data
    print("Processing Train Data...")
    with open(os.path.join(src_train_dir, "allweather_100.txt"), 'r') as f:
        train_files = [line.strip() for line in f.readlines()]
        
    new_train_list = []
    for file_path in train_files:
        # file_path is like "input/image_name.jpg"
        img_name = os.path.basename(file_path)
        gt_name = img_name.replace('input', 'gt') # Assuming naming convention holds, or based on previous code exploration
        # Actually logic in AllWeatherDataset: gt_names = [i.strip().replace('input', 'gt') for i in input_names]
        # But wait, looking at file content: "input/0000...jpg".
        # We need to find where the actual files are. 
        # based on previous `ls -F`, `allweather` has `input/` and `gt/` dirs.
        
        src_input = os.path.join(src_train_dir, file_path)
        dst_input = os.path.join(dst_train_dir, file_path)
        
        # Verify src input exists
        if not os.path.exists(src_input):
            print(f"Warning: {src_input} does not exist. Skipping.")
            continue
            
        shutil.copy2(src_input, dst_input)
        
        # Handle GT
        # GT file is likely just replacing 'input' with 'gt' in the path
        gt_rel_path = file_path.replace('input', 'gt')
        src_gt = os.path.join(src_train_dir, gt_rel_path)
        dst_gt = os.path.join(dst_train_dir, gt_rel_path)
        
        if os.path.exists(src_gt):
            shutil.copy2(src_gt, dst_gt)
        else:
            print(f"Warning: GT {src_gt} does not exist.")
            
        new_train_list.append(file_path)

    with open(os.path.join(dst_train_dir, "train.txt"), 'w') as f:
        f.write('\n'.join(new_train_list))
        
    # Process Val Data
    print("Processing Val Data...")
    with open(os.path.join(src_val_dir, "allweather_val_100.txt"), 'r') as f:
        val_files = [line.strip() for line in f.readlines()]
        
    new_val_list = []
    for file_path in val_files:
        src_input = os.path.join(src_val_dir, file_path)
        dst_input = os.path.join(dst_val_dir, file_path)
        
        if not os.path.exists(src_input):
            print(f"Warning: {src_input} does not exist. Skipping.")
            continue
            
        shutil.copy2(src_input, dst_input)
        
        gt_rel_path = file_path.replace('input', 'gt')
        src_gt = os.path.join(src_val_dir, gt_rel_path)
        dst_gt = os.path.join(dst_val_dir, gt_rel_path)
        
        if os.path.exists(src_gt):
            shutil.copy2(src_gt, dst_gt)
        else:
             print(f"Warning: GT {src_gt} does not exist.")

        new_val_list.append(file_path)
        
    with open(os.path.join(dst_val_dir, "val.txt"), 'w') as f:
        f.write('\n'.join(new_val_list))

    print("Toy data setup complete.")

if __name__ == "__main__":
    setup_toy_data()
