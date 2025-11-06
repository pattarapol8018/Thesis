import pandas as pd
import glob
import os

folder_path = r'D:\scrap data'  

all_files = glob.glob(os.path.join(folder_path, "*.csv"))

df_list = []
for file in all_files:
    df = pd.read_csv(file)
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

combined_df.to_csv(os.path.join(folder_path, "Dataset_test.csv"), index=False, encoding='utf-8-sig')

print("รวมไฟล์เสร็จแล้ว")
