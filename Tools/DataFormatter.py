import pandas as pd
import glob

name_files = glob.glob("E:\ML.NETRepo\aclImdb\train\neg\*.txt")



df = pd.DataFrame()

for file in name_files:
    file_df = pd.read_table(file, sep=",", names=columns)

    df = df.append(file_df)

df.to_csv(r'D:\filename.ext')