import pandas as pd

df = pd.read_csv("Data_s/at2_s.csv")
# sort df according to the column "time"
df = df.sort_values(by=['sendtime_2'])
# save df 
df.to_csv("Data_s/at2_sort.csv", index=False)