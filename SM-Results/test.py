import pandas as  pd

df = pd.read_csv("/home/fykendhia/Documents/yabanji/social_mapper/SM-Results/results-social-mapper.csv", header=0, sep=",")
if (df["Twitter"][0] and len(df["Twitter"][0]) > 1):
    url = df["Twitter"][0]

print(url)