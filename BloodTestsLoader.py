import pandas as pd
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader

SJ_df = SubjectLoader().get_data(groupby_reg="first").df
BT_df = BloodTestsLoader().get_data(groupby_reg="first").df

print(SJ_df.columns)
print(SJ_df.head(10))


print(BT_df.columns)
print(BT_df.head(10))

joined_df = pd.merge(SJ_df,BT_df,on="RegistrationCode",how="right")
print(joined_df.columns)
print(joined_df.head(10))

BT_df.to_csv("BT.csv",index=True)