from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader


BT_df = BloodTestsLoader().get_data().df

print(BT_df.columns)
print(BT_df.head(10))

BT_df.to_csv("BT.csv",index=True)