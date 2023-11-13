from LabData.DataLoaders.Loader import Loader

from  LabData.DataLoaders import DemographicsLoader
from  LabData.DataLoaders import LifeStyleLoader
import matplotlib.pyplot as plt
####options for df {'english, 'hebrew', 'number' (default)}####
demographics_english = DemographicsLoader.DemographicsLoader().get_data(df='english')
demographics_number = DemographicsLoader.DemographicsLoader().get_data(df='number')
lifestyle_english = LifeStyleLoader.LifeStyleLoader().get_data(df='english')
lifestyle_number = LifeStyleLoader.LifeStyleLoader().get_data(df='number')
lifestyle_english.df['alcohol_drink'].value_counts().plot(kind='bar')
plt.title('reported alcohol consumption')
plt.show()
lifestyle_number.df['alcohol_drink'].value_counts().plot(kind='bar')
plt.title('reported alcohol consumption (numerical)')
plt.show()
demographics_english.df['living_place_today'].value_counts().plot(kind='pie')
plt.show()


