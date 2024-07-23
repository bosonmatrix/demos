import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

if __name__=='__main__':
    sla_points=pd.read_csv(r"F:\phD_career\multi_source_adjustment\codes\BA-sv2\123_XQProject\XQSatBA\Tri\residual.txt",sep='   ')
    print(sla_points)
    sla_points.plot(x='Z',y='dZ',kind='scatter')
    plt.show()
    