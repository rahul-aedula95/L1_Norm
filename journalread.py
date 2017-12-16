#import pandas_ml as pdl
import pandas as pd
import numpy as np


if __name__ == "__main__":

		filename = 'schol.csv'

		df1 = pd.read_csv(filename)

		df2 = df1[['Self citation','Total Citation','NLIQ','ICR ','OCQ','SNIP']].copy()

		n = df2.values

		n = np.transpose(n)
		np.savetxt('inputmat.txt',n)