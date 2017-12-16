import pandas as pd
import numpy as np
import sys


if __name__ == "__main__":

		filename1 = 'sorted.txt'
		filename2 = 'schol.csv'
		df1 = pd.read_csv(filename2)
		#print len(df1)

		#df2 = 
		#df2 = df1[['Self citation','Total Citation','NLIQ','ICR ','OCQ','SNIP']].copy()

		#n = df2.values
		n = np.loadtxt(filename1)
		#print np.shape(n)
		count = 0	
		#print ""
		for i in xrange(0,len(n)):

			for j in xrange(0,len(df1)):

				if n[i][1] == df1['Self citation'][j] and  n[i][2] == df1['Total Citation'][j]  :

					sys.stdout.write(str(count)+ " " +str(df1['Journal name'][j]) + str(n[i]) + '\n')
					count = count +1
					break


		
		#print i			