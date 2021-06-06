import pandas as pd
import numpy as np
data = pd.read_csv('play.csv')
concepts = np.array(data)[:,:-1]
target = np.array(data)[:,-1]
def train(tar,con):
    for i,val in enumerate(tar):
        if val=='yes':
            specific = con[i].copy()
            break
    for i,val in enumerate(con):
        if tar[i]=='yes':
            for x in range(len(specific)):
                if val[x] != specific[x]:
                    specific[x] = '?'
                else:
                    pass
        print("Specific[",(i+1),"]:",str(specific))
    return specific
print(train(target,concepts)) 
