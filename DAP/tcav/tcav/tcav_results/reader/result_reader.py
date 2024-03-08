import os
import pandas as pd
import pickle


concepts = ['ocean-s', 'desert-s', 'forest-s','black-c', 'brown-c', 'white-c', 'blue-c', 'orange-c', 'red-c', 'yellow-c']


data = {}
res = {}
val = {}
for folder in os.listdir('../eval_test_awa_inceptionv3_test_1/'):
    if os.path.isdir('../eval_test_awa_inceptionv3_test_1/'+folder):
        label = folder.split('_')[-1]
        print(label)


        with open('../eval_test_awa_inceptionv3_test_1/'+folder+'/tcav_res_'+label+'.pkl', 'rb') as f:
            data[label] = pickle.load(f)
            #print(data)
        print("_________________________________________")
        
        
        res[label] = {}
        val[label] = {}
        for c in concepts:
            val[label][c] = 0
            for i in range(len(data[label])):
                
                if c in data[label][i]['cav_concept']:
                    #print(data[label][i]['cav_accuracies'],data[label][i]['i_up'])
                    
                    if data[label][i]['cav_accuracies'][list(data[label][i]['cav_accuracies'].keys())[0]] == 1:
                        print("________________")
                        print(data[label][i]['i_up'],data[label][i]['cav_accuracies'])
                        
                        
                            
                        if data[label][i]['i_up'] > val[label][c]:
                            val[label][c] = round(data[label][i]['i_up'],3)
                        print(val)
                        print("________________")
                

df = pd.DataFrame.from_dict(val).transpose()
df.to_excel("res.xlsx")
















