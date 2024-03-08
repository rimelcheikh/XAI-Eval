# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:07:36 2023

@author: rielcheikh
"""

import numpy as np

def repro_check(results):
    moy, std = {}, {}
    
    for res in results:
        
        # helper function, returns if this is a random concept
        def is_random_concept(concept):
            return 'random500_' in concept
        
        result_summary = {}
          
        # random
        random_i_ups = {}
    
        for result in res:
          if result['cav_concept'] not in result_summary:
            result_summary[result['cav_concept']] = {}
          
          if result['bottleneck'] not in result_summary[result['cav_concept']]:
            result_summary[result['cav_concept']][result['bottleneck']] = []
          
          result_summary[result['cav_concept']][result['bottleneck']].append(result)
          
          # store random
          if is_random_concept(result['cav_concept']):
            if result['bottleneck'] not in random_i_ups:
              random_i_ups[result['bottleneck']] = []
              
            random_i_ups[result['bottleneck']].append(result['i_up'])
            
            
        # print concepts and classes with indentation
        for concept in result_summary:
          moy[concept], std[concept] = [], []
          # if not random
          if not is_random_concept(concept):
    
            for bottleneck in result_summary[concept]:
              #tcav scores using CAVs for concept with each of the random sets as negative concepts
              i_ups = [item['i_up'] for item in result_summary[concept][bottleneck]]
              
            moy[concept].append(str(np.mean(i_ups))[0:5])
            std[concept].append(str(np.std(i_ups))[0:5])
    
    
    print("_____________________________________\n", moy)
    print("_____________________________________\n", std)