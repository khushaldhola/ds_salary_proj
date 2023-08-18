# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:09:23 2023

@author: khush
"""

import glassdoor_scraping as gs 
import pandas as pd 

path = "C:/Users/khush/Documents/ds_salary_poj/chromedriver"

df = gs.get_jobs('data scientist',1000, False, path, 15)