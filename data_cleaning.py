# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:27:29 2023

@author: khush
"""

import pandas as pd

df = pd.read_csv("glassdoor_jobs.csv")

# List of Tasks :
# salary parsing, we can use regex as well for this kind of tasks
# Company name Text only
# state of field
# age of company
# parsing of job description

# salary parsing
# if hourly rate given then 1 else 0, adding "hourly" column to df
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

# also in salary there are some -1's which are not very usefull 
df = df[df['Salary Estimate'] != '-1'] # count goes to 742 from 956 rows
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
# remove k ans $
minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))

min_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary)/2

# Company name Text only
# we can see which is have rating is 3 characer long
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis = 1)
# In pandas, the axis parameter is used to specify whether you want to apply a function along rows (axis=0) or along columns (axis=1).
# The axis=1 parameter tells pandas to apply this function to each row. It ensures that the lambda function works row-wise across the DataFrame.

# state of field
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
# Which state has how much of counts/rows
df.job_state.value_counts()

# if actual location is at HeadQuaters
df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

# age of company
df['age'] = df.Founded.apply(lambda x: x if x <1 else 2023 - x)

# parsing of job description (python, etc..)
#python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
 
#r studio 
df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
df.R_yn.value_counts()

#spark 
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df.spark.value_counts()

#aws 
df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df.aws.value_counts()

#excel
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df.excel.value_counts()


#Cleaning DONE
df.columns

df_out = df.drop(['Unnamed: 0'], axis =1)

df_out.to_csv('salary_data_cleaned.csv',index = False)
# index = False, because we don't want that Unnamed index

pd.read_csv('salary_data_cleaned.csv')
