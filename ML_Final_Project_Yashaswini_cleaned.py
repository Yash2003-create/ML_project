cell_type": "code
id": "bc82757e
name": "stdout
output_type": "stream
Dataset saved as adult_income.csv

# Loading the data set

import pandas as pd



# Define column names as per the `adult.names` file

columns = [

    \"age\", \"workclass\", \"fnlwgt\", \"education\", \"education_num\", \"marital_status\",

    \"occupation\", \"relationship\", \"race\", \"sex\", \"capital_gain\", \"capital_loss\",

    \"hours_per_week\", \"native_country\", \"income\"

]



# Load the dataset from the .data file

url = \"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data\"

data = pd.read_csv(url, header=None, names=columns, na_values=\" ?\")



# Save as CSV

data.to_csv(\"adult_income.csv\", index=False)

print(\"Dataset saved as adult_income.csv\")

# 32,561 rows and 15 columns.

cell_type": "code
id": "8a0fe555
name": "stdout
output_type": "stream
Data preprocessing and normalization (3NF) complete!

# Create a normalized database (3NF).

import pandas as pd

from sklearn.preprocessing import LabelEncoder



# Load the dataset

file_path = 'adult_income.csv'

data = pd.read_csv(file_path)



# Step 1: Handle Missing Values

# Fill missing values in categorical columns with the mode

missing_columns = ['workclass', 'occupation', 'native_country']

for col in missing_columns:

    data[col].fillna(data[col].mode()[0], inplace=True)



# Step 2: Remove Duplicates

data.drop_duplicates(inplace=True)



# Step 3: Standardize Categorical Columns

# Strip leading/trailing whitespaces in categorical columns

categorical_columns = [

    'workclass', 'education', 'marital_status', 'occupation',

    'relationship', 'race', 'sex', 'native_country', 'income'

]

for col in categorical_columns:

    data[col] = data[col].str.strip()



# Step 4: Encode Categorical Variables

# Initialize a LabelEncoder for all categorical columns

label_encoders = {}

for col in categorical_columns:

    le = LabelEncoder()

    data[col] = le.fit_transform(data[col])

    label_encoders[col] = le  # Store encoders for inverse transformation if needed



# Step 5: Feature Engineering (Optional Enhancements)

# Create bins for continuous variables like 'age'

data['age_group'] = pd.cut(data['age'], bins=[0, 25, 45, 65, 100], labels=['Youth', 'Adult', 'Middle-Aged', 'Senior'])



# Step 6: Normalize the Dataset (3NF)

# Break the dataset into normalized tables based on logical grouping



# Table 1: Demographics

demographics = data[['age', 'age_group', 'race', 'sex', 'native_country']]

demographics = demographics.drop_duplicates().reset_index(drop=True)



# Table 2: Work Information

work_info = data[['workclass', 'occupation', 'education', 'education_num']]

work_info = work_info.drop_duplicates().reset_index(drop=True)



# Table 3: Financial Information

financial_info = data[['fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']]

financial_info = financial_info.drop_duplicates().reset_index(drop=True)



# Table 4: Income (Target Variable)

income_info = data[['income']].drop_duplicates().reset_index(drop=True)



# Step 7: Save Normalized Tables

demographics.to_csv(\"demographics.csv\", index=False)

work_info.to_csv(\"work_info.csv\", index=False)

financial_info.to_csv(\"financial_info.csv\", index=False)

income_info.to_csv(\"income_info.csv\", index=False)



print(\"Data preprocessing and normalization (3NF) complete!\")

cell_type": "code
id": "a7aa4dd6
name": "stdout
output_type": "stream
SQLite database created with normalized tables!

Data fetched and merged successfully! Saved to 'merged_data.csv'.

#Write SQL join statement to fetch data from the database and into Pandas DataFrame.



import sqlite3

import pandas as pd



# Step 1: Create SQLite Database

db_name = \"adult_income.db\"

conn = sqlite3.connect(db_name)



# Load normalized tables into SQLite database

tables = {

    \"demographics\": \"demographics.csv\",

    \"work_info\": \"work_info.csv\",

    \"financial_info\": \"financial_info.csv\",

    \"income_info\": \"income_info.csv\"

}



for table_name, file_name in tables.items():

    df = pd.read_csv(file_name)

    df.to_sql(table_name, conn, if_exists=\"replace\", index=False)



print(\"SQLite database created with normalized tables!\")



# Step 2: Write SQL Join Statement to Fetch Data

# SQL query to join all normalized tables

query = \"\"\"

SELECT 

    d.age, d.age_group, d.race, d.sex, d.native_country,

    w.workclass, w.occupation, w.education, w.education_num,

    f.fnlwgt, f.capital_gain, f.capital_loss, f.hours_per_week,

    i.income

FROM demographics d

JOIN work_info w ON w.rowid = d.rowid

JOIN financial_info f ON f.rowid = d.rowid

JOIN income_info i ON i.rowid = d.rowid

\"\"\"



# Execute the query and fetch the result into a Pandas DataFrame

merged_data = pd.read_sql_query(query, conn)



# Step 3: Save the Fetched Data to a CSV for Further Processing

merged_data.to_csv(\"merged_data.csv\", index=False)



print(\"Data fetched and merged successfully! Saved to 'merged_data.csv'.\")

cell_type": "code
id": "16df1b2b
name": "stdout
output_type": "stream
First 5 rows of the dataset:

   age  workclass  fnlwgt  education  education_num  marital_status  \\

0   39          6   77516          9             13               4   

1   50          5   83311          9             13               2   

2   38          3  215646         11              9               0   

3   53          3  234721          1              7               2   

4   28          3  338409          9             13               2   



   occupation  relationship  race  sex  capital_gain  capital_loss  \\

0           0             1     4    1          2174             0   

1           3             0     4    1             0             0   

2           5             1     4    1             0             0   

3           5             0     2    1             0             0   

4           9             5     2    0             0             0   



   hours_per_week  native_country  income    age_group  

0              40              38       0        Adult  

1              13              38       0  Middle-Aged  

2              40              38       0        Adult  

3              40              38       0  Middle-Aged  

4              40               4       0        Adult  



Dataset Info:

<class 'pandas.core.frame.DataFrame'>

Int64Index: 32537 entries, 0 to 32560

Data columns (total 16 columns):

 #   Column          Non-Null Count  Dtype   

---  ------          --------------  -----   

 0   age             32537 non-null  int64   

 1   workclass       32537 non-null  int32   

 2   fnlwgt          32537 non-null  int64   

 3   education       32537 non-null  int32   

 4   education_num   32537 non-null  int64   

 5   marital_status  32537 non-null  int32   

 6   occupation      32537 non-null  int32   

 7   relationship    32537 non-null  int32   

 8   race            32537 non-null  int32   

 9   sex             32537 non-null  int32   

 10  capital_gain    32537 non-null  int64   

 11  capital_loss    32537 non-null  int64   

 12  hours_per_week  32537 non-null  int64   

 13  native_country  32537 non-null  int32   

 14  income          32537 non-null  int32   

 15  age_group       32537 non-null  category

dtypes: category(1), int32(9), int64(6)

memory usage: 2.9 MB

None



Summary Statistics for Numerical Columns:

                age     workclass        fnlwgt     education  education_num  \\

count  32537.000000  32537.000000  3.253700e+04  32537.000000   32537.000000   

mean      38.585549      3.094446  1.897808e+05     10.297507      10.081815   

std       13.637984      1.107549  1.055565e+05      3.870142       2.571633   

min       17.000000      0.000000  1.228500e+04      0.000000       1.000000   

25%       28.000000      3.000000  1.178270e+05      9.000000       9.000000   

50%       37.000000      3.000000  1.783560e+05     11.000000      10.000000   

75%       48.000000      3.000000  2.369930e+05     12.000000      12.000000   

max       90.000000      7.000000  1.484705e+06     15.000000      16.000000   



       marital_status    occupation  relationship          race           sex  \\

count    32537.000000  32537.000000  32537.000000  32537.000000  32537.000000   

mean         2.611427      6.139288      1.446538      3.665827      0.669238   

std          1.506301      3.973173      1.607064      0.848847      0.470495   

min          0.000000      0.000000      0.000000      0.000000      0.000000   

25%          2.000000      3.000000      0.000000      4.000000      0.000000   

50%          2.000000      6.000000      1.000000      4.000000      1.000000   

75%          4.000000      9.000000      3.000000      4.000000      1.000000   

max          6.000000     13.000000      5.000000      4.000000      1.000000   



       capital_gain  capital_loss  hours_per_week  native_country  \\

count  32537.000000  32537.000000    32537.000000    32537.000000   

mean    1078.443741     87.368227       40.440329       36.419184   

std     7387.957424    403.101833       12.346889        6.053816   

min        0.000000      0.000000        1.000000        0.000000   

25%        0.000000      0.000000       40.000000       38.000000   

50%        0.000000      0.000000       40.000000       38.000000   

75%        0.000000      0.000000       45.000000       38.000000   

max    99999.000000   4356.000000       99.000000       40.000000   



             income  

count  32537.000000  

mean       0.240926  

std        0.427652  

min        0.000000  

25%        0.000000  

50%        0.000000  

75%        0.000000  

max        1.000000  



Missing Values Count:

age               0

workclass         0

fnlwgt            0

education         0

education_num     0

marital_status    0

occupation        0

relationship      0

race              0

sex               0

capital_gain      0

capital_loss      0

hours_per_week    0

native_country    0

income            0

age_group         0

dtype: int64



Distribution of the target variable (income):

0    0.759074

1    0.240926

Name: income, dtype: float64

image/png": "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NLhkZmbKy8urLlcJAABgqdWpohEjRrhMG2OUk5Oj7du3689//nOddAwAAKCiWgUXp9PpMt2kSRO1a9dOTzzxhOLj4+ukYwAAABXVKrgsWrSorvsBAABwUbUKLuV27Nih7OxsORwORUdHq0uXLnXVLwAAgEpqFVzy8vL061//Whs2bNAVV1whY4wKCgrUt29fLVu2TC1atKjrfgIAANTurqKJEyeqsLBQu3fv1okTJ5Sfn69du3apsLBQKSkpdd1HAAAASbUccUlPT9eHH36o9u3bW23R0dF64YUXuDgXAADUm1qNuJSVlcnd3b1Su7u7u8rKyi67UwAAAFWpVXC5+eab9cc//lFHjhyx2r7//nvdf//96tevX511DgAA4Hy1Ci7z5s3TqVOn1Lp1a7Vp00ZXX321oqKidOrUKf3tb3+r6z4CAABIquU1LhEREfrss8+UkZGhr776SsYYRUdHq3///nXdPwAAAEuNRlzWr1+v6OhoFRYWSpIGDBigiRMnKiUlRd26dVOHDh30ySef1EtHAQAAahRcnnvuOY0ZM0Z+fn6V5jmdTo0bN05z5syps84BAACcr0bB5YsvvtCgQYOqnR8fH68dO3ZcdqcAAACqUqPgcvTo0Spvgy7XtGlTHTt27LI7BQAAUJUaBZcrr7xSX375ZbXzd+7cqbCwsMvuFAAAQFVqFFxuueUWPfbYY/rpp58qzTt79qymTp2qhISEOuscAADA+Wp0O/R///d/a8WKFWrbtq3uu+8+tWvXTg6HQ9nZ2XrhhRdUWlqqRx99tL76CgAA/sPVKLiEhIRo8+bNuvfee/Xwww/LGCNJcjgcGjhwoObPn6+QkJB66SgAAECNH0AXGRmpNWvWKD8/X998842MMbrmmmvk7+9fH/0DAACw1OrJuZLk7++vbt261WVfAAAALqhW31UEAADQEAguAADANgguAADANgguAADANgguAADANgguAADANgguAADANgguAADANho0uHz88ccaOnSowsPD5XA4tHLlSpf5xhhNmzZN4eHh8vb2Vp8+fbR7926XmqKiIk2cOFFBQUHy8fHRsGHDdPjwYZea/Px8JSUlyel0yul0KikpSSdPnnSpOXjwoIYOHSofHx8FBQUpJSVFxcXF9bHbAACglho0uJw5c0bXXXed5s2bV+X8GTNmaM6cOZo3b562bdum0NBQDRgwQKdOnbJqUlNTlZaWpmXLlmnTpk06ffq0EhISVFpaatUkJiYqKytL6enpSk9PV1ZWlpKSkqz5paWlGjJkiM6cOaNNmzZp2bJlWr58uSZPnlx/Ow8AAGrMYcq/KbGBORwOpaWlafjw4ZJ+Hm0JDw9XamqqHnzwQUk/j66EhITo2Wef1bhx41RQUKAWLVrojTfe0J133ilJOnLkiCIiIrRmzRoNHDhQ2dnZio6O1pYtW9S9e3dJ0pYtWxQXF6evvvpK7dq10/vvv6+EhAQdOnRI4eHhkqRly5Zp1KhRysvLk5+f3yXtQ2FhoZxOpwoKCi55mZqI/dPrdb5OoLHZMfPuhu4CgAZwqX9DG+01Lvv371dubq7i4+OtNk9PT/Xu3VubN2+WJO3YsUMlJSUuNeHh4YqJibFqMjMz5XQ6rdAiST169JDT6XSpiYmJsUKLJA0cOFBFRUXasWNHtX0sKipSYWGhywsAANSfRhtccnNzJUkhISEu7SEhIda83NxceXh4VPpm6oo1wcHBldYfHBzsUlNxO/7+/vLw8LBqqjJ9+nTruhmn06mIiIga7iUAAKiJRhtcyjkcDpdpY0yltooq1lRVX5uaih5++GEVFBRYr0OHDl2wXwAA4PI02uASGhoqSZVGPPLy8qzRkdDQUBUXFys/P/+CNUePHq20/mPHjrnUVNxOfn6+SkpKKo3EnM/T01N+fn4uLwAAUH8abXCJiopSaGioMjIyrLbi4mJt3LhRPXv2lCTFxsbK3d3dpSYnJ0e7du2yauLi4lRQUKBPP/3Uqtm6dasKCgpcanbt2qWcnByrZu3atfL09FRsbGy97icAALh0TRty46dPn9Y333xjTe/fv19ZWVkKCAhQq1atlJqaqmeeeUbXXHONrrnmGj3zzDNq1qyZEhMTJUlOp1OjR4/W5MmTFRgYqICAAE2ZMkUdO3ZU//79JUnt27fXoEGDNGbMGL300kuSpLFjxyohIUHt2rWTJMXHxys6OlpJSUmaOXOmTpw4oSlTpmjMmDGMogAA0Ig0aHDZvn27+vbta01PmjRJkpScnKzFixfrgQce0NmzZzV+/Hjl5+ere/fuWrt2rXx9fa1l5s6dq6ZNm2rkyJE6e/as+vXrp8WLF8vNzc2qWbJkiVJSUqy7j4YNG+by7Bg3NzetXr1a48ePV69eveTt7a3ExETNmjWrvg8BAACogUbzHJd/BzzHBbh8PMcF+M9k++e4AAAAVERwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAtkFwAQAAttG0oTsAAP8ODj7RsaG7ANS7Vo992dBdYMQFAADYB8EFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYBsEFAADYRqMOLtOmTZPD4XB5hYaGWvONMZo2bZrCw8Pl7e2tPn36aPfu3S7rKCoq0sSJExUUFCQfHx8NGzZMhw8fdqnJz89XUlKSnE6nnE6nkpKSdPLkyV9iFwEAQA006uAiSR06dFBOTo71+vLLL615M2bM0Jw5czRv3jxt27ZNoaGhGjBggE6dOmXVpKamKi0tTcuWLdOmTZt0+vRpJSQkqLS01KpJTExUVlaW0tPTlZ6erqysLCUlJf2i+wkAAC6uaUN34GKaNm3qMspSzhij5557To8++qhGjBghSXrttdcUEhKipUuXaty4cSooKNCrr76qN954Q/3795ckvfnmm4qIiNCHH36ogQMHKjs7W+np6dqyZYu6d+8uSXrllVcUFxenPXv2qF27dr/czgIAgAtq9CMue/fuVXh4uKKiovTrX/9a+/btkyTt379fubm5io+Pt2o9PT3Vu3dvbd68WZK0Y8cOlZSUuNSEh4crJibGqsnMzJTT6bRCiyT16NFDTqfTqqlOUVGRCgsLXV4AAKD+NOrg0r17d73++uv64IMP9Morryg3N1c9e/bU8ePHlZubK0kKCQlxWSYkJMSal5ubKw8PD/n7+1+wJjg4uNK2g4ODrZrqTJ8+3bouxul0KiIiotb7CgAALq5RB5fBgwfr9ttvV8eOHdW/f3+tXr1a0s+nhMo5HA6XZYwxldoqqlhTVf2lrOfhhx9WQUGB9Tp06NBF9wkAANReow4uFfn4+Khjx47au3evdd1LxVGRvLw8axQmNDRUxcXFys/Pv2DN0aNHK23r2LFjlUZzKvL09JSfn5/LCwAA1B9bBZeioiJlZ2crLCxMUVFRCg0NVUZGhjW/uLhYGzduVM+ePSVJsbGxcnd3d6nJycnRrl27rJq4uDgVFBTo008/tWq2bt2qgoICqwYAADQOjfquoilTpmjo0KFq1aqV8vLy9NRTT6mwsFDJyclyOBxKTU3VM888o2uuuUbXXHONnnnmGTVr1kyJiYmSJKfTqdGjR2vy5MkKDAxUQECApkyZYp16kqT27dtr0KBBGjNmjF566SVJ0tixY5WQkMAdRQAANDKNOrgcPnxYv/nNb/TDDz+oRYsW6tGjh7Zs2aLIyEhJ0gMPPKCzZ89q/Pjxys/PV/fu3bV27Vr5+vpa65g7d66aNm2qkSNH6uzZs+rXr58WL14sNzc3q2bJkiVKSUmx7j4aNmyY5s2b98vuLAAAuCiHMcY0dCf+XRQWFsrpdKqgoKBerneJ/dPrdb5OoLHZMfPuhu5CrRx8omNDdwGod60e+/LiRbV0qX9DbXWNCwAA+M9GcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcAEAALZBcKlg/vz5ioqKkpeXl2JjY/XJJ580dJcAAMD/Ibic5+2331ZqaqoeffRRff7557rxxhs1ePBgHTx4sKG7BgAARHBxMWfOHI0ePVr33HOP2rdvr+eee04RERFasGBBQ3cNAABIatrQHWgsiouLtWPHDj300EMu7fHx8dq8eXOVyxQVFamoqMiaLigokCQVFhbWSx9Li87Wy3qBxqS+Pj/17dRPpQ3dBaDe1efns3zdxpgL1hFc/s8PP/yg0tJShYSEuLSHhIQoNze3ymWmT5+uxx9/vFJ7REREvfQR+E/g/NsfGroLAKoz3Vnvmzh16pSczuq3Q3CpwOFwuEwbYyq1lXv44Yc1adIka7qsrEwnTpxQYGBgtcvAPgoLCxUREaFDhw7Jz8+vobsD4Dx8Pv/9GGN06tQphYeHX7CO4PJ/goKC5ObmVml0JS8vr9IoTDlPT095enq6tF1xxRX11UU0ED8/P34xAo0Un89/LxcaaSnHxbn/x8PDQ7GxscrIyHBpz8jIUM+ePRuoVwAA4HyMuJxn0qRJSkpKUteuXRUXF6eXX35ZBw8e1B/+wDl3AAAaA4LLee68804dP35cTzzxhHJychQTE6M1a9YoMjKyobuGBuDp6ampU6dWOh0IoOHx+fzP5TAXu+8IAACgkeAaFwAAYBsEFwAAYBsEFwAAYBsEFwAAYBsEF6AK8+fPV1RUlLy8vBQbG6tPPvmkobsE4P98/PHHGjp0qMLDw+VwOLRy5cqG7hJ+QQQXoIK3335bqampevTRR/X555/rxhtv1ODBg3Xw4MGG7hoASWfOnNF1112nefPmNXRX0AC4HRqooHv37rr++uu1YMECq619+/YaPny4pk+f3oA9A1CRw+FQWlqahg8f3tBdwS+EERfgPMXFxdqxY4fi4+Nd2uPj47V58+YG6hUAoBzBBTjPDz/8oNLS0kpfrBkSElLpCzgBAL88ggtQBYfD4TJtjKnUBgD45RFcgPMEBQXJzc2t0uhKXl5epVEYAMAvj+ACnMfDw0OxsbHKyMhwac/IyFDPnj0bqFcAgHJ8OzRQwaRJk5SUlKSuXbsqLi5OL7/8sg4ePKg//OEPDd01AJJOnz6tb775xprev3+/srKyFBAQoFatWjVgz/BL4HZooArz58/XjBkzlJOTo5iYGM2dO1c33XRTQ3cLgKQNGzaob9++ldqTk5O1ePHiX75D+EURXAAAgG1wjQsAALANggsAALANggsAALANggsAALANggsAALANggsAALANggsAALANggsAALANggsAALANggsAjRo1SsOHD2/obtSb4uJizZgxQ9ddd52aNWumoKAg9erVS4sWLVJJScklrePAgQNyOBzKysqq384CuCC+ZBHAv7Xi4mINHDhQX3zxhZ588kn16tVLfn5+2rJli2bNmqUuXbqoc+fODd3NGisuLpaHh0dDdwP4xTHiAqCSPn36KCUlRQ888IACAgIUGhqqadOmudScPHlSY8eOVUhIiLy8vBQTE6P33nvPmr98+XJ16NBBnp6eat26tWbPnu2yfOvWrfXUU0/p7rvvVvPmzRUZGalVq1bp2LFjuvXWW9W8eXN17NhR27dvd1lu8+bNuummm+Tt7a2IiAilpKTozJkz1e7Lc889p48//ljr1q3ThAkT1LlzZ1111VVKTEzU1q1bdc0110iS0tPTdcMNN+iKK65QYGCgEhIS9O2331rriYqKkiR16dJFDodDffr0seYtWrRI7du3l5eXl6699lrNnz+/Up87d+4sLy8vde3aVStXrqw0erNx40b96le/kqenp8LCwvTQQw/p3LlzLj+T++67T5MmTVJQUJAGDBig3//+90pISHDZ1rlz5xQaGqqFCxdWe0wAWzMA/uMlJyebW2+91Zru3bu38fPzM9OmTTNff/21ee2114zD4TBr1641xhhTWlpqevToYTp06GDWrl1rvv32W/Puu++aNWvWGGOM2b59u2nSpIl54oknzJ49e8yiRYuMt7e3WbRokbWNyMhIExAQYF588UXz9ddfm3vvvdf4+vqaQYMGmb///e9mz549Zvjw4aZ9+/amrKzMGGPMzp07TfPmzc3cuXPN119/bf75z3+aLl26mFGjRlW7b506dTLx8fEXPQb/+Mc/zPLly83XX39tPv/8czN06FDTsWNHU1paaowx5tNPPzWSzIcffmhycnLM8ePHjTHGvPzyyyYsLMwsX77c7Nu3zyxfvtwEBASYxYsXG2OMKSwsNAEBAeauu+4yu3fvNmvWrDFt27Y1ksznn39ujDHm8OHDplmzZmb8+PEmOzvbpKWlmaCgIDN16lSXn0nz5s3Nn/70J/PVV1+Z7Oxs889//tO4ubmZI0eOWHWrVq0yPj4+5tSpUxfdZ8COCC4AqgwuN9xwg0tNt27dzIMPPmiMMeaDDz4wTZo0MXv27KlyfYmJiWbAgAEubX/6059MdHS0NR0ZGWnuuusuazonJ8dIMn/+85+ttszMTCPJ5OTkGGOMSUpKMmPHjnVZ7yeffGKaNGlizp49W2VfvL29TUpKSnW7Xq28vDwjyXz55ZfGGGP279/vEjbKRUREmKVLl7q0PfnkkyYuLs4YY8yCBQtMYGCgS/9eeeUVl3U98sgjpl27dlZAM8aYF154wTRv3twKTr179zadO3eu1M/o6Gjz7LPPWtPDhw+/YJAD7I5TRQCq1KlTJ5fpsLAw5eXlSZKysrLUsmVLtW3btspls7Oz1atXL5e2Xr16ae/evSotLa1yGyEhIZKkjh07Vmor3+6OHTu0ePFiNW/e3HoNHDhQZWVl2r9/f5V9McbI4XBcdH+//fZbJSYm6qqrrpKfn591aujgwYPVLnPs2DEdOnRIo0ePdunTU089ZZ1m2rNnjzp16iQvLy9ruV/96lcu68nOzlZcXJxLP3v16qXTp0/r8OHDVlvXrl0r9eGee+7RokWLJP18nFavXq3f//73F91fwK64OBdAldzd3V2mHQ6HysrKJEne3t4XXLaqsGCMueA2yuuraivfbllZmcaNG6eUlJRK62rVqlWVfWnbtq2ys7Mv2F9JGjp0qCIiIvTKK68oPDxcZWVliomJUXFxcbXLlPfrlVdeUffu3V3mubm5Sbq0Y3GhmvPbfXx8KvXh7rvv1kMPPaTMzExlZmaqdevWuvHGGy+2u4BtEVwA1FinTp10+PBhff3111WOukRHR2vTpk0ubZs3b1bbtm2tP+i1cf3112v37t26+uqrL3mZxMREPfLII/r888/VpUsXl3nnzp1TUVGRfvrpJ2VnZ+ull16y/uhX7H/5HTznjxiFhIToyiuv1L59+/Tb3/62yu1fe+21WrJkiYqKiuTp6SlJlS44jo6O1vLly10CzObNm+Xr66srr7zygvsXGBio4cOHa9GiRcrMzNTvfve7ix0SwNY4VQSgxnr37q2bbrpJt99+uzIyMrR//369//77Sk9PlyRNnjxZ69at05NPPqmvv/5ar732mubNm6cpU6Zc1nYffPBBZWZmasKECcrKytLevXv1zjvvaOLEidUuk5qaql69eqlfv3564YUX9MUXX2jfvn36+9//ru7du2vv3r3y9/dXYGCgXn75ZX3zzTdav369Jk2a5LKe4OBgeXt7Kz09XUePHlVBQYEkadq0aZo+fbqef/55ff311/ryyy+1aNEizZkzR9LPwamsrExjx45Vdna2PvjgA82aNUvS/x9NGT9+vA4dOqSJEyfqq6++0qpVqzR16lRNmjRJTZpc/Nf0Pffco9dee03Z2dlKTk6u1bEFbKPhLq8B0FhUdXHuH//4R5eaW2+91SQnJ1vTx48fN7/73e9MYGCg8fLyMjExMea9996z5v/jH/8w0dHRxt3d3bRq1crMnDnTZX2RkZFm7ty5Lm2STFpamjVd1QWxn376qRkwYIBp3ry58fHxMZ06dTJPP/30Bffvp59+MtOnTzcdO3Y0Xl5eJiAgwPTq1cssXrzYlJSUGGOMycjIMO3btzeenp6mU6dOZsOGDZX688orr5iIiAjTpEkT07t3b6t9yZIlpnPnzsbDw8P4+/ubm266yaxYscKa/89//tN06tTJeHh4mNjYWLN06VIjyXz11VdWzYYNG0y3bt2Mh4eHCQ0NNQ8++KDVN2Oq/pmUKysrM5GRkeaWW2654HEA/h04jKnixDMAoN4sWbJEv/vd71RQUHDR64UuxY8//qjw8HAtXLhQI0aMqIMeAo0X17gAQD17/fXXddVVV+nKK6/UF198oQcffFAjR4687NBSVlam3NxczZ49W06nU8OGDaujHgONF8EFAOpZbm6uHnvsMeXm5iosLEz/9V//paeffvqy13vw4EFFRUWpZcuWWrx4sZo25Vc6/v1xqggAANgGdxUBAADbILgAAADbILgAAADbILgAAADbILgAAADbILgAAADbILgAAADbILgAAADb+H+tJw3xMCfB3QAAAABJRU5ErkJggg==
<Figure size 600x400 with 1 Axes>
output_type": "display_data
name": "stdout
output_type": "stream


Categorical Variables Distributions:



workclass distribution:

3    0.753266

5    0.078065

1    0.064327

6    0.039893

4    0.034299

0    0.029505

7    0.000430

2    0.000215

Name: workclass, dtype: float64

image/png": "iVBORw0KGgoAAAANSUhEUgAAAqoAAAGHCAYAAAB4X3AfAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAy30lEQVR4nO3de1wV1f7/8fcWZaMIW7wgEltErbwgamqClaKIaXjpbmaK53TOyY5advlW1jexOt+DXTTzlJbVsazjpYuYZVmggppY5iU1zSwvoGmWCZgWKq7fH/3Ypx1osAVmkNfz8ZjHw71mzcxnWEy9H2vPDA5jjBEAAABgM7WsLgAAAAAoDUEVAAAAtkRQBQAAgC0RVAEAAGBLBFUAAADYEkEVAAAAtkRQBQAAgC0RVAEAAGBLBFUAAADYEkEVqMFeeeUVORwOzxIQEKCwsDD17t1bqampOnToUIltJk2aJIfDUa7jHD9+XJMmTVJmZma5tivtWC1atNDAgQPLtZ8/MnfuXE2bNq3UdQ6HQ5MmTarQ41W0ZcuWqWvXrgoMDJTD4dCiRYusLqmEUaNGqX79+hW2rxYtWlTIvgDYW22rCwBgvdmzZ6tNmzY6efKkDh06pNWrV+vxxx/XU089pQULFqhv376evn/5y1/Uv3//cu3/+PHjeuSRRyRJ8fHxZd7Ol2P5Yu7cudq6davGjx9fYl12drYiIiIqvQZfGWN044036qKLLtLixYsVGBioiy++2OqyAKBCEFQBKDo6Wl27dvV8vu6663TXXXfp8ssv17XXXqudO3eqadOmkqSIiIhKD27Hjx9XvXr1quRYfyQ2NtbS4/+Rb7/9Vj/++KOuueYaJSQkWF1OCcVjCQC+4Kt/AKVq3ry5pkyZoqNHj+qFF17wtJf2dfzy5csVHx+vRo0aqW7dumrevLmuu+46HT9+XHv27FGTJk0kSY888ojnNoNRo0Z57W/Dhg26/vrrFRISolatWp3xWMXS0tIUExOjgIAAtWzZUtOnT/daX3xbw549e7zaMzMz5XA4PLchxMfHa8mSJdq7d6/XbRDFSvvqf+vWrRoyZIhCQkIUEBCgTp066dVXXy31OPPmzdNDDz2k8PBwBQcHq2/fvtqxY8eZf/C/sXr1aiUkJCgoKEj16tVTjx49tGTJEs/6SZMmeYL8/fffL4fDccavxI0xatq0qcaMGeNpKyoqUkhIiGrVqqXvvvvO0z516lTVrl1beXl5nrbFixcrLi5O9erVU1BQkBITE5Wdne11jLONZWk+/vhjNW7cWAMHDtSxY8c87XPnzlVcXJzq16+v+vXrq1OnTnr55ZfP+rN67rnn1LNnT4WGhiowMFAdOnTQE088oZMnT3r127hxowYOHKjQ0FA5nU6Fh4crKSlJ+/bt8/R588031b17d7lcLtWrV08tW7bUn//857MeH0DlIKgCOKOrrrpKfn5+Wrly5Rn77NmzR0lJSfL399e///1vLV26VJMnT1ZgYKBOnDihZs2aaenSpZKkW2+9VdnZ2crOztbDDz/stZ9rr71WrVu31ptvvqnnn3/+rHVt2rRJ48eP11133aW0tDT16NFDd955p5566qlyn+OMGTN02WWXKSwszFPb7wPYb+3YsUM9evTQF198oenTp2vhwoVq166dRo0apSeeeKJE/wcffFB79+7VSy+9pFmzZmnnzp0aNGiQioqKzlpXVlaW+vTpo/z8fL388suaN2+egoKCNGjQIC1YsEDSr7dGLFy4UJI0btw4ZWdnKy0trdT9ORwO9enTRxkZGZ62zz77THl5eQoICNCyZcs87RkZGerSpYsaNGgg6dfgOGTIEAUHB2vevHl6+eWXdeTIEcXHx2v16tUljlWWsXzjjTeUkJCgG2+8Ue+8844CAwMlSRMnTtTw4cMVHh6uV155RWlpaUpOTtbevXvP+vP65ptvdPPNN+u1117Te++9p1tvvVVPPvmkbrvtNk+fY8eOKTExUd99952ee+45paena9q0aWrevLmOHj0q6ddbPYYOHaqWLVtq/vz5WrJkiSZOnKhTp06d9fgAKokBUGPNnj3bSDLr1q07Y5+mTZuatm3bej6npKSY3/6n46233jKSzKZNm864j++//95IMikpKSXWFe9v4sSJZ1z3W5GRkcbhcJQ4XmJiogkODjbHjh3zOrfdu3d79VuxYoWRZFasWOFpS0pKMpGRkaXW/vu6b7rpJuN0Ok1OTo5XvwEDBph69eqZvLw8r+NcddVVXv3eeOMNI8lkZ2eXerxisbGxJjQ01Bw9etTTdurUKRMdHW0iIiLM6dOnjTHG7N6920gyTz755Fn3Z4wxL730kpHkqf0f//iHadOmjRk8eLD505/+ZIwx5sSJEyYwMNA8+OCDxhhjioqKTHh4uOnQoYMpKiry7Ovo0aMmNDTU9OjRw9N2trFMTk42gYGBxhhjJk+ebPz8/Mzjjz/u1WfXrl3Gz8/PDB8+/KznkZycfMbxKq755MmTZs6cOcbPz8/8+OOPxhhjPvvsMyPJLFq06IzbPvXUU0aSZxwBWIsZVQBnZYw56/pOnTrJ399ff/vb3/Tqq69q165dPh3nuuuuK3Pf9u3bq2PHjl5tN998swoKCrRhwwafjl9Wy5cvV0JCgtxut1f7qFGjdPz48RKzsYMHD/b6HBMTI0lnnSE8duyYPvnkE11//fVeT8r7+flpxIgR2rdvX5lvH/it4ofiimdV09PTlZiYqL59+yo9PV3SrzOKx44d8/TdsWOHvv32W40YMUK1av33fxn169fXddddp7Vr1+r48eNexznTWBpjdNtttyklJUVz587Vfffd57U+PT1dRUVFXrcnlNXGjRs1ePBgNWrUSH5+fqpTp45GjhypoqIiffXVV5Kk1q1bKyQkRPfff7+ef/55bdu2rcR+unXrJkm68cYb9cYbb2j//v3lrgVAxSGoAjijY8eO6fDhwwoPDz9jn1atWikjI0OhoaEaM2aMWrVqpVatWumZZ54p17GaNWtW5r5hYWFnbDt8+HC5jltehw8fLrXW4p/R74/fqFEjr89Op1OS9PPPP5/xGEeOHJExplzHKYvIyEjPeBWH6uKgWhx+MzIyVLduXfXo0cPrOGeq5fTp0zpy5IhX+5nG8sSJE1qwYIHat2+vAQMGlFj//fffS1K5H6DLycnRFVdcof379+uZZ57RqlWrtG7dOj333HOS/vuzdrlcysrKUqdOnfTggw+qffv2Cg8PV0pKiude1p49e2rRokU6deqURo4cqYiICEVHR2vevHnlqglAxSCoAjijJUuWqKio6A9fKXXFFVfo3XffVX5+vtauXau4uDiNHz9e8+fPL/OxyvNu1oMHD56xrTgYBgQESJIKCwu9+v3www9lPk5pGjVqpAMHDpRo//bbbyVJjRs3Pqf9S/I84FQZx0lISNCyZcuUlZWl06dPKz4+Xm3btlV4eLjS09OVkZGhK664whOoi3+eZ6qlVq1aCgkJ8Wo/01g6nU6tWLFCubm56tu3b4mAW/zQ3W8fbCqLRYsW6dixY1q4cKFuueUWXX755eratav8/f1L9O3QoYPmz5+vw4cPa9OmTRo6dKgeffRRTZkyxdNnyJAhWrZsmfLz85WZmamIiAjdfPPNZ713GUDlIKgCKFVOTo7uvfdeuVwurwdSzsbPz0/du3f3zGQVfw1fllnE8vjiiy/0+eefe7XNnTtXQUFBuuSSSyTJ8/T75s2bvfotXry4xP6cTmeZa0tISNDy5cs9gbHYnDlzVK9evQp5nVVgYKC6d++uhQsXetV1+vRpvf7664qIiNBFF13k07779u2r7777TtOmTVNsbKyCgoIk/XpeaWlpWrdundd7cy+++GJdcMEFmjt3rtdtIMeOHdPbb7/teRNAWXXu3FlZWVnat2+f4uPjvf6oRL9+/eTn56eZM2eW65yKg3Hx75n0620GL7744lm36dixo55++mk1aNCg1FtGnE6nevXqpccff1zSr7cXAKhavEcVgLZu3apTp07p1KlTOnTokFatWqXZs2fLz89PaWlpnpmu0jz//PNavny5kpKS1Lx5c/3yyy/697//Lem/90QGBQUpMjJS77zzjhISEtSwYUM1btzY578uFB4ersGDB2vSpElq1qyZXn/9daWnp+vxxx/3hKZu3brp4osv1r333qtTp04pJCREaWlppT6l3qFDBy1cuFAzZ85Uly5dVKtWLa/3yv5WSkqK3nvvPfXu3VsTJ05Uw4YN9Z///EdLlizRE088IZfL5dM5/V5qaqoSExPVu3dv3XvvvfL399eMGTO0detWzZs3r9x/HaxYnz595HA49NFHH3n+CIP061glJyd7/l2sVq1aeuKJJzR8+HANHDhQt912mwoLC/Xkk08qLy9PkydPLncNbdu21apVq9S3b1/17NlTGRkZioiIUIsWLfTggw/qscce088//6xhw4bJ5XJp27Zt+uGHH7zq/a3ExET5+/tr2LBhuu+++/TLL79o5syZJWZs33vvPc2YMUNXX321WrZsKWOMFi5cqLy8PCUmJkr69a0D+/btU0JCgiIiIpSXl6dnnnlGderUUa9evcp9rgDOkZVPcgGwVvGT8cWLv7+/CQ0NNb169TL//Oc/zaFDh0ps8/sn8bOzs80111xjIiMjjdPpNI0aNTK9evUyixcv9touIyPDdO7c2TidTiPJJCcne+3v+++//8NjGfPrU/9JSUnmrbfeMu3btzf+/v6mRYsWZurUqSW2/+qrr0y/fv1McHCwadKkiRk3bpxZsmRJiaf+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
<Figure size 800x400 with 1 Axes>
output_type": "display_data
name": "stdout
output_type": "stream


education distribution:

11    0.322525

15    0.223807

9     0.164520

12    0.052924

8     0.042475

1     0.036113

7     0.032793

0     0.028675

5     0.019824

14    0.017703

6     0.015797

2     0.013308

10    0.012693

4     0.010204

3     0.005102

13    0.001537

Name: education, dtype: float64

image/png": "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aWmmUqVKxfq7qA+XLl1qjDHmk08+MZLMK6+84tbu2WefLfZ9pKamlvjd/b7v1q1bZySZJ5988oJ1FiksLDRnzpwxa9euNZLMl19+6To3dOjQYv/bKFK3bl2Tmprq2n/88ceNJLNp0ya3dg8++KCx2Wxm9+7dxpj//102bdrUnD171tVu8+bNRpKZP3/+RdcOoGyYNgDgshljznu+RYsWCgoK0l/+8hfNmjVL33//fZmec/vtt1902yZNmqh58+Zux/r376+8vDxlZGSU6fkXa9WqVeratatiY2Pdjg8aNEinTp1yGwmUpJtvvtltv1mzZpKk/fv3l/qM/Px8bdq0SXfccYeqVq3qOh4QEKCBAwfqhx9+uOipB+f6+OOPlZiYqBYtWujs2bOurWfPnm5TKVavXi1JGjBggNv1/fv3v+RnFvnkk08kSUOHDj1vu++//179+/dXVFSUAgICFBgYqM6dO0uSdu3aVaZnr1q1So0bN9Yf//hHt+ODBg2SMcY1olukd+/ebqPiF9NnADyD8ArgsuTn5+vo0aOKiYkptU39+vW1YsUK1apVS0OHDlX9+vVVv359vfLKK5f0rNL+yrckUVFRpR47evToJT33Uh09erTEWou+o98/PzIy0m2/6K/1f/3111KfcezYMRljLuk5F+Onn37SV199pcDAQLctLCxMxhj9/PPPrntXrly5WO0lfe8X68iRIwoICDjvPU6ePKnrr79emzZt0jPPPKM1a9Zoy5YtWrBggaTzf2fn440+A+AZzHkFcFmWLFmigoKCCy5vdf311+v6669XQUGBtm7dqldffVUjR45U7dq11a9fv4t61qWsHXv48OFSjxUFjypVqkiSnE6nW7uigFZWkZGROnToULHjBw8elCTVqFHjsu4vSdWrV1elSpU8/pwaNWooODhY77zzTqnnpf99xrNnz+ro0aNuQa6k771KlSrFvmOp+Pdcs2ZNFRQU6PDhw6X+QWXVqlU6ePCg1qxZ4xptlaTjx49f8LOdjzf6DIBnMPIKoMwOHDigMWPGyOFw6IEHHrioawICAnTddde53gQv+it8T49cff311/ryyy/djs2bN09hYWH6wx/+IEmuN+C/+uort3bnvtlexG63X3RtXbt2dYWsc82ePVshISEeWVorNDRU1113nRYsWOBWV2FhoebOnas6dero2muvveT73nTTTfruu+8UGRmp1q1bF9uKvrOkpCRJ0nvvved2/bx584rdMz4+Xjk5Ofrpp59cx06fPq3ly5e7tSt62WvGjBml1lf0B5jfv3R27moXRS7lf1Ndu3bVzp07i00pmT17tmw2m+vzArAeI68ALsqOHTtc8x9zcnL02WefaebMmQoICNDChQuLrQxwrtdff12rVq1S7969FRcXp99++801slf04wZhYWGqW7eu/vOf/6hr166KiIhQjRo1LrisU2liYmJ08803a8KECYqOjtbcuXOVnp6uyZMnKyQkRJLUpk0bNWjQQGPGjNHZs2dVvXp1LVy4UOvXry92v6ZNm2rBggWaMWOGWrVqpUqVKrmte3uu8ePH6+OPP1ZSUpLGjRuniIgIvffee1qyZIleeOEFj72NnpaWpu7duyspKUljxoxRUFCQpk+frh07dmj+/PmX/CtnkjRy5Eh99NFH6tSpk0aNGqVmzZqpsLBQBw4c0Keffqq//vWvuu6669SjRw916tRJjz76qPLz89W6dWv93//9n+bMmVPsnn379tW4cePUr18/PfLII/rtt980depUFRQUuLW7/vrrNXDgQD3zzDP66aefdNNNN8lut+uLL75QSEiIhg8frvbt26t69eoaMmSIxo8fr8DAQL333nvF/qAi/a/PJGny5MlKTk5WQECAmjVrpqCgoGJtR40apdmzZ6t3796aNGmS6tatqyVLlmj69Ol68MEHy/QHAQDlxNr3xQD4uqI38ou2oKAgU6tWLdO5c2fz3HPPmZycnGLX/P4t8o0bN5pbb73V1K1b19jtdhMZGWk6d+5sFi9e7HbdihUrTMuWLY3dbjeSXG+DF93vyJEjF3yWMf97k7x3797mww8/NE2aNDFBQUEmPj7evPTSS8Wu//bbb02PHj1MeHi4qVmzphk+fLhZsmRJsdUGfvnlF3PHHXeYatWqGZvN5vZMlbBKwvbt201KSopxOBwmKCjING/e3MycOdOtTdFqA//+97/djhe90f779iX57LPPzA033GBCQ0NNcHCwadu2rfnvf/9b4v0uZrUBY4w5efKk+dvf/mYaNGhggoKCjMPhME2bNjWjRo0yhw8fdrU7fvy4uf/++021atVMSEiI6d69u/nmm29K/D6WLl1qWrRoYYKDg029evXMtGnTSuy7goIC8/LLL5vExETXs9u1a+f2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9PLLLys9PV01atQotX1aWpocDodri42N9WK1AAAAKE8+H16TkpKUmZmpDRs26MYbb9Rdd92lnJycUtuPHTtWubm5ri07O9uL1QIAAKA8+Xx4DQ0N1dVXX622bdvq7bffVuXKlfX222+X2t5utys8PNxtAwAAQMXg8+H194wxbnNaAQAAcOWw9IWtkydPau/eva79rKwsZWZmKiIiQpGRkXr22Wd18803Kzo6WkePHtX06dP1ww8/6M4777SwagAAAFjF0vC6detWJSUlufZHjx4tSUpNTdXrr7+ub775RrNmzdLPP/+syMhItWnTRp999pmaNGliVckAAACwkKXhtUuXLjLGlHp+wYIFXqwGAAAAvs7v5rwCAADgyuU3P1JwuWoO6cTKAwAAAH6uwofXomkJ/NIWAACAbyrKaeebTlqkwofXo0ePShK/tAUAAODjTpw4IYfDcd42FT68RkRESJIOHDhwwS8DvikvL0+xsbHKzs5m6ocfov/8H33o3+g//3cl9KExRidOnFBMTMwF21b48Fqp0v/eSXM4HBW2w68U/GKaf6P//B996N/oP/9X0fvwYgcZWW0AAAAAfoPwCgAAAL9R4cOr3W7X+PHjZbfbrS4FZUQf+jf6z//Rh/6N/vN/9KE7m7mYNQkAAAAAH1DhR14BAABQcRBeAQAA4DcIrwAAAPAbhFcAAAD4jQodXqdPn66EhARVqVJFrVq10meffWZ1SVektLQ0tWnTRmFhYapVq5b69Omj3bt3u7UxxmjChAmKiYlRcHCwunTpoq+//tqtjdPp1PDhw1WjRg2Fhobq5ptv1g8//ODW5tixYxo4cKAcDoccDocGDhyo48ePl/dHvKKkpaXJZrNp5MiRrmP0n+/78ccfdc899ygyMlIhISFq0aKFtm3b5jpPH/qus2fP6m9/+5sSEhIUHBysevXqadKkSSosLHS1of98y7p165SSkqKYmBjZbDYtWrTI7bw3++vAgQNKSUlRaGioatSooYcfflinT58uj4/tPaaCev/9901gYKB56623zM6dO82IESNMaGio2b9/v9WlXXF69uxpZs6caXbs2GEyMzNN7969TVxcnDl58qSrzfPPP2/CwsLMRx99ZLZv32769u1roqOjTV5enqvNkCFDzFVXXWXS09NNRkaGSUpKMs2bNzdnz551tbnxxhtNYmKi2bBh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
<Figure size 800x400 with 1 Axes>
output_type": "display_data
name": "stdout
output_type": "stream


marital_status distribution:

2    0.460092

4    0.327842

0    0.136491

5    0.031503

6    0.030519

3    0.012847

1    0.000707

Name: marital_status, dtype: float64

image/png": "iVBORw0KGgoAAAANSUhEUgAAAqYAAAGHCAYAAABiY5CRAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAA3SElEQVR4nO3deVhWdf7/8dcNwo0gIKCoJAJWkwtipuZa7jamlFNpWqPm1LeccTdznRK9SlomUyttrEZrHJextHQqG9xtcks0McwlQcw0cgOXBITP7w8v7l+3LMItch/k+biu+7o653zuz3mfDyd9+TkLNmOMEQAAAOBmHu4uAAAAAJAIpgAAALAIgikAAAAsgWAKAAAASyCYAgAAwBIIpgAAALAEgikAAAAsgWAKAAAASyCYAgAAwBIIpkAFs2DBAtlsNsfHx8dHtWvXVqdOnRQfH6/09PQC34mLi5PNZivVfi5evKi4uDht2LChVN8rbF+RkZHq1atXqfq5lkWLFmnmzJmFbrPZbIqLiyvT/ZW1tWvXqkWLFvLz85PNZtMnn3zi7pKK1bFjR3Xs2NGx7Or58Vupqamy2WxasGDBddf3W8nJyYqLi1NqaqrLfXz99deKi4vT2bNny6wuANdGMAUqqPnz52vLli1KSEjQ22+/rTvvvFOvvPKKGjZsqDVr1ji1feqpp7Rly5ZS9X/x4kVNnTq11MHDlX25orhgumXLFj311FM3vAZXGWPUt29feXl5aeXKldqyZYs6dOjg7rKKNWfOHM2ZM8ex7Or5UR6Sk5M1derU6w6mU6dOJZgC5ayKuwsA4Jro6Gi1aNHCsfzwww9r9OjRat++vR566CEdPHhQtWrVkiTVrVtXdevWvaH1XLx4Ub6+vuWyr2tp3bq1W/d/LT/99JNOnz6tP/zhD+rSpYu7yylW/s+1UaNG7i4FQCXAjClwE6lXr55ef/11nTt3Tn//+98d6wu7vL5u3Tp17NhRISEhqlq1qurVq6eHH35YFy9eVGpqqmrWrClJmjp1quO2gSeeeMKpv8TERD3yyCMKCgrSrbfeWuS+8q1YsUIxMTHy8fFR/fr1NXv2bKft+bcpXD3TtWHDBtlsNsfsXMeOHfXZZ5/pyJEjTrc15CvsUv7evXv14IMPKigoSD4+Prrzzjv1wQcfFLqfxYsXa/LkyQoLC1NAQIC6du2q/fv3Fz3wv/HVV1+pS5cu8vf3l6+vr9q2bavPPvvMsT0uLs4R3MePHy+bzabIyMgi+8uvadGiRRo/frzq1KmjatWqKTY2Vj///LPOnTunp59+WjVq1FCNGjU0ePBgnT9/3qmPt99+W/fee69CQ0Pl5+enJk2a6NVXX1VOTo5Tu44dOyo6OlqbNm1S27Zt5evrqz/96U+ObfmX8q91fhw6dEiDBw/W7bffLl9fX91yyy2KjY1VUlJSicbwWubOnaumTZuqWrVq8vf3V4MGDTRp0iRJV86hPn36SJI6derkqC3/doGEhAQ9+OCDqlu3rnx8fHTbbbfpmWee0cmTJx39x8XF6bnnnpMkRUVFOfrIP/+KulUkMjLSMQbSlVA/duxYRUVFycfHR8HBwWrRooUWL15cJuMA3IyYMQVuMvfff788PT21adOmItukpqaqZ8+euueee/SPf/xD1atX17Fjx7R69WplZ2erTp06Wr16tX7/+9/rySefdFwWzw8j+R566CH169dPQ4YM0YULF4qta/fu3Ro1apTi4uJUu3Zt/etf/9LIkSOVnZ2tsWPHluoY58yZo6efflo//PCDVqxYcc32+/fvV9u2bRUaGqrZs2crJCRECxcu1BNPPKGff/5Z48aNc2o/adIktWvXTu+9954yMzM1fvx4xcbGat++ffL09CxyPxs3blS3bt0UExOj999/X3a7XXPmzFFsbKwWL16sRx99VE899ZSaNm2qhx56SMOHD9djjz0mu91+zWOYNGmSOnXqpAULFig1NVVjx45V//79VaVKFTVt2lSLFy/Wrl27NGnSJPn7+zuF/h9++EGPPfaYoqKi5O3trW+//VYvvfSSvv/+e/3jH/9w2s/x48f1xz/+UePGjdP06dPl4VFw/uJa58dPP/2kkJAQvfzyy6pZs6ZOnz6tDz74QK1atdKuXbt0xx13XPN4i7JkyRL95S9/0fDhw/W3v/1NHh4eOnTokJKTkyVJPXv21PTp0zVp0iS9/fbbuuuuuyTJ8Q+nH374QW3atNFTTz2lwMBApaamasaMGWrfvr2SkpLk5eWlp556SqdPn9abb76p5cuXq06dOpJU6lnjMWPG6J///KdefPFFNWvWTBcuXNDevXt16tQpl48fuOkZABXK/PnzjSSzY8eOItvUqlXLNGzY0LE8ZcoU89v/3T/66CMjyezevbvIPn755RcjyUyZMqXAtvz+XnjhhSK3/VZERISx2WwF9tetWzcTEBBgLly44HRsKSkpTu3Wr19vJJn169c71vXs2dNEREQUWvvVdffr18/Y7XaTlpbm1K5Hjx7G19fXnD171mk/999/v1O7f//730aS2bJlS6H7y9e6dWsTGhpqzp0751h3+fJlEx0dberWrWvy8vKMMcakpKQYSea1114rtr/f1hQbG+u0ftSoUUaSGTFihNP63r17m+Dg4CL7y83NNTk5OebDDz80np6e5vTp045tHTp0MJLM2rVrC3yvQ4cOpkOHDo7l4s6Pq12+fNlkZ2eb22+/3YwePdqxPn8c5s+ff80+8g0bNsxUr1692DbLli0rcL4UJi8vz+Tk5JgjR44YSebTTz91bHvttdcKPReNKXh+5YuIiDCDBg1yLEdHR5vevXsXWwMAZ1zKB25Cxphit995553y9vbW008/rQ8++ECHDx92aT8PP/xwids2btxYTZs2dVr32GOPKTMzU4mJiS7tv6TWrVunLl26KDw83Gn9E088oYsXLxZ4WOuBBx5wWo6JiZEkHTlypMh9XLhwQdu2bdMjjzyiatWqOdZ7enpqwIAB+vHHH0t8O0Bhrn6rQcOGDSVdmSG8ev3p06edLufv2rVLDzzwgEJCQuTp6SkvLy8NHDhQubm5OnDggNP3g4KC1LlzZ5frlKTLly9r+vTpatSokby9vVWlShV5e3vr4MGD2rdv33X1fffdd+vs2bPq37+/Pv30U6dL8CWRnp6uIUOGKDw8XFWqVJGXl5ciIiIk6bprK6zWL774QhMmTNCGDRv066+/lmn/wM2IYArcZC5cuKBTp04pLCysyDa33nqr1qxZo9DQUA0dOlS33nqrbr31Vs2aNatU+8q/xFkStWvXLnLdjb60eerUqUJrzR+jq/cfEhLitJx/qb24YHHmzBkZY0q1n9IIDg52Wvb29i52/aVLlyRJaWlpuueee3Ts2DHNmjVLmzdv1o4dO/T2229LKnhMpfmZFmXMmDF6/vnn1bt3b61atUrbtm3Tjh071LRp0+sOZwMGDNA//vEPHTlyRA8//LBCQ0PVqlUrJSQkXPO7eXl56t69u5YvX65x48Zp7dq12r59u7Zu3Sqp+J+vK2bPnq3x48frk08+UadOnRQcHKzevXvr4MGDZbof4GbCPabATeazzz5Tbm6u0zsnC3PPPffonnvuUW5urr755hu9+eabGjVqlGrVqqV+/fqVaF+leTfqiRMnilyXHwR9fHwkSVlZWU7tSjsrdrWQkBAdP368wPqffvpJklSjRo3r6l+6MtPo4eFxw/dTWp988okuXLig5cuXO2YGpSv3/BamtO+7LczChQs1cOBATZ8+3Wn9yZMnVb169evuf/DgwRo8eLAuXLigTZs2acqUKerVq5cOHDjgdIxX27t3r7799lstWLBAgwYNcqw/dOhQqfZvt9sLnKNSwX94+Pn5aerUqZo6dap+/vlnx+xpbGysvv/++1LtE6gsmDEFbiJpaWkaO3asAgMD9cwzz5ToO56enmrVqpVjBi3/snpJZglL47vvvtO3337rtG7RokXy9/d3PKCS/3T6nj17nNqtXLmyQH92u73EtXXp0kXr1q1zBMR8H374oXx9fcvk9VJ+fn5q1aqVli9f7lRXXl6eFi5cqLp16+p3v/vdde+ntPKD5m8fsDLG6N13372ufos7P2w2W4EHuj777DMdO3bsuvZ5NT8/P/Xo0UOTJ09Wdna2vvvuu2JrK2wsJDm9wSJfcccXGRlZ4Bxdt25dgbch/FatWrX0xBNPqH///tq/f78uXrx4rcMDKiVmTIEKau/evbp8+bIuX76s9PR0bd68WfPnz5enp6dWrFhR4An633rnnXe0bt069ezZU/Xq1dOlS5ccT2d37dpVkuTv76+IiAh9+umn6tKli4KDg1WjRo1iX21UnLCwMD3wwAOKi4tTnTp1tHDhQiUkJOiVV16Rr6+vJKlly5a64447NHbsWF2+fFlBQUFasWKFvvrqqwL9NWnSRMuXL9fcuXPVvHlzeXh4OL3X9bemTJmi//znP+rUqZNeeOEFBQcH61//+pc+++wzvfrqqwoMDHTpmK4WHx+vbt26qVOnTho7dqy8vb01Z84c7d27V4sXLy6T2cjS6tatm7y9vdW/f3+NGzdOly5d0ty5c3XmzJnr6re486NXr15asGCBGjRooJiYGO3cuVOvvfZambzf9v/+7/9UtWpVtWvXTnXq1NGJEycUHx+vwMBAtWzZUtKVd/xK0rx58+Tv7y8fHx9FRUWpQYMGuvXWWzVhwgQZYxQcHKxVq1YVehtAkyZNJEmzZs3SoEGD5OXlpTvuuEP+/v4aMGCAnn/+eb3wwgvq0KGDkpOT9dZbbxU4j1q1aqVevXopJiZGQUFB2rdvn/75z3+qTZs2jnMewFXc++wVgNLKf3I9/+Pt7W1CQ0NNhw4dzPTp0016enqB71z9pPyWLVvMH/7wBxMREWHsdrsJCQkxHTp0MCtXrnT63po1a0yzZs2M3W43khxPHOf398svv1xzX8ZceVq5Z8+e5qOPPjKNGzc23t7eJjIy0syYMaPA9w8cOGC6d+9uAgICTM2aNc3w4cPNZ599VuAp69OnT5tHHnnEVK9e3dhsNqd9qpCnppOSkkxsbKwJDAw03t7epmnTpgWeBs9/An7ZsmVO60vz9PjmzZtN586djZ+fn6latapp3bq1WbVqVaH9leap/KtrKurtDIX9bFatWmWaNm1qfHx8zC233GKee+4588UXXxQY0w4dOpjGjRsXWsfVT+UbU/T5cebMGfPkk0+a0NBQ4+vra9q3b282b95coA9Xnsr/4IMPTKdOnUytWrWMt7e3CQsLM3379jV79uxxajdz5kwTFRVlPD09nfaRnJxsunXrZvz9/U1QUJDp06ePSUtLK/ScmThxogkLCzMeHh5OY5WVlWXGjRtnwsPDTdWqVU2HDh3M7t27CzyVP2HCBNOiRQsTFBRk7Ha7qV+/vhk9erQ5efJkiY8XqGxsxlzj8V0AAACgHHCPKQAAACyBe0wBAJZw+fLlYrd7eHgU+puoANw8+D8cAOB2qamp8vLyKvYzbdo0d5cJ4AZjxhQA4HZhYWHasWPHNdsAuLnx8BMAAAAsgUv5AAAAsIQKfSk/Ly9PP/30k/z9/d3y4moAAAAUzxijc+fOKSws7JoPMFboYPrTTz8pPDzc3WUAAADgGo4ePXrN3wBXoYOpv7+/pCsHGhAQ4OZqAAAAcLXMzEyFh4c7cltxKnQwzb98HxAQQDAFAACwsJLcdsnDTwAAALCECj1jmu/evy6Wp72qu8sAAACwvJ2vDXR3CUVixhQAAACWQDAFAACAJRBMAQAAYAkEUwAAAFgCwRQAAACWQDAFAACAJRBMAQAAYAkEUwAAAFgCwRQAAACWQDAFAACAJRBMAQAAYAkEUwAAAFgCwRQAAACW4NZgGh8fr5YtW8rf31+hoaHq3bu39u/f786SAAAA4CZuDaYbN27U0KFDtXXrViUkJOjy5cvq3r27Lly44M6yAAAA4AZV3Lnz1atXOy3Pnz9foaGh2rlzp+699143VQUAAAB3cGswvVpGRoYkKTg4uNDtWVlZysrKcixnZmaWS10AAAC48Szz8JMxRmPGjFH79u0VHR1daJv4+HgFBgY6PuHh4eVcJQAAAG4UywTTYcOGac+ePVq8eHGRbSZOnKiMjAzH5+jRo+VYIQAAAG4kS1zKHz58uFauXKlNmzapbt26Rbaz2+2y2+3lWBkAAADKi1uDqTFGw4cP14oVK7RhwwZFRUW5sxwAAAC4kVuD6dChQ7Vo0SJ9+umn8vf314kTJyRJgYGBqlq1qjtLAwAAQDlz6z2mc+fOVUZGhjp27Kg6deo4PkuXLnVnWQAAAHADt1/KBwAAACQLPZUPAACAyo1gCgAAAEsgmAIAAMASCKYAAACwBIIpAAAALIFgCgAAAEsgmAIAAMASCKYAAACwBIIpAAAALIFgCgAAAEtw668kLSubXuyvgIAAd5cBAACA68CMKQAAACyBYAoAAABLIJgCAADAEgimAAAAsASCKQAAACyBYAoAAABLIJgCAADAEgimAAAAsISb4gX7R19uLX8fT3eXAQCo5Oq9kOTuEoAKjRlTAAAAWALBFAAAAJZAMAUAAIAlEEwBAABgCQRTAAAAWALBFAAAAJZAMAUAAIAlEEwBAABgCQRTAAAAWALBFAAAAJZAMAUAAIAlEEwBAABgCQRTAAAAWIJlgml8fLxsNptGjRrl7lIAAADgBpYIpjt27NC8efMUExPj7lIAAADgJm4PpufPn9fjjz+ud999V0FBQe4uBwAAAG7i9mA6dOhQ9ezZU127dr1m26ysLGVmZjp9AAAAcHOo4s6dL1myRImJidqxY0eJ2sfHx2vq1Kk3uCoAAAC4g9tmTI8ePaqRI0dq4cKF8vHxKdF3Jk6cqIyMDMfn6NGjN7hKAAAAlBe3zZju3LlT6enpat68uWNdbm6uNm3apLfeektZWVny9PR0+o7dbpfdbi/vUgEAAFAO3BZMu3TpoqSkJKd1gwcPVoMGDTR+/PgCoRQAAAA3N7cFU39/f0VHRzut8/PzU0hISIH1AAAAuPm5/al8AAAAQHLzU/lX27Bhg7tLAAAAgJswYwoAAABLIJgCAADAEgimAAAAsASCKQAAACyBYAoAAABLIJgCAADAEgimAAAAsASCKQAAACyBYAoAAABLIJgCAADAEiz1K0ldFT5hqwICAtxdBgAAAK4DM6YAAACwBIIpAAAALIFgCgAAAEsgmAIAAMASCKYAAACwBIIpAAAALIFgCgAAAEsgmAIAAMASbooX7Hd7p5uqVL0pDgWVyP+G/8/dJQAAYCnMmAIAAMASCKYAAACwBIIpAAAALIFgCgAAAEsgmAIAAMASCKYAAACwBIIpAAAALIFgCgAAAEsgmAIAAMASCKYAAACwBIIpAAAALIFgCgAAAEsgmAIAAMASLBFM58yZo6ioKPn4+Kh58+bavHmzu0sCAABAOXN7MF26dKlGjRqlyZMna9euXbrnnnvUo0cPpaWlubs0AAAAlCOXgmliYqKSkpIcy59++ql69+6tSZMmKTs7u1R9zZgxQ08++aSeeuopNWzYUDNnzlR4eLjmzp3rSmkAAACooFwKps8884wOHDggSTp8+LD69esnX19fLVu2TOPGjStxP9nZ2dq5c6e6d+/utL579+76+uuvC7TPyspSZmam0wcAAAA3B5eC6YEDB3TnnXdKkpYtW6Z7771XixYt0oIFC/Txxx+XuJ+TJ08qNzdXtWrVclpfq1YtnThxokD7+Ph4BQYGOj7h4eGulA8AAAALcimYGmOUl5cnSVqzZo3uv/9+SVJ4eLhOnjxZ6v5sNluB/q9eJ0kTJ05URkaG43P06FEXqgcAAIAVVXHlSy1atNCLL76orl27auPGjY77QVNSUgrMfhanRo0a8vT0LDA7mp6eXmg/drtddrvdlZIBAABgcS7NmM6cOVOJiYkaNmyYJk+erNtuu02S9NFHH6lt27Yl7sfb21vNmzdXQkKC0/qEhIRS9QMAAICKz6UZ05iYGKen8vO99tpr8vT0LFVfY8aM0YABA9SiRQu1adNG8+bNU1pamoYMGeJKaQAAAKigXAqmRfHx8Sn1dx599FGdOnVK06ZN0/HjxxUdHa3PP/9cERERZVkaAAAALM6lYOrh4VHow0n5cnNzS9XfX/7yF/3lL39xpRQAAADcJFwKpitWrHBazsnJ0a5du/TBBx9o6tSpZVIYAAAAKheXgumDDz5YYN0jjzyixo0ba+nSpXryySevuzAAAABULi49lV+UVq1aac2aNWXZJQAAACqJMgumv/76q958803VrVu3rLoEAABAJeLSpfygoCCnh5+MMTp37px8fX21cOHCMisOAAAAlYdLwfSNN95wCqYeHh6qWbOmWrVqpaCgoDIrDgAAAJWHS8G0c+fOCg8PL/SVUWlpaapXr951FwYAAIDKxaV7TKOiovTLL78UWH/q1ClFRUVdd1EAAACofFwKpsaYQtefP3/epd/+BAAAAJTqUv6YMWMkSTabTS+88IJ8fX0d23Jzc7Vt2zbdeeedZVogAAAAKodSBdNdu3ZJujJjmpSUJG9vb8c2b29vNW3aVGPHji3bCksgYUiCAgICyn2/AAAAKDulCqbr16+XJA0ePFizZs0iDAIAAKDMuPRU/vz588u6DgAAAFRyLgVTSdqxY4eWLVumtLQ0ZWdnO21bvnz5dRcGAACAysWlp/KXLFmidu3aKTk5WStWrFBOTo6Sk5O1bt06BQYGlnWNAAAAqARcCqbTp0/XG2+8of/85z/y9vbWrFmztG/fPvXt25eX6wMAAMAlLgXTH374QT179pQk2e12XbhwQTabTaNHj9a8efPKtEAAAABUDi4F0+DgYJ07d06SdMstt2jv3r2SpLNnz+rixYtlVx0AAAAqDZcefrrnnnuUkJCgJk2aqG/fvho5cqTWrVunhIQEdenSpaxrBAAAQCVgM0X9ftFinD59WpcuXVJYWJjy8vL0t7/9TV999ZVuu+02Pf/88woKCroRtRaQmZmpwMBAfdamrfyquPyCgRuuw6aN7i4BAADALfLzWkZGxjXfge9SMLUKgikAAIC1lSaYunSPqaenp9LT0wusP3XqlDw9PV3pEgAAAJWcS8G0qEnWrKwseXt7X1dBAAAAqJxKdf179uzZkiSbzab33ntP1apVc2zLzc3Vpk2b1KBBg7KtEAAAAJVCqYLpG2+8IenKjOk777zjdNne29tbkZGReuedd8q2QgAAAFQKpQqmKSkpkqROnTpp+fLl5fb0PQAAAG5+Lt1jun79eqdQmpubq927d+vMmTNlVhgAAAAqF5eC6ahRo/T+++9LuhJK7733Xt11110KDw/Xhg0byrI+AAAAVBIuBdNly5apadOmkqRVq1YpNTVV33//vUaNGqXJkyeXaYEAAACoHFwKpqdOnVLt2rUlSZ9//rn69Omj3/3ud3ryySeVlJRUpgUCAACgcnApmNaqVUvJycnKzc3V6tWr1bVrV0nSxYsXecE+AAAAXOLS7/EcPHiw+vbtqzp16shms6lbt26SpG3btvEeUwAAALjEpWAaFxen6OhoHT16VH369JHdbpd05VeVTpgwoUwLBAAAQOXg0qV8SXrkkUc0evRo1a1b17Fu0KBBevDBBx3LTZo00dGjR4vsIy4uTjabzemTf+8qAAAAKheXZkxLKjU1VTk5OcW2ady4sdasWeNY5h5VAACAyumGBtMSFVClCrOkAAAAcP1Sflk5ePCgwsLCFBUVpX79+unw4cNFts3KylJmZqbTBwAAADcHtwbTVq1a6cMPP9SXX36pd999VydOnFDbtm116tSpQtvHx8crMDDQ8QkPDy/nigEAAHCj2Iwx5kZ17u/vr2+//Vb169cvUfsLFy7o1ltv1bhx4zRmzJgC27OyspSVleVYzszMVHh4uD5r01Z+Vdx+V0KROmza6O4SAAAA3CIzM1OBgYHKyMhQQEBAsW0tleb8/PzUpEkTHTx4sNDtdrvd8WoqAAAA3Fxu6KX8v//976pVq1aJ22dlZWnfvn2qU6fODawKAAAAVlTiGdPZs2eXuNMRI0ZIkh577LFi240dO1axsbGqV6+e0tPT9eKLLyozM1ODBg0q8b4AAABwcyhxMH3jjTdK1M5mszmC6bX8+OOP6t+/v06ePKmaNWuqdevW2rp1qyIiIkpaFgAAAG4SJQ6mKSkpZb7zJUuWlHmfAAAAqJjc/h5TAAAAQLqOp/J//PFHrVy5UmlpacrOznbaNmPGjOsuDAAAAJWLS8F07dq1euCBBxQVFaX9+/crOjpaqampMsborrvuKusaAQAAUAm4dCl/4sSJevbZZ7V37175+Pjo448/1tGjR9WhQwf16dOnrGsEAABAJeBSMN23b5/jlU5VqlTRr7/+qmrVqmnatGl65ZVXyrRAAAAAVA4uBVM/Pz/HrwYNCwvTDz/84Nh28uTJsqkMAAAAlYpL95i2bt1a//vf/9SoUSP17NlTzz77rJKSkrR8+XK1bt26rGsEAABAJeBSMJ0xY4bOnz8vSYqLi9P58+e1dOlS3XbbbSV+ET8AAADwWy4F0/r16zv+29fXV3PmzCmzggAAAFA5uXSPaf369XXq1KkC68+ePesUWgEAAICScmnGNDU1Vbm5uQXWZ2Vl6dixY9ddVGm1X/2FAgICyn2/AAAAKDulCqYrV650/PeXX36pwMBAx3Jubq7Wrl2ryMjIMisOAAAAlUepgmnv3r0lSTabzfEe03xeXl6KjIzU66+/XmbFAQAAoPIoVTDNy8uTJEVFRWnHjh2qUaPGDSkKAAAAlY9L95impKSUdR0AAACo5EocTGfPnq2nn35aPj4+mj17drFtR4wYcd2FAQAAoHKxGWNMSRpGRUXpm2++UUhIiCIjI2Wz2Qrv0GbT4cOHy7TIomRmZiowMFAZGRk8lQ8AAGBBpclrJZ4x/e3l+9TUVJeLAwAAAApT6hfs5+TkqH79+kpOTr4R9QAAAKCSKvXDT15eXsrKyiryUr47/H3SF6pq93V3GQUMez3W3SUAAABUGC79StLhw4frlVde0eXLl8u6HgAAAFRSLr0uatu2bVq7dq3++9//qkmTJvLz83Pavnz58jIpDgAAAJWHS8G0evXqevjhh8u6FgAAAFRiLgXT+fPnl3UdAAAAqORcuscUAAAAKGsuzZhK0kcffaR///vfSktLU3Z2ttO2xMTE6y4MAAAAlYtLM6azZ8/W4MGDFRoaql27dunuu+9WSEiIDh8+rB49epR1jQAAAKgEXAqmc+bM0bx58/TWW2/J29tb48aNU0JCgkaMGKGMjIyyrhEAAACVgEvBNC0tTW3btpUkVa1aVefOnZMkDRgwQIsXLy676gAAAFBpuBRMa9eurVOnTkmSIiIitHXrVklSSkqKjDFlVx0AAAAqDZeCaefOnbVq1SpJ0pNPPqnRo0erW7duevTRR/WHP/yhTAsEAABA5eDSU/nz5s1TXl6eJGnIkCEKCQnR5s2bFRsbqz//+c9lWiAAAAAqB5eCqYeHh7Kzs5WYmKj09HTZ7XZ17dpVkrR69WrFxsaWaZEAAAC4+bkUTFevXq0BAwY47jP9LZvNptzc3BL3dezYMY0fP15ffPGFfv31V/3ud7/T+++/r+bNm7tSGgAAACool+4xHTZsmPr27avjx48rLy/P6VOaUHrmzBm1a9dOXl5e+uKLL5ScnKzXX39d1atXd6UsAAAAVGAuzZimp6drzJgxqlWr1nXt/JVXXlF4eLjmz5/vWBcZGXldfQIAAKBicmnG9JFHHtGGDRuue+crV65UixYt1KdPH4WGhqpZs2Z69913i2yflZWlzMxMpw8AAABuDi7NmL711lvq06ePNm/erCZNmsjLy8tp+4gRI0rUz+HDhzV37lyNGTNGkyZN0vbt2zVixAjZ7XYNHDiwQPv4+HhNnTrVlZIBAABgcTbjwhvx33vvPQ0ZMkRVq1ZVSEiIbDbb/+/QZtPhw4dL1I+3t7datGihr7/+2rFuxIgR2rFjh7Zs2VKgfVZWlrKyshzLmZmZCg8P16tDl6iq3be0h3HDDXudtxMAAIDKLTMzU4GBgcrIyFBAQECxbV2aMf3rX/+qadOmacKECfLwcOluAElSnTp11KhRI6d1DRs21Mcff1xoe7vdLrvd7vL+AAAAYF0upcrs7Gw9+uij1xVKJaldu3bav3+/07oDBw4oIiLiuvoFAABAxeNSshw0aJCWLl163TsfPXq0tm7dqunTp+vQoUNatGiR5s2bp6FDh1533wAAAKhYXLqUn5ubq1dffVVffvmlYmJiCjz8NGPGjBL107JlS61YsUITJ07UtGnTFBUVpZkzZ+rxxx93pSwAAABUYC4F06SkJDVr1kyStHfvXqdtv30QqiR69eqlXr16uVIGAAAAbiIuBdP169eXdR0AAACo5K7v6SUAAACgjBBMAQAAYAkEUwAAAFgCwRQAAACWQDAFAACAJRBMAQAAYAkEUwAAAFgCwRQAAACWQDAFAACAJbj0m5+s5pnpPRQQEODuMgAAAHAdmDEFAACAJRBMAQAAYAkEUwAAAFgCwRQAAACWQDAFAACAJRBMAQAAYAkEUwAAAFgCwRQAAACWcFO8YP+1/xsgHy8vl78/eeFHZVgNAAAAXMGMKQAAACyBYAoAAABLIJgCAADAEgimAAAAsASCKQAAACyBYAoAAABLIJgCAADAEgimAAAAsASCKQAAACyBYAoAAABLIJgCAADAEgimAAAAsASCKQAAACzBrcF07ty5iomJUUBAgAICAtSmTRt98cUX7iwJAAAAbuLWYFq3bl29/PLL+uabb/TNN9+oc+fOevDBB/Xdd9+5sywAAAC4QRV37jw2NtZp+aWXXtLcuXO1detWNW7c2E1VAQAAwB3cGkx/Kzc3V8uWLdOFCxfUpk2bQttkZWUpKyvLsZyZmVle5QEAAOAGc/vDT0lJSapWrZrsdruGDBmiFStWqFGjRoW2jY+PV2BgoOMTHh5eztUCAADgRnF7ML3jjju0e/dubd26VX/+8581aNAgJScnF9p24sSJysjIcHyOHj1aztUCAADgRnH7pXxvb2/ddtttkqQWLVpox44dmjVrlv7+978XaGu322W328u7RAAAAJQDt8+YXs0Y43QfKQAAACoHt86YTpo0ST169FB4eLjOnTunJUuWaMOGDVq9erU7ywIAAIAbuDWY/vzzzxowYICOHz+uwMBAxcTEaPXq1erWrZs7ywIAAIAbuDWYvv/+++7cPQAAACzEcveYAgAAoHIimAIAAMASCKYAAACwBIIpAAAALIFgCgAAAEsgmAIAAMASCKYAAACwBIIpAAAALIFgCgAAAEsgmAIAAMASbMYY4+4iXJWZmanAwEBlZGQoICDA3eUAAADgKqXJa8yYAgAAwBIIpgAAALAEgikAAAAsgWAKAAAASyCYAgAAwBIIpgAAALAEgikAAAAsgWAKAAAASyCYAgAAwBIIpgAAALAEgikAAAAsgWAKAAAASyCYAgAAwBIIpgAAALAEgikAAAAsgWAKAAAASyCYAgAAwBIIpgAAALAEgikAAAAsgWAKAAAASyCYAgAAwBIIpgAAALAEtwbTTZs2KTY2VmFhYbLZbPrkk0/cWQ4AAADcyK3B9MKFC2ratKneeustd5YBAAAAC6jizp336NFDPXr0cGcJAAAAsAi3BtPSysrKUlZWlmM5MzPTjdUAAACgLFWoh5/i4+MVGBjo+ISHh7u7JAAAAJSRChVMJ06cqIyMDMfn6NGj7i4JAAAAZaRCXcq32+2y2+3uLgMAAAA3QIWaMQUAAMDNy60zpufPn9ehQ4ccyykpKdq9e7eCg4NVr149N1YGAACA8ubWYPrNN9+oU6dOjuUxY8ZIkgYNGqQFCxa4qSoAAAC4g1uDaceOHWWMcWcJAAAAsAjuMQUAAIAlEEwBAABgCQRTAAAAWALBFAAAAJZAMAUAAIAlEEwBAABgCQRTAAAAWALBFAAAAJZAMAUAAIAlEEwBAABgCQRTAAAAWALBFAAAAJZAMAUAAIAlEEwBAABgCVXcXcD1MMZIkjIzM91cCQAAAAqTn9Pyc1txKnQwPXXqlCQpPDzczZUAAACgOOfOnVNgYGCxbSp0MA0ODpYkpaWlXfNA4SwzM1Ph4eE6evSoAgIC3F1OhcLYuY6xcx1j5zrGznWMnesYu//PGKNz584pLCzsmm0rdDD18Lhyi2xgYGCl/6G7KiAggLFzEWPnOsbOdYyd6xg71zF2rmPsrijpBCIPPwEAAMASCKYAAACwhAodTO12u6ZMmSK73e7uUiocxs51jJ3rGDvXMXauY+xcx9i5jrFzjc2U5Nl9AAAA4Aar0DOmAAAAuHkQTAEAAGAJBFMAAABYAsEUAAAAllChg+mcOXMUFRUlHx8fNW/eXJs3b3Z3SeUmPj5eLVu2lL+/v0JDQ9W7d2/t37/fqY0xRnFxcQoLC1PVqlXVsWNHfffdd05tsrKyNHz4cNWoUUN+fn564IEH9OOPPzq1OXPmjAYMGKDAwEAFBgZqwIABOnv27I0+xHITHx8vm82mUaNGOdYxdkU7duyY/vjHPyokJES+vr668847tXPnTsd2xq5wly9f1l//+ldFRUWpatWqql+/vqZNm6a8vDxHG8buik2bNik2NlZhYWGy2Wz65JNPnLaX5zilpaUpNjZWfn5+qlGjhkaMGKHs7Owbcdhlorixy8nJ0fjx49WkSRP5+fkpLCxMAwcO1E8//eTUB2NX+Hn3W88884xsNptmzpzptL6yjl2ZMhXUkiVLjJeXl3n33XdNcnKyGTlypPHz8zNHjhxxd2nl4r777jPz5883e/fuNbt37zY9e/Y09erVM+fPn3e0efnll42/v7/5+OOPTVJSknn00UdNnTp1TGZmpqPNkCFDzC233GISEhJMYmKi6dSpk2natKm5fPmyo83vf/97Ex0dbb7++mvz9ddfm+joaNOrV69yPd4bZfv27SYyMtLExMSYkSNHOtYzdoU7ffq0iYiIME888YTZtm2bSUlJMWvWrDGHDh1ytGHsCvfiiy+akJAQ85///MekpKSYZcuWmWrVqpmZM2c62jB2V3z++edm8uTJ5uOPPzaSzIoVK5y2l9c4Xb582URHR5tOnTqZxMREk5CQYMLCwsywYcNu+Bi4qrixO3v2rOnatatZunSp+f77782WLVtMq1atTPPmzZ36YOwKP+/yrVixwjRt2tSEhYWZN954w2lbZR27slRhg+ndd99thgwZ4rSuQYMGZsKECW6qyL3S09ONJLNx40ZjjDF5eXmmdu3a5uWXX3a0uXTpkgkMDDTvvPOOMebKH1JeXl5myZIljjbHjh0zHh4eZvXq1cYYY5KTk40ks3XrVkebLVu2GEnm+++/L49Du2HOnTtnbr/9dpOQkGA6dOjgCKaMXdHGjx9v2rdvX+R2xq5oPXv2NH/605+c1j300EPmj3/8ozGGsSvK1QGhPMfp888/Nx4eHubYsWOONosXLzZ2u91kZGTckOMtS8WFq3zbt283khyTOozdFUWN3Y8//mhuueUWs3fvXhMREeEUTBm7slEhL+VnZ2dr586d6t69u9P67t276+uvv3ZTVe6VkZEhSQoODpYkpaSk6MSJE05jZLfb1aFDB8cY7dy5Uzk5OU5twsLCFB0d7WizZcsWBQYGqlWrVo42rVu3VmBgYIUf66FDh6pnz57q2rWr03rGrmgrV65UixYt1KdPH4WGhqpZs2Z69913HdsZu6K1b99ea9eu1YEDByRJ3377rb766ivdf//9khi7kirPcdqyZYuio6MVFhbmaHPfffcpKyvL6faViiwjI0M2m03Vq1eXxNgVJy8vTwMGDNBzzz2nxo0bF9jO2JWNKu4uwBUnT55Ubm6uatWq5bS+Vq1aOnHihJuqch9jjMaMGaP27dsrOjpakhzjUNgYHTlyxNHG29tbQUFBBdrkf//EiRMKDQ0tsM/Q0NAKPdZLlixRYmKiduzYUWAbY1e0w4cPa+7cuRozZowmTZqk7du3a8SIEbLb7Ro4cCBjV4zx48crIyNDDRo0kKenp3Jzc/XSSy+pf//+kjjvSqo8x+nEiRMF9hMUFCRvb++bYiwvXbqkCRMm6LHHHlNAQIAkxq44r7zyiqpUqaIRI0YUup2xKxsVMpjms9lsTsvGmALrKoNhw4Zpz549+uqrrwpsc2WMrm5TWPuKPNZHjx7VyJEj9d///lc+Pj5FtmPsCsrLy1OLFi00ffp0SVKzZs303Xffae7cuRo4cKCjHWNX0NKlS7Vw4UItWrRIjRs31u7duzVq1CiFhYVp0KBBjnaMXcmU1zjdrGOZk5Ojfv36KS8vT3PmzLlm+8o+djt37tSsWbOUmJhY6vor+9iVVoW8lF+jRg15enoW+JdDenp6gX9l3OyGDx+ulStXav369apbt65jfe3atSWp2DGqXbu2srOzdebMmWLb/PzzzwX2+8svv1TYsd65c6fS09PVvHlzValSRVWqVNHGjRs1e/ZsValSxXFcjF1BderUUaNGjZzWNWzYUGlpaZI474rz3HPPacKECerXr5+aNGmiAQMGaPTo0YqPj5fE2JVUeY5T7dq1C+znzJkzysnJqdBjmZOTo759+yolJUUJCQmO2VKJsSvK5s2blZ6ernr16jn+3jhy5IieffZZRUZGSmLsykqFDKbe3t5q3ry5EhISnNYnJCSobdu2bqqqfBljNGzYMC1fvlzr1q1TVFSU0/aoqCjVrl3baYyys7O1ceNGxxg1b95cXl5eTm2OHz+uvXv3Otq0adNGGRkZ2r59u6PNtm3blJGRUWHHukuXLkpKStLu3bsdnxYtWujxxx/X7t27Vb9+fcauCO3atSvwWrIDBw4oIiJCEuddcS5evCgPD+c/cj09PR2vi2LsSqY8x6lNmzbau3evjh8/7mjz3//+V3a7Xc2bN7+hx3mj5IfSgwcPas2aNQoJCXHaztgVbsCAAdqzZ4/T3xthYWF67rnn9OWXX0pi7MpMuT1mVcbyXxf1/vvvm+TkZDNq1Cjj5+dnUlNT3V1aufjzn/9sAgMDzYYNG8zx48cdn4sXLzravPzyyyYwMNAsX77cJCUlmf79+xf6SpW6deuaNWvWmMTERNO5c+dCX20RExNjtmzZYrZs2WKaNGlSoV49UxK/fSrfGMauKNu3bzdVqlQxL730kjl48KD517/+ZXx9fc3ChQsdbRi7wg0aNMjccsstjtdFLV++3NSoUcOMGzfO0Yaxu+LcuXNm165dZteuXUaSmTFjhtm1a5fjyfHyGqf81/Z06dLFJCYmmjVr1pi6deta+rU9xY1dTk6OeeCBB0zdunXN7t27nf7uyMrKcvTB2BV+3l3t6qfyjam8Y1eWKmwwNcaYt99+20RERBhvb29z1113OV6VVBlIKvQzf/58R5u8vDwzZcoUU7t2bWO32829995rkpKSnPr59ddfzbBhw0xwcLCpWrWq6dWrl0lLS3Nqc+rUKfP4448bf39/4+/vbx5//HFz5syZcjjK8nN1MGXsirZq1SoTHR1t7Ha7adCggZk3b57TdsaucJmZmWbkyJGmXr16xsfHx9SvX99MnjzZKRAwdlesX7++0D/fBg0aZIwp33E6cuSI6dmzp6lataoJDg42w4YNM5cuXbqRh39dihu7lJSUIv/uWL9+vaMPxq7w8+5qhQXTyjp2ZclmjDHlMTMLAAAAFKdC3mMKAACAmw/BFAAAAJZAMAUAAIAlEEwBAABgCQRTAAAAWALBFAAAAJZAMAUAAIAlEEwBAABgCQRTAAAAWALBFADKyIkTJzR8+HDVr19fdrtd4eHhio2N1dq1a8u1DpvNpk8++aRc9wkAZaGKuwsAgJtBamqq2rVrp+rVq+vVV19VTEyMcnJy9OWXX2ro0KH6/vvv3V0iAFiezRhj3F0EAFR0999/v/bs2aP9+/fLz8/PadvZs2dVvXp1paWlafjw4Vq7dq08PDz0+9//Xm+++aZq1aolSXriiSd09uxZp9nOUaNGaffu3dqwYYMkqWPHjoqJiZGPj4/ee+89eXt7a8iQIYqLi5MkRUZG6siRI47vR0REKDU19UYeOgCUGS7lA8B1On36tFavXq2hQ4cWCKWSVL16dRlj1Lt3b50+fVobN25UQkKCfvjhBz366KOl3t8HH3wgPz8/bdu2Ta+++qqmTZumhIQESdKOHTskSfPnz9fx48cdywBQEXApHwCu06FDh2SMUYMGDYpss2bNGu3Zs0cpKSkKDw+XJP3zn/9U48aNtWPHDrVs2bLE+4uJidGUKVMkSbfffrveeustrV27Vt26dVPNmjUlXQnDtWvXvo6jAoDyx4wpAFyn/DuibDZbkW327dun8PBwRyiVpEaNGql69erat29fqfYXExPjtFynTh2lp6eXqg8AsCKCKQBcp9tvv102m63YgGmMKTS4/na9h4eHrr7tPycnp8B3vLy8nJZtNpvy8vJcKR0ALIVgCgDXKTg4WPfdd5/efvttXbhwocD2s2fPqlGjRkpLS9PRo0cd65OTk5WRkaGGDRtKkmrWrKnjx487fXf37t2lrsfLy0u5ubml/h4AuBvBFADKwJw5c5Sbm6u7775bH3/8sQ4ePKh9+/Zp9uzZatOmjbp27aqYmBg9/vjjSkxM1Pbt2zVw4EB16NBBLVq0kCR17txZ33zzjT788EMdPHhQU6ZM0d69e0tdS2RkpNauXasTJ07ozJkzZX2oAHDDEEwBoAxERUUpMTFRnTp10rPPPqvo6Gh169ZNa9eu1dy5cx0vvQ8KCtK9996rrl27qn79+lq6dKmjj/vuu0/PP/+8xo0bp5YtW+rcuXMaOHBgqWt5/fXXlZCQoPDwcDVr1qwsDxMAbijeYwoAAABLYMYUAAAAlkAwBQAAgCUQTAEAAGAJBFMAAABYAsEUAAAAlkAwBQAAgCUQTAEAAGAJBFMAAABYAsEUAAAAlkAwBQAAgCUQTAEAAGAJ/w/VpTRqv83WVQAAAABJRU5ErkJggg==
<Figure size 800x400 with 1 Axes>
output_type": "display_data
name": "stdout
output_type": "stream


occupation distribution:

9     0.183760

2     0.125826

3     0.124935

0     0.115807

11    0.112180

7     0.101146

6     0.061468

13    0.049083

5     0.042075

4     0.030488

12    0.028491

10    0.019947

8     0.004518

1     0.000277

Name: occupation, dtype: float64

image/png": "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JiQkxJw4ccLtPjfffLNbu08++cRIMmvXri32foXatm1rIiMjzcmTJ13Hzp07Z5o1a2Zq1aplCgoKjDHGZGRkGEnmlVdeueD1jDHm6aefNpLM+vXr3Y4//PDDxmazmV27dhljjJk+fbqRZCZPnlzitVauXGkkmWefffaC9yyp3+Pi4syAAQNc+4X9dccdd7i1+9///mckmX/84x/FXr+goMCcPXvW7N+/30gy//nPf1znXnnllWL/HTDGmM6dO5vOnTu79t99910jyXzyySdu7V566SUjySxatMjtZ6pZs6bJzc11HTt06JCpUKGCSU1NLbZOAKXHtAEAl8QYc8Hz11xzjQIDA/WXv/xF06ZNK/Ir3ot11113XXTbpk2bqkWLFm7H7rvvPuXm5mrz5s2luv/FWrp0qbp27arY2Fi34wMHDtTp06e1du1at+O33nqr235CQoIkaf/+/SXeIy8vT+vXr9fdd9+typUru45XrFhR/fr1048//njRUw9+X3uTJk107bXXFqndGKOlS5dKkhYsWKCgoCA98MADJV5rwYIFkqQhQ4Zcch0X0rdvX7f99u3bKy4uTsuWLXMdy87O1uDBgxUbG6tKlSopICBAcXFxkqSdO3eW6r5Lly5VaGio7r77brfjhVMblixZ4nY8KSlJYWFhrv2aNWsqMjLygv0KoHQIrwAuWl5eno4ePaqYmJgS29SrV0+LFy9WZGSkhgwZonr16qlevXp68803L+lehb/WvRhRUVElHjt69Ogl3fdSHT16tNhaC7+j39+/WrVqbvuFv9b/5ZdfSrzH8ePHZYy5pPtcjIut/fDhw4qJiVGFCiX/L+Pw4cOqWLFisX1xOUrq28LaCgoK1L17d82ePVtPPfWUlixZoq+//lrr1q2TdOHv9UKOHj2qqKioInOsIyMjValSpT/sV+m3vi3t/QGUjPAK4KLNnz9f+fn5f7i8VadOnfTFF18oJydH69atU7t27TRs2DDNmjXrou91KWvHHjp0qMRjhaEiKChIkoo8bHP+Qz2lUa1aNR08eLDI8QMHDkiSqlevflnXl6SqVauqQoUKHr/PxdZeo0YNHThwoMhDXOerUaOG8vPzi+2L89nt9iJ9IJUcvkvq28J+3b59u7755hu98sorGjp0qBITE9WmTZtiw+SlqFatmn7++eciv2nIzs7WuXPnPNKvAEqH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unbt6ukaAQAAAEmlXOf12LFj+vXXXxUTE6OCggK9+uqrWr16terXr69Ro0apatWqZVFrqVzKumEAAADwvkvJa6UKr/6E8AoAAODbyvwlBZKUn5+vOXPmaOfOnbLZbGrcuLFuu+02VapU6ksCAAAAF1SqpLl9+3bddtttOnTokBo2bCjpt5cH1KhRQ/PmzVPz5s09WiQAAAAglXLaQNu2bRUZGalp06a55rceP35cAwcOVHZ2ttauXevxQkurcBi6+7T3FBASYnU5uIJ8eTfrEQMAcDHKfNrAN998o40bN7o9mFW1alWNHTtWbdq0Kc0lAQAAgD9UqnVeGzZsqJ9//rnI8ezsbNWvX/+yiwIAAACKU+rXwz722GP67LPP9OOPP+rHH3/UZ599pmHDhumll15Sbm6uawMAAAA8pVTTBgpf19qrVy/ZbDZJUuHU2Z49e7r2bTab8vPzPVEnAAAAULrwumzZMo/cfOXKlXrllVe0adMmHTx4UHPmzNHtt9/uOp+SkqJZs2YpKytLgYGBatWqlcaOHavrrrvOI/cHAACAfylVeO3cubNHbp6Xl6cWLVro/vvv11133VXk/NVXX60JEyaobt26+uWXX/T666+re/fu2rNnj2rUqOGRGgAAAOA/ShVeV65cecHz119//UVdJzk5WcnJySWev++++9z2x48frylTpmjr1q3q2rXrRd0DAAAA5UepwmtiYmKRY4VzXyWVyTzXM2fO6L333pPD4VCLFi1KbOd0OuV0Ol37PDQGAABQfpRqtYHjx4+7bdnZ2Vq4cKHatGmjRYsWebTAL7/8UpUrV1ZQUJBef/11paWlqXr16iW2T01NlcPhcG2xsbEerQcAAADWKdXIq8PhKHKsW7dustvteuKJJ7Rp06bLLqxQUlKS0tPTdeTIEU2ePFm9evXS+vXrFRkZWWz7kSNHavjw4a793NxcAiwAAEA5UaqR15LUqFFDu3bt8uQlFRoaqvr166tt27aaMmWKKlWqpClTppTY3m63Kzw83G0DAABA+VCqkdetW7e67RtjdPDgQY0bN+6C81E9wRjjNqcVAAAAV45ShddrrrlGNpvN9WKCQm3bttW///3vi77OqVOntGfPHtd+RkaG0tPTFRERoWrVqmns2LG69dZbFR0draNHj2rixIn68ccfdc8995SmbAAAAPi5UoXXjIwMt/0K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
<Figure size 800x400 with 1 Axes>
output_type": "display_data
name": "stdout
output_type": "stream


relationship distribution:

0    0.405292

1    0.254848

3    0.155638

4    0.105879

5    0.048191

2    0.030150

Name: relationship, dtype: float64

image/png": "iVBORw0KGgoAAAANSUhEUgAAAqYAAAGHCAYAAABiY5CRAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAzBUlEQVR4nO3deVhV5f7//9eWYSMIKBgigThkToiWZpnmkEMOaWV90jLFOv7Kk2M2OH0KrVNYnePHrNSmY5aVdk5alh0KFS1zTLScMi0RKglzAMNChPv7x7nYP7egwmbLWsjzcV37ulz3utda73VD+upew3YYY4wAAAAAi9WwugAAAABAIpgCAADAJgimAAAAsAWCKQAAAGyBYAoAAABbIJgCAADAFgimAAAAsAWCKQAAAGyBYAoAAABbIJgCl7A333xTDofD9QkICFBkZKS6d++upKQkZWdnl9hm+vTpcjgc5TrOyZMnNX36dK1Zs6Zc25V2rIYNG+rmm28u134u5N1339Xs2bNLXedwODR9+nSvHs/bVq1apfbt2ysoKEgOh0MffvjhRTtW8e9Menp6ubddv369pk+fruPHj5dY161bN3Xr1q3C9VVUt27dFBcXd8F+6enpcjgcevPNNy9+UQBcfK0uAMDFt2DBAjVv3lwFBQXKzs7WunXr9Oyzz+rvf/+7lixZop49e7r6jhw5Un369CnX/k+ePKkZM2ZIUrnChyfH8sS7776rnTt3asKECSXWbdiwQdHR0Re9Bk8ZY3TnnXfqyiuv1PLlyxUUFKRmzZpZXVap1q9frxkzZmjEiBGqXbu227q5c+daU5SH6tevrw0bNqhJkyZWlwJUKwRToBqIi4tT+/btXcu33367HnroIXXu3FmDBg3Svn37VK9ePUlSdHT0RQ9qJ0+eVGBgYKUc60Kuu+46S49/Ib/88ouOHj2q2267TT169Cj39gUFBXI4HPL1tfav+5YtW1p6/PJyOp22/90ALkVcygeqqQYNGugf//iHTpw4oVdeecXVXtrl9dWrV6tbt24KDw9XzZo11aBBA91+++06efKk0tPTddlll0mSZsyY4bptYMSIEW77S0tL0x133KE6deq4ZqHOd9vAsmXLFB8fr4CAADVu3Fhz5sxxW3+uS85r1qyRw+Fw3VbQrVs3rVixQgcPHnS7raFYaZfyd+7cqVtuuUV16tRRQECA2rZtq4ULF5Z6nPfee0/Tpk1TVFSUQkJC1LNnT+3du/fcA3+GdevWqUePHgoODlZgYKCuv/56rVixwrV++vTpruA+adIkORwONWzY8Jz7K67p7bff1sMPP6zLL79cTqdT+/fvlyStXLlSPXr0UEhIiAIDA9WpUyetWrXqgnWmpKTolltuUXR0tAICAnTFFVfogQce0G+//eZW66OPPipJatSokWucz/w5nD2bfvToUT344IO6/PLL5e/vr8aNG2vatGnKz8936+dwODRmzBi9/fbbatGihQIDA9WmTRt98sknbv0OHz6s+++/XzExMXI6nbrsssvUqVMnrVy5ssQ5bdmyRTfccIMCAwPVuHFjzZw5U0VFRa71pV3KL/593bZtmwYNGqSQkBCFhobqnnvu0eHDhy84jgAujBlToBrr16+ffHx89MUXX5yzT3p6uvr3768bbrhB//znP1W7dm39/PPPSk5O1qlTp1S/fn0lJyerT58++stf/qKRI0dKkiusFhs0aJCGDBmiUaNGKS8v77x1bd++XRMmTND06dMVGRmpd955R+PHj9epU6f0yCOPlOsc586dq/vvv18//PCDli1bdsH+e/fu1fXXX6+IiAjNmTNH4eHhWrRokUaMGKFff/1Vjz32mFv/qVOnqlOnTnr99deVm5urSZMmacCAAdqzZ498fHzOeZy1a9eqV69eio+P1xtvvCGn06m5c+dqwIABeu+99zR48GCNHDlSbdq00aBBgzR27FjdfffdcjqdFzyHKVOmqGPHjpo/f75q1KihiIgILVq0SMOHD9ctt9yihQsXys/PT6+88opuuukmffbZZ+edjf3hhx/UsWNHjRw5UqGhoUpPT9esWbPUuXNn7dixQ35+fho5cqSOHj2qF198UUuXLlX9+vUlnXum9M8//1T37t31ww8/aMaMGYqPj9eXX36ppKQkbd++3S2gS9KKFSu0ZcsWPfnkk6pVq5aee+453Xbbbdq7d68aN24sSRo2bJjS0tL09NNP68orr9Tx48eVlpamI0eOuO0rKytLQ4cO1cMPP6zExEQtW7ZMU6ZMUVRUlIYPH37B8b3tttt05513atSoUdq1a5cef/xx7d69W5s2bZKfn98FtwdwHgbAJWvBggVGktmyZcs5+9SrV8+0aNHCtZyYmGjO/Kvh3//+t5Fktm/ffs59HD58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evXVV71WGAAAAKoXj4Lp3Xff7fpa0qysLPXs2VObN2/W1KlT9eSTT3q1QAAAAFQPHgXTnTt3qkOHDpKk999/X61bt9b69ev17rvv6s033/RmfQAAAKgmPAqmBQUFrq8GXblypQYOHChJat68uQ4dOuS96gAAAFBteBRMW7Vqpfnz5+vLL79USkqK+vTpI0n65ZdfFB4e7tUCAQAAUD14FEyfffZZvfLKK+rWrZvuuusutWnTRpK0fPly1yV+AAAAoDw8+uanbt266bffflNubq7q1Knjar///vsr9E1QAAAAqL48CqaS5OPj4xZKJalhw4YVrQcAAADVlEeX8n/99VcNGzZMUVFR8vX1lY+Pj9sHAAAAKC+PZkxHjBihjIwMPf7446pfv74cDoe36wIAAEA141EwXbdunb788ku1bdvWy+UAAACguvIomMbExMgY4+1aPJYyKkUhISFWlwEAAIAK8Oge09mzZ2vy5MlKT0/3cjkAAACorjyaMR08eLBOnjypJk2aKDAwUH5+fm7rjx496pXiAAAAUH14FExnz57t5TIAAABQ3XkUTBMSErxdBwAAAKo5j1+wX1hYqA8//FB79uyRw+FQy5YtNXDgQN5jCgAAAI94FEz379+vfv366eeff1azZs1kjNH333+vmJgYrVixQk2aNPF2nQAAALjEefRU/rhx49SkSRNlZmYqLS1N27ZtU0ZGhho1aqRx48Z5u0YAAABUAx7NmK5du1YbN25UWFiYqy08PFwzZ85Up06dvFYcAAAAqg+PZkydTqdOnDhRov3333+Xv79/hYsCAABA9ePRjOnNN9+s+++/X2+88YY6dOggSdq0aZNGjRqlgQMHerXAsljXp6+CfD1+jgu4oK5frLW6BAAALnkezZjOmTNHTZo0UceOHRUQEKCAgAB16tRJV1xxhV544QVv1wgAAIBqwKNpxtq1a+ujjz7Svn379N1338kYo5YtW+qKK67wdn0AAACoJip0/btp06Zq2rSpt2oBAABANVbmYDpx4kQ99dRTCgoK0sSJE8/bd9asWRUuDAAAANVLmYPptm3bVFBQ4PozAAAA4E1lDqapqaml/hkAAADwBo+eyr/vvvtKfY9pXl6e7rvvvgoXBQAAgOrHo2C6cOFC/fHHHyXa//jjD7311lsVLgoAAADVT7meys/NzZUxRsYYnThxQgEB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1+6inqcdFIfSffv2aeXKlQoPD3dbzzjbQGU+aeVNxa+LeuONN8zu3bvNhAkTTFBQkElPT7e6NNv561//akJDQ82aNWvMoUOHXJ+TJ0+6+sycOdOEhoaapUuXmh07dpi77rqr1FeVREdHm5UrV5q0tDRz4403lvoKjfj4eLNhwwazYcMG07p160vy1S9lceZT+cYwxt6wefNm4+vra55++mmzb98+884775jAwECzaNEiVx/GueISEhLM5Zdf7npd1NKlS03dunXNY4895urDOJfPiRMnzLZt28y2bduMJDNr1iyzbds219PglTWexa8x6tGjh0lLSzMrV6400dHRl8xrjM43zgUFBWbgwIEmOjrabN++3e3fw/z8fNc+GGdrVdlgaowxL7/8somNjTX+/v7m6quvdr3+CO4klfpZsGCBq09RUZFJTEw0kZGRxul0mi5dupgdO3a47eePP/4wY8aMMWFhYaZmzZrm5ptvNhkZGW59jhw5YoYOHWqCg4NNcHCwGTp0qDl27FglnKX9nB1MGWPv+Pjjj01cXJxxOp2mefPm5tVXX3VbzzhXXG5urhk/frxp0KCBCQgIMI0bNzbTpk1z+8ebcS6f1NTUUv8eTkhIMMZU7ngePHjQ9O/f39SsWdOEhYWZMWPGmD///PNinn6lOd84Hzhw4Jz/Hqamprr2wThby2GMMZU3PwsAAACUrkreYwoAAIBLD8EUAAAAtkAwBQAAgC0QTAEAAGALBFMAAADYAsEUAAAAtkAwBQAAgC0QTAEAAGALBFMAAADYAsEUALwkKytLY8eOVePGjeV0OhUTE6MBAwZo1apVlVqHw+HQhx9+WKnHBABv8LW6AAC4FKSnp6tTp06qXbu2nnvuOcXHx6ugoECfffaZRo8ere+++87qEgHA9hzGGGN1EQBQ1fXr10/ffvut9u7dq6CgILd1x48fV+3atZWRkaGxY8dq1apVqlGjhvr06aMXX3xR9erVkySNGDFCx48fd5vtnDBhgrZv3641a9ZIkrp166b4+HgFBATo9ddfl7+/v0aNGqXp06dLkho2bKiDBw+6to+NjVV6evrFPHUA8Bou5QNABR09elTJyckaPXp0iVAqSbVr15YxRrfeequOHj2qtWvXKiUlRT/88IMGDx5c7uMtXLhQQUFB2rRpk5577jk9+eSTSklJkSRt2bJFkrRgwQIdOnTItQwAVQGX8gGggvbv3y9jjJo3b37OPitXrtS3336rAwcOKCYmRpL09ttvq1WrVtqyZYuuueaaMh8vPj5eiYmJkqSmTZvqpZde0qpVq9SrVy9ddtllkv4bhiMjIytwVgBQ+ZgxBYAKKr4jyuFwnLPPnj17FBMT4wqlktSyZUvVrl1be/bsKdfx4uPj3Zbr16+v7Ozscu0DAOyIYAoAFdS0aVM5HI7zBkxjTKnB9cz2GjVq6Ozb/gsKCkps4+fn57bscDhUVFTkSekAYCsEUwCooLCwMN100016+eWXlZeXV2L98ePH1bJlS2VkZCgzM9PVvnv3buXk5KhFixaSpMsuu0yHDh1y23b79u3lrsfPz0+FhYXl3g4ArEYwBQAvmDt3rgoLC9WhQwd98MEH2rdvn/bs2aM5c+aoY8eO6tmzp+Lj4zV06FClpaVp8+bNGj58uLp27ar27dtLkm688UZ9/fXXeuutt7Rv3z4lJiZq586d5a6lYcOGWrVqlbKysnTs2DFvnyoAXDQEUwDwgkaNGiktLU3du3fXww8/rLi4OPXq1UurVq3SvHnzXC+9r1Onjrp06aKePXuqcePGWrJkiWsfN910kx5//HE99thjuuaaa3TixAkNHz683LX84x//UEpKimJiYnTVVVd58zQB4KLiPaYAAACwBWZMAQAAYAsEUwAAANgCwRQAAAC2QDAFAACALRBMAQAAYAsEUwAAANgCwRQAAAC2QDAFAACALRBMAQAAYAsEUwAAANgCwRQAAAC28P8AklHVru4EBP4AAAAASUVORK5CYII=
<Figure size 800x400 with 1 Axes>
output_type": "display_data
name": "stdout
output_type": "stream


race distribution:

4    0.854258

2    0.095952

1    0.031902

0    0.009558

3    0.008329

Name: race, dtype: float64

image/png": "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
<Figure size 800x400 with 1 Axes>
output_type": "display_data
name": "stdout
output_type": "stream


sex distribution:

1    0.669238

0    0.330762

Name: sex, dtype: float64

image/png": "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iIgKx7Ru3Vrr1q1T8+bNNX78eLVu3VqtW7cu97nFcwkPDz/vsWFhYRVuO3z4cJWOW1WHDx8+61zP/Bn99vhNmjRxu+90OiVJP//8c4XHOHr0qIwxVTrO+Xj88cf13HPPadu2bUpMTFSTJk3Ut29fffHFF659njp1Ss8//7x8fHzcbgMHDpQk/fjjj277HDt2rE6ePKkGDRrowQcfrPKcANiPMAVghdWrV6usrOycP/HUs2dPffDBByooKNC2bduUkJCgSZMmaenSped9rKr8NmpeXl6F286EoJ+fnySptLTUbdxvw6qqmjRpotzc3HLbv//+e0lS06ZNL2j/khQSEiIvL68aP06DBg00efJk7dy5U0eOHNGSJUuUk5OjAQMGqKSkRCEhIfL29taYMWO0ffv2s97OBKr0yz9cRo0apTZt2sjf31/33ntv9U8agLUIUwAel52drYcffljBwcH64x//eF7P8fb2Vvfu3V3f4D7ztvr5XCWsin379unLL7902/b2228rMDBQ11xzjSS5vp2+e/dut3GrVq0qtz+n03nec+vbt682bNjgCsQzFi9erIYNG9bIz0sFBASoe/fuWr58udu8Tp8+rTfffFORkZFq06bNBR2jUaNGuu222zR+/HgdOXJEmZmZatiwoXr37q1du3YpNjZWcXFx5W6/vgI8btw4ZWdna/ny5Xr11Ve1atUq/fOf/7ygeQGwD79jCqBW7d271/VZwvz8fH388cdauHChvL29tWLFCtfPPZ3NSy+9pA0bNmjQoEFq0aKFjh8/rtdee02SXD/MHxgYqOjoaP373/9W37591bhxYzVt2rTSnzaqTEREhIYOHark5GSFh4frzTffVGpqqp599lk1bNhQktS1a1ddddVVevjhh3Xq1CmFhIRoxYoV2rp1a7n9dezYUcuXL9f8+fPVpUsXeXl5uf2u668lJSXpww8/VO/evTV16lQ1btxYb731llavXq0ZM2YoODi4Wuf0W9OmTVO/fv3Uu3dvPfzww/L19dW8efO0d+9eLVmypMr/9y1JGjJkiOs3a5s1a6asrCzNnj1b0dHRuvLKKyVJc+bM0XXXXaeePXvq/vvvV8uWLVVUVKSvv/5aH3zwgTZs2CBJWrBggd58800tXLhQHTp0UIcOHTRhwgQ99thjuvbaa9WtW7ca+XMAYAFPf/sKQP1w5pvr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eLjy8/OrtA8AsBFhCgAX6Morr5TD4ag0MI0xZw3XX2/38vLSbz/2f/LkyXLP8fHxcbvvcDh0+vTp6kwdAKxCmALABWrcuLEGDBigF198UcXFxeUeP3bsmNq3b6/s7Gzl5OS4tu/fv18FBQVq166dJKlZs2bKzc11e256enqV5+Pj46OysrIqPw8API0wBYAaMG/ePJWVlalbt256//33lZGRoQMHDmju3LlKSEjQjTfeqNjYWI0cOVI7d+7U559/rrvvvls33HCD4uLiJEl9+vTRF198ocWLFysjI0NJSUnau3dvlefSsmVLrV+/Xnl5eTp69GhNnyoAXDSEKQDUgFatWmnnzp3q3bu3pkyZopiYGPXr10/r16/X/PnzXT96HxISouuvv1433nijLr/8ci1btsy1jwEDBuivf/2rHn30UXXt2lVFRUW6++67qzyXmTNnKjU1VVFRUercuXNNniYAXFT8jikAAACswBVTAAAAWIEwBQAAgBUIUwAAAFiBMAUAAIAVCFMAAABYgTAFAACAFQhTAAAAWIEwBQAAgBUIUwAAAFiBMAUAAIAVCFMAAABY4f8BzklKoflmGgAAAAAASUVORK5CYII=
<Figure size 800x400 with 1 Axes>
output_type": "display_data
name": "stdout
output_type": "stream


native_country distribution:

38    0.913883

25    0.019639

29    0.006085

10    0.004211

1     0.003719

32    0.003504

7     0.003258

18    0.003073

4     0.002920

8     0.002766

22    0.002489

34    0.002459

2     0.002305

21    0.002244

5     0.002151

39    0.002059

23    0.001906

12    0.001906

30    0.001844

3     0.001813

35    0.001567

13    0.001352

19    0.001322

31    0.001137

26    0.001045

28    0.000953

9     0.000891

11    0.000891

6     0.000861

20    0.000738

16    0.000615

0     0.000584

37    0.000584

24    0.000553

36    0.000553

40    0.000492

27    0.000430

15    0.000400

17    0.000400

33    0.000369

14    0.000031

Name: native_country, dtype: float64

image/png": "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9z/JcImIiIjoCWi2r4fVVAarOre+fPky3NzckJGRgR49etTaB0tlERERERkuvak2UJ8yWPXBUllEREREhktvktf6lMGqD5bKIiIiIjJcerFsoL5lsOqDpbKIiIiIDJdOk1chBEJCQhATE4OkpCS4ubk12lifj4lutL6JiIiIqGlotWzgwVJWj2PKlCnYtGkTtmzZAktLS+Tl5dUogzVnzhxIJBJVYtuzZ0+0bdsWeXl5TyQGIiIiImo+tEpe7ezsMHHiRBw+fPixBl+9ejUUCgX8/Pzg4OCg2rZv365q8/BLCwCgoKAA33zzTYPGeiOOdV6JiIiImjutlg1s3boV69evx6BBg+Dq6oqJEyfizTffhKOjY4P6qU+J2R49euDy5cvIzMzUJlQiIiIiMiBazbyOHDkSUVFRuHbtGt577z1s3boVrq6ueP755xEdHY2KioonGmRWVhYcHR3h5uaGcePG4c8//6y1rVKpRFFRkdpGRERERIbhsUpltWnTBjNnzsTJkyfx5ZdfYv/+/XjppZfg6OiI+fPno7S09LED7Nu3LzZu3Ii9e/fiu+++Q15eHnx8fFBQUKCxPeu8EhERERmux3o9bF5eHjZu3IjIyEhkZ2fjhRdewKRJk3Dt2jUsXrwYDg4O2Ldv35OMFyUlJXB3d8eHH36I0NDQGsc1vWHL2dkZI7cMQuxrfD0sERERkb5p9NfDRkdHIzIyEnv37oWnpyemTJmCN954A61atVK16dGjB3r27KlN93WysLBA165dkZWVpfE467wSERERGS6tlg289dZbaNeuHX755RdkZmZi6tSpaokrADz11FOYN29enf2sXr0a3bp1g5WVFaysrODt7Y09e/aojl+/fh3BwcFwdHREixYtMHToUPz22284d+4cHBwcGhTzphGs80pERETU3DV42UBFRQW+/fZbjB07Fvb29o81+O7du2FsbIwOHToAADZs2IClS5ciIyMDnp6e8PHxQU5ODj755BN06NABX3/9Nfbu3QtjY2OcOXMGrq6ujxyjIdPQRERERNT0GpKvabXmtUWLFjh37ly9kseGsra2xtKlS/Hcc8/Bw8MDQ4cOxcmTJ3Hz5k20bdsWBQUFmDdvHubPn1+v/pi8EhEREem3Rl/z2rdvX2RkZDzR5LWyshI7duxASUkJvL29VQ9dff3113B3d1e1c3BweGSp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yckaX0KQm5sLV1dXbNu2DWPHjtXYH9+wRURERNS8NOQNW1qVylqyZAkGDBgADw8PPPfccwCAQ4cOoaioCImJiQ3qqyE1XB0cHODq6qp6WEwT1nklIiIiMlxaLRvw9PTEqVOn8MorryA/Px/FxcV488038fvvv8PLy6ve/aSkpGDkyJFwdHSERCLBzp071Wq4Lly4EJ07d4aFhQVat24NX19fXLlyBQ4ODtqETURERETNnFbLBurr/fffx6effoq2bdtqPP7qq69CJpPBx8cH7733Hl588UXExMQgPj4e3t7eePXVVxEYGIgePXrg4sWL+Oijj3Djxg1cvHgRbm5u9YqhIdPQRERERNT0GpKvaTXzWl+bNm2q82n/li1bIiUlBdOnTwdwv05sdQ1XY2NjVFRU4N///jcCAgKwcOFCDBo0CEII/Pnnn40ZNhERERHpKa3WvNbXoyZ1161bp/pviUSCRYsWqWq4mpubY+/evarjZWVlWLlyJX7++Wd079691j41PbBFRERERIahUWden4SffvoJLVu2hJmZGf773/8iISGh1mUIAEtlERERERkyvU9e/f39kZmZiSNHjmDo0KGqh8RqExYWBoVCodpycnKaMFoiIiIiakx6n7xaWFigQ4cO6NevH9atWwcTExO15QYPk8lksLKyUtuIiIiIyDDoffL6sAdLaRERERHR30ujJq9vvPFGnTOf8fHx8PX1hY2NDQAgNjYWmZmZyM7ORklJCaZPn44RI0bAzs4OZmZmcHJyQnZ2Nl5++eXGDJuIiIiI9JTWyeuhQ4fwxhtvwNvbG3/99RcA4IcffsDhw4dVbVavXl3nw1UnT55ESkoKbt68CQCIjIxEz549MX/+fBgZGeGHH37A/v37cevWLVhZWcHExARt2rRB+/bttQ2biIiIiJoxrZLXqKgoBAYGwtzcHBkZGap/xi8uLsbnn39e737mzJkDIYSqpFZMTAyEEFi/fj1ycnJw+/ZtnDhxAuXl5cjPz8fFixdRVlaGrVu3ahM2ERERETVzWiWv//73v/HNN9/gu+++g6mpqWq/j48PTpw48UQCq06IzczMVPuMjY0hlUrVZnc1nVdUVKS2EREREZFh0Cp5PX/+PAYMGFBjv5WVFQoLCx83JgBA586d4erqirCwMNy+fRtlZWVYvHgx8vLykJubW+t5rPNKREREZLi0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h+OHH37QQXREREREpG/0ptrAw6qqqhAXF4dOnTohMDAQtra26Nu3L3bu3FnneazzSkRERGS49DZ5zc/Px507d7B48WIMHToU+/btwwsvvICxY8ciOTm51vNY55WIiIjIcEnEg/92r0MPLxu4du0a2rVrh9deew1btmxRtRs1ahQsLCw01oMF7s+8KpVK1eeioiI4OztDoVDAysqqUb8DERERETVcUVER5HJ5vfI1vajzqknbtm1hYmICT09Ptf1PP/00Dh8+XOt5rPNKREREZLj0dtmAVCpFnz59cP78eURERMDNzQ1mZmZYs2YNzM3NdR0eEREREemATmde79y5gwsXLqg+X7p0CZmZmbC2toaLiwtmz56Nl19+GVu3bsWnn36K0tJSfP755zhy5Aiys7Ph4uKiw+iJiIiIqKnpdOY1LS0NPXv2RM+ePQEAoaGh6NmzJ+bPnw/g/ksMXFxcYGFhgX/961/46aefEBMTA1dXV6xevVqXoRMRERGRDuh05tXPzw91PS9WVlaG7Oxs7NixAy+88IJqf2JiIo4cOaLxHE0PbBERERGRYdDbNa8AcPPmTVRWVsLOzk5tv52dHfLy8jSew1JZRERERIZLr5PXahKJRO2zEKLGvmphYWFQKBSqLScnpylCJCIiIqImoLelsoD75bKMjY1rzLLm5+fXmI2txlJZRERERIZLr2depVIpevXqhYSEBLX9CQkJ8PHx0VFURERERKQrOk1eV69ejW7dusHKygpWVlbw9vbGnj17VMclEgl+/fVXrF69GhKJRLVduHABkydP1mHkRERERKQLOk1enZycsHjxYqSlpSEtLQ0DBw7E6NGj8dtvvwEAcnNzkZubi88//xxOTk4wNjYGAGzcuBGurq66DJ2IiIiIdEAi6qpVpQPW1tZYunQpJk2aVOPYmDFjUFxcjAMHDtS7v4a8K5eIiIiIml5D8jW9eWCrsrISO3bsQElJCby9vWscv379OuLi4rBhw4Y6+2GdVyIiIiLDpfMHtk6fPo2WLVtCJpNh8uTJiImJgaenZ412GzZsgKWlJcaOHVtnf6zzSkRERGS4dL5soPotWoWFhYiKisLatWuRnJxcI4Ht3LkzBg8ejK+++qrO/jTNvDo7O3PZABEREZGeasiyAZ0nrw8LCAiAu7s71qxZo9p36NAhDBgwAJmZmejevXuD+uOa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
<Figure size 800x400 with 1 Axes>
output_type": "display_data
name": "stdout
output_type": "stream


Target variable distribution in the original dataset:

0    0.759074

1    0.240926

Name: income, dtype: float64



Target variable distribution in the training set:

0    0.759078

1    0.240922

Name: income, dtype: float64



Target variable distribution in the testing set:

0    0.759066

1    0.240934

Name: income, dtype: float64



Train/test split complete. Files saved as:

X_train.csv, X_test.csv, y_train.csv, y_test.csv

#Explore the data to determine if you need to stratify it by some attribute when doing train/test split. Perform the train/test split.

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split





# Step 1: Explore the Dataset

print(\"First 5 rows of the dataset:\")

print(data.head())



print(\"\
Dataset Info:\")

print(data.info())



print(\"\
Summary Statistics for Numerical Columns:\")

print(data.describe())



# Step 2: Check for Missing Values

print(\"\
Missing Values Count:\")

print(data.isnull().sum())



# Step 3: Explore Target Variable Distribution

print(\"\
Distribution of the target variable (income):\")

income_distribution = data['income'].value_counts(normalize=True)

print(income_distribution)



# Visualize Target Variable Distribution

plt.figure(figsize=(6, 4))

sns.countplot(x='income', data=data)

plt.title(\"Distribution of Income Target Variable\")

plt.xlabel(\"Income Category\")

plt.ylabel(\"Count\")

plt.show()



# Step 4: Explore Categorical Attributes

categorical_columns = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

print(\"\
Categorical Variables Distributions:\")

for col in categorical_columns:

    print(f\"\
{col} distribution:\")

    print(data[col].value_counts(normalize=True))



    # Visualize distributions

    plt.figure(figsize=(8, 4))

    sns.countplot(y=col, data=data, order=data[col].value_counts().index)

    plt.title(f\"Distribution of {col}\")

    plt.xlabel(\"Count\")

    plt.ylabel(col)

    plt.show()



# Step 5: Perform Train/Test Split with Stratification

# Separate features (X) and target variable (y)

X = data.drop('income', axis=1)  # Features

y = data['income']  # Target variable



# Perform the split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=42, stratify=y

)



# Verify Stratification

print(\"\
Target variable distribution in the original dataset:\")

print(y.value_counts(normalize=True))

print(\"\
Target variable distribution in the training set:\")

print(y_train.value_counts(normalize=True))

print(\"\
Target variable distribution in the testing set:\")

print(y_test.value_counts(normalize=True))



# Step 6: Save the Train/Test Splits for Future Use

X_train.to_csv(\"X_train.csv\", index=False)

X_test.to_csv(\"X_test.csv\", index=False)

y_train.to_csv(\"y_train.csv\", index=False)

y_test.to_csv(\"y_test.csv\", index=False)



print(\"\
Train/test split complete. Files saved as:\")

print(\"X_train.csv, X_test.csv, y_train.csv, y_test.csv\")

cell_type": "code
id": "4fd6aab2
name": "stdout
output_type": "stream


### Basic Info for Target Variable 'income' ###

count    32537.000000

mean         0.240926

std          0.427652

min          0.000000

25%          0.000000

50%          0.000000

75%          0.000000

max          1.000000

Name: income, dtype: float64



Value Counts:

0    24698

1     7839

Name: income, dtype: int64

image/png": "iVBORw0KGgoAAAANSUhEUgAAAskAAAHUCAYAAADIlbU1AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAA+MUlEQVR4nO3de1hVZf7//9eWswQ7DnJKRDIlDC3DRtAOHvGEZlZaTIzOmFqWxiiT4zhj6jRamdZMjmZ9UstDODNpaRojHrL4iGYUKoZ+c0ZTE8QUQU0Bcf3+6MP6uRfgAZGN9nxc174u1r3ea933Wmz15c3aNzbDMAwBAAAAMDVy9gAAAACAhoaQDAAAAFgQkgEAAAALQjIAAABgQUgGAAAALAjJAAAAgAUhGQAAALAgJAMAAAAWhGQAAADAgpAMXAcWLlwom81mvjw9PRUSEqIuXbpo+vTpKiwsrHLM5MmTZbPZrqifH3/8UZMnT9ann356RcdV11fz5s2VmJh4Ree5lKVLl+r111+vdp/NZtPkyZPrtL+6tn79erVv317e3t6y2Wz68MMPq9R07tzZ4Xtd06uhXeuVvHf++te/ymazKT09vcaat99+WzabTcuXL6+T8TVv3lxDhw6t1bE2m03PPvvsJes+/fRT2Wy2K/7zc6Xqqx/g587V2QMAcPkWLFig22+/XeXl5SosLFRmZqZefvllvfrqq1q2bJm6d+9u1j755JPq1avXFZ3/xx9/1JQpUyT9FNYuV236qo2lS5cqNzdXKSkpVfZlZWWpadOm13wMtWUYhgYNGqRWrVpp5cqV8vb2VlRUVJW6OXPmqKSkxNxevXq1XnzxRfN7X6mhXeuVvHeeeOIJjR8/XvPnz6/xfbNgwQI1adJE/fr1q5PxrVixQr6+vnVyLme7++67lZWVpdatWzt7KMANjZAMXEdiYmLUvn17c/vhhx/Wb3/7W917770aOHCgvv32WwUHB0v6KURd6yD1448/qnHjxvXS16XExcU5tf9LOXz4sI4fP66HHnpI3bp1q7HOGnx2794tqer3vrYqv2fOFBAQoAcffFAffvihjh07poCAAIf9u3fvVlZWlsaNGyc3N7er6uvMmTPy8vJSu3btruo8DYmvr2+Df78DNwIetwCuc82aNdPMmTN18uRJzZs3z2yv7hGIDRs2qHPnzgoICJCXl5eaNWumhx9+WD/++KP279+vJk2aSJKmTJli/li/8kfUlef76quv9Mgjj8jPz08tWrSosa9KK1asUNu2beXp6albb71Vf/vb3xz2Vz5Ksn//fod264+UO3furNWrV+u7775zeOygUnWPIOTm5urBBx+Un5+fPD09ddddd+ndd9+ttp/3339fEydOVFhYmHx9fdW9e3ft2bOn5ht/gczMTHXr1k0+Pj5q3LixOnbsqNWrV5v7J0+ebP4nYvz48bLZbGrevPllnbs6GRkZevDBB9W0aVN5enrqtttu08iRI/XDDz841F3se1ZaWqpx48YpJCREjRs31v3336/s7OxqH0soKCjQyJEj1bRpU7m7uysyMlJTpkzRuXPnJOmS753qDBs2TGVlZVq6dGmVfQsWLJAk/eY3vzHP2aFDB/n7+8vX11d333233nnnHRmG4XBc5SM+y5cvV7t27eTp6WnObluv6+zZsxo3bpzuuusu2e12+fv7Kz4+Xh999FGNY543b55atWolDw8PtW7dWmlpaTXWXujLL79U//795e/vL09PT7Vr107/+Mc/LuvY6lT3uMXQoUN10003ae/everTp49uuukmhYeHa9y4cSotLXU4vrS0VFOnTlV0dLQ8PT0VEBCgLl26aPPmzWbN2bNnNWHCBEVGRsrd3V233HKLnnnmGZ04ccLhXJX3/OOPP1a7du3k5eWl6Ohoffzxx5J++vMdHR0tb29v/eIXv9CXX355ze8PUFeYSQZuAH369JGLi4s+++yzGmv279+vvn376r777tP8+fN188036/vvv1d6errKysoUGhqq9PR09erVS8OGDdOTTz4pSWb4qTRw4EA99thjeuqpp3T69OmLjisnJ0cpKSmaPHmyQkJCtGTJEj333HMqKytTamrqFV3jnDlzNGLECP3nP//RihUrLlm/Z88edezYUUFBQfrb3/6mgIAALV68WEOHDtWRI0f0/PPPO9T/4Q9/UKdOnfQ///M/Kikp0fjx49WvXz/l5eXJxcWlxn42bdqkHj16qG3btnrnnXfk4eGhOXPmqF+/fnr//fc1ePBgPfnkk7rzzjs1cOBAjR49WklJSfLw8Lii67/Qf/7zH8XHx+vJJ5+U3W7X/v37NWvWLN17773auXNnldnX6r5nv/71r7Vs2TI9//zz6tq1q7755hs99NBDDo96SD8F5F/84hdq1KiRJk2apBYtWigrK0svvvii9u/frwULFlz2e+dC3bt3V0REhObPn6/Ro0eb7RUVFVq0aJHi4uLMWfX9+/dr5MiRatasmSRpy5YtGj16tL7//ntNmjTJ4bxfffWV8vLy9Mc//lGRkZHy9vautv/S0lIdP35cqampuuWWW1RWVqZ169Zp4MCBWrBggX71q1851K9cuVIbN27U1KlT5e3trTlz5ujxxx+Xq6urHnnkkRqvc+PGjerVq5c6dOigN998U3a7XWlpaRo8eLB+/PFHh+Be+R8n638YL1d5ebn69++vYcOGady4cfrss8/05z//WXa73bxP586dU+/evfX5558rJSVFXbt21blz57RlyxYdOHBAHTt2lGEYGjBggNavX68JEybovvvu044dO/TCCy8oKytLWVlZDu/f7du3a8KECZo4caLsdrumTJmigQMHasKECVq/fr2mTZsmm82m8ePHKzExUfv27ZOXl9cV3x+g3hkAGrwFCxYYkoxt27bVWBMcHGxER0eb2y+88IJx4R/xf/3rX4YkIycnp8ZzHD161JBkvPDCC1X2VZ5v0qRJNe67UEREhGGz2ar016NHD8PX19c4ffq0w7Xt27fPoW7jxo2GJGPjxo1mW9++fY2IiIhqx24d92OPPWZ4eHgYBw4ccKjr3bu30bhxY+PEiRMO/fTp08eh7h//+IchycjKyqq2v0pxcXFGUFCQcfLkSbPt3LlzRkxMjNG0aVPj/PnzhmEYxr59+wxJxowZMy56PqtLfe/Pnz9vlJeXG999950hyfjoo4/MfTV9z3bt2mVIMsaPH+/Q/v777xuSjCFDhphtI0eONG666Sbju+++c6h99dVXDUnGrl27DMO4+HunJpXj++qrr8y2VatWGZKMt99+u9pjKioqjPLycmPq1KlGQECAeX8N46f3nIuLi7Fnz54qx0VERDhcl9W5c+eM8vJyY9iwYUa7du0c9kkyvLy8jIKCAof622+/3bjtttvMtures7fffrvRrl07o7y83OGciYmJRmhoqFFRUWG2tWjRwmjRokWNY7xYP0OGDDEkGf/4xz8cavv06WNERUWZ2++9995F769hGEZ6erohyXjllVcc2pctW2ZIMt566y2zLSIiwvDy8jIOHTpktuXk5BiSjNDQUPPPuWEYxocffmhIMlauXGm2Xcn9Aeobj1sANwjD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
<Figure size 800x500 with 1 Axes>
output_type": "display_data
image/png": "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e1Xbj42NVUBAgObMmWOPffXVVzp8+LDL/tx333368ssv9cc//lErV67U2bNnr3lff8nhcKhHjx4uY02bNtVPP/1kP7+a43phlvDi9/mRRx5RtWrVSrzP9957r+688077eVhYmKSfT1v95TWDF8Yv9LN3717t2rXLnjE8f/68/ejevbsyMjK0e/fuq9p3ALjdEMgAoILr0qWLpJ9D15o1a1RYWKhOnTqpS5cu9i/cKSkpatu2rX162YVT6QIDA0usLygoqMSpdpequ+D777/Xhg0bFBsbq/Dw8BLLjx07pqCgoEue5nglixYtUp8+fXTnnXdq/vz5WrdunTZt2qTHH39c586ds+vatWunzz77TOfPn9fAgQNVp04dhYeHu1y/NWHCBL3++utav369YmNjVatWLXXu3PmS10H9kru7u+Lj47V48WL79Ma5c+cqMDBQMTExdt1f//pXPfvss/rss8/UsWNH+fj4qFevXtqzZ8817fMFVatWVeXKlV3GvLy8XPb7ao7riRMn5O7urtq1a7uMOxwOBQQElHiffXx8XJ57enpecfxCP0eOHJEkjR07Vh4eHi6P4cOHS5KOHz9+5Z0GgNsUgQwAKrg6deqoUaNGSklJUXJyslq0aKGaNWuqc+fOysjI0IYNG7R+/Xo7uElSrVq1JP18bdjFDh8+7HL9mPTzL/CXExUVpTlz5uj999/XE088oeLiYpfltWvX1uHDh0uM/5r58+crJCREH3/8sXr16qXWrVurRYsWJW5sIUm/+93vtHz5cuXk5GjlypWqU6eOBgwYoHXr1kn6OViNHj1aW7Zs0cmTJ7Vw4UIdPHhQMTExOnPmzBX7GDJkiM6dO6fExERlZWVpyZIlGjhwoNzc3OyaatWqafLkydq1a5cyMzP17rvvav369SVmuUrT1RzXWrVq6fz58zp27JjLuGVZyszMLPE+X68L65kwYYI2bdp0yce9995bKtsCgLKGQAYAUJcuXbRixQolJyera9eukqRGjRqpbt26mjhxogoLC10C2YXTD+fPn++ynk2bNmnnzp0lTjv8NYMGDVJiYqJ9s42ioiJ7WWxsrM6dO6e5c+de0zodDoc8PT1dwmBmZuYl77J4gZeXl9q3b6+//OUvkqRvv/22RE3NmjX18MMP6+mnn9bJkye1f//+K/YRFhamVq1aac6cOfroo4+Un5+v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8f/973/16KOPytvbW4sXLy71Rm8Hubm5cjqdysnJYWYNQJkXOe5D0y1cVuq0gaZbAADghlxtNriua8hmzpypU6dOqX79+rrrrrv0m9/8RiEhITp16pTefvvt624aAAAAACqS6zplMTg4WFu2bFFycrJ27doly7LUuHFj+7otAAAAAMCvu6YZshUrVqhx48bKzc2VJHXt2lUjRozQyJEj1bJlSzVp0kSrV6++KY0CAAAAQHlzTYFsxowZl73zoNPp1BNPPKE33nij1JoDAAAAgPLsmgLZ1q1b1a1bt8suj46OVmpq6g03BQAAAAAVwTUFsiNHjlzydvcXuLu769ixYzfcFAAAAABUBNcUyO6880599913l12+bds2BQYG3nBTAAAAAFARXFMg6969uyZOnKhz586VWHb27Fm9+OKLiouLK7XmAAAAAKA8u6bb3j///PNatGiRGjVqpGeeeUahoaFyOBzauXOn/va3v6moqEjPPffczeoVAAAAAMqVawpk/v7+Wrt2rZ566ilNmDBBlmVJkhwOh2JiYvTOO+/I39//pjQKAAAAAOXNNX8wdL169fTFF18oKytLe/fulWVZatiwoby9vW9GfwAAAABQbl1zILvA29tbLVu2LM1eAAAAAKBCuaabegAAAAAASg+BDAAAAAAMIZABAAAAgCEEMgAAAAAwhEAGAAAAAIYQyAAAAADAEAIZAAAAABhCIAMAAAAAQwhkAAAAAGAIgQwAAAAADCGQAQAAAIAhBDIAAAAAMIRABgAAAACGEMgAAAAAwBACGQAAAAAYQiADAAAAAEMIZAAAAABgCIEMAAAAAAwhkAEAAACAIQQyAAAAADCEQAYAAAAAhhDIAAAAAMAQAhkAAAAAGEIgAwAAAABDCGQAAAAAYAiBDAAAAAAMIZABAAAAgCEEMgAAAAAwhEAGAAAAAIYQyAAAAADAEAIZAAAAABhCIAMAAAAAQwhkAAAAAGBImQ9k9evXl8PhKPF4+umnJUmDBw8usax169Yu68jPz9eIESPk6+uratWqqWfPnjp06JBLTVZWluLj4+V0OuV0OhUfH6/s7OxbtZsAAAAAKqAyH8g2bdqkjIwM+5GcnCxJeuSRR+yabt26udR88cUXLusYNWqUFi9erMTERK1Zs0anT59WXFycioqK7JoBAwYoLS1NSUlJSkpKUlpamuLj42/N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69e7jFepUuWa+h03bpw+/fRTvfrqq1q9erXS0tIUERFh30CkevXq2rJlixYuXKjAwEBNnDhR99xzj7Kzs+Xm5qbk5GR9+eWXaty4sd5++22FhoZq375919QDAMAMAhkAoNy5cB3ZjBkz1L59ezkcDrVv314rV650CWSNGzfWmjVrXF67du1aNWrU6LLXe/3SU089pT//+c/q2bOnyzVqTZs21aFDh/T9999fVb+rV6/W4MGD9eCDDyoiIkIBAQHav3+/S427u7u6dOmi1157Tdu2bdP+/fu1YsUKSZLD4VDbtm01efJkffvtt/L09HQ5xRIAUHZxUw8AQLnjdDp17733av78+Xrrrbck/RzSHnnkERUWFqpDhw6SpDFjxqhly5Z6+eWX1bdvX61bt04zZ87UO++8c9XbGjFihIqKihQXF6cvv/xS999/v9q3b6927drpoYce0htvvKHf/OY32rVrlxwOh7p161ZiHb/5zW+0aNEi9ejRQw6HQy+88IJ9l0hJ+vzzz/Xjjz+qXbt28vb21hdffKHi4mKFhoZqw4YNWr58uaKjo+Xn56cNGzbo2LFjCgsLu7GDCAC4JZghAwCUSx07dlRRUZEdvry9vdW4cWPVrl3bDivNmzfXJ598osTERIWHh2vixIl66aWXXO6weDVGjRqlyZMnq3v37lq7dq2kn2+n37JlS/Xv31+NGzfW+PHjXa5L+6U333xT3t7eatOmjXr06KGYmBg1b97cXl6zZk0tWrRInTp1UlhYmP7+979r4cKFatKkiWrUqKFvvvlG3bt3V6NGjfT8889r+vTpio2NvfaDBgC45bjLIgAAAAAYwgwZAAAAABhCIAMAAAAAQwhkAAAAAGAIgQwAAAAADCGQAQAAAIAhBDIAAAAAMIRABgAAAACGEMgAAAAAwBACGQAAAAAYQiADAAAAAEMIZAAAAABgyP8DB7laPrb9uAMAAAAASUVORK5CYII=
<Figure size 1000x600 with 1 Axes>
output_type": "display_data
image/png": "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TIw6d+6svn376l//+pcOHDiguLi4bJ8NGjhwoBYuXKhHH31Uw4YN0wMPPKCLFy9q1apVio6OVuPGjdWmTRtNnDhRnTp1Us+ePXXy5Em9++67V53pyJodWrhwoSIiIlS4cOFrvkgfN26cmjVrpsaNG2vw4MHy9vbW1KlTtX37ds2fP182my0fbt0r7rvvPtWvX1/jxo3ToUOHNGPGDKftS5cu1dSpU/XYY48pIiJCxhgtWrRIZ86cUbNmzfKtjn+aOHGiGjRooLp162rYsGG65557dOzYMS1ZskTTp09XQECAXn/9dS1dulSNGzfWqFGjVLx4cX3++ef6+uuvFRcXJ7vdni+1PPjgg+rZs6e6deumjRs36uGHH5a/v78SEhK0evVqVa1aVX369MmXYwFAgeLS5TkAAJbbt2+fee6558xdd91lvLy8TKlSpUxUVJQZM2aMU7+dO3eaZs2amcKFC5vixYub7t27m//+97/ZViPMzMw0cXFxJiIiwhQuXNjUrl3brFixIttqhMYYc/r0afPiiy+acuXKGS8vLxMUFGTatGljdu/e7ejz6aefmooVKxofHx8TERFhxo0bZz755BMjyezbt8/Rb//+/aZ58+YmICDASHKs0ne11QiNMebnn382jzzyiPH39ze+vr6mXr16jhUCs2StRrhhwwan9qzV9/4+7pzMmDHDSDK+vr7ZVn7cvXu3efrpp0358uWNr6+vsdvt5oEHHjCzZs267n6vtRphv379svW92iqRO3fuNE8++aQpUaKE8fb2NuXKlTNdu3Y1ly5dcvTZtm2badu2rbHb7cbb29tUr149222ZdXt88cUXTu3Xuv1ef/11I8kcP37cqf3TTz81devWddwn5cuXN126dDEbN2687m0BAO7IZszfvjUSAAAAAJAv+MwWAAAAAFiAsAUAAAAAFiBsAQAAAIAFCFsAAAAAYAHCFgAAAABYgLAFAAAAABbgS41zKTMzU0ePHlVAQEC+fhkmAAAAAPdijNHZs2cVGhoqD49rz18RtnLp6NGjKlu2rKvLAAAAAFBAHDp0SGXKlLnmdsJWLgUEBEi6coMGBga6uBoAAAAArpKSkqKyZcs6MsK1ELZyKevUwcDAQMIWAAAAgOt+vIgFMgAAAADAAoQtAAAAALAAYQsAAAAALMBntgAAAADkSkZGhi5fvuzqMizn5eUl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bdo09evXTz179tS2bdu0ZMkS3XPPPY7tHh4eev/997V9+3bNnj1bK1as0NChQyVJUVFRmjRpkgIDA5WQkKCEhAQNHjw42zGMMXrsscd06tQprVq1SsuXL9eff/6pp556yqnfn3/+qS+//FJLly7V0qVLtWrVKr399tt5HhMA4NbjNEIAgNs4f/68Jk6cqBUrVqh+/fqSpIiICK1evVrTp09XWFiYIiIi9N5778lms6lixYratm2b3nnnnTwdZ8yYMRo0aJBefPFFR1udOnUcvw8cONDxe3h4uN5880316dNHU6dOlbe3t+x2u2w2m0JCQq55jB9++EFbt27Vvn37VLZsWUnSnDlzVLlyZW3YsMFxvMzMTM2aNUsBAQGSpJiYGP34449666238jQmAMCtR9gCALiNnTt36tKlS2rWrJlTe1pammrUqKGLFy+qXr16stlsjm1ZoSy3kpKSdPToUTVp0uSafVauXKmxY8dq586dSklJUXp6ui5duqTz58/L398/V8fZtWuXypYt6whaklSpUiUVLVpUu3btcoStsLAwR9CSpNKlSyspKSlPYwIAuAZhCwDgNjIzMyVJX3/9te666y6nbT4+Purfv/919+Hh4SFjjFPb5cuXHb/7+vrmeP0DBw6odevW6t27t958800VL15cq1evVvfu3Z32cz3GGKdQeK12Ly8vp+02m81xOwAACjbCFgDAbVSqVEk+Pj46ePCgGjZseNXtX375pVPbunXrnC6XKlVKiYmJTqEmPj7esT0gIEBhYWH68ccf1bhx42zH2Lhxo9LT0zVhwgR5eFz56PP//M//OPXx9vZWRkbGdcdy8OBBHTp0yDG7tXPnTiUnJysyMjLH6wIA3ANhCwDgNgICAjR48GC99NJLyszMVIMGDZSSkqI1a9aoSJEi6t27tyZMmKCXX35ZvXr10qZNmzRr1iynfTRq1EjHjx9XXFycnnjiCS1btkzffvutAgMDHX1Gjx6t3r17KygoSK1atdLZs2f1yy+/qH///ipfvrzS09M1efJktW3bVr/88os+/PBDp2OEhYXp3Llz+vHHH1W9enX5+fnJz8/PqU/Tpk1VrVo1PfPMM5o0aZLS09PVt29fNWzYULVr17bsNgQA3DqsRggAcCtvvvmmRo0apXHjxikyMlItWrTQV199pfDwcJUrV07/+c9/9NVXX6l69er68MMPNXbsWKfrR0ZGaurUqfrggw9UvXp1rV+/Pttqgc8++6wmTZqkqVOnqnLlyoqOjtbvv/8uSbr//vs1ceJEvfPOO6pSpYo+//xzjRs3zun6UVFR6t27t5566imVKlVKcXFx2cZhs9n05ZdfqlixYnr44YfVtGlTRUREaOHChfl8iwEAXMVm/nniOgAAAADgpjGzBQAAAAAWIGwBAAAAgAUIWwAAAABgAcIWAAAAAFiAsAUAAAAAFiBsAQAAAIAFCFsAAAAAYAHCFgAAAABYgLAFAAAAABYgbAEAAACABQhbAAAAAGABwhYAAAAAWOD/AhSBnHQ/SllDAAAAAElFTkSuQmCC
<Figure size 1000x600 with 1 Axes>
output_type": "display_data
image/png": "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+QNTGnDuSibOW8nFuSu5OHclF+euZLqRz5vT6fzTGh57/78qV64sDw+PfKNhGRkZ+UbNAAAAAKAoEMj+l7e3t6KiopScnOw2Pzk5WS1btjTUFQAAAIAbGZcs/sGwYcMUGxur5s2bKzo6WrNnz9b+/fv1xBNPmG7NOB8fH7300kv5LtFE8ce5K5k4byUX567k4tyVXJy7konz9juesniBmTNnatKkSUpLS1PDhg01depUtW7d2nRbAAAAAG5ABDIAAAAAMIR7yAAAAADAEAIZAAAAABhCIAMAAAAAQwhkAAAAAGAIgQwAAAAADCGQ4ZLOnTuns2fPmm4DKHV4AC5wfaSlpWnHjh2m20AhuFwuSfx7WRKdOnWKz5d/QCDDRe3YsUMPP/yw7r77bvXr108LFy403RIu0/n/pFCynDx5UsePH1d2drYcDofpdnAFjh49qu+//14//PCDcnNzTbeDy/TLL78oMjJSY8aM0aZNm0y3gyvw7bffqm3btjp58iT/XpYwqamp6tWrl9avX6+cnBzT7RQLBDIUaPfu3WrZsqW8vb3VoUMH/fzzz5o8ebL69etnujX8id27dyshIUFpaWmmW8EV2LFjh7p376677rpL9erV03vvvSeJ3/yWBKmpqWrfvr169uypyMhITZo0iV+KlBC7d+9WVlaWsrKyNG3aNH377bf2Mn72iq+tW7eqdevWuvXWW1WuXDl7Pues+Nu+fbtat26tatWqqVatWvLx8THdUrHAF0MjH8uy9OKLL2rXrl368MMPJf0+tDx37lzNmjVL9erV0/vvv2+4SxTkxx9/VIsWLZSZmannn39ew4YNU+XKlU23hT+xY8cOtW7dWn369NGtt96qTZs2adq0adqwYYOaNGliuj1cwvlz169fP/Xr10+ff/65RowYoX379iksLMx0e/gTR48eVb9+/XTffffZ/7+NGjVKDRo0UF5ensqU4ffWxc22bdvUsmVLDR48WJMmTbLnnzlzRmXLljXYGf7MyZMn1b17d9WuXVszZ86UJH3//ffKyclRQEBAqf43k0CGAvXr108//vijvv76a3ve6dOntWDBAs2YMUOdOnXSxIkTDXaIC508eVJDhw5VXl6emjdvriFDhmj48OEaOXIkoawYO3r0qHr16qW6devq9ddft+fffffdioyM1Ouvvy7Lsrgkpxg6cuSI/t//+39q2rSpEhISJP3+C617771XY8eOla+vrwIDA0v1h4zizOVy6ejRo7rjjju0cuVKbdiwQRMnTlSTJk20fft2Va1aVf/+979Nt4k/SE9PV9OmTdW4cWMlJSXJ5XLpmWee0e7du7V7927169dPMTExatq0qelWUYCcnBy1b99eb7zxhho1aqT77rvPvty7QYMGevzxx9W/f3/TbRrhaboBFC/nP/g1a9ZMu3bt0vfff6+6detKknx9fdWjRw/t3r1bX375pTIyMhQUFGS4Y5xXpkwZRUVFKTAwUA8++KCqVKmihx56SJIIZcXY2bNndezYMf3lL3+RJPu38rVq1dJvv/0mSYSxYsrhcOiee+6xz50kvfzyy/riiy+Unp6uI0eOqEGDBhozZozuuOMOg52iIGXKlFGVKlV06623KjU1VQ888IB8fHzUt29f5eTkaMCAAaZbRAGio6N14MAB/fe//9Vbb72lc+fO6bbbblNkZKQ++OADpaamavz48apTp47pVnGBY8eOadeuXTpy5IhGjBghSZozZ47S0tK0cuVKjRkzRk6n0+3f1NKCsXi4Of/B795779UPP/ygSZMm6fjx4/byChUqKC4uThs3btTatWtNtYkC+Pr6qm/fvnrwwQclST179tTChQv1j3/8Q6+++qr94T4vL0979uwx2Sr+IDg4WPPnz9edd94p6f8eyHLTTTflu1zqxIkT170/XFxgYKCeeuopRURESJIWLVqkl156SQsXLtSKFSv03nvvKTMzUytWrDDcKQpy/v87Dw8PrVq1SpK0ePFiuVwuhYWF6euvv9aGDRsMdogLhYSEaMaMGapfv74eeughuVwuvf/++3rllVc0efJk/e1vf9Pq1au1detW062iAEFBQWrXrp2WLl2qH374Qc8884waN26se+65R0OHDlX79u21YsUKuVyuUnc/ICNkKFDt2rX1wQcfqHPnzvLz81N8fLw9wuLt7a2mTZuqYsWKZptEPudvbna5XCpTpowefPBBWZal3r17y+FwKC4uTv/4xz+0b98+zZs3T35+foY7hiT7A31eXp68vLwk/X4ODx8+bNdMnDhRPj4+Gjp0qDw9+ae7uPD397f/HB0drU2bNqlZs2aSpNatWys4OFibN2821R4u4fwVIXfffbd+/vlnDR48WJ999pk2b96slJQUjRgxQt7e3mrUqBH3JhUjVatW1cSJE1WtWjV16NBBAQEB9pUF3bp10wsvvKCvvvpKPXv2NN0qLuBwOPTss8+qTZs2OnXqlAYOHGgvq1atmoKDg7Vx40aVKVOm1F0Zwv/quKi2bdvqww8/VI8ePXTo0CH16NFDjRo10rx583Tw4EHVrl3bdIu4CA8PD1mWpby8PD300ENyOByKjY3V0qVL9dNPP2njxo2EsWKoTJky9odEh8MhDw8PSdLYsWP18ssva8uWLYSxYqxGjRqqUaOGpN8/7Ofm5qp8+fJq2LCh4c5QkPMf+MLDw9WvXz8FBwfrk08+UXh4uMLDw+VwONS4cWPCWDEUGhqqkSNHytfXV9L//dt57NgxBQYGKioqynCHuJjmzZvr888/11133aXZs2erVq1aatCggaTfL+G/5ZZbdO7cOfuXk6UFD/XAn/r22281bNgw7dmzR56envLy8tLChQu5abYEOP/j7XA41K5dO6WkpGjVqlWKjIw03Bku5vxveuPj45WWlqaIiAiNGTNGa9eutUdeUDKMHTtW77zzjpYvX26PgqL4OXv2rObNm6fmzZurUaNGPESnBBs7dqwWLlyo5ORk1axZ03Q7uISvvvpKvXr1UrVq1RQZGanc3FwtXbpUa9asKZW/xCKQ4bJkZ2fr6NGjOnHihEJCQnhARAnicrk0YsQIJSQkKCUlRY0aNTLdEi7DK6+8ohdffFEVKlTQ8uXL1bx5c9Mt4TL9+9//1qpVq7Ro0SIlJyfzy6sSgEfcl2yLFi3SqlWr9MEHH2jFihX8zJUQu3bt0vz587V+/XpFRERo8ODBpTKMSQQy4IbncrmUmJioqKgovtOqBNm0aZNuu+02paamqn79+qbbwRXYvn27xo8fr5deeolzB1wH27Zt0+jRo/Xqq6/al7+h5MjLy5OkUv1LEQIZUApwCU7JdPLkSftBLShZzp49W+rugQBMys3Nlbe3t+k2gEIhkAEAAACAIaV3bBAAAAAADCOQAQAAAIAhBDIAAAAAMIRABgAAAACGEMgAAAAAwBACGQAAAAAYQiADAJRqNWvWVEJCQrHZDgCgdCGQAQBKtY0bN2rgwIH2tMPh0EcffWSsn8IGuzZt2iguLq7I+wEAXFuephsAAMCE3NxceXt7q0qVKqZbAQCUYoyQAQCKnTZt2mjIkCGKi4tTpUqVFBwcrNmzZ+vkyZPq16+f/P39Vbt2bX3++eeSJJfLpf79+ys8PFy+vr6qU6eOXn/9dbdtPvroo+rWrZsmTpyo0NBQ3XLLLZLcR6Rq1qwpSXrggQfkcDjs6Z9++kn333+/goODVb58ed16661avnx5oY8vPj5e1atXl4+Pj0JDQzV06FD7uPft26dnnnlGDodDDodDkvTbb7+pV69eqlatmvz8/BQZGamFCxe6Hdvq1av1+uuv2+vt3btXiYmJqlixotu+P/roI3u7krR161a1bdtW/v7+qlChgqKiorRp06ZCHxsA4MoQyAAAxdI777yjypUra8OGDRoyZIj++te/qkePHmrZsqW+/fZbderUSbGxsTp16pTy8vJUrVo1ffDBB9qxY4fGjh2r0aNH64MPPnDb5ooVK7Rz504lJyfrk08+ybfPjRs3SpLmzp2rtLQ0e/rEiRO69957tXz5cm3ZskWdOnVSly5dtH///is+rn//+9+aOnWqZs2apR9++EEfffSRIiMjJUmLFy9WtWrVNH78eKWlpSktLU2SdObMGUVFRemTTz5RamqqBg4cqNjYWH3zzTeSpNdff13R0dEaMGCAvV5YWNhl9fPwww+rWrVq2rhxozZv3qznn39eXl5eV3xcAIDC4ZJFAECx1LhxY40ZM0aSNGrUKP39739X5cqVNWDAAEnS2LFj9eabb2rbtm26/fbbNW7cOHvd8PBwrV27Vh988IF69uxpzy9Xrpz++c9/ytvbu8B9nr98sWLFigoJCXHrpXHjxvb0yy+/rCVLlmjp0qV66qmnrui49u/fr5CQELVv315eXl6qXr26brvtNklSQECAPDw85O/v77b/m266ScOHD7enhwwZoqSkJH344Ydq0aKFnE6nvL295efn57be5fYzYsQI1a1bV5IUERFxResDAK4OI2QAgGKpUaNG9p89PDwUGBhojyRJUnBwsCQpIyNDkvTWW2+pefPmqlKlisqXL685c+bkG8GKjIy8aBi7lJMnT2rkyJGqX7++KlasqPLly+v7778v1AhZjx49dPr0adWqVUsDBgzQkiVLdO7cuUuu43K59Morr6hRo0YKDAxU+fLltWzZskLt/0LDhg3T448/rvbt2+vvf/+7fvrpp6veJgDg8hHIAADF0oWXzTkcDrd55++DysvL0wcffKBnnnlGjz32mJYtW6aUlBT169dPubm5btsoV65coXoZMWKE/vOf/+iVV17R119/rZSUFEVGRubb/uUICwvTrl27NGPGDPn6+mrw4MFq3bq1zp49e9F1XnvtNU2dOlUjR47UypUrlZKSok6dOv3p/suUKSPLstzmXbif+Ph4bd++Xffdd59Wrlyp+vXra8mSJVd8XACAwuGSRQBAiff111+rZcuWGjx4sD2vsCM9Xl5ecrlc+bb/6KOP6oEHHpD0+z1le/fuLXS/vr6+6tq1q7p27aonn3xSdevW1XfffadmzZrJ29u7wP3ff//9euSRRyT9HkJ/+OEH1atXz64paL0qVaro+PHjOnnypB1GU1JS8vVzyy236JZbbtEzzzyjXr16ae7cufaxAgCuLUbIAAAl3s0336xNmzbpiy++0O7du/Xiiy/aD+S4UjVr1tSKFSuUnp6uzMxMe/uLFy9WSkqKtm7dqt69eysvL69Q209MTNTbb7+t1NRU/fzzz5o3b558fX1Vo0YNe/9fffWVfvnlFx05csTef3JystauXaudO3dq0KBBSk9Pz9f3N998o7179+rIkSPKy8tTixYt5Ofnp9GjR+vHH3/UggULlJiYaK9z+vRpPfXUU1q1apX27dun//mf/9HGjRvdgh4A4NoikAEASrwnnnhC3bt314MPPqgWLVrot99+cxstuxKvvfaakpOTFRYWpqZNm0qSpk6dqkqVKqlly5bq0qWLOnXqpGbNmhVq+xUrVtScOXPUqlUrNWrUSCtWrNDHH3+swMBASdL48eO1d+9e1a5d237IyIsvvqhmzZqpU6dOatOmjUJCQtStWze37Q4fPlweHh6qX7++qlSpov379ysgIEDz58/XZ599Zj8qPz4+3l7Hw8NDv/32m/r06aNbbrlFPXv2VOfOnd0ekAIAuLYc1oUXlwMAAAAArgtGyAAAAADAEAIZAABF6L333lP58uULfDVo0MB0ewCAYoZLFgEAKELHjx/X4cOHC1zm5eVlP7wDAACJQAYAAAAAxnDJIgAAAAAYQiADAAAAAEMIZAAAAABgCIEMAAAAAAwhkAEAAACAIQQyAAAAADCEQAYAAAAAhvx/7dLEbXZ9xmgAAAAASUVORK5CYII=
<Figure size 1000x600 with 1 Axes>
output_type": "display_data
image/png": "iVBORw0KGgoAAAANSUhEUgAAA1sAAAIqCAYAAADSNVDVAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABboklEQVR4nO3deVhV5d7/8c9WJiXdCsaUOFWaA1qiKZaK4VhoZWVmUZZpaWrm9DhV5DHt0Rw6mh7rWGSOp5N2GinNofw5k+SslXOBqDGIESDcvz96WKftFCLLDfh+Xde+Lvda373W92aj8PFe694OY4wRAAAAAKBYlXN3AwAAAABQFhG2AAAAAMAGhC0AAAAAsAFhCwAAAABsQNgCAAAAABsQtgAAAADABoQtAAAAALABYQsAAAAAbEDYAgAAAAAbELYAAKXSokWLNGPGjAvuczgcio2Nvar9XA2xsbFyOBzubgMAUEgOY4xxdxMAAFyu6Oho7dy5U4cOHTpv38aNG1W9enVVr1796jdmo2PHjunYsWNq2bKlu1sBABSCh7sbAACguJXVMFIWAyQAlGVcRggAZci6desUFRWlSpUqqWLFimrVqpU+++yz8+p+/vln9evXT6GhofLy8lJISIgefPBBHT9+3KpJS0vTsGHDVKdOHXl7eysgIEB333239u7dK0las2aNHA6H1qxZ43LsQ4cOyeFwKC4uztrWu3dvXXfdddq1a5eioqLk6+ur66+/XgMHDtRvv/3m8vo333xTbdq0UUBAgHx9fRUWFqbJkycrNzfXqomMjNRnn32mw4cPy+FwWI8CF7qMcOfOnbr33ntVtWpV+fj46NZbb9V7773nUlMwpsWLF2vs2LEKCQlR5cqV1b59e+3bt++SX/uPPvpIDodDX3/99Xn75syZI4fDoe3bt0uSDhw4oJ49eyokJETe3t4KDAxUVFSUEhMTL3mOC11GWKtWLUVHRys+Pl5NmzZVhQoVdMstt+idd9457/WFed+PHDmixx57TAEBAfL29lb9+vU1depU5efnWzUF7/GUKVP0v//7v6pVq5YqVKigyMhI7d+/X7m5uRo1apRCQkLkdDp1//33KyUl5bx+li5dqoiICPn6+uq6665Tp06dtG3btkt+DQCgNGFmCwDKiLVr16pDhw5q3Lix5s2bJ29vb82ePVtdu3bV4sWL9fDDD0v64xfu5s2bKzc3V2PGjFHjxo116tQpffnll0pNTVVgYKBOnz6tO++8U4cOHdL//M//qEWLFsrMzNQ333yjpKQk3XLLLZfdX25uru6++24988wzGjVqlNavX68JEybo8OHD+uSTT6y6n376Sb169VLt2rXl5eWl77//Xq+++qr27t1rBYjZs2erX79++umnn7R8+fK/PPe+ffvUqlUrBQQE6O9//7v8/f21YMEC9e7dW8ePH9fIkSNd6seMGaM77rhD//znP5WRkaH/+Z//UdeuXbVnzx6VL1/+gueIjo5WQECA3n33XUVFRbnsi4uLU9OmTdW4cWNJ0t133628vDxNnjxZNWrU0MmTJ7V+/XqlpaVdzpfU8v3332vYsGEaNWqUAgMD9c9//lN9+vTRTTfdpDZt2kgq3Pt+4sQJtWrVSjk5Ofrb3/6mWrVq6dNPP9Xw4cP1008/afbs2S7nffPNN9W4cWO9+eabVjjv2rWrWrRoIU9PT73zzjs6fPiwhg8frqeffloff/yx9dqJEydq3LhxevLJJzVu3Djl5ORoypQpat26tTZv3qwGDRoU6WsBACWKAQCUCS1btjQBAQHm9OnT1razZ8+aRo0amerVq5v8/HxjjDFPPfWU8fT0NLt3777oscaPH28kmRUrVly0ZvXq1UaSWb16tcv2gwcPGknm3XfftbY98cQTRpJ54403XGpfffVVI8msW7fugufIy8szubm5Zv78+aZ8+fLm119/tfbdc889pmbNmhd8nSTz8ssvW8979uxpvL29zZEjR1zqunTpYipWrGjS0tJcxnT33Xe71P3rX/8yksyGDRsueL4CQ4cONRUqVLCOZ4wxu3fvNpLMzJkzjTHGnDx50kgyM2bMuOSxLuTll1825/7orlmzpvHx8TGHDx+2tmVlZRk/Pz/zzDPPWNsK876PGjXKSDKbNm1y2d6/f3/jcDjMvn37jDH/fY+bNGli8vLyrLoZM2YYSaZbt24urx8yZIiRZNLT040xxhw5csR4eHiYQYMGudSdPn3aBAUFmR49ehTmywEAJR6XEQJAGXDmzBlt2rRJDz74oK677jpre/ny5RUTE6Njx45Zl8F98cUXateunerXr3/R433xxReqW7eu2rdvX6x9Pvrooy7Pe/XqJUlavXq1tW3btm3q1q2b/P39Vb58eXl6eurxxx9XXl6e9u/f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DodDsbGxio2NvWiNj4+PZs6cqZkzZxZzdwAAAABwcSU6bAEAAKDowkfMt+W4CVMet+W4QFlTqhbIAAAAAIDSgrAFAAAAADYgbAEAAACADQhbAAAAAGADwhYAAAAA2ICwBQAAAAA2IGwBAAAAgA0IWwAAAABgA8IWAAAAANiAsAUAAAAANiBsAQAAAIANCFsAAAAAYAMPdzcAAAAAAFfqyPgwW45b46UdRX4tM1sAAAAAYAPCFgAAAADYgLAFAAAAADYgbAEAAACADVggAwAAALhGlMRFJMoyZrYAAAAAwAaELQAAAACwAZcRAgBKPS6LAQCURMxsAQAAAIANCFsAAAAAYAPCFgAAAADYgLAFAAAAADZggQwAAABcFhalAQqHmS0AAAAAsAFhCwAAAABsQNgCAAAAABsQtgAAAADABoQtAAAAALABYQsAAAAAbMDS7wAA4JLCR8y35bgJUx635bgAUFIQtgAAKMH4PCMAKL24jBAAAAAAbMDMFgD8H2YQAABAcWJmCwAAAABsQNgCAAAAABsQtgAAAADABtyzBQDXAO5HAwDg6mNmCwAAAABsQNgCAAAAABsQtgAAAADABoQtAAAAALDBNRW2Zs+erdq1a8vHx0fh4eH69ttv3d0SAAAAgDLqmglbS5cu1ZAhQzR27Fht27ZNrVu3VpcuXXTkyBF3twYAAACgDLpmwta0adPUp08fPf3006pfv75mzJih0NBQzZkzx92tAQAAACiDronP2crJyVFCQoJGjRrlsr1jx45av379BV+TnZ2t7Oxs63l6erokKSMjw6UuLzurmLv9w2nPPFuOK50/hktpM26xLT18M+ERW457OUrbe3c575tdY5PK9vgY2+UrCWOTyvb4yvLYLqcHO/HeXT7GVjRleXzX2tgKthljLvlah/mrijLgl19+0Q033KD/9//+n1q1amVtnzhxot577z3t27fvvNfExsbqlVdeuZptAgAAAChFjh49qurVq190/zUxs1XA4XC4PDfGnLetwOjRozV06FDreX5+vn799Vf5+/tf9DXFJSMjQ6GhoTp69KgqV65s67ncoSyPryyPTSrb42NspVdZHh9jK73K8vgYW+lVlsd3tcdmjNHp06cVEhJyybprImxVq1ZN5cuXV3Jyssv2lJQUBQYGXvA13t7e8vb2dtlWpUoVu1q8oMqVK5e5vwh/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66+/Xs2bN9fOnTt1//33KzY2VsuXL9eOHTsUHR3t7haLLCIiQqdOndJ//vMfRUdHa8+ePQoPD9cDDzygf/3rX3rttde0b98+d7dZJGlpadq3b59OnjypESNGSJLefvtt/etf/1Lr1q01btw4/fvf/3Zzl0WTnZ2tkSNH6tVXX1V+fr7KlSunCRMmaP/+/ZozZ46k0vtvivTHz7sjR44oMTFRkvTqq6/qiy++0AcffKBZs2apZ8+eiouLc2uPRWGM0W+//aacnBz99NNPOnv2rHx8fPTzzz9r6dKlio6OVoMGDfT555+7u9Ur5unpqaSkJOt3F2OMatSooZdeeklt2rTRp59+al1KXxq/V7dt26ZatWpp0qRJ+uyzz/T111/r6aefVkZGhrtbuyL5+flasmSJbrrpJo0ZM0bly5eXJN1+++3y9PQs1bcH5OXl6cMPP1SXLl00atQoa3vBAkN33HGH+vfvr08++USSG78v3RLx4OLHH3801apVM08++aTJyMhw2ffLL7+YcuXKmeXLl7unuSuUmZnp8nzJkiXG4XCY4cOHm5MnTxpj/vgf3AMHDrijvSuyf/9+688F/3vy0ksvmZiYGJe606dPX9W+isOfvw8XL15sHA6HWbJkiTl16pRZu3atuf3220vUFH1RPf7449bMcZ8+fUzVqlVNgwYNzFNPPWU2bdrk5u6K5pdffjGPP/648fHxMR06dDCnTp2y9i1fvtwEBgaapUuXurHDosvPzzc9e/Y0AwcONNHR0SY+Pt7ad/ToUfPYY4+ZZ5991pw9e7ZU/s/6n+Xn55u0tDRz3333mR49epT6MR04cMC0atXK3HTTTeaBBx4wDofDfPTRRyY/P98cP37cDB482ERGRpqTJ0+WynGuW7fOlCtXzrRp08bExMQYX19f8/TTTxtjjNmxY4e57rrrzN69e0vl2Iz5Y5Y8JyfHPPnkk+b+++83WVlZJj8/37rk+vDhw6ZLly6mW7dubu606FJSUszq1aut5xs2bDB+fn7moYceMmlpadb20vgerl279ryrpPLy8kzt2rVdxlwaHT161GzYsMF6/re//c2UL1/ejB071vz97383zZs3N1FRUSYpKcltPRK2SohVq1YZb29v89xzz7lcZnfy5EkTHh5e6v8y/PkXhYJf3keMGGF+/vln88ILL5ju3bubM2fOuLnLovnz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
<Figure size 1000x600 with 1 Axes>
output_type": "display_data
image/png": "iVBORw0KGgoAAAANSUhEUgAAA1sAAAIjCAYAAAD1OgEdAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABO2UlEQVR4nO3de1iUdf7/8ddwFBRGQQEpVCoyVDykheimmOcia20zo0jN1DI1UrM1W0MrKC219ZS6rpontjZ1XWtJrLTMM0XlIe1gHhI8JA5oBAj3749+3t9GxCO3g/B8XNdcl/O53/d9v+9pEl5+7vmMzTAMQwAAAACAcuXm6gYAAAAAoDIibAEAAACABQhbAAAAAGABwhYAAAAAWICwBQAAAAAWIGwBAAAAgAUIWwAAAABgAcIWAAAAAFiAsAUAAAAAFiBsAUAV17dvXzVo0OCy9l2yZImmTJlyzm02m01JSUmX3Vd5sdlsGjJkyAXr5s+fL5vNpp9++sn6pspJgwYN1LdvX1e3AQAog4erGwAAXLuWLFmi7du3KzExsdS2jRs36vrrr7/6TV2mu+++Wxs3blTdunVd3cpFW758ufz9/V3dBgCgDIQtAKhkfv31V/n6+rq6DbVu3drVLVySOnXqqE6dOq5u45K0aNHC1S0AAM6D2wgB4BqWlJQkm82mL774Qn/5y19Uq1Yt3XjjjZIkwzA0Y8YMNW/eXD4+PqpVq5b+8pe/6Mcff7zgcadPn6527dopKChI1atXV1RUlCZMmKCioiKzJjY2Vu+//7727dsnm81mPs44122E27dv17333qtatWqpWrVqat68uRYsWOBUs3btWtlsNi1dulRjxoxRaGio/P391alTJ+3evdup9ssvv1RcXJyCgoLk7e2t0NBQ3X333Tp48GCpa1q4cKEiIyPl6+urZs2aadWqVU7bz3UbYWxsrJo0aaLPPvtMrVu3lo+Pj6677jr97W9/U3Fx8Xlfw/vuu0/169dXSUlJqW3R0dG69dZbzefvvvuuoqOjZbfb5evrqxtuuEGPPfbYeY8vlb6N8FJeO0lKS0tTx44dzfNGRkYqJSXFqWblypWKiYmRr6+v/Pz81LlzZ23cuNGp5sz78Ouvv9YDDzwgu92ugIAADR8+XKdPn9bu3bvVrVs3+fn5qUGDBpowYUKpXnJzczVy5EiFh4fLy8tL1113nRITE3Xq1KkLvg4AUFERtgCgEujZs6duuukmvfvuu3rrrbckSYMGDVJiYqI6deqkFStWaMaMGdqxY4fatGmjw4cPn/d4P/zwg+Lj47Vw4UKtWrVK/fv318SJEzVo0CCzZsaMGWrbtq1CQkK0ceNG81GW3bt3q02bNtqxY4f+/ve/a9myZWrUqJH69u17zl++n3/+ee3bt0//+Mc/NHv2bH333Xe65557zJBz6tQpde7cWYcPH9b06dOVnp6uKVOmqF69esrLy3M61vvvv69p06Zp/Pjxeu+99xQQEKA///nPFxU8s7Oz1bt3bz388MP6z3/+o7/85S96+eWX9fTTT593v8cee0z79+/Xxx9/7DT+7bffasuWLerXr5+k32+3fPDBB3XDDTcoNTVV77//vsaOHavTp09fsLeyXOi1k6S5c+fqrrvuUklJid566y3997//1bBhw5yC6pIlS3TvvffK399fS5cu1dy5c5WTk6PY2FitX7++1Hl79eqlZs2a6b333tOAAQM0efJkPfPMM7rvvvt09913a/ny5brzzjv13HPPadmyZeZ+v/76q9q3b68FCxZo2LBh+t///qfnnntO8+fPV48ePWQYxmW/FgDgUgYA4Jr14osvGpKMsWPHOo1v3LjRkGS88cYbTuMHDhwwfHx8jFGjRpljffr0MerXr1/mOYqLi42ioiLj7bffNtzd3Y3jx4+b2+6+++4y95VkvPjii+bz3r17G97e3sb+/fud6rp37274+voaJ06cMAzDMD755BNDknHXXXc51b3zzjuGJGPjxo2GYRjGtm3bDEnGihUryuz9TB/BwcFGbm6uOZadnW24ubkZKSkp5ti8efMMScbevXvNsfbt2xuSjP/85z9OxxwwYIDh5uZm7Nu3r8zzFhUVGcHBwUZ8fLzT+KhRowwvLy/j2LFjhmEYxuuvv25IMq//UtSvX9/o06eP+fxiX7u8vDzD39/f+NOf/mSUlJSc89jFxcVGaGioERUVZRQXF5vjeXl5RlBQkNGmTRtz7Mz78Oz3W/PmzQ1JxrJly8yxoqIio06dOkbPnj3NsZSUFMPNzc3YunWr0/7//ve/DUnGBx98cJGvCABULMxsAUAlcP/99zs9X7VqlWw2mx555BGdPn3afISEhKhZs2Zau3bteY/35ZdfqkePHgoMDJS7u7s8PT316KOPqri4WHv27LmsHj/++GN17NhRYWFhTuN9+/bVr7/+WmpWrEePHk7PmzZtKknat2+fJOmmm25SrVq19Nxzz+mtt97Szp07yzx3hw4d5OfnZz4PDg5WUFCQeazz8fPzK9VLfHy8SkpK9Omnn5a5n4eHhx555BEtW7ZMDodDklRcXKyFCxfq3nvvVWBgoCTptttuk/T7rNA777yjn3/++YI9XciFXrsNGzYoNzdXgwcPdrr18492796tQ4cOKSEhQW5u//frQo0aNXT//fdr06ZN+vXXX532iYuLc3oeGRkpm82m7t27m2MeHh666aabnF77VatWqUmTJmrevLnT+7Vr166y2WwXfL8CQEVF2AKASuDsFfQOHz4swzAUHBwsT09Pp8emTZt07NixMo+1f/9+3XHHHfr555/15ptv6rPPPtPWrVs1ffp0SVJ+fv5l9fjLL7+cc6W/0NBQc/sfnQkjZ3h7ezud3263a926dWrevLmef/55NW7cWKGhoXrxxRedPlt2rmOdOd7FXEtwcHCpsZCQkHP2fLbHHntMv/32m1JTUyVJH374obKyssxbCCWpXbt2WrFihU6fPq1HH31U119/vZo0aaKlS5desLeyXOi1O3r0qCSdd7XIM9dW1n+zkpIS5eTkOI0HBAQ4Pffy8pKvr6+qVatWavy3334znx8+fFhff/11qfeqn5+fDMM47/sVACoyViMEgErg7NmJ2rVry2az6bPPPjN/0f6jc42dsWLFCp06dUrLli1T/fr1zfHMzMwr6jEwMFBZWVmlxg8dOmT2fKmioqKUmpoqwzD09ddfa/78+Ro/frx8fHz017/+9Yr6PeNcn2/Lzs6WdO4Q90eNGjXS7bffrnnz5mnQoEGaN2+eQkND1aVLF6e6e++9V/fee68KCgq0adMmpaSkKD4+Xg0aNFBMTEy5XMcfnVl18VwLiZxx5trK+m/m5uamWrVqlUs/tWvXlo+Pj/75z3+WuR0ArkXMbAFAJRQXFyfDMPTzzz+rVatWpR5RUVFl7nsmuP0xkBmGoTlz5pSqvdjZIUnq2LGjPv74YzNcnfH222/L19f3ipaKt9lsatasmSZPnqyaNWvqiy++uOxjnS0vL08rV650GluyZInc3NzUrl27C+7fr18/bd68WevXr9d///tf9enTR+7u7ues9fb2Vvv27fXaa69J+v12Tiu0adNGdrtdb731VpmLTzRs2FDXXXedlixZ4lRz6tQpvffee+YKheUhLi5OP/zwgwIDA8/5fr3cL90GAFdjZgsAKqG2bdtq4MCB6tevn7Zt26Z27dqpevXqysrK0vr16xUVFaUnn3zynPt27txZXl5eeuihhzRq1Cj99ttvmjlzZqlbxqTfZ5aWLVummTNnqmXLlnJzc1OrVq3OedwXX3xRq1atUocOHTR27FgFBARo8eLFev/99zVhwgTZ7fZLusZVq1ZpxowZuu+++3TDDTfIMAwtW7ZMJ06cUOfOnS/pWOcTGBioJ598Uvv379fNN9+sDz74QHPmzNGTTz6pevXqXXD/hx56SMOHD9dDDz2kgoICp6XaJWns2LE6ePCgOnbsqOuvv14nTpzQm2++KU9PT7Vv377cruOPatSooTfeeEOPP/64OnXqpAEDBig4OFjff/+9vvrqK02bNk1ubm6aMGGCHn74YcXFxWnQoEEqKCjQxIkTdeLECb366qvl1k9iYqLee+89tWvXTs8884yaNm2qkpIS7d+/X6tXr9aIESMUHR1dbucDgKuFsAUAldSsWbPUunVrzZo1SzNmzFBJSYlCQ0PVtm1b3X777WXud8stt+i9997TCy+8oJ49eyowMFDx8fEaPny400IHkvT0009rx44dev755+VwOGQYxnlnSjZs2KDnn39eTz31lPLz8xUZGal58+aVCiAXIyIiQjVr1tSECRN06NAheXl5qWHDhpo/f7769OlzyccrS0hIiKZPn66RI0fqm2++UUBAgJ5//nmNGzfuova32+3685//rCVLlqht27a6+eabnbZHR0dr27Zteu6553T06FHVrFlTrVq10scff6zGjRuX23WcrX///goNDdVrr72mxx9/XIZhqEGDBk6vXXx8vKpXr66UlBQ9+OCDcnd3V+vWrfXJJ5+oTZs25dZL9erV9dlnn+nVV1/V7NmztXfvXvn4+KhevXrq1KkTM1sArlk2o6yfigAAVHGxsbE6duyYtm/f7upWAADXID6zBQAAAAAWIGwBAAAAgAW4jRAAAAAALMDMFgAAAABYgLAFAAAAABYgbAEAAACABfierYtUUlKiQ4cOyc/PTzabzdXtAAAAAHARwzCUl5en0NBQubmVPX9F2LpIhw4dUlhYmKvbAAAAAFBBHDhwQNdff32Z2wlbF8nPz0/S7y+ov7+/i7sBAAAA4Cq5ubkKCwszM0JZCFsX6cytg/7+/oQtAAAAABf8eBELZAAAAACABQhbAAAAAGABbiMEAAAAcFGKi4tVVFTk6jYs5+npKXd39ys+DmELAAAAwHkZhqHs7GydOHHC1a1cNTVr1lRISMgVfe0TYQsAAADAeZ0JWkFBQfL19a3U3ztrGIZ+/fVXHTlyRJJUt27dyz4WYQsAAABAmYqLi82gFRgY6Op2rgofHx9J0pEjRxQUFHTZtxSyQAYAAACAMp35jJavr6+LO7m6zlzvlXxGjbAFAAAA4IIq862D51Ie10vYAgAAAAALELYAAAAAwAKELQAAAACXLTY2VomJia5uo0JiNUIAAAAAl23ZsmXy9PR0dRsVEmELAAAAwGULCAhwdQsVFrcRAgAAALhsf7yNsEGDBkpOTtZjjz0mPz8/1atXT7Nnz3aqP3jwoHr37q2AgABVr15drVq10ubNm83tM2fO1I033igvLy81bNhQCxcudNrfZrNp1qxZiouLk6+vryIjI7Vx40Z9//33io2NVfXq1RUTE6MffvjBab///ve/atmypapVq6YbbrhB48aN0+nTp615Uf4/whYAAACAcvPGG2+oVatW+vLLLzV48GA9+eST+vbbbyVJJ0+eVPv27XXo0CGtXLlSX331lUaNGqWSkhJJ0vLly/X0009rxIgR2r59uwYNGqR+/frpk08+cTrHSy+9pEcffVSZmZm65ZZbFB8fr0GDBmn06NHatm2bJGnIkCFm/YcffqhHHnlEw4YN086dOzVr1izNnz9fr7zyiqWvhc0wDMPSM1QSubm5stvtcjgc8vf3d3U7AAAAwFXx22+/ae/evQoPD1e1atVKbY+NjVXz5s01ZcoUNWjQQHfccYc5G2UYhkJCQjRu3Dg98cQTmj17tkaOHKmffvrpnLcftm3bVo0bN3aaDevVq5dOnTql999/X9LvM1svvPCCXnrpJUnSpk2bFBMTo7lz5+qxxx6TJKWmpqpfv37Kz8+XJLVr107du3fX6NGjzeMuWrRIo0aN0qFDhy75ui82GzCzBQAAAKDcNG3a1PyzzWZTSEiIjhw5IknKzMxUixYtyvyc165du9S2bVunsbZt22rXrl1lniM4OFiSFBUV5TT222+/KTc3V5KUkZGh8ePHq0aNGuZjwIABysrK0q+//noFV3t+LJABAAAAoNycvTKhzWYzbxP08fG54P42m83puWEYpcb+eI4z2841dua8JSUlGjdunHr27FnqfOearSsvzGwBAAAAuCqaNm2qzMxMHT9+/JzbIyMjtX79eqexDRs2KDIy8orOe+utt2r37t266aabSj3c3KyLRMxsARZr+ezbrm6hXGVMfNTVLQAAgGvUQw89pOTkZN13331KSUlR3bp19eWXXyo0NFQxMTF69tln1atXL916663q2LGj/vvf/2rZsmVas2bNFZ137NixiouLU1hYmB544AG5ubnp66+/1jfffKOXX365nK6uNGa2AAAAAFwVXl5eWr16tYKCgnTXXXcpKipKr776qtzd3SVJ9913n958801NnDhRjRs31qxZszRv3jzFxsZe0Xm7du2qVatWKT09Xbfddptat26tSZMmqX79+uVwVWVjNcKLxGqEuFzMbAEAgGvZhVYjrKxYjRAAAAAAKijCFgAAAABYgLAFAAAAABYgbAEAAACABQhbAAAAAGABwhYAAAAAWIAvNbZYZVr2myW/AQAAgIvHzBYAAAAAWICwBQAAAAAW4DZCAAAAAJftan9s5lr6aAszWwAAAAAqvRkzZig8PFzVqlVTy5Yt9dlnn1l+TsIWAAAAgErtX//6lxITEzVmzBh9+eWXuuOOO9S9e3ft37/f0vMStgAAAABUapMmTVL//v31+OOPKzIyUlOmTFFYWJhmzpxp6XkJWwAAAAAqrcLCQmVkZKhLly5O4126dNGGDRssPTdhCwAAAECldezYMRUXFys4ONhpPDg4WNnZ2Zaem7AFAAAAoNKz2WxOzw3DKDVW3ghbAAAAACqt2rVry93dvdQs1pEjR0rNdpU3whYAAACASsvLy0stW7ZUenq603h6erratGlj6bn5UmMAAAAAldrw4cOVkJCgVq1aKSYmRrNnz9b+/fv1xBNPWHpel4atBg0aaN++faXGBw8erOnTp8swDI0bN06zZ89WTk6OoqOjNX36dDVu3NisLSgo0MiRI7V06VLl5+erY8eOmjFjhq6//nqzJicnR8OGDdPKlSslST169NDUqVNVs2ZNy68RAAAAqMwyJj7q6hYu6MEHH9Qvv/yi8ePHKysrS02aNNEHH3yg+vXrW3pel95GuHXrVmVlZZmPM1N7DzzwgCRpwoQJmjRpkqZNm6atW7cqJCREnTt3Vl5e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2r59uyZNmqQdO3ZI4pftquDrr79W27ZtNXjwYL3xxhvm+G+//cbPg4tgM/i/BOfQr18/ff/99/rss8/Msfz8fC1ZskTTp09X165dlZKS4sIOYaVTp05p2LBhKikpUatWrTR06FCNHDlSo0aNUu3atV3dHix2/PhxPfTQQ7rlllv05ptvmuN33nmnoqKi9Oabb8owDH7IVnLHjh3T/fffrxYtWmjKlCmSfv/F+q677tLYsWPl4+OjwMBAhYWFubZRWKq4uFjHjx/Xn/70J3388cfasmWLUlJS1Lx5c+3YsUN169bVv//9b1e3CYtkZ2erRYsWatasmdLS0lRcXKxnnnlGe/bs0Z49e9SvXz/FxcWpRYsWrm61wvJwdQOoWM78AnXrrbdq9+7d+vbbb3XLLbdIknx8fPTAAw9oz549+uSTT3TkyBEFBQW5uGNYwc3NTS1btlRgYKAefPBB1alTR71795YkAlcVUFRUpBMnTugvf/mLJKmkpERubm664YYb9Msvv0gSQasKsNls6tatm/k+kKSXX35ZH374obKzs3Xs2DE1btxYL7zwgv70pz+5sFNYyc3NTXXq1NFtt92m7du3689//rO8vb3Vp08fFRQUaMCAAa5uERaLiYnRgQMH9J///EdvvfWWTp8+rdtvv11RUVF65513tH37do0fP14NGzZ0dasVErcRwsmZX6Duuusufffdd5owYYLy8vLM7f7+/kpMTNTWrVu1YcMGV7UJi/n4+KhPnz7mLaO9evXS0qVL9frrr+u1114zf+EuKSnR3r17XdkqLBAcHKxFixbpjjvukPR/i6Rcd911cnNz/rFx8uTJq94fro7AwEANGTJEERERkqTU1FS9+OKLWrp0qT766CMtXrxYOTk5+uijj1zcKax05vcCd3d3rV27VpK0bNkyFRcXKywsTJ999pm2bNniwg5hpZCQEE2fPl2NGjVS7969VVxcrH/961965ZVXNHHiRL300ktat26dvvrqK1e3WmExs4VzuvHGG/XOO++oe/fu8vX1VVJSkjmb4eXlpRYtWqhmzZqubRKWOvMB2OLiYrm5uenBBx+UYRiKj4+XzWZTYmKiXn/9de3bt08LFy6Ur6+viztGeTrzC3ZJSYk8PT0l/f5eOHz4sFmTkpIib29vDRs2TB4e/DipjPz8/Mw/x8TEaNu2bbr11lslSe3atVNwcLAyMjJc1R6ugjN3vNx555368ccfNXjwYH3wwQfKyMhQZmamnn32WXl5ealp06aqVq2aq9uFBerWrauUlBRdf/316ty5swICAsw7Hu677z6NGTNGn376qXr16uXqViskfjqiTB06dNC7776rBx54QIcOHdIDDzygpk2bauHChTp48KBuvPFGV7eIq8Dd3V2GYaikpES9e/eWzWZTQkKCVq5cqR9++EFbt24laFVibm5u5i9bNptN7u7ukqSxY8fq5Zdf1pdffknQqiLq16+v+vXrS/r9F/DCwkLVqFFDTZo0cXFnsNKZma3w8HD169dPwcHBWrVqlcLDwxUeHi6bzaZmzZoRtCq50NBQjRo1Sj4+PpL+72fDiRMnFBgYqJYtW7q4w4qLBTJwQV988YWGDx+uvXv3ysPDQ56enlq6dCkfhqxizvxVYbPZ1LFjR2VmZmrt2rWKiopycWew2pl/wUxKSlJWVpYiIiL0wgsvaMOGDeYsB6qesWPHasGCBVqzZo05E4rKq6ioSAsXLlSrVq3UtGlTFsmBpN//Hli6dKnS09PVoEEDV7dTIfHPkbigW2+9VStXrtTx48d18uRJhYSEsEBCFWSz2VRcXKxnn31Wn3zyiTIzMwlaVcSZz2l5enpqzpw58vf31/r16wlaVdS///1vrV27VqmpqUpPTydoVRGenp7q27ev+fcBQatqS01N1dq1a/XOO+/oo48+ImidBwtk4KL4+/urQYMGatKkCUGrimvcuLG++OILNW3a1NWt4Crr2rWrJGnDhg1q1aqVi7uBq0RGRuro0aP69NNPucOhijl7gRxUXY0aNdLBgwf12Wef8ffABXAbIYBLwq0jVdupU6fMxVNQdRUVFZkLpwComgoLC+Xl5eXqNio8whYAAAAAWID5YAAAAACwAGELAAAAACxA2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAFQKsbGxSkxMrDDHuVhJSUlq3rz5eWuudk8AgPLh4eoGAABwhbVr16pDhw7KyclRzZo1zfFly5ZVuC/srYg9AQAujLAFAKjwCgsL5eXldVXOFRAQcFXOcykqYk8AgAvjNkIAQIUTGxurIUOGaPjw4apdu7Y6d+6snTt36q677lKNGjUUHByshIQEHTt2rMxjLFq0SK1atZKfn59CQkIUHx+vI0eOSJJ++ukndejQQZJUq1Yt2Ww29e3b1zz3H2/Zy8nJ0aOPPqpatWrJ19dX3bt313fffWdunz9/vmrWrKkPP/xQkZGRqlGjhrp166asrCyzZu3atbr99ttVvXp11axZU23bttW+ffuc+l24cKEaNGggu92u3r17Ky8vz+n1+GNPDRo00EsvvaT4+HjVqFFDoaGhmjp16iW/zgAAaxG2AAAV0oIFC+Th4aHPP/9cr776qtq3b6/mzZtr27ZtSktL0+HDh9WrV68y9y8sLNRLL72kr776SitWrNDevXvNQBUWFqb33ntPkrR7925lZWXpzTffPOdx+vbtq23btmnlypXauHGjDMPQXXfdpaKiIrPm119/1euvv66FCxfq008/1f79+zVy5EhJ0unTp3Xfffepffv2+vrrr7Vx40YNHDhQNpvN3P+HH37QihUrtGrVKq1atUrr1q3Tq6++et7XZ+LEiWratKm++OILjR49Ws8884zS09Mv6rUFAFwd3EYIAKiQbrrpJk2YMEGSNHbsWN16661KTk42t//zn/9UWFiY9uzZo5tvvrnU/o899pj55xtuuEF///vfdfvtt+vkyZOqUaOGeWteUFCQ02e2/ui7777TypUr9fnnn6tNmzaSpMWLFyssLEwrVqzQAw88IEkqKirSW2+9pRtvvFGSNGTIEI0fP16SlJubK4fDobi4OHN7ZGSk03lKSko0f/58+fn5SZISEhL00Ucf6ZVXXinz9Wnbtq3++te/SpJuvvlmff7555o8ebI6d+5c5j4AgKuLmS0AQIXUqlUr888ZGRn65JNPVKNGDfNxyy23SPp9VuhcvvzyS917772qX7++/Pz8FBsbK0nav3//Rfewa9cueXh4KDo62hwLDAxUw4YNtWvXLnPM19fXDFKSVLduXfOWxYCAAPXt21ddu3bVPffcozfffNPpFkPp99sCzwSts/cvS0xMTKnnf+wJAOB6hC0AQIVUvXp1888lJSW65557lJmZ6fT47rvv1K5du1L7njp1Sl26dFGNGjW0aNEibd26VcuXL5f0++2FF8swjDLH/3gb4NkrBdpsNqd9582bp40bN6pNmzb617/+pZtvvlmbNm067/4lJSUX3ecf9wMAVBzcRggAqPBuvfVWvffee2rQoIE8PC78o+vbb7/VsWPH9OqrryosLEyStG3bNqeaM6sbFhcXl3mcRo0a6fTp09q8ebN5G+Evv/yiPXv2lLoV8EJatGihFi1aaPTo0YqJidGSJUvUunXrSzrGH/0xrJ15fma2DwBQMTCzBQCo8J566ikdP35cDz30kLZs2aIff/xRq1ev1mOPPXbOsFSvXj15eXlp6tSp+vHHH7Vy5Uq99NJLTjX169eXzWbTqlWrdPToUZ08ebLUcSIiInTvvfdqwIABWr9+vb766is98sgjuu6663TvvfdeVO979+7V6NGjtXHjRu3bt0+rV6++rLB2ts8//1wTJkzQnj17NH36dL377rt6+umnr+iYAIDyRdgCAFR4oaGh+vzzz1VcXKyuXbuqSZMmevrpp2W32+XmVvpHWZ06dTR//ny9++67atSokV599VW9/vrrTjXXXXedxo0bp7/+9a8KDg7WkCFDznnuefPmqWXLloqLi1NMTIwMw9AHH3xw0V8y7Ovrq2+//Vb333+/br75Zg0cOFBDhgzRoEGDLv2F+IMRI0YoIyNDLVq00EsvvaQ33nhDXbt2vaJjAgDKl80o64Z0AABQITVo0ECJiYlO370FAKh4mNkCAAAAAAsQtgAAAADAAtxGCAAAAAAWYGYLAAAAACxA2AIAAAAACxC2AAAAAMAChC0AAAAAsABhCwAAAAAsQNgCAAAAAAsQtgAAAADAAoQtAAAAALDA/wN9VIir59YGtwAAAABJRU5ErkJggg==
<Figure size 1000x600 with 1 Axes>
output_type": "display_data
image/png": "iVBORw0KGgoAAAANSUhEUgAAA2QAAAIjCAYAAABswtioAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABMyklEQVR4nO3de1RVdR738c+Rm0hyEpFboZJjhII3bBStxFQQRaes1GhIzbDGkiGlZqzGrCmZyTR7tBrrMU2laGa6TjokZjfHO0aJmlljqQniBQ9CCgjn+aNxPx1R8wL8EN6vtc5a7N/+nr2/v7NO6qff3hub0+l0CgAAAABQ75qZbgAAAAAAmioCGQAAAAAYQiADAAAAAEMIZAAAAABgCIEMAAAAAAwhkAEAAACAIQQyAAAAADCEQAYAAAAAhhDIAAAAAMAQAhkAAHUsNjZWsbGxptsAADRANqfT6TTdBAAAjdm2bdskSZ06dTLcCQCgoSGQAQDq1Y8//qgWLVqYbgMAgAaBSxYBAHVm+vTpstls2rx5s2699Va1atVKHTp0kCRt2rRJo0ePVvv27eXt7a327dvr9ttv1/fff1/jOD/88IMmTJig0NBQeXp6KiQkRLfeeqv2799v1ZSUlCg9PV1hYWHy9PTUFVdcobS0NJWVlZ21x7S0NPn4+KikpKTGvlGjRikwMFCVlZWSpFWrVik2NlatW7eWt7e32rZtq1tuuUU//vjjWc9x6iWL3333nWw2m5555hnNnj1bYWFhuuyyyxQTE6N169bVeP/69es1bNgwtW7dWs2bN1eHDh2UlpbmUrN69WoNGDBALVu2VIsWLdSnTx8tW7bMpWbRokWy2WxatWqVUlJS1Lp1a/n6+urOO+9UWVmZCgsLNXLkSF1++eUKDg5Wenq6NfeTKioq9OSTT+qaa66Rl5eX2rRpo3HjxunAgQNn/QwAAKfnbroBAEDjN2LECI0ePVr33nuvFZC+++47hYeHa/To0fLz81NBQYFefPFFXXvttdq2bZv8/f0l/RTGrr32WlVWVurhhx9Wly5ddOjQIX3wwQcqLi5WYGCgfvzxR/Xr10979+61arZu3app06Zpy5YtWrlypWw222l7u+uuu/Tcc8/p73//u+6++25r/MiRI3r33Xd13333ycPDQ999952GDh2q66+/Xq+88oouv/xy/fDDD8rOzlZFRcUFrfo9//zzuuaaazRnzhxJ0p/+9CcNGTJEu3btkt1ulyR98MEHGjZsmCIiIjR79my1bdtW3333nVasWGEd55NPPtGgQYPUpUsXLViwQF5eXnrhhRc0bNgwvf766xo1apTLee+++26NGDFCWVlZ+vzzz/Xwww/rxIkT2rFjh0aMGKEJEyZo5cqV+utf/6qQkBBNnjxZklRdXa3f/OY3+uyzz/TQQw+pT58++v777/XYY48pNjZWmzZtkre393l/DgDQpDkBAKgjjz32mFOSc9q0ab9Ye+LECWdpaanTx8fH+dxzz1njd911l9PDw8O5bdu2M743IyPD2axZM+fGjRtdxv/5z386JTmXL19+1nP36NHD2adPH5exF154wSnJuWXLFpdj5eXl/eJcTtWvXz9nv379rO1du3Y5JTmjoqKcJ06csMY3bNjglOR8/fXXrbEOHTo4O3To4Dx27NgZj9+7d29nQECA8+jRo9bYiRMnnJGRkc4rr7zSWV1d7XQ6nc6FCxc6JTknTZrk8v6bbrrJKck5e/Zsl/Fu3bo5e/ToYW2//vrrTknON99806Vu48aNTknOF1544Rw+DQDAz3HJIgCgzt1yyy01xkpLS/WHP/xBv/rVr+Tu7i53d3dddtllKisr0/bt2626f//73+rfv78iIiLOePz3339fkZGR6tatm06cOGG94uPjZbPZ9PHHH5+1v3HjxmnNmjXasWOHNbZw4UJde+21ioyMlCR169ZNnp6emjBhgl599VX997//Pc9PoaahQ4fKzc3N2u7SpYskWZdtfv311/r22281fvx4NW/e/LTHKCsr0/r163Xrrbfqsssus8bd3NyUnJysvXv3usxLkhITE122T362Q4cOrTH+80tI33//fV1++eUaNmyYy+fcrVs3BQUF/eLnDACoiUAGAKhzwcHBNcaSkpI0b9483X333frggw+0YcMGbdy4UW3atNGxY8esugMHDujKK6886/H379+vL7/8Uh4eHi6vli1byul06uDBg2d9/x133CEvLy8tWrRI0k9PRdy4caPGjRtn1XTo0EErV65UQECA7rvvPnXo0EEdOnTQc889dx6fhKvWrVu7bHt5eUmSNf+T92Wdbf7FxcVyOp2n/YxDQkIkSYcOHXIZ9/Pzc9n29PQ84/jx48et7f379+vIkSPy9PSs8VkXFhb+4ucMAKiJe8gAAHXu1Pu3HA6H3n//fT322GP64x//aI2Xl5fr8OHDLrVt2rTR3r17z3p8f39/eXt765VXXjnj/rNp1aqVfvOb32jx4sV68skntXDhQjVv3ly33367S93111+v66+/XlVVVdq0aZPmzp2rtLQ0BQYGavTo0Wc9x4Vo06aNJJ11/q1atVKzZs1UUFBQY9++ffsk/fL8z5W/v79at26t7Ozs0+5v2bJlrZwHAJoSVsgAAPXOZrPJ6XRaK0In/d//+39VVVXlMpaQkKCPPvqoxmV3P5eYmKhvv/1WrVu3Vs+ePWu82rdv/4s9jRs3Tvv27dPy5cu1dOlS3Xzzzbr88stPW+vm5qZevXrp+eeflyRt3rz5F49/Ia6++mp16NBBr7zyisrLy09b4+Pjo169eumtt95yWVmsrq7W0qVLdeWVV+rqq6+ulX4SExN16NAhVVVVnfZzDg8Pr5XzAEBTwgoZAKDe+fr66oYbbtDMmTPl7++v9u3b65NPPtGCBQtqhKAnnnhC//73v3XDDTfo4YcfVlRUlI4cOaLs7GxNnjxZ11xzjdLS0vTmm2/qhhtu0AMPPKAuXbqourpau3fv1ooVKzRlyhT16tXrrD3FxcXpyiuv1MSJE1VYWOhyuaIk/e1vf9OqVas0dOhQtW3bVsePH7dW5AYOHFirn8/PPf/88xo2bJh69+6tBx54QG3bttXu3bv1wQcfKDMzU5KUkZGhQYMGqX///kpPT5enp6deeOEF5efn6/XXXz/jEybP1+jRo5WZmakhQ4bo97//vX7961/Lw8NDe/fu1UcffaTf/OY3uvnmm2vlXADQVBDIAABGvPbaa/r973+vhx56SCdOnFDfvn2Vk5NT48ESV1xxhTZs2KDHHntMf/nLX3To0CG1adNG1113nXXPk4+Pjz777DP95S9/0UsvvaRdu3ZZvyds4MCB57RC1qxZM915552aMWOGQkNDNWDAAJf93bp104oVK/TYY4+psLBQl112mSIjI/Xee+8pLi6u1j6XU8XHx+vTTz/VE088odTUVB0/flxXXnmlhg8fbtX069dPq1at0mOPPaaxY8equrpaXbt21XvvvVfjAR4Xw83NTe+9956ee+45LVmyRBkZGXJ3d9eVV16pfv36KSoqqtbOBQBNhc3pdDpNNwEAAAAATRH3kAEAAACAIQQyAAAAADCEQAYAAAAAhhDIAAAAAMAQAhkAAAAAGEIgAwAAAABD+D1ktai6ulr79u1Ty5Yta+2XcAIAAAC49DidTh09elQhISFq1uzM62AEslq0b98+hYaGmm4DAAAAQAOxZ88eXXnllWfcTyCrRS1btpT004fu6+truBsAAAAAppSUlCg0NNTKCGdCIKtFJy9T9PX1JZABAAAA+MVbmXioBwAAAAAYQiADAAAAAEO4ZBEAAABAramqqlJlZaXpNuqch4eH3NzcLvo4BDIAAAAAF83pdKqwsFBHjhwx3Uq9ufzyyxUUFHRRv/KKQAYAAADgop0MYwEBAWrRokWj/r28TqdTP/74o4qKiiRJwcHBF3wsAhkAAACAi1JVVWWFsdatW5tup154e3tLkoqKihQQEHDBly/yUA8AAAAAF+XkPWMtWrQw3En9Ojnfi7lnjkAGAAAAoFY05ssUT6c25ksgAwAAAABDCGQAAAAAYAiBDAAAAECdio2NVVpamuk2GiSesggAAACgTr311lvy8PAw3UaDRCADAAAAUKf8/PxMt9BgcckiAAAAgDr180sW27dvrxkzZuiuu+5Sy5Yt1bZtW7300ksu9Xv37tXo0aPl5+cnHx8f9ezZU+vXr7f2v/jii+rQoYM8PT0VHh6uJUuWuLzfZrNp/vz5SkxMVIsWLRQREaG1a9fqm2++UWxsrHx8fBQTE6Nvv/3W5X3/+te/FB0drebNm+uqq67S448/rhMnTtTNh/I/BDIAAAAA9WrWrFnq2bOnPv/8c02cOFG/+93v9NVXX0mSSktL1a9fP+3bt0/vvfeevvjiCz300EOqrq6WJL399tv6/e9/rylTpig/P1/33HOPxo0bp48++sjlHH/+85915513Ki8vT9dcc42SkpJ0zz33aOrUqdq0aZMk6f7777fqP/jgA/32t79Vamqqtm3bpvnz52vRokV66qmn6vSzsDmdTmednqEJKSkpkd1ul8PhkK+vr+l2AAAAgHpx/Phx7dq1S2FhYWrevHmN/bGxserWrZvmzJmj9u3b6/rrr7dWtZxOp4KCgvT444/r3nvv1UsvvaT09HR99913p73UsW/fvurcubPLqtrIkSNVVlamZcuWSfpphezRRx/Vn//8Z0nSunXrFBMTowULFuiuu+6SJGVlZWncuHE6duyYJOmGG25QQkKCpk6dah136dKleuih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LgPUpcrKSuuBMgAuTQQyAAAAADCENUgAAAAAMIRABgAAAACGEMgAAAAAwBACGQAAAAAYQiADAAAAAEMIZAAAAABgCIEMAAAAAAwhkAEAAACAIQQyAADOUUVFhekWAACNDIEMAIAziI2N1f3336/JkyfL399fgwYN0uzZsxUVFSUfHx+FhoZq4sSJKi0tdXnff/7zH/Xr108tWrRQq1atFB8fr+LiYkmS0+nU008/rauuukre3t7q2rWr/vnPf5qYHgCgASCQAQBwFq+++qrc3d31n//8R/Pnz1ezZs30f/7P/1F+fr5effVVrVq1Sg899JBVn5eXpwEDBqhz585au3atVq9erWHDhqmqqkqS9Oijj2rhwoV68cUXtXXrVj3wwAP67W9/q08++cTUFAEABtmcTqfTdBMAADREsbGxcjgc+vzzz89Y849//EO/+93vdPDgQUlSUlKSdu/erdWrV9eoLSsrk7+/v1atWqWYmBhr/O6779aPP/6o1157rfYnAQBo0NxNNwAAQEPWs2dPl+2PPvpIM2bM0LZt21RSUqITJ07o+PHjKisrk4+Pj/Ly8nTbbbed9ljbtm3T8ePHNWjQIJfxiooKde/evc7mAABouAhkAACchY+Pj/Xz999/ryFDhujee+/Vn//8Z/n5+Wn16tUaP368KisrJUne3t5nPFZ1dbUkadmyZbriiitc9nl5edVB9wCAho5ABgDAOdq0aZNOnDihWbNmqVmzn27D/vvf/+5S06VLF3344Yd6/PHHa7y/U6dO8vLy0u7du9WvX7966RkA0LARyAAAOEcdOnTQiRMnNHfuXA0bNkz/+c9/9Le//c2lZurUqYqKitLEiRN17733ytPTUx999JFuu+02+fv7Kz09XQ888ICqq6t13XXXqaSkRGvWrNFll12mMWPGGJoZAMAUnrIIAMA56tatm2bPnq2//vWvioyMVGZmpjIyMlxqrr76aq1YsUJffPGFfv3rXysmJkbvvvuu3N1/+n+gf/7znzVt2jRlZGQoIiJC8fHx+te//qWwsDATUwIAGMZTFgEAAADAEFbIAAAAAMAQAhkAAAAAGEIgAwAAAABDCGQAAAAAYAiBDAAAAAAMIZABAAAAgCEEMgAAAAAwhEAGAAAAAIYQyAAAAADAEAIZAAAAABhCIAMAAAAAQ/4fy+JHkabTMd8AAAAASUVORK5CYII=
<Figure size 1000x600 with 1 Axes>
output_type": "display_data
image/png": "iVBORw0KGgoAAAANSUhEUgAAA2QAAAIjCAYAAABswtioAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABEu0lEQVR4nO3de1hVdd7//9eWk0CyFQiICdTKMVRMo1K0UlNRJ3S8vcuM2pk5apkSqXnIydQpmTSV0rHUGikP0cyUfc2KxA6Wg0eUyjI7WWqCWOLGIyCu3x/drl9bPAt8EJ6P69rX5fqs91rr/aHruve87s9aazssy7IEAAAAAKhydUw3AAAAAAC1FYEMAAAAAAwhkAEAAACAIQQyAAAAADCEQAYAAAAAhhDIAAAAAMAQAhkAAAAAGEIgAwAAAABDCGQAAAAAYAiBDACACnb//ferUaNGptsAAFwCHJZlWaabAACgJvn+++9VVFSk1q1bm24FAFDNEcgAAAAAwBBuWQQAVJm9e/dq8ODBioqKkp+fny6//HK1b99eK1eu9KhbuXKlOnfurKCgIAUEBKh9+/b64IMP7P3ffvutgoKCdOedd3oc9+GHH8rLy0tPPPHEaXtIS0uTw+HQd999V27fmDFj5Ovrq19++UWStHnzZiUmJiosLEx+fn6KjIzU7bffrl27dp1xnqe6ZdHhcGjYsGFauHChYmJiFBAQoOuuu07Lly8vd/zXX3+tu+++W+Hh4fLz81N0dLTuu+8+FRcX2zVbtmzRn//8ZzVo0EB169ZVq1at9Morr3ic5+OPP5bD4dCSJUs0ZswYXXHFFbrsssvUs2dP7dmzRwcOHNDgwYMVGhqq0NBQDRgwQAcPHvQ4h2VZmjNnjlq1aiV/f381aNBAd9xxh3744Ycz/g0AAOeGQAYAqDIul0tvvfWWJkyYoBUrVuill15Sly5d9Ouvv9o1ixYtUkJCgoKCgvTKK6/oX//6l4KDg9WtWzc7lDVp0kTz58/Xf/7zHz3//POSpPz8fCUlJemWW27RxIkTT9vDvffeK19fX6Wnp3uMl5WVadGiRerZs6dCQ0N16NAhde3aVXv27NE//vEPZWVlKS0tTdHR0Tpw4MAFzf+dd97R7NmzNXnyZL3xxhsKDg7W//zP/3iEm88++0w33nij1q5dq8mTJ+u9995TamqqiouLVVJSIknatm2b2rVrpy+//FLPP/+83nzzTTVr1kz333+/pk6dWu66jz/+uAoKCpSenq7p06fr448/1t13363//d//ldPp1GuvvabRo0dr4cKFevzxxz2OHTJkiFJSUtSlSxe99dZbmjNnjr788ku1a9dOe/bsuaC/AwDgdywAAKrIZZddZqWkpJx2/6FDh6zg4GCrZ8+eHuNlZWXWddddZ910000e4w899JDl6+trrVmzxrrtttussLAwa/fu3Wfto0+fPtaVV15plZWV2WPvvvuuJcl6++23LcuyrI0bN1qSrLfeeut8pmhZlmX179/fatiwoceYJCs8PNwqKiqyx/Lz8606depYqamp9thtt91m1a9f3yooKDjt+fv162f5+flZO3bs8Bjv0aOHFRAQYO3fv9+yLMv66KOPLEnl/p4pKSmWJCs5OdljvHfv3lZwcLC9vWbNGkuSNX36dI+6nTt3Wv7+/tbo0aPP8FcAAJwLVsgAAFXmpptuUnp6up566imtXbtWpaWlHvuzs7O1b98+9e/fX8eOHbM/x48fV/fu3bVhwwYdOnTIrp85c6aaN2+uTp066eOPP9aiRYt0xRVXnLWPAQMGaNeuXR63Si5YsEARERHq0aOHJOmaa65RgwYNNGbMGL344ov66quvLnr+nTp1Ur169ezt8PBwhYWF6aeffpIkHT58WKtWrVLfvn11+eWXn/Y8H374oTp37qyoqCiP8fvvv1+HDx/WmjVrPMYTExM9tmNiYiRJt99+e7nxffv22bctLl++XA6HQ/fee6/Hf4+IiAhdd911+vjjj8/vDwAAKIdABgCoMq+//rr69++vl156SfHx8QoODtZ9992n/Px8SbJvgbvjjjvk4+Pj8XnmmWdkWZb27dtnn8/Pz09JSUk6evSoWrVqpa5du55THz169NAVV1yhBQsWSJIKCwu1bNky3XffffLy8pIkOZ1OrVq1Sq1atdLjjz+u5s2bKzIyUk8++WS5IHmuQkJCyo35+fnpyJEjdh9lZWW68sorz3ieX3/99ZTBMzIy0t7/e8HBwR7bvr6+Zxw/evSopN/+e1iWpfDw8HL/PdauXWs/awcAuHDephsAANQeoaGhSktLU1pamnbs2KFly5Zp7NixKigoUGZmpkJDQyVJs2bNUtu2bU95jvDwcPvfW7Zs0YQJE3TjjTdqw4YNmjFjhkaMGHHWPry8vORyufT8889r//79WrJkiYqLizVgwACPutjYWGVkZMiyLH3++edKT0/X5MmT5e/vr7Fjx17EX+LUgoOD5eXlddaXhoSEhCgvL6/c+O7duyXJ/jterNDQUDkcDn366afy8/Mrt/9UYwCA88MKGQDAiOjoaA0bNkxdu3bVpk2bJEnt27dX/fr19dVXX+mGG2445efEKs6hQ4d05513qlGjRvroo480bNgwjR07VuvWrTun6w8YMEBHjx7Va6+9pvT0dMXHx+vaa689Za3D4dB1112nmTNnqn79+na/Fc3f318dOnTQv//97zOuPnXu3FkffvihHcBOePXVVxUQEHDaMHu+EhMTZVmWfv7551P+t4iNja2Q6wBAbcYKGQCgSrjdbnXq1ElJSUm69tprVa9ePW3YsEGZmZnq06ePJOmyyy7TrFmz1L9/f+3bt0933HGHwsLCtHfvXn322Wfau3evXnjhBUnSgw8+qB07dmj9+vUKDAzU9OnTtWbNGvXr10+bN29W/fr1z9jPtddeq/j4eKWmpmrnzp2aN2+ex/7ly5drzpw56t27t6666ipZlqU333xT+/fvP+dbIy/EjBkzdPPNN6tNmzYaO3asrrnmGu3Zs0fLli3T3LlzVa9ePT355JNavny5OnXqpAkTJig4OFiLFy/WO++8o6lTp8rpdFZIL+3bt9fgwYM1YMAAbdy4UbfeeqsCAwOVl5en1atXKzY2Vg899FCFXAsAaisCGQCgStStW1dt2rTRwoUL9eOPP6q0tFTR0dEaM2aMRo8ebdfde++9io6O1tSpUzVkyBAdOHBAYWFhatWqle6//35J0ksvvaRFixZpwYIFat68uaTfnn96/fXXdf3112vAgAFaunTpWXsaMGCABg8eLH9/f911110e+5o0aaL69etr6tSp2r17t3x9fdW0aVOlp6erf//+FfeHOcl1112n9evX68knn9S4ceN04MABRURE6LbbbrNXB5s2bars7Gw9/vjjevjhh3XkyBHFxMRowYIF9t+oosydO1dt27bV3LlzNWfOHB0/flyRkZFq3769brrppgq9FgDURg7LsizTTQAAAABAbcQzZAAAAABgCIEMAAAAAAwhkAEAAACAIQQyAAAAADCEQAYAAAAAhhDIAAAAAMAQfoesAh0/fly7d+9WvXr15HA4TLcDAAAAwBDLsnTg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2rWTr6+vunbtqh9++EHTpk3TgAEDTLcGAKhC33zzjdLS0pSXl2e6FeCSww9DA7gglmXpiSee0LZt2/Tvf/9bknT48GEtWLBAc+fOVUxMjF5//XXDXQIAKtt3332nNm3aqLCwUGPHjtWIESMUGhpqui3gksEKGYAL4nA49PPPPys/P98eCwgI0AMPPKBHHnlE3377rcaNG2ewQwBAZTt06JBSU1PVq1cvzZo1S3//+981depU/fLLL6ZbAy4Z3qYbAHDpsSxLDodD119/vbZt26avv/5a1157rSTJ399fd955p7755ht99NFHKigoUFhYmOGOAQCVoU6dOoqLi1NISIjuuusuXX755erXr58kafTo0ayUAeeAWxYBXLDvv/9ebdu2Vc+ePfXcc8+pXr169r68vDxdeeWVeuONN9S7d29zTQIAKtWhQ4cUGBhob7/++uu6++67NXLkSI0dO1YhISE6fvy4fvrpJzVu3Nhgp0D1xAoZgAt29dVX61//+pd69OihgIAATZw40f7/hvr6+qp169aqX7++2SYBAJXqRBgrKytTnTp1dNddd8myLCUlJcnhcCglJUXPPvusfvrpJy1cuFABAQGGOwaqFwIZgIvSqVMn/fvf/9add96p3bt3684771TLli21cOFC7dq1S1dffbXpFgEAVcDLy0uWZen48ePq16+fHA6HXC6Xli1bpu+//14bNmwgjAGnwC2LACrEpk2bNGLECG3fvl3e3t7y8fHRa6+9ptatW5tuDQBQhU78T0uHw6HOnTsrNzdXH3/8sWJjYw13BlRPBDIAFaaoqEj79u3TwYMHFRERwcPcAFBLlZWV6bHHHlNaWppyc3PVsmVL0y0B1Ra3LAKoMEFBQQoKCjLdBgCgGmjevLk2bdpEGAPOghUyAAAAVLgTP5EC4Mz4YWgAAABUOMIYcG4IZAAAAABgCIEMAAAAAAwhkAEAAACAIQQyAAAAADCEQAYAAAAAhhDIAAAAAMAQAhkAAAAAGEIgAwAAAABDCGQAAAAAYAiBDACAc/Sf//xHsbGx8vf3V0hIiLp06aJDhw5JkhYsWKCYmBjVrVtX1157rebMmWMf98ADD6hly5YqLi6WJJWWliouLk733HOPkXkAAKoPAhkAAOcgLy9Pd999tx544AFt3bpVH3/8sfr06SPLsjR//nyNHz9eTz/9tLZu3aopU6boiSee0CuvvCJJev7553Xo0CGNHTtWkvTEE0/ol19+8QhtAIDayWFZlmW6CQAAqrtNmzYpLi5OP/74oxo2bOixLzo6Ws8884zuvvtue+ypp57Su+++q+zsbEnSmjVr1KFDB40dO1apqan64IMPdOutt1bpHAAA1Q+BDACAc1BWVqZu3bpp/fr16tatmxISEnTHHXfo2LFjCgsLk7+/v+rU+f9vPDl27JicTqf27Nljjz3++ONKTU3VmDFj9Pe//93ENAAA1Yy36QYAALgUeHl5KSsrS9nZ2VqxYoVmzZql8ePH6+2335YkzZ8/X23atCl3zAnHjx/Xf//7X3l5eenbb7+t0t4BANUXz5ABAHCOHA6H2rdvr0mTJmnz5s3y9fXVf//7X/3hD3/QDz/8oGuuucbj07hxY/vYadOmaevWrVq1apXef/99LViwwOBMAADVBStkAACcg3Xr1umDDz5QQkKCwsLCtG7dOu3du1cxMTGaOHGikpOTFRQUpB49eqi4uFgbN25UYWGhRowYodzcXE2YMEH/+c9/1L59ez333HN65JFH1KFDB1111VWmpwYAMIhnyAAAOAdbt27Vo48+qk2bNqmoqEgNGzbU8OHDNWzYMEnSkiVLNG3aNH311VcKDAxUbGysUlJS1KNHD8XFxenmm2/W3Llz7fP16dNHe/bs0SeffOJxayMAoHYhkAEAAACAITxDBgAAAACGEMgAAAAAwBACGQAAAAAYQiADAAAAAEMIZAAAAABgCIEMAAAAAAwhkAEAAACAIQQyAAAAADCEQAYAAAAAhhDIAAAAAMAQAhkAAAAAGPL/Ae54ZfPjGDtCAAAAAElFTkSuQmCC
<Figure size 1000x600 with 1 Axes>
output_type": "display_data
image/png": "iVBORw0KGgoAAAANSUhEUgAAA2YAAAIqCAYAAABPOFldAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABoVklEQVR4nO3deZyN9f//8ecxZgbDTNYZk7EUyU7IlqWFIZNQKctYQkolFQolpBTS8iHtliJ9Pp/SppRC5SPFZGhsbWPL2Jmxzvr6/eE75+eY7ZwxXBOP++3mdnOu6zqv87quc857zvNcy3GZmQkAAAAA4JgiTjcAAAAAAJc6ghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQBc4hYsWKCXXnop23kul0vjx4+/oP3803zxxRcX3TbieQeAC89lZuZ0EwAA50RFRSkuLk7btm3LMm/16tWqVKmSKlWqdOEb+4d44IEHNHPmTF1Mf0553gHgwivqdAMAgMKrefPmTrdwUTEznTp1SsWLF3e6lVzxvAPAhcehjADwDzJ+/Hi5XC5t3LhRPXv2VEhIiEJDQ3X33XcrMTHRY9mZM2eqTZs2qlChgoKCglSvXj1NmTJFqamp7mXatWunxYsXa/v27XK5XO5/mc48pG39+vVyuVx6++23s/T15ZdfyuVy6dNPP3VP+/3339WrVy9VqFBBgYGBqlWrlmbOnJmv9V6wYIFatGihkiVLqmTJkmrYsGGWPt555x01aNBAxYoVU5kyZdStWzdt3rzZY5l27dqpXbt2Wer3799fVatWdd/etm2bXC6Xpk2bpunTp6tatWoqWbKkWrRoodWrV3vcL3Odztx+mXsfXS6XHnjgAb322muqVauWAgMDNWfOHNWoUUORkZFZ+jh27JhCQkJ0//3357gtGjVqpNatW2eZnp6erssvv1zdu3d3T5s1a5YaNGigkiVLqlSpUrr66qs1ZsyYHGtnOvtQxjlz5sjlcmn58uW67777VK5cOZUtW1bdu3fX7t27s9y/oJ6v/v37q2TJktqyZYsiIyMVFBSkihUr6rnnnpN0es/eddddp6CgIF111VWaO3dull727NmjIUOGqFKlSgoICFC1atU0YcIEpaWl5bkdAOBCIpgBwD/QbbfdpquuukoffvihHn/8cS1YsEAPP/ywxzJ//vmnevXqpXfffVeff/65Bg4cqKlTp2rIkCHuZV599VW1atVKYWFh+vHHH93/stOgQQM1atRIs2fPzjJvzpw5qlChgm6++WZJ0qZNm9S0aVPFxcXphRde0Oeff67OnTtr2LBhmjBhgk/rOm7cOPXu3Vvh4eGaM2eOFi1apH79+mn79u3uZSZPnqyBAweqTp06+uijj/Tyyy9rw4YNatGihX7//XefHu9MM2fO1NKlS/XSSy9p/vz5On78uG6++WZ3CH7yySd1++23S5LH9qtYsaK7xscff6xZs2Zp3Lhx+uqrr9SmTRs9+OCDWrp0aZbe5s2bp6SkpFyD2YABA7Ry5cos9/3666+1e/duDRgwQJK0cOFCDR06VG3bttWiRYv08ccf6+GHH9bx48fzvT0GDRokf39/LViwQFOmTNGKFSvUp08fj2UK+vlKTU1V9+7d1blzZ33yySfq1KmTRo8erTFjxqhfv366++67tWjRItWsWVP9+/dXTEyM+7579uzRtddeq6+++krjxo3Tl19+qYEDB2ry5MkaPHhwvrcDAJwXBgD4x3jqqadMkk2ZMsVj+tChQ61YsWKWkZGR7f3S09MtNTXV5s2bZ35+fnbo0CH3vM6dO1uVKlWyvZ8ke+qpp9y3X3nlFZNkW7dudU87dOiQBQYG2qOPPuqeFhkZaZUqVbLExESPeg888IAVK1bM4/Fz89dff5mfn5/17t07x2UOHz5sxYsXt5tvvtlj+o4dOywwMNB69erlnta2bVtr27Ztlhr9+vXz2Abx8fEmyerVq2dpaWnu6T///LNJsvfff9897f7777ec/pxKspCQkCzrm5SUZKVKlbKHHnrIY3rt2rXt+uuvz3FdzcwOHDhgAQEBNmbMGI/pPXr0sNDQUEtNTTWz09v6sssuy7VWTs5+3mfPnm2SbOjQoR7LTZkyxSRZQkKCmRX889WvXz+TZB9++KF7WmpqqpUvX94k2S+//OKefvDgQfPz87NHHnnEPW3IkCFWsmRJ2759u8djTZs2zSTZxo0bvdgaAHBhsMcMAP6BunTp4nG7fv36OnXqlPbt2+eetm7dOnXp0kVly5aVn5+f/P391bdvX6Wnp+u3337L1+P27t3bfThepvfff1/JycnuPTWnTp3St99+q27duqlEiRJKS0tz/7v55pt16tQpj8MBc7N06VKlp6fnugfpxx9/1MmTJ9W/f3+P6REREbrhhhv07bff+ryemTp37iw/Pz/37fr160uSx96fvNxwww0qXbq0x7RSpUppwIABmjNnjnsP1rJly7Rp0yY98MADudYrW7asbrnlFs2dO1cZGRmSpMOHD+uTTz5R3759VbTo6dPHr732Wh05ckQ9e/bUJ598ogMHDnjdc06ye91J/397nI/ny+VyuffESlLRokVVvXp1VaxYUY0aNXJPL1OmjCpUqODx3Hz++ee6/vrrFR4e7vE67NSpkyTpu+++82HtAeD8IpgBwD9Q2bJlPW4HBgZKkk6ePClJ2rFjh1q3bq2///5bL7/8sn744QetWbPGfT5U5nK+KlOmjLp06aJ58+YpPT1d0unDGK+99lrVqVNHknTw4EGlpaXpX//6l/z9/T3+ZX7A9jYk7N+/X5JyvTrgwYMHJcnj8MFM4eHh7vn5kdd29kZ2fUnSgw8+qKNHj2r+/PmSpBkzZqhSpUq69dZb86x599136++//9bSpUsl/f9wfGbYiY6O1jvvvKPt27frtttuU4UKFdSsWTP3ffIjr+1xPp6vEiVKqFixYh7TAgICVKZMmSz3DwgI0KlTp9y39+7dq88++yzL6zDztVoQYRUACgpXZQSAi9DHH3+s48eP66OPPlKVKlXc02NjY8+59oABA/Sf//xHS5cuVeXKlbVmzRrNmjXLPb906dLy8/NTdHR0jntOqlWr5tVjlS9fXpK0a9cuRUREZLtMZlhISEjIMm/37t0qV66c+3axYsWyXCRFOr8f0M+8mMqZqlevrk6dOmnmzJnq1KmTPv30U02YMMFjD11OIiMjFR4ertmzZysyMlKzZ89Ws2bNVLt2bY/lBgwYoAEDBuj48eP6/vvv9dRTTykqKkq//fabx+uioBT083WuypUrp/r16+uZZ57Jdn54eHiBPRYAnCuCGQBchDLDQOYeDen0pdrffPPNLMsGBgb6tAeoQ4cOuvzyyzV79mxVrlxZxYoVU8+ePd3zS5Qooeuvv17r1q1T/fr1FRAQkO/16NChg/z8/DRr1iy1aNEi22VatGih4sWL67333tMdd9zhnr5r1y4tW7bMfXEOSapatar+85//KDk52b1tDh48qFWrVik4ODhfPZ6518jXy+A/9NBD6tChg/r16yc/Pz+vL0iRGXxfeukl/fDDD1q7dq1ef/31HJcPCgpSp06dlJKSoq5du2rjxo3nJZgV9PN1rqKiovTFF1/oyiuvzHI4KQAUNgQzALgItW/fXgEBAerZs6dGjRqlU6dOadasWTp8+HCWZevVq6ePPvpIs2bNUuPGjVWkSBE1adIkx9p+fn7q27evpk+fruDgYHXv3l0hISEey7z88su67rrr1Lp1a913332qWrWqjh49qj/++EOfffaZli1b5tV6VK1aVWPGjNHTTz+tkydPun8iYNOmTTpw4IAmTJigyy67TE8++aTGjBmjvn37qmfPnjp48KAmTJigYsWK6amnnnLXi46O1uuvv64+ffpo8ODBOnjwoKZMmZLvUCad3n6S9Pzzz6tTp07y8/PzOpC2b99etWvX1vLly9WnTx9VqFDB68e9++679fzzz6tXr14qXry47rzzTo/5gwcPVvHixdWqVStVrFhRe/bs0eTJkxUSEqKmTZv6tpJeKujn61xNnDhRS5cuVcuWLTVs2DDVrFlTp06d0rZt2/TFF1/otdde40e0ARQeTl99BADgvcyrMu7fv99jeuZV8+Lj493TPvvsM2vQoIEVK1bMLr/8chs5cqR9+eWXJsmWL1/uXu7QoUN2++2322WXXWYul8vjCoM66+p8mX777TeTZJJs6dKl2fYaHx9vd999t11++eXm7+9v5cuXt5YtW9qkSZN8Xu958+ZZ06ZNrVixYlayZElr1KiRzZ4922OZt956y+rXr28BAQEWEhJit956a7ZX3Zs7d67VqlXLihUrZrVr17YPPvggx6syTp06Ncv9z94mycnJNmjQICtfvrx7+2U+D5Ls/vvvz3Xdxo8fb5Js9erVXm+PTC1btjRJ2V4Fce7cuXb99ddbaGioBQQEWHh4uPXo0cM2bNiQZ92z1zHz9bVmzRqP5ZYvX57l9WRWcM9Xv379LCgoKEt/bdu2tTp16mSZXqVKFevcubPHtP3799uwYcOsWrVq5u/vb2XKlLHGjRvb2LFj7dixY3luCwC4UFxmZhc+DgIAAElq0qSJXC6X1qxZ43QrAAAHcSgjAAAXWFJSkuLi4vT5558rJiZGixYtcrolAIDDCGYAAEekp6crt4M2XC6XV1co/Cf65ZdfdP3116ts2bJ66qmn1LVrV6dbAgA4jEMZAQCOqFq1aq4/1Ny2bVutWLHiwjUEAICD2GMGAHDEZ599puTk5BznlypV6gJ2AwCAs9hjBgAAAAAOK+J0AwAAAABwqeNQxgKUkZGh3bt3q1SpUnK5XE63AwAAAMAhZqajR48qPDxcRYrkvT+MYFaAdu/erYiICKfbAAAAAFBI7Ny5U5UqVcpzOYJZAco8UX3nzp0KDg52uBsAAAAATklKSlJERITXF7MimBWgzMMXg4ODCWYAAAAAvD7FiYt/AAAAAIDDCGYAAAAA4DCCGQAAAAA4jHPMLrCMjAylpKQ43cYFERAQ4NWlQQEAAIBLHcHsAkpJSVF8fLwyMjKcbuWCKFKkiKpVq6aAgACnWwEAAAAKNYLZBWJmSkhIkJ+fnyIiIi76PUmZP7adkJCgypUr84PbAAAAQC4IZhdIWlqaTpw4ofDwcJUoUcLpdi6I8uXLa/fu3UpLS5O/v7/T7QAAAACF1sW926YQSU9Pl6RL6rC+zHXNXHcAAAAA2SOYXWCX0iF9l9K6AgAAAOeCYAYAAAAADiOYFQLt2rXT8OHDnW4DAAAAgEO4+Ech8NFHH3FxDAAAAOASRjArBMqUKeN0CwAAAAAcxKGMhcCZhzJWrVpVzz77rO6++26VKlVKlStX1htvvOGx/K5du3TXXXepTJkyCgoKUpMmTfTTTz+558+aNUtXXnmlAgICVLNmTb377rse93e5XHr99dcVFRWlEiVKqFatWvrxxx/1xx9/qF27dgoKClKLFi30559/etzvs88+U+PGjVWsWDFdccUVmjBhgtLS0s7PRgEAAAAuIQSzQuiFF15QkyZNtG7dOg0dOlT33XeftmzZIkk6duyY2rZtq927d+vTTz/V+vXrNWrUKGVkZEiSFi1apIceekiPPvqo4uLiNGTIEA0YMEDLly/3eIynn35affv2VWxsrK6++mr16tVLQ4YM0ejRo7V27VpJ0gMPPOBe/quvvlKfPn00bNgwbdq0Sa+//rrmzJmjZ5555gJtFQAAAODi5TIzc7qJi0VSUpJCQkKUmJio4OBgj3mnTp1SfHy8qlWrpmLFinnMa9eunRo2bKiXXnpJVatWVevWrd17ucxMYWFhmjBhgu6991698cYbGjFihLZt25btIZCtWrVSnTp1PPay9ejRQ8ePH9fixYslnd5j9sQTT+jpp5+WJK1evVotWrTQ22+/rbvvvluStHDhQg0YMEAnT56UJLVp00adOnXS6NGj3XXfe+89jRo1Srt37852e+S2zgAAAMDFLLdskB32mBVC9evXd//f5XIpLCxM+/btkyTFxsaqUaNGOZ6XtnnzZrVq1cpjWqtWrbR58+YcHyM0NFSSVK9ePY9pp06dUlJSkiQpJiZGEydOVMmSJd3/Bg8erISEBJ04ceIc1hYAAAAAF/8ohM6+QqPL5XIfqli8ePE873/2DzubWZZpZz5G5rzspmU+bkZGhiZMmKDu3btneTz2hgEAAADnhj1m/zD169dXbGysDh06lO38WrVqaeXKlR7TVq1apVq1ap3T415zzTXaunWrqlevnuVfkSK8jAAAAIBzwR6zf5iePXvq2WefVdeuXTV58mRVrFhR69atU3h4uFq0aKGRI0eqR48euuaaa3TjjTfqs88+00cffaRvvvnmnB533LhxioqKUkREhO644w4VKVJEGzZs0K+//qpJkyYV0NoBAADgUtN45Lwc58VM7XsBO3EWuzr+YQICAvT111+rQoUKuvnmm1WvXj0999xz8vPzkyR17dpVL7/8sqZOnao6dero9ddf1+zZs9WuXbtzetzIyEh9/vnnWrp0qZo2barmzZtr+vTpqlKlSgGsFQAAAHBp46qMBSi/V2W8WF2K6wwAAADfXKx7zLgqIwAAAAD8wxDMAAAAAMBhBDMAAAAAcBjBDAAAAAAcRjADAAAAAIcRzAAAAADAYQQzAAAAAHAYwQwAAAAAHEYwAwAAAACHEcwAAAAAwGFFnW4ApzUeOe+CPVbM1L75ut+rr76qqVOnKiEhQXXq1NFLL72k1q1bF3B3AAAAwKWHPWbwygcffKDhw4dr7NixWrdunVq3bq1OnTppx44dTrcGAAAA/OMRzOCV6dOna+DAgRo0aJBq1aqll156SREREZo1a5bTrQEAAAD/eAQz5CklJUUxMTHq0KGDx/QOHTpo1apVDnUFAAAAXDwIZsjTgQMHlJ6ertDQUI/poaGh2rNnj0NdAQAAABcPghm85nK5PG6bWZZpAAAAAHxHMEOeypUrJz8/vyx7x/bt25dlLxoAAAAA3xHMkKeAgAA1btxYS5cu9Zi+dOlStWzZ0qGuAAAAgIsHv2MGrzzyyCOKjo5WkyZN1KJFC73xxhvasWOH7r33XqdbAwAAAP7xCGbwyp133qmDBw9q4sSJSkhIUN26dfXFF1+oSpUqTrcGAAAA/OMRzAqJmKl9nW4hT0OHDtXQoUOdbgMAAAC46HCOGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADivqdAM4bcfEehfssSqP+9Xn+3z//feaOnWqYmJilJCQoEWLFqlr164F3xwAAABwCWKPGbxy/PhxNWjQQDNmzHC6FQAAAOCiwx4zeKVTp07q1KmT020AAAAAFyX2mAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMO4KiO8cuzYMf3xxx/u2/Hx8YqNjVWZMmVUuXJlBzsDAAAA/vkc3WM2efJkNW3aVKVKlVKFChXUtWtXbd261WMZM9P48eMVHh6u4sWLq127dtq4caPHMsnJyXrwwQdVrlw5BQUFqUuXLtq1a5fHMocPH1Z0dLRCQkIUEhKi6OhoHTlyxGOZHTt26JZbblFQUJDKlSunYcOGKSUl5bys+z/N2rVr1ahRIzVq1EiS9Mgjj6hRo0YaN26cw50BAAAA/3yO7jH77rvvdP/996tp06ZKS0vT2LFj1aFDB23atElBQUGSpClTpmj69OmaM2eOrrrqKk2aNEnt27fX1q1bVapUKUnS8OHD9dlnn2nhwoUqW7asHn30UUVFRSkmJkZ+fn6SpF69emnXrl1asmSJJOmee+5RdHS0PvvsM0lSenq6OnfurPLly2vlypU6ePCg+vXrJzPTv/71r/O+LSqP+/W8P8a5aNeunczM6TYAAACAi5LLCtGn7f3796tChQr67rvv1KZNG5mZwsPDNXz4cD322GOSTu8dCw0N1fPPP68hQ4YoMTFR5cuX17vvvqs777xTkrR7925FREToiy++UGRkpDZv3qzatWtr9erVatasmSRp9erVatGihbZs2aKaNWvqyy+/VFRUlHbu3Knw8HBJ0sKFC9W/f3/t27dPwcHBefaflJSkkJAQJSYmZln+1KlTio+PV7Vq1VSsWLGC3GyF1qW4zgAAAPBN45HzcpwXM7XvBeykYOWWDbJTqC7+kZiYKEkqU6aMpNPnMe3Zs0cdOnRwLxMYGKi2bdtq1apVkqSYmBilpqZ6LBMeHq66deu6l/nxxx8VEhLiDmWS1Lx5c4WEhHgsU7duXXcok6TIyEglJycrJibmPK0xAAAAABSii3+YmR555BFdd911qlu3riRpz549kqTQ0FCPZUNDQ7V9+3b3MgEBASpdunSWZTLvv2fPHlWoUCHLY1aoUMFjmbMfp3Tp0goICHAvc7bk5GQlJye7byclJXm9vgAAAACQqdDsMXvggQe0YcMGvf/++1nmuVwuj9tmlmXa2c5eJrvl87PMmSZPnuy+mEhISIgiIiJy7QkAAAAAslMogtmDDz6oTz/9VMuXL1elSpXc08PCwiQpyx6rffv2ufduhYWFKSUlRYcPH851mb1792Z53P3793ssc/bjHD58WKmpqVn2pGUaPXq0EhMT3f927tyZ57oWolP6zrtLaV0BAACAc+FoMDMzPfDAA/roo4+0bNkyVatWzWN+tWrVFBYWpqVLl7qnpaSk6LvvvlPLli0lSY0bN5a/v7/HMgkJCYqLi3Mv06JFCyUmJurnn392L/PTTz8pMTHRY5m4uDglJCS4l/n6668VGBioxo0bZ9t/YGCggoODPf7lJPPqkJfS5fcz1zVz3QEAAABkz9FzzO6//34tWLBAn3zyiUqVKuXeYxUSEqLixYvL5XJp+PDhevbZZ1WjRg3VqFFDzz77rEqUKKFevXq5lx04cKAeffRRlS1bVmXKlNGIESNUr1493XTTTZKkWrVqqWPHjho8eLBef/11Sacvlx8VFaWaNWtKkjp06KDatWsrOjpaU6dO1aFDhzRixAgNHjzYq6uo5KVo0aIqUaKE9u/fL39/fxUpUih2Vp43GRkZ2r9/v0qUKKGiRQvNqYwAAABAoeToJ+ZZs2ZJOv0bWWeaPXu2+vfvL0kaNWqUTp48qaFDh+rw4cNq1qyZvv76a/dvmEnSiy++qKJFi6pHjx46efKkbrzxRs2ZM8djT838+fM1bNgw99Ubu3TpohkzZrjn+/n5afHixRo6dKhatWql4sWLq1evXpo2bVqBrKvL5VLFihUVHx/vvnDJxa5IkSKqXLlynucDAgAAAJe6QvU7Zv903vxWQUZGxiVzOGNAQMBFv2cQAAAA54bfMTuNY8wusCJFivBjywAAAAA8sDsDAAAAABxGMAMAAAAAhxHMAAAAAMBhBDMAAAAAcBjBDAAAAAAcRjADAAAAAIcRzAAAAADAYQQzAAAAAHAYwQwAAAAAHEYwAwAAAACHEcwAAAAAwGEEMwAAAABwGMEMAAAAABxGMAMAAAAAhxHMAAAAAMBhBDMAAAAAcBjBDAAAAAAcRjADAAAAAIcRzAAAAADAYQQzAAAAAHAYwQwAAAAAHEYwAwAAAACHEcwAAAAAwGEEMwAAAABwGMEMAAAAABxGMAMAAAAAhxHMAAAAAMBhBDMAAAAAcBjBDAAAAAAcRjADAAAAAIcRzAAAAADAYQQzAAAAAHAYwQwAAAAAHEYwAwAAAACHEcwAAAAAwGEEMwAAAABwGMEMAAAAABxGMAMAAAAAhxHMAAAAAMBhBDMAAAAAcBjBDAAAAAAcRjADAAAAAIcRzAAAAADAYQQzAAAAAHAYwQwAAAAAHEYwAwAAAACHEcwAAAAAwGEEMwAAAABwGMEMAAAAABxGMAMAAAAAhxHMAAAAAMBhBDMAAAAAcBjBDAAAAAAcRjADAAAAAIcRzAAAAADAYQQzAAAAAHAYwQwAAAAAHEYwAwAAAACHEcwAAAAAwGEEMwAAAABwGMEMAAAAABxGMAMAAAAAhxHMAAAAAMBhBDMAAAAAcBjBDAAAAAAcRjADAAAAAIcRzAAAAADAYQQzAAAAAHAYwQwAAAAAHEYwAwAAAACHEcwAAAAAwGEEMwAAAABwGMEMAAAAABxGMAMAAAAAhxHMAAAAAMBhBDMAAAAAcBjBDAAAAAAcRjADAAAAAIcRzAAAAADAYQQzAAAAAHAYwQwAAAAAHEYwAwAAAACHEcwAAAAAwGEEMwAAAABwGMEMAAAAABxGMAMAAAAAhzkazL7//nvdcsstCg8Pl8vl0scff+wxv3///nK5XB7/mjdv7rFMcnKyHnzwQZUrV05BQUHq0qWLdu3a5bHM4cOHFR0drZCQEIWEhCg6OlpHjhzxWGbHjh265ZZbFBQUpHLlymnYsGFKSUk5H6sNAAAAAB4cDWbHjx9XgwYNNGPGjByX6dixoxISEtz/vvjiC4/5w4cP16JFi7Rw4UKtXLlSx44dU1RUlNLT093L9OrVS7GxsVqyZImWLFmi2NhYRUdHu+enp6erc+fOOn78uFauXKmFCxfqww8/1KOPPlrwKw0AAAAAZynq5IN36tRJnTp1ynWZwMBAhYWFZTsvMTFRb7/9tt59913ddNNNkqT33ntPERER+uabbxQZGanNmzdryZIlWr16tZo1ayZJevPNN9WiRQtt3bpVNWvW1Ndff61NmzZp586dCg8PlyS98MIL6t+/v5555hkFBwcX4FoDAAAAgKdCf47ZihUrVKFCBV111VUaPHiw9u3b554XExOj1NRUdejQwT0tPDxcdevW1apVqyRJP/74o0JCQtyhTJKaN2+ukJAQj2Xq1q3rDmWSFBkZqeTkZMXExJzvVQQAAABwiXN0j1leOnXqpDvuuENVqlRRfHy8nnzySd1www2KiYlRYGCg9uzZo4CAAJUuXdrjfqGhodqzZ48kac+ePapQoUKW2hUqVPBYJjQ01GN+6dKlFRAQ4F4mO8nJyUpOTnbfTkpKyve6AgAAALh0Fepgduedd7r/X7duXTVp0kRVqlTR4sWL1b179xzvZ2ZyuVzu22f+/1yWOdvkyZM1YcKEPNcDAAAAAHJT6A9lPFPFihVVpUoV/f7775KksLAwpaSk6PDhwx7L7du3z70HLCwsTHv37s1Sa//+/R7LnL1n7PDhw0pNTc2yJ+1Mo0ePVmJiovvfzp07z2n9AAAAAFya/lHB7ODBg9q5c6cqVqwoSWrcuLH8/f21dOlS9zIJCQmKi4tTy5YtJUktWrRQYmKifv75Z/cyP/30kxITEz2WiYuLU0JCgnuZr7/+WoGBgWrcuHGO/QQGBio4ONjjHwAAAAD4ytFDGY8dO6Y//vjDfTs+Pl6xsbEqU6aMypQpo/Hjx+u2225TxYoVtW3bNo0ZM0blypVTt27dJEkhISEaOHCgHn30UZUtW1ZlypTRiBEjVK9ePfdVGmvVqqWOHTtq8ODBev311yVJ99xzj6KiolSzZk1JUocOHVS7dm1FR0dr6tSpOnTokEaMGKHBgwcTtgAAAACcd44Gs7Vr1+r66693337kkUckSf369dOsWbP066+/at68eTpy5IgqVqyo66+/Xh988IFKlSrlvs+LL76ookWLqkePHjp58qRuvPFGzZkzR35+fu5l5s+fr2HDhrmv3tilSxeP307z8/PT4sWLNXToULVq1UrFixdXr169NG3atPO9CQAAAABALjMzp5u4WCQlJSkkJESJiYnsaQMAAAC80HjkvBznxUztewE7KVi+ZoN/1DlmAAAAAHAxIpgBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgsHwFsyuuuEIHDx7MMv3IkSO64oorzrkpAAAAALiU5CuYbdu2Tenp6VmmJycn6++//z7npgAAAADgUlLUl4U//fRT9/+/+uorhYSEuG+np6fr22+/VdWqVQusOQAAAAC4FPgUzLp27SpJcrlc6tevn8c8f39/Va1aVS+88EKBNQcAAAAAlwKfgllGRoYkqVq1alqzZo3KlSt3XpoCAAAAgEuJT8EsU3x8fEH3AQAAAACXrHwFM0n69ttv9e2332rfvn3uPWmZ3nnnnXNuDAAAAAAuFfkKZhMmTNDEiRPVpEkTVaxYUS6Xq6D7AgAAAIBLRr6C2WuvvaY5c+YoOjq6oPsBAAAAgEtOvn7HLCUlRS1btizoXgAAAADgkpSvYDZo0CAtWLCgoHsBAAAAgEtSvg5lPHXqlN544w198803ql+/vvz9/T3mT58+vUCaAwAAAIBLQb6C2YYNG9SwYUNJUlxcnMc8LgQCAAAAAL7JVzBbvnx5QfcBAAAAAJesfJ1jBgAAAAAoOPnaY3b99dfnesjismXL8t0QAAAAAFxq8hXMMs8vy5SamqrY2FjFxcWpX79+BdEXAAAAAFwy8hXMXnzxxWynjx8/XseOHTunhgAAAADgUlOg55j16dNH77zzTkGWBAAAAICLXoEGsx9//FHFihUryJIAAAAAcNHL16GM3bt397htZkpISNDatWv15JNPFkhjAAAAAHCpyFcwCwkJ8bhdpEgR1axZUxMnTlSHDh0KpDEAAAAAuFTkK5jNnj27oPsAAAAAgEtWvoJZppiYGG3evFkul0u1a9dWo0aNCqovAAAAALhk5CuY7du3T3fddZdWrFihyy67TGamxMREXX/99Vq4cKHKly9f0H0CAAAAwEUrX1dlfPDBB5WUlKSNGzfq0KFDOnz4sOLi4pSUlKRhw4YVdI8AAAAAcFHL1x6zJUuW6JtvvlGtWrXc02rXrq2ZM2dy8Q8AAAAA8FG+9phlZGTI398/y3R/f39lZGScc1MAAAAAcCnJVzC74YYb9NBDD2n37t3uaX///bcefvhh3XjjjQXWHAAAAABcCvIVzGbMmKGjR4+qatWquvLKK1W9enVVq1ZNR48e1b/+9a+C7hEAAAAALmr5OscsIiJCv/zyi5YuXaotW7bIzFS7dm3ddNNNBd0fAAAAAFz0fNpjtmzZMtWuXVtJSUmSpPbt2+vBBx/UsGHD1LRpU9WpU0c//PDDeWkUAAAAAC5WPgWzl156SYMHD1ZwcHCWeSEhIRoyZIimT59eYM0BAAAAwKXAp2C2fv16dezYMcf5HTp0UExMzDk3BQAAAACXEp+C2d69e7O9TH6mokWLav/+/efcFAAAAABcSnwKZpdffrl+/fXXHOdv2LBBFStWPOemAAAAAOBS4lMwu/nmmzVu3DidOnUqy7yTJ0/qqaeeUlRUVIE1BwAAAACXAp8ul//EE0/oo48+0lVXXaUHHnhANWvWlMvl0ubNmzVz5kylp6dr7Nix56tXAAAAALgo+RTMQkNDtWrVKt13330aPXq0zEyS5HK5FBkZqVdffVWhoaHnpVEAAAAAuFj5/APTVapU0RdffKHDhw/rjz/+kJmpRo0aKl269PnoDwAAAAAuej4Hs0ylS5dW06ZNC7IXAAAAALgk+XTxDwAAAABAwSOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOMzRYPb999/rlltuUXh4uFwulz7++GOP+Wam8ePHKzw8XMWLF1e7du20ceNGj2WSk5P14IMPqly5cgoKClKXLl20a9cuj2UOHz6s6OhohYSEKCQkRNHR0Tpy5IjHMjt27NAtt9yioKAglStXTsOGDVNKSsr5WG0AAAAA8OBoMDt+/LgaNGigGTNmZDt/ypQpmj59umbMmKE1a9YoLCxM7du319GjR93LDB8+XIsWLdLChQu1cuVKHTt2TFFRUUpPT3cv06tXL8XGxmrJkiVasmSJYmNjFR0d7Z6fnp6uzp076/jx41q5cqUWLlyoDz/8UI8++uj5W3kAAAAA+D8uMzOnm5Akl8ulRYsWqWvXrpJO7y0LDw/X8OHD9dhjj0k6vXcsNDRUzz//vIYMGaLExESVL19e7777ru68805J0u7duxUREaEvvvhCkZGR2rx5s2rXrq3Vq1erWbNmkqTVq1erRYsW2rJli2rWrKkvv/xSUVFR2rlzp8LDwyVJCxcuVP/+/bVv3z4FBwd7tQ5JSUkKCQlRYmKi1/cBAAAALmWNR87LcV7M1L4XsJOC5Ws2KLTnmMXHx2vPnj3q0KGDe1pgYKDatm2rVatWSZJiYmKUmprqsUx4eLjq1q3rXubHH39USEiIO5RJUvPmzRUSEuKxTN26dd2hTJIiIyOVnJysmJiYHHtMTk5WUlKSxz8AAAAA8FWhDWZ79uyRJIWGhnpMDw0Ndc/bs2ePAgICVLp06VyXqVChQpb6FSpU8Fjm7McpXbq0AgIC3MtkZ/Lkye7z1kJCQhQREeHjWgIAAABAIQ5mmVwul8dtM8sy7WxnL5Pd8vlZ5myjR49WYmKi+9/OnTtz7QsAAAAAslNog1lYWJgkZdljtW/fPvferbCwMKWkpOjw4cO5LrN3794s9ffv3++xzNmPc/jwYaWmpmbZk3amwMBABQcHe/wDAAAAAF8V2mBWrVo1hYWFaenSpe5pKSkp+u6779SyZUtJUuPGjeXv7++xTEJCguLi4tzLtGjRQomJifr555/dy/z0009KTEz0WCYuLk4JCQnuZb7++msFBgaqcePG53U9AQAAAKCokw9+7Ngx/fHHH+7b8fHxio2NVZkyZVS5cmUNHz5czz77rGrUqKEaNWro2WefVYkSJdSrVy9JUkhIiAYOHKhHH31UZcuWVZkyZTRixAjVq1dPN910kySpVq1a6tixowYPHqzXX39dknTPPfcoKipKNWvWlCR16NBBtWvXVnR0tKZOnapDhw5pxIgRGjx4MHvBAAAAAJx3jgaztWvX6vrrr3fffuSRRyRJ/fr105w5czRq1CidPHlSQ4cO1eHDh9WsWTN9/fXXKlWqlPs+L774oooWLaoePXro5MmTuvHGGzVnzhz5+fm5l5k/f76GDRvmvnpjly5dPH47zc/PT4sXL9bQoUPVqlUrFS9eXL169dK0adPO9yYAAAAAgMLzO2YXA37HDAAAAPANv2N2WqE9xwwAAAAALhUEMwAAAABwGMEMAAAAABxGMAMAAAAAhxHMAAAAAMBhBDMAAAAAcBjBDAAAAAAcRjADAAAAAIcRzAAAAADAYQQzAAAAAHAYwQwAAAAAHEYwAwAAAACHEcwAAAAAwGEEMwAAAABwGMEMAAAAABxGMAMAAAAAhxHMAAAAAMBhBDMAAAAAcBjBDAAAAAAcRjADAAAAAIcRzAAAAADAYQQzAAAAAHAYwQwAAAAAHEYwAwAAAACHEcwAAAAAwGEEMwAAAABwGMEMAAAAABxGMAMAAAAAhxHMAAAAAMBhBDMAAAAAcBjBDAAAAAAcRjADAAAAAIcRzAAAAADAYQQzAAAAAHAYwQwAAAAAHEYwAwAAAACHEcwAAAAAwGEEMwAAAABwGMEMAAAAABxGMAMAAAAAhxHMAAAAAMBhBDMAAAAAcBjBDAAAAAAcRjADAAAAAIcRzAAAAADAYQQzAAAAAHBYUacbAAAAAIDs7JhYL9f5lcf9eoE6Of/YYwYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4rFAHs/Hjx8vlcnn8CwsLc883M40fP17h4eEqXry42rVrp40bN3rUSE5O1oMPPqhy5copKChIXbp00a5duzyWOXz4sKKjoxUSEqKQkBBFR0fryJEjF2IVAQAAAKBwBzNJqlOnjhISEtz/fv31V/e8KVOmaPr06ZoxY4bWrFmjsLAwtW/fXkePHnUvM3z4cC1atEgLFy7UypUrdezYMUVFRSk9Pd29TK9evRQbG6slS5ZoyZIlio2NVXR09AVdTwAAAACXrqJON5CXokWLeuwly2RmeumllzR27Fh1795dkjR37lyFhoZqwYIFGjJkiBITE/X222/r3Xff1U033SRJeu+99xQREaFvvvlGkZGR2rx5s5YsWaLVq1erWbNmkqQ333xTLVq00NatW1WzZs0Lt7IAAAAALkmFfo/Z77//rvDwcFWrVk133XWX/vrrL0lSfHy89uzZow4dOriXDQwMVNu2bbVq1SpJUkxMjFJTUz2WCQ8PV926dd3L/PjjjwoJCXGHMklq3ry5QkJC3MvkJDk5WUlJSR7/AAAAAMBXhTqYNWvWTPPmzdNXX32lN998U3v27FHLli118OBB7dmzR5IUGhrqcZ/Q0FD3vD179iggIEClS5fOdZkKFSpkeewKFSq4l8nJ5MmT3eelhYSEKCIiIt/rCgAAAODSVaiDWadOnXTbbbepXr16uummm7R48WJJpw9ZzORyuTzuY2ZZpp3t7GWyW96bOqNHj1ZiYqL7386dO/NcJwAAAAA4W6EOZmcLCgpSvXr19Pvvv7vPOzt7r9a+ffvce9HCwsKUkpKiw4cP57rM3r17szzW/v37s+yNO1tgYKCCg4M9/gEAAACAr/5RwSw5OVmbN29WxYoVVa1aNYWFhWnp0qXu+SkpKfruu+/UsmVLSVLjxo3l7+/vsUxCQoLi4uLcy7Ro0UKJiYn6+eef3cv89NNPSkxMdC8DAAAAAOdTob4q44gRI3TLLbeocuXK2rdvnyZNmqSkpCT169dPLpdLw4cP17PPPqsaNWqoRo0aevbZZ1WiRAn16tVLkhQSEqKBAwfq0UcfVdmyZVWmTBmNGDHCfWikJNWqVUsdO3bU4MGD9frrr0uS7rnnHkVFRXFFRgAAAAAXRKEOZrt27VLPnj114MABlS9fXs2bN9fq1atVpUoVSdKoUaN08uRJDR06VIcPH1azZs309ddfq1SpUu4aL774oooWLaoePXro5MmTuvHGGzVnzhz5+fm5l5k/f76GDRvmvnpjly5dNGPGjAu7sgAAAAAuWS4zM6ebuFgkJSUpJCREiYmJnG8GAAAAeKHxyHk5zltUamqu96087teCbqfA+JoN/lHnmAEAAADAxYhgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDDijrdAAAAgLcaj5yX6/yYqX0vUCcAULDYYwYAAAAADiOYAQAAAIDDCGYAAAAA4DCCGQAAAAA4jGAGAAAAAA4jmAEAAACAwwhmAAAAAOAwgtlZXn31VVWrVk3FihVT48aN9cMPPzjdEgAAAICLHD8wfYYPPvhAw4cP16uvvqpWrVrp9ddfV6dOnbRp0yZVrlzZ6fYAAAB8ktsPcvNj3EDhQjA7w/Tp0zVw4EANGjRIkvTSSy/pq6++0qxZszR58mSHu8M/2aXwh/FSWEcAQOGS298eib8/+GchmP2flJQUxcTE6PHHH/eY3qFDB61atSrb+yQnJys5Odl9OzExUZKUlJRUYH21eeL9XOfPL/lyrvMjHl/tda3vJ/UssL4uhloFKT35ZI7zfH295LaOTq2flPs6bhxTO9f7nvk6LUiXwmvrUpDbtvdlDCysCuvrtLC+5nMba6Tcx5t/wuuhoBXk2FyQf38KqlZer4eC/Ezmi8L6/imscnsej/qn53rfs5/jwvQ5KbM3M/NqeZd5u+RFbvfu3br88sv1v//9Ty1btnRPf/bZZzV37lxt3bo1y33Gjx+vCRMmXMg2AQAAAPyD7Ny5U5UqVcpzOfaYncXlcnncNrMs0zKNHj1ajzzyiPt2RkaGDh06pLJly+Z4n6SkJEVERGjnzp0KDg4+p16p9c+vVRh7oha1zmetwtgTtah1PmsVxp6oRa3zWasw9uRULTPT0aNHFR4e7lVdgtn/KVeunPz8/LRnzx6P6fv27VNoaGi29wkMDFRgYKDHtMsuu8yrxwsODj7nFwW1Lp5ahbEnalHrfNYqjD1Ri1rns1Zh7Ila1DqftQpjT07UCgkJ8boel8v/PwEBAWrcuLGWLl3qMX3p0qUehzYCAAAAQEFjj9kZHnnkEUVHR6tJkyZq0aKF3njjDe3YsUP33nuv060BAAAAuIgRzM5w55136uDBg5o4caISEhJUt25dffHFF6pSpUqBPUZgYKCeeuqpLIdAUuvSrFUYe6IWtc5nrcLYE7WodT5rFcaeqEWt81mrMPZUmGudiasyAgAAAIDDOMcMAAAAABxGMAMAAAAAhxHMAAAAAMBhBDMAAAAAcBjBDAAAOK4gr0VWGK9rlpGRUaD1CuM6AmfjdeobgtkFkpaWptTUVKfbyFFheuMkJCRo06ZNBVIrPT1dUsGs34kTJwrsOdy1a5fWrVtXILUKUkZGRoF/eACyc/z48QKvWZjGsX+CwrK9kpOTJUkul+uce9q7d6+71rnau3evDhw4cM51JCk+Pl5vvfWW0tPTz3kdM8foglhH4HzJHOML+nVaWMat84VgdgFs2rRJvXv31g033KABAwbo/fffz3etzKBREI4fP66jR48qKSnpnN84hw4d0pYtW/T7778rJSUl33X+/vtv1atXT0888YTWrl17Tj398ssvuv7663X8+PFzXr+4uDj17NlTq1evdn+IyK+NGzeqZcuWeu+99ySd27eou3bt0gcffKAPP/xQGzZsOKe+Nm3apP79+6t9+/a65557tHDhwnOql5uLbWA1swJ7bx46dEj79+8vkFpbt27V//73vwKp9ccff+jjjz8+p/d3pq1bt+ree+/Vrl27zqnOiRMndPjwYZ06dUpS4fkAsGPHDi1evFhvvfWWEhISzimEZq5bQdi7d6/WrFmjzz77TFL+g9COHTv03//+V9OnTz/n53Dr1q0aNGiQli9ffk49SdL69evVsGFDLVu27Jx6kqRff/1VrVq10vz583Xs2LFzqrVhwwbVrl1bEyZMkHRu6/j7779r3Lhx6tevn+bNm6eDBw/mu6+dO3fqm2++0TvvvKO9e/fqxIkT+a6VnYtpnGeM901BjfFS4RzncxrjC+Q1bzivtm7daiEhIdanTx+bMGGCtWnTxho1amT9+/fPV61p06bZ7t27z7mvjRs3WocOHaxRo0YWHh5u7733npmZZWRk+Fzr119/tUaNGlm9evUsMDDQnn76aUtLS8tXX8uWLbOiRYvaDTfcYH379rWYmBj3PF96i42NtaCgIHvkkUc8pudn/eLi4qx06dI2dOhQ27Vrl8/3P7uvEiVKWLVq1SwsLMz27t2b71obNmywKlWqWJMmTSw0NNS6dOlif/75Z75qbd682UqXLm0DBw60F154wW666Sa78sor7YEHHsh3f2ZmW7ZssYcfftjuvPNOmzx5cr6fTzOzvXv32uHDh8+pn0x//fWXTZ8+3R555BFbuHDhOdXaunWrPfTQQ9a5c2ebMGGCHThwIN+1/vzzT7viiivsySeftL///vuc+lq3bp2VKFHCXnnllXOqY3b6tRYaGmr33XffOY8/sbGxVrx4cXO5XDZ79ux814mLi7Obb77ZatWqZbfeeqstXrw437W2bNlijz/+uEVHR9vUqVNt3bp17nm+vk7Xr19voaGh1qhRI7vsssssIiLCRowYYX/99ZfPfcXFxVnz5s1txYoVPt/3bBs2bLD69etbnTp1rGTJktakSRM7ceKEmfm2jhs2bLCqVatay5YtrUyZMlatWjVLSEjIV0/JycnWvXt3c7lc1q9fP/vf//7nnufrdo+NjbVixYrZqFGjsszztdbWrVutbNmyNnLkSDt48KBP982urxIlSlifPn2sRo0a9swzz+S71oYNG6xs2bLWo0cPa968udWtW9c+//zzfNVav369VaxY0dq2bWvh4eFWtWpVe+yxx2znzp0+17rYx3nGeN8U1BhvVjjH+bzG+Px8zjwTwew8ysjIsLFjx9rtt9/unnb8+HGbMWOG1atXz3r06OF1rd9//93KlCljLpfLRo8ebfv37893Xxs3brSyZcvaww8/bAsWLLBHHnnE/P39PV6kvtYaMWKEbdy40aZNm2Yul8t27NiRr94OHjxoXbp0sddff92uueYa6927t8XFxZmZWXp6ulc11q9fb0FBQTZy5EiP6SdPnvS5n2PHjlmHDh3svvvuc0/bvHmzxcbG+ryOmYPVmDFjbP/+/VanTh2bNGmSZWRk+PxG3rZtm11++eX2+OOP27Fjx+yLL76wsLAw+/nnn32qY2Z26tQp6927tw0bNsw97eTJk9agQQNzuVzWq1cvn2uanX5thISEWFRUlPXp08fCwsKsdevW9sILL7iX8Xa9N23aZAEBAXb77bdbYmJivvrJtGHDBqtUqZLddNNN1rJlSytSpIhNmTIl37UqVKhgt99+uw0ZMsQCAgJs/Pjx+e5t1qxZ5nK5rFGjRvbMM894fOD15XWS+WHw0UcfzXcvmbZv326VK1fO9gPvmb1521fx4sVt1KhRNmLECGvdunW+PtRv3LjRSpcubffff7+99tpr1qpVK+vTp0++etq4caNddtlldscdd9i9995r4eHh1rBhQ5s1a5bPtQ4fPmyNGze2kSNH2qFDh8zMbMKECda6dWvr0qWL/f77716u4en3+NVXX20BAQF2+eWX2w8//OD1fc/222+/WWhoqI0ZM8Y2b95sW7Zssauvvtqio6N9qrNlyxarUKGCPfnkk3bo0CHLyMiw8PBwmzdvXr57e/rpp61z58529dVXW1RUlH3//fc+19i4caMVK1bM/d7LyMiw7du3288//2wpKSk+j68jRoywnj17mtnpvzsff/yxTZ061b799lufPrRmfkE4duxYMzO76667rH379paamupTP2Zm+/bts4YNG9oTTzzhntauXbt8Bb2EhASrW7eujR8/3pKSkszM7P777zeXy2XdunWzP/74w+taF/s4/08e472tVRjHeLPCOc57O8afSzgjmJ1n/fv3t+uuu85j2okTJ+ytt96yRo0a2eOPP55njWPHjtndd99t/fv3txkzZpjL5bKRI0fmK5wdPHjQOnTo4PEh3Mzs+uuvd0/z9gW1f/9+a9OmjT300EPuaRkZGdaxY0dbtWqVrVu3zqfwkpaWZvv27bOrrrrKdu3aZR999JE1bdrUBg8ebC1btrTbbrstzxoJCQkWFhZmkZGR7poPPvigRUZGWrVq1WzixIn2yy+/eN3TqVOn7LrrrrNffvnF0tLSLDIy0po2bWqlSpWy5s2b21tvveVVnfXr11tgYKCNGTPGzE7/sb/99tutadOm7mV8eSO/9tpr1q5dO4/73Hzzzfb666/b3LlzbdmyZV7XMjO78cYb3X9sMgPsqFGjrHv37nbNNdfY1KlTfaqXkpJiffv2tYEDB7qnbd++3e6991675pprbNKkSe7pea33nj17rFWrVnbjjTdauXLl7I477sj3H+1t27ZZ9erVbdSoUe69um+//baFhYX59KHZ7PS3sVWrVrXRo0e7p40fP96GDh1qKSkpHst6+9yuX7/e+vXrZ5MmTbLw8HB7+umnff72+LfffrPAwED3h8GUlBT76KOP7F//+pctXLjQ5720n332md18883uWmPHjrVu3brZoEGDbO7cue7l8lrHtWvXWnBwsPs98P7771tISIitXLnSzLz/4uXEiRPWtWtXj3Hnk08+se7du9vevXvt6NGjXvd09OhRi4yM9PhAsmPHDrvsssssLCzM5w+927dvtypVqthXX33lMX3u3LnWpk0b69Wrl1cf7FNSUuyFF16wrl272oYNG+z222+3cuXK5SucHT9+3Pr162dDhgzxOJLhhRdeyPK3KTdHjx616Ohoe/DBBy0tLc29bbt06WKTJ0+2xx57zL755huv/y5l3v/FF1+0iRMnWnx8vNWsWdO6detmmzZtsscee8y2bt2aZ50jR45Yy5YtLSIiwj2tR48eVrduXStWrJjVqFHD3nrrLTty5IjX69qxY0ebPn26mZm1atXKWrZsaZUqVbK6detahw4dbMuWLXnW+PPPP83lcrnfh2ZmP/zwg7lcLvvwww+97iXThg0brGbNmrZ27Vr3tAEDBlj//v0tKirKnnrqKa/6MjNbtWqVNWzY0P766y/3a+Kvv/6yatWqWdOmTa1fv35e7Sm82Md5xnhnxnizwjvOF9QYnxvOMTtP7P+OM73mmmuUnp6uLVu2uOcVL15cd9xxh9q3b6/ly5dr3759udYqUqSIGjdurI4dO+r+++/XwoULNW3aNE2ZMsXnE5NTU1N15MgR3X777ZL+//lNV1xxhftYdW+P33W5XO6eMk2aNElfffWVhg4dqltuuUWDBw/WypUrvapXpEgRlS9fXk2bNlVcXJy6deum8ePHa9GiRfr1118VFRXlVZ0WLVro4MGD+uSTTxQVFaXNmzercePGuu222/Tvf/9bzz33nLZu3epVrSNHjmjr1q06cOCARo4cKUl688039e9//1utW7fWE088of/+97951klOTtaoUaP0zDPPKCMjQ0WKFNGkSZP022+/adasWZJ8O27azLRjxw7FxsZKkp555hl9+eWX+s9//qMZM2borrvu0pw5c7yqc+LECaWkpOjPP/9UWlqaihUrpr///lsffPCBoqKiVLt2bX3xxRde9yZJ/v7+SkhIcL8PzEyVK1fWuHHj1KZNG33++eeaP3++V+u9bt06Va1aVZMnT9bixYv17bffatCgQUpKSvKpp4yMDC1cuFDVq1fXmDFj5OfnJ0m69tpr5e/v79P5A+np6frwww/VqVMnPf744+7pmRd1adWqle677z6Pc3m8YWZatWqVxowZoyFDhuiNN97Q3Llz1b17d40dOzbP+6elpWnGjBkqWbKkGjZsKEm69dZbNWHCBL388suKjo7WoEGDtGLFCq/X9ZdfftGhQ4ckSTfffLP+97//qXLlytq+fbtefPFFjRkzJs91PH78uNq2bauBAwfqmWeekSTdddddatKkicaNG6e0tDQVKeLdn6PAwEAdPHhQZcqUcU/74Ycf9Msvv6hx48a69dZbNXr06Dx7kk6POYcOHXJvqxMnTigiIkI33XST6tSpo8WLF+vLL7/0qi9J8vPzU/HixbV7925Jp58PSerbt6969+6tuLg4LV26VFLu5yL4+/urfv36io6OVr169fTvf/9bbdu2Vbdu3bweTzMVK1ZMxYsXV/Xq1d2veUlq2LChtm3bpiNHjnh1UaOSJUsqKipKffr0kZ+fn1wul55++ml98cUXWrt2rVasWKF77rlHb7/9tlfnzWY+N23atNHatWtVtWpV/fe//9XWrVvVsWNHvfrqqx7jR05CQkLUrVs31ahRQ/369VOTJk104sQJTZw4UevXr1fLli313HPPebXdM0VERGj79u2aPHmygoKC9O9//1vbt2/XxIkT5XK59Nxzz+V57t8VV1yhd955R5MmTZJ0esxo1qyZunbtqgULFujo0aN59nGmkydPKjU1VT/99JP279+vyZMn67333lPlypVVrlw5rVq1SiNGjPDqM8GePXu0a9culSxZ0v2a2L9/vypVqqR27dppxYoV2rhxo6S8X6cFMc6bmdatW6cqVaoUmnGeMd77Md7MCnSMl06P8wcOHCh047zL5fJ6jM/39QPOKdYhT3/88YeVK1fOBgwY4D5kINPu3butSJEitmjRojzrHDt2zOP2woULzeVy2YgRI9zHO6enp3t1HsNvv/3m/n/mtz7jxo3LcljLmd9I5OTMdXr//ffN5XLZwoUL7eDBg/bdd9/Ztdde6/Nu/759+7r3JA4cONBKly5ttWvXtrvvvtt++umnPO+/e/du69u3rxUrVszat2/v8c3fokWLLDQ01D744AOvesnIyLC77rrLHnjgAYuKirIlS5a45+3cudP69Olj9957r8c3yN7WPXLkiHXt2tV69Ojh8/3/+usva9mypVWvXt1uu+02c7lc9vHHH1tGRobt3bvXhg0bZu3atbMDBw54VXflypVWpEgRa9OmjUVHR1tQUJANGjTIzE6fQ1iyZEnbsmWLV7XS0tIsJSXFBgwYYN26dbOTJ09aRkaG+9uy7du3W6dOnaxLly5ereu+ffts+fLl7ts//vijlSlTxu644w6Pb8G96e27777Lspc6PT3dqlWr5vEY3ti5c6f9+OOP7ttPP/20+fn52dixY+2VV16xpk2b2o033ujzYRwdOnSw+Ph4MzObMmWKBQUFWUhISJZv6HLy22+/2T333GPNmze3iIgI69y5s23dutXS0tLs119/tTp16ni19znT0qVL7YYbbrC33nrL2rdv7z7P8siRIzZhwgRr3ry5bdy4Mc86metkZu5vsd9880276qqr3Oek5PWNanp6uiUmJlpkZKR169bNZsyYYaNHj7bixYvb7Nmz7csvv7QJEybYNddcY5988kmutTLfK+Hh4R57hHfu3Gm1a9e2uXPnWv369d3vA2/dcsst1rBhQ/e34Gcetnb77bdbixYtfKqXKTU11b3nLPMb6NTUVFu6dKn7kJqzZb4nMs8lO3PaihUrrEaNGh570Xbt2pXtc5Dde2v9+vVWq1Yt++yzz9zreN9991nNmjWz/L3KrV5sbKxVr17dvXfklltuMX9/f2vXrp2tWbMm1zpn9vryyy9b7dq1rWPHjlm+sY6MjPRq72Dmtnj22WetQYMGdtttt9lzzz3nsczLL79sVatWzfU8o9wOVXz11VctJCTEvXfLl70I/fr1sxo1atgNN9xgJUqUsE8//dQ9b/78+Xb55Zd7dUTIqVOnrHr16hYZGWnffvutffXVVxYUFGTjxo0zM7MWLVrYkCFD8qyTnJxsAwYMsK5du57zOL93794CG+dXrFjhsZfLLH/j/I4dOwrlGL9161a755577Nprry2wMb5du3bnPMaf+fkzv2N8psOHD1tkZKR17dr1nMZ5s9NHUoWHh3scyurLOL97926P9Y+KijovY3wmgtkFsGzZMgsMDLT777/f4zCPAwcOWOPGjX0aKM78AJ8ZhEaOHGl///23Pfzww9a9e3c7fvy4V7XOfIOMHTvWOnTo4L797LPP2gsvvODTsfDbtm3zOOnX7PQf2VtuucWr+2eu15w5c2zcuHF23333WcWKFe2vv/6yjz76yK688kq79957vTpX7O+//7YxY8a4t+2Z61q7dm27//77vVwrszVr1lhQUJC5XC6PP4RmZo8++qi1adMm38cTf/jhh+ZyudwftHwRHx9v//nPf2z8+PEe5zGamT333HPWoEEDn86r+/nnn61Pnz42aNAgmzlzpnv6J598YrVq1crzUKCzL/iyYsUK8/Pzs5dfftk9LfN5+Pnnn83lcuV4XmNOF4/JvP/q1avdf7QTExMtJSXFXn31Vfv666+9rpX5nGVkZNgVV1zhcd9vvvnG9u3b53WtAwcO2PDhw+3LL790T9u0aZO5XC6Pad7UateunfvwkYEDB1pwcLCFhYXZlClTcjxZ/Oxaf/zxh0VHR1tUVJTHFzGZ6+ZyuWzDhg1e1dq8ebOFh4db7dq17aabbvKYt2PHDitRooQtWLAgz1rZvUeOHj1qEREReb4fz+5p9erV1qlTJ+vVq5fVrFnT3n77bfe8PXv2WOXKlW3y5Mle1co8PPzuu++2J554wkqVKmWDBw82M7P//Oc/7g/h2X2gOHbsmCUlJXkccrV//36rVq2atW/f3pKTkz2Wf/PNN6158+ZZpudUy8xzu6WkpLjD2fLly23IkCF29dVXZzl0KadaZ67DihUr7Morr3TXHzFihLVr187j70dOdczMEhMT3a/HzC/45s6daw0aNMjyJWRutU6dOuX+AmfAgAFWqVIle//9961evXrWrl27bL+My6nWvHnz7PPPP3evU+bfr8cffzzHYJZTrdatW5vL5bK+fft6HLb2yy+/WO3atbO9SEZu2+vM57FZs2Z211135fp3I6damzZtspUrV9rVV19t27Zt85heo0aNbM8zzq7W+vXrrX79+laxYkWrWLGiPfbYY+55t99+u8fhiWc6ePCgbd682T2u/Pjjj/ke5w8ePGibNm3Kcshq5nvUl3E+p1q+jvM51cnPGH/mtjrzNdS2bVufx/jMvjID/c6dO/M9xp/ZV3p6usXHx+d7jD948KBt3LjRtm7daqdOncoy39sx/sy+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1rt3b/dyOZ1DmJ9aZ1qzZk226+hrrcTEROvQoUOWD5a+1jlx4oRNmzYt2yMjvKnl7RERvvZ1/PhxmzRpUrYflJx+Ds2yv0CIt7XO/ttzLrXOtGLFimz3eBw4cMCeffZZj+d44sSJ5nK5rEGDBhYREWEdOnSwP//803bv3m3fffedff3119mO87nVatiwoVWqVMkiIyPdPw3x888/W7169bId572ttWLFCjt27JiFhISYv79/lnHel5727Nlj3bp1szp16mQ7xue1rSpXrmxRUVFeXejDl752795tkZGRVr169WzHrbz6uvzyy+2mm26yrVu32tGjR+23336z9evXZzvG+/ocmp1+HrM7FSWvWpl9bdmyxX7++WebNGmSvfHGG9mOgb70derUKXvllVfs+uuvz/EzdObY1Lt3b/ehnIsXL7Zy5cpZyZIls5yiUVAIZv9wBXmYX0GEg0z5PefqfDqXkzGRs9yuGJo5qKempnp8U5WfWmdefTTzw0NOV6PztlZqaqodOHDAlixZkmto96ZWWlqa7d2713bu3JnthQG8qZUZZr29wqq3fe3bt8/i4+NzfQ683V7e7Nn19nk8e++DL3UyX1spKSleHX3gy3OYmpqa62vL2yvbnnkRjJzG6vzUOte+Mg9xymmc9raON78zVZBXAf6n91VY19GbWrldgXnFihXWpEkT91Udz6XW2VdzTkxMzPW96E2tp556ysxOfyGT0zjvS09xcXG5jvG51Vq+fLlPV6v2pa/MPZf5rdWkSRP3tirIvs61VtOmTc/La+vAgQO5vrYy5Xal8DOv2llQn8eL5u8i+ygsfPntq7wEBQUVWC1/f/8Cq1VQAgICnG7hopT5uklPT1eRIkV05513yszUq1cvuVwuDR8+XNOmTdP27ds1b948lShRIsfXrbe14uPjtWDBApUuXfqc+9q2bZvee+89lShR4pxrxcfH6/3331exYsUKZHu9++67hbKvgnoeM9exePHi2dby5TnM3FYX4rVVo0YNSad/pyZzrEtPT9fevXvdy0yePFmBgYEaNmyYihYtmmNf+al1rn0FBATooYceynGcdqInahXuWqVKlXL/v0WLFlq7dq2uueYaSVLbtm1VsWJFrVu3Lsf7e1urTZs2Cg0NVUxMjCQpODi4wGr16NHjnOqsXbtWklSnTp1899SuXTuPnvLiS18NGjQ4p1oVK1bUL7/8UmB9FdQ6hoWFFWhfmdurbNmyudYyM7lcLt1www3666+/NHToUH3xxReKiYlRbGysRo4cqYCAADVq1EiBgYEF9nmcYAagQPj5+cnMlJGRobvuuksul0vR0dH69NNP9eeff2rNmjVeh/+8av38888qXrz4Odf6448/tHbt2lzDj6995RZ+fKm1Zs2aQttXQT2P3q5jYX1tFSlSxP3H2+VyuX/Mdty4cZo0aZLWrVuX64fdwl6rMPZELWdrSVKVKlVUpUoVSac/vKakpKhkyZKqW7eu1zUKc63C2BO1LnytzKBVrVo1DRgwQKGhofr8889VrVo1VatWTS6XSw0aNFBgYKDPveWqQPa7AcD/KagrhlKLWv+EngryyraFsVZh7IlaztY6W36vwPxPqVUYe6LWhat1Pq8Unh2CGYACV5BXDKUWtQp7T2YFe2XbwlirMPZELWdrFcQVmAtzrcLYE7WcqVXQVwrPTZGC3f8GAKfVqVNHv/zyi+rXr08tahVorcLYU2RkpCRp1apVatKkyUVXqzD2RC1na9WqVUv79+/X999/r0aNGl10tQpjT9RyplaRIhcuLrnMzC7YowG4ZNj/ndNALWoVdK3C2JMkHT9+vMAuolQYaxXGnqjlbK3U1NQCu9hXYaxVGHuilrO1zjeCGQAAAAA4jEMZAQAAAMBhBDMAAAAAcBjBDAAAAAAcRjADAAAAAIcRzAAAAADAYQQzAAAAAHAYwQwAcNGrWrWqXnrpJafbAAAgRwQzAMBFY86cObrsssuyTF+zZo3uueeeC9+Qg/r376+uXbs63QYAwEtFnW4AAIDzrXz58k63UGilpqbK39/f6TYA4JLHHjMAQKHRrl07DRs2TKNGjVKZMmUUFham8ePHu+dPnz5d9erVU1BQkCIiIjR06FAdO3ZMkrRixQoNGDBAiYmJcrlccrlc7vueeShjz549ddddd3k8bmpqqsqVK6fZs2dLksxMU6ZM0RVXXKHixYurQYMG+u9//+v1emzcuFGdO3dWcHCwSpUqpdatW+vPP/+UJGVkZGjixImqVKmSAgMD1bBhQy1ZssR93xUrVsjlcunIkSPuabGxsXK5XNq2bZuk/79n8KuvvlKtWrVUsmRJdezYUQkJCZKk8ePHa+7cufrkk0/c22LFihXatm2bXC6X/v3vf6tdu3YqVqyY3njjDQUHB2dZv88++0xBQUE6evSo1+sNAMg/ghkAoFCZO3eugoKC9NNPP2nKlCmaOHGili5dKkkqUqSIXnnlFcXFxWnu3LlatmyZRo0aJUlq2bKlXnrpJQUHByshIUEJCQkaMWJElvq9e/fWp59+6g50kvTVV1/p+PHjuu222yRJTzzxhGbPnq1Zs2Zp48aNevjhh9WnTx999913efb/999/q02bNipWrJiWLVummJgY3X333UpLS5Mkvfzyy3rhhRc0bdo0bdiwQZGRkerSpYt+//13n7bTiRMnNG3aNL377rv6/vvvtWPHDvf6jhgxQj169HCHtYSEBLVs2dJ938cee0zDhg3T5s2b1a1bN911113uUJpp9uzZuv3221WqVCmf+gIA5JMBAFBItG3b1q677jqPaU2bNrXHHnss2+X//e9/W9myZd23Z8+ebSEhIVmWq1Klir344otmZpaSkmLlypWzefPmuef37NnT7rjjDjMzO3bsmBUrVsxWrVrlUWPgwIHWs2fPPNdh9OjRVq1aNUtJScl2fnh4uD3zzDNZ1nHo0KFmZrZ8+XKTZIcPH3bPX7dunUmy+Ph493pKsj/++MO9zMyZMy00NNR9u1+/fnbrrbd6PE58fLxJspdeeslj+k8//WR+fn72999/m5nZ/v37zd/f31asWJHn+gIACgZ7zAAAhUr9+vU9blesWFH79u2TJC1fvlzt27fX5ZdfrlKlSqlv3746ePCgjh8/7nV9f39/3XHHHZo/f74k6fjx4/rkk0/Uu3dvSdKmTZt06tQptW/fXiVLlnT/mzdvnvtwxNzExsaqdevW2Z63lZSUpN27d6tVq1Ye01u1aqXNmzd7vQ6SVKJECV155ZXu22dup7w0adLE4/a1116rOnXqaN68eZKkd999V5UrV1abNm186gkAkH8EMwBAoXJ2oHG5XMrIyND27dt18803q27duvrwww8VExOjmTNnSjp9jpgvevfurW+++Ub79u3Txx9/rGLFiqlTp06STp8DJkmLFy9WbGys+9+mTZu8Os+sePHieS7jcrk8bpuZe1qRIkXc0zJlt37Zbacz75OboKCgLNMGDRrkPpxx9uzZGjBgQJY+AQDnD8EMAPCPsHbtWqWlpemFF15Q8+bNddVVV2n37t0eywQEBCg9PT3PWi1btlRERIQ++OADzZ8/X3fccYcCAgIkSbVr11ZgYKB27Nih6tWre/yLiIjIs3b9+vX1ww8/ZBumgoODFR4erpUrV3pMX7VqlWrVqiXp/19BMvNCHtLpvXC+8nZbZOrTp4927NihV155RRs3blS/fv18fkwAQP4RzAAA/whXXnml0tLS9K9//Ut//fWX3n33Xb322msey1StWlXHjh3Tt99+qwMHDujEiRPZ1nK5XOrVq5dee+01LV26VH369HHPK1WqlEaMGKGHH35Yc+fO1Z9//ql169Zp5syZmjt3bp59PvDAA0pKStJdd92ltWvX6vfff9e7776rrVu3SpJGjhyp559/Xh988IG2bt2qxx9/XLGxsXrooYckyR0Ax48fr99++02LFy/WCy+84PP2qlq1qjZs2KCtW7fqwIEDee5VLF26tLp3766RI0eqQ4cOqlSpks+PCQDIP4IZAOAfoWHDhpo+fbqef/551a1bV/Pnz9fkyZM9lmnZsqXuvfde3XnnnSpfvrymTJmSY73evXtr06ZNuvzyy7Oc8/X0009r3Lhxmjx5smrVqqXIyEh99tlnqlatWp59li1bVsuWLdOxY8fUtm1bNW7cWG+++ab70MNhw4bp0Ucf1aOPPqp69eppyZIl+vTTT1WjRg1Jpw9RfP/997VlyxY1aNBAzz//vCZNmuTr5tLgwYNVs2ZNNWnSROXLl9f//ve/PO8zcOBApaSk6O677/b58QAA58Zl3h6QDgAALmrz58/XQw89pN27d7sP7QQAXBhFnW4AAAA468SJE4qPj9fkyZM1ZMgQQhkAOIBDGQEA8MG9997rcRn9M//de++9TreXL1OmTFHDhg0VGhqq0aNHO90OAFySOJQRAAAf7Nu3T0lJSdnOCw4OVoUKFS5wRwCAiwHBDAAAAAAcxqGMAAAAAOAwghkAAAAAOIxgBgAAAAAOI5gBAAAAgMMIZgAAAADgMIIZAAAAADiMYAYAAAAADiOYAQAAAIDD/h+gkBEPg+iufAAAAABJRU5ErkJggg==
<Figure size 1000x600 with 1 Axes>
output_type": "display_data
image/png": "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
<Figure size 1200x800 with 2 Axes>
output_type": "display_data
name": "stdout
output_type": "stream


### Observations ###

- Feature 'age' appears to have capped values at 90.

- Feature 'workclass' appears to have capped values at 7.

- Feature 'fnlwgt' appears to have capped values at 1484705.

- Feature 'education_num' appears to have capped values at 16.

- Feature 'marital_status' appears to have capped values at 6.

- Feature 'occupation' appears to have capped values at 13.

- Feature 'relationship' appears to have capped values at 5.

- Feature 'capital_gain' appears to have capped values at 99999.

- Feature 'capital_loss' appears to have capped values at 4356.

- Feature 'hours_per_week' appears to have capped values at 99.

- Feature 'native_country' appears to have capped values at 40.

- Feature 'native_country' is highly imbalanced with 91.39% in one category.

- No missing values found in the dataset.



### Data Cleanup Tasks ###

- Inspect the feature for possible outlier treatment or normalization.

- Inspect the feature for possible outlier treatment or normalization.

- Inspect the feature for possible outlier treatment or normalization.

- Inspect the feature for possible outlier treatment or normalization.

- Inspect the feature for possible outlier treatment or normalization.

- Inspect the feature for possible outlier treatment or normalization.

- Inspect the feature for possible outlier treatment or normalization.

- Inspect the feature for possible outlier treatment or normalization.

- Inspect the feature for possible outlier treatment or normalization.

- Inspect the feature for possible outlier treatment or normalization.

- Inspect the feature for possible outlier treatment or normalization.

- Consider rebalancing the feature with techniques like oversampling or under-sampling.

- Handle missing values using imputation techniques or removal.

#Explore the data using yprofile and correlation matrix. Make observations about features, distributions, capped values, and missing values. Create a list of data cleanup tasks.\t10\t\t

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns







# Step 1: Analyze the Target Variable (y-profile)

target_column = 'income'



print(f\"\
### Basic Info for Target Variable '{target_column}' ###\")

print(data[target_column].describe())

print(\"\
Value Counts:\")

print(data[target_column].value_counts())



# Visualize Target Variable Distribution

plt.figure(figsize=(8, 5))

sns.countplot(x=target_column, data=data)

plt.title(f\"Distribution of Target Variable: {target_column}\")

plt.xlabel(target_column)

plt.ylabel(\"Count\")

plt.show()



# Relationships Between Target and Categorical Features

categorical_columns = [

    'workclass', 'education', 'marital_status', 'occupation',

    'relationship', 'race', 'sex', 'native_country'

]



for col in categorical_columns:

    plt.figure(figsize=(10, 6))

    sns.countplot(x=col, hue=target_column, data=data)

    plt.title(f\"{col} vs {target_column}\")

    plt.xlabel(col)

    plt.ylabel(\"Count\")

    plt.legend(title=target_column)

    plt.xticks(rotation=45)

    plt.show()



# Step 2: Correlation Matrix for Numerical Features

# Select only numerical columns for the correlation matrix

numerical_data = data.select_dtypes(include=[np.number])

correlation_matrix = numerical_data.corr()



# Visualize the Correlation Matrix

plt.figure(figsize=(12, 8))

sns.heatmap(correlation_matrix, annot=True, fmt=\".2f\", cmap=\"coolwarm\", cbar=True)

plt.title(\"Correlation Matrix of Numerical Features\")

plt.show()



# Step 3: Observations About Features, Distributions, Capped Values, and Missing Values

observations = []



# Check for capped values in numerical features

for col in numerical_data.columns:

    max_value = data[col].max()

    if max_value > data[col].quantile(0.95):

        observations.append(f\"Feature '{col}' appears to have capped values at {max_value}.\")



# Check for imbalanced distributions in categorical features

for col in categorical_columns:

    imbalance = data[col].value_counts(normalize=True).iloc[0]

    if imbalance > 0.9:

        observations.append(f\"Feature '{col}' is highly imbalanced with {imbalance * 100:.2f}% in one category.\")



# Check for missing values

missing_values_count = data.isnull().sum()

if missing_values_count.sum() == 0:

    observations.append(\"No missing values found in the dataset.\")

else:

    for col, count in missing_values_count.items():

        if count > 0:

            observations.append(f\"Feature '{col}' has {count} missing values.\")



# Step 4: Create a List of Data Cleanup Tasks

cleanup_tasks = []



# Suggest cleanup actions based on observations

for obs in observations:

    if \"capped values\" in obs:

        cleanup_tasks.append(\"Inspect the feature for possible outlier treatment or normalization.\")

    elif \"highly imbalanced\" in obs:

        cleanup_tasks.append(\"Consider rebalancing the feature with techniques like oversampling or under-sampling.\")

    elif \"missing values\" in obs:

        cleanup_tasks.append(\"Handle missing values using imputation techniques or removal.\")



# Print Observations and Cleanup Tasks

print(\"\
### Observations ###\")

for obs in observations:

    print(f\"- {obs}\")



print(\"\
### Data Cleanup Tasks ###\")

for task in cleanup_tasks:

    print(f\"- {task}\")

cell_type": "code
id": "171d143f
# Printing this list of cleanup tasks

# ### Observations ###

# - Feature 'age' appears to have capped values at 90.

# - Feature 'fnlwgt' appears to have capped values at 1484705.

# - Feature 'education_num' appears to have capped values at 16.

# - Feature 'capital_gain' appears to have capped values at 99999.

# - Feature 'capital_loss' appears to have capped values at 4356.

# - Feature 'hours_per_week' appears to have capped values at 99.

# - Feature 'native_country' is highly imbalanced with 91.22% in one category.

# - Feature 'workclass' has 1836 missing values.

# - Feature 'occupation' has 1843 missing values.

# - Feature 'native_country' has 583 missing values.



# ### Data Cleanup Tasks ###

# - Inspect the feature for possible outlier treatment or normalization.

# - Inspect the feature for possible outlier treatment or normalization.

# - Inspect the feature for possible outlier treatment or normalization.

# - Inspect the feature for possible outlier treatment or normalization.

# - Inspect the feature for possible outlier treatment or normalization.

# - Inspect the feature for possible outlier treatment or normalization.

# - Consider rebalancing the feature with techniques like oversampling or under-sampling.

# - Handle missing values using imputation techniques or removal.

# - Handle missing values using imputation techniques or removal.

# - Handle missing values using imputation techniques or removal.

cell_type": "code
id": "b75a104b
# Experiment1:Experiment #1: Create a pipeline for preprocessing (StandardScaler, MinMaxScaler, LogTransformation, OneHotEncoding) and Logistic Regression. Log F1-score/(TP,TN,FN,FP)  in MLFlow on DagsHub. – Cross validation 3/10 folds. Results—mean/std of CV results and results on the whole training data – add in parameter hyper tuning



from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

import numpy as np

import pandas as pd



# Load Dataset

file_path = \"adult_income.csv\"  # Replace with your dataset path

data = pd.read_csv(file_path)



# Separate Features and Target

X = data.drop('income', axis=1)

y = data['income']



# Convert Target Variable to Numerical Labels

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)



# Define Preprocessing for Numerical and Categorical Columns

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

categorical_cols = X.select_dtypes(include=['object', 'category']).columns



numerical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean')),

    ('log', FunctionTransformer(np.log1p, validate=True)),  # Log Transformation

    ('scaler', StandardScaler())

])



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Combine Preprocessors

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ]

)

cell_type": "code
id": "9d0166b6
# Intergrate model

# Experiment 1 - Step2



from sklearn.linear_model import LogisticRegression



# Create the Model Pipeline

pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('classifier', LogisticRegression(max_iter=1000))

])

cell_type": "code
id": "4c42863e
name": "stdout
output_type": "stream
3-Fold CV Mean F1: 0.6442, Std: 0.0103

10-Fold CV Mean F1: 0.6438, Std: 0.0184

# Experiment1-3



from sklearn.model_selection import cross_val_score, train_test_split



# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)



# Perform Cross-Validation

cv_scores_3 = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1')

cv_scores_10 = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='f1')



print(f\"3-Fold CV Mean F1: {cv_scores_3.mean():.4f}, Std: {cv_scores_3.std():.4f}\")

print(f\"10-Fold CV Mean F1: {cv_scores_10.mean():.4f}, Std: {cv_scores_10.std():.4f}\")

cell_type": "code
id": "fa0ee5a5
name": "stdout
output_type": "stream


Confusion Matrix:

[[6863  554]

 [ 936 1416]]



Classification Report:

              precision    recall  f1-score   support



       <=50K       0.88      0.93      0.90      7417

        >50K       0.72      0.60      0.66      2352



    accuracy                           0.85      9769

   macro avg       0.80      0.76      0.78      9769

weighted avg       0.84      0.85      0.84      9769





F1 Score on Test Data: 0.6553



True Positives: 1416, True Negatives: 6863, False Positives: 554, False Negatives: 936

# Experiment1-4 Train,Model and Evalute

from sklearn.metrics import classification_report, confusion_matrix, f1_score



# Train Model

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)



# Confusion Matrix and Classification Report

conf_matrix = confusion_matrix(y_test, y_pred)

print(\"\
Confusion Matrix:\")

print(conf_matrix)



print(\"\
Classification Report:\")

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))



# Calculate F1 Score

f1 = f1_score(y_test, y_pred)

print(f\"\
F1 Score on Test Data: {f1:.4f}\")



# Display TP, TN, FP, FN

tn, fp, fn, tp = conf_matrix.ravel()

print(f\"\
True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}\")

cell_type": "code
id": "3c907656
name": "stdout
output_type": "stream


Best Parameters from Grid Search:

{'classifier__C': 100, 'classifier__solver': 'saga'}



Confusion Matrix for Best Model:

[[6857  560]

 [ 938 1414]]



F1 Score for Best Model: 0.6537

name": "stderr
output_type": "stream
C:\\Users\\lenovo\\anaconda3\\Lib\\site-packages\\sklearn\\linear_model\\_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge

  warnings.warn(

# Experiment1-step5 Parameter tuning

from sklearn.model_selection import GridSearchCV



# Define Parameter Grid for Hyperparameter Tuning

param_grid = {

    'classifier__C': [0.01, 0.1, 1, 10, 100],

    'classifier__solver': ['liblinear', 'saga']

}



# Perform Grid Search

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)

grid_search.fit(X_train, y_train)



# Best Parameters

print(\"\
Best Parameters from Grid Search:\")

print(grid_search.best_params_)



# Best Model Evaluation

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)



# Confusion Matrix and F1 Score for Best Model

conf_matrix_best = confusion_matrix(y_test, y_pred_best)

f1_best = f1_score(y_test, y_pred_best)

print(\"\
Confusion Matrix for Best Model:\")

print(conf_matrix_best)

print(f\"\
F1 Score for Best Model: {f1_best:.4f}\")

cell_type": "code
id": "7fbca147
# Experiment #2 step1

# Importing the required libraries and pre-processing pipelines



from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, f1_score

from sklearn.model_selection import train_test_split, cross_val_score

import pandas as pd

import mlflow

import mlflow.sklearn







# Separate Features and Target

X = data.drop('income', axis=1)

y = data['income']



# Convert Target Variable to Numerical Labels

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)



# Define Preprocessing for Numerical and Categorical Columns

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

categorical_cols = X.select_dtypes(include=['object', 'category']).columns



numerical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean')),

    ('scaler', StandardScaler())

])



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Combine Preprocessors

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ]

)

cell_type": "code
id": "d4ae2adb
# Experiment 2 - step2 Define the classifiers and the pipelines



# Define Classifiers

classifiers = {

    \"LogisticRegression\": LogisticRegression(max_iter=1000),

    \"RidgeClassifier\": RidgeClassifier(),

    \"RandomForestClassifier\": RandomForestClassifier(n_estimators=100, random_state=42),

    \"XGBClassifier\": XGBClassifier(use_label_encoder=False, eval_metric=\"logloss\", random_state=42)

}



# Create Pipelines

pipelines = {name: Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)]) 

             for name, clf in classifiers.items()}

cell_type": "code
id": "b1063254
# Experiment 2 - step3 train/test split

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

cell_type": "code
id": "e9a19b1d
name": "stdout
output_type": "stream
Requirement already satisfied: pip in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (24.3.1)

Collecting dagshub==0.4.0

  Using cached dagshub-0.4.0-py3-none-any.whl.metadata (11 kB)

Collecting PyYAML>=5 (from dagshub==0.4.0)

  Using cached PyYAML-6.0.2-cp311-cp311-win_amd64.whl.metadata (2.1 kB)

Collecting appdirs>=1.4.4 (from dagshub==0.4.0)

  Using cached appdirs-1.4.4-py2.py3-none-any.whl.metadata (9.0 kB)

Collecting click>=8.0.4 (from dagshub==0.4.0)

  Using cached click-8.1.7-py3-none-any.whl.metadata (3.0 kB)

Collecting httpx>=0.23.0 (from dagshub==0.4.0)

  Using cachb==0.4.0)

  Using cached tenacity-9.0.0-py3-none-any.whl.metadata (1.2 kB)

Collecting gql[requests] (from dagshub==0.4.0)

  Using cached gql-3.5.0-py2.py3-none-any.whl.metadata (9.2 kB)

Collecting dataclasses-json (from dagshub==0.4.0)

  Using cached dataclasses_json-0.6.7-py3-none-any.whl.metadata (25 kB)

Collecting pandas (from dagshub==0.4.0)

  Using cached pandas-2.2.3-cp311-cp311-win_amd64.whl.metadata (19 kB)

Collecting treelib>=1.6.4 (from dagshub==0.4.0)

  Using cached treelib-1.7.0-py3-none-any.whl.metadata (1.3 kB)

Collecting pathvalidate>=3.0.0 (from dagshub==0.4.0)

  Using cached pathvalidate-3.2.1-py3-none-any.whl.metadata (12 kB)

Collecting python-dateutil (from dagshub==0.4.0)

  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)

Collecting boto3 (from dagshub==0.4.0)

  Using cached boto3-1.35.86-py3-none-any.whl.metadata (6.7 kB)

Collecting dagshub-annotation-converter>=0.1.0 (from dagshub==0.4.0)

  Using cached dagshub_annotation_converter-0.1.2-py3-none-any.whl.metadata (2.5 kB)

Collecting colorama (from click>=8.0.4->dagshub==0.4.0)

  Using cached colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)

Collecting lxml (from dagshub-annotation-converter>=0.1.0->dagshub==0.4.0)

  Using cached lxml-5.3.0-cp311-cp311-win_amd64.whl.metadata (3.9 kB)

Collecting pillow (from dagshub-annotation-converter>=0.1.0->dagshub==0.4.0)

  Using cached pillow-11.0.0-cp311-cp311-win_amd64.whl.metadata (9.3 kB)

Collecting pydantic>=2.0.0 (from dagshub-annotation-converter>=0.1.0->dagshub==0.4.0)

  Using cached pydantic-2.10.4-py3-none-any.whl.metadata (29 kB)

Collecting typing-extensions (from dagshub-annotation-converter>=0.1.0->dagshub==0.4.0)

  Using cached typing_extensions-4.12.2-py3-none-any.whl.metadata (3.0 kB)

Collecting gitdb<5,>=4.0.1 (from GitPython>=3.1.29->dagshub==0.4.0)

  Using cached gitdb-4.0.11-py3-none-any.whl.metadata (1.2 kB)

Collecting anyio (from httpx>=0.23.0->dagshub==0.4.0)

  Using cached anyio-4.7.0-py3-none-any.whl.metadata (4.7 kB)

Collecting certifi (from httpx>=0.23.0->dagshub==0.4.0)

  Using cached certifi-2024.12.14-py3-none-any.whl.metadata (2.3 kB)

Collecting httpcore==1.* (from httpx>=0.23.0->dagshub==0.4.0)

  Using cached httpcore-1.0.7-py3-none-any.whl.metadata (21 kB)

Collecting idna (from httpx>=0.23.0->dagshub==0.4.0)

  Using cached idna-3.10-py3-none-any.whl.metadata (10 kB)

Collecting h11<0.15,>=0.13 (from httpcore==1.*->httpx>=0.23.0->dagshub==0.4.0)

  Using cached h11-0.14.0-py3-none-any.whl.metadata (8.2 kB)

Collecting markdown-it-py>=2.2.0 (from rich>=13.1.0->dagshub==0.4.0)

  Using cached markdown_it_py-3.0.0-py3-none-any.whl.metadata (6.9 kB)

Collecting pygments<3.0.0,>=2.13.0 (from rich>=13.1.0->dagshub==0.4.0)

  Using cached pygments-2.18.0-py3-none-any.whl.metadata (2.5 kB)

Collecting six (from treelib>=1.6.4->dagshub==0.4.0)

  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)

Collecting botocore<1.36.0,>=1.35.86 (from boto3->dagshub==0.4.0)

  Using cached botocore-1.35.86-py3-none-any.whl.metadata (5.7 kB)

Collecting jmespath<2.0.0,>=0.7.1 (from boto3->dagshub==0.4.0)

  Using cached jmespath-1.0.1-py3-none-any.whl.metadata (7.6 kB)

Collecting s3transfer<0.11.0,>=0.10.0 (from boto3->dagshub==0.4.0)

  Using cached s3transfer-0.10.4-py3-none-any.whl.metadata (1.7 kB)

Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json->dagshub==0.4.0)

  Using cached marshmallow-3.23.2-py3-none-any.whl.metadata (7.1 kB)

Collecting typing-inspect<1,>=0.4.0 (from dataclasses-json->dagshub==0.4.0)

  Using cached typing_inspect-0.9.0-py3-none-any.whl.metadata (1.5 kB)

Collecting graphql-core<3.3,>=3.2 (from gql[requests]->dagshub==0.4.0)

  Using cached graphql_core-3.2.5-py3-none-any.whl.metadata (10 kB)

Collecting yarl<2.0,>=1.6 (from gql[requests]->dagshub==0.4.0)

  Using cached yarl-1.18.3-cp311-cp311-win_amd64.whl.metadata (71 kB)

Collecting backoff<3.0,>=1.11.1 (from gql[requests]->dagshub==0.4.0)

  Using cached backoff-2.2.1-py3-none-any.whl.metadata (14 kB)

Collecting requests<3,>=2.26 (from gql[requests]->dagshub==0.4.0)

  Using cached requests-2.32.3-py3-none-any.whl.metadata (4.6 kB)

Collecting requests-toolbelt<2,>=1.0.0 (from gql[requests]->dagshub==0.4.0)

  Using cached requests_toolbelt-1.0.0-py2.py3-none-any.whl.metadata (14 kB)

Collecting numpy>=1.23.2 (from pandas->dagshub==0.4.0)

  Using cached numpy-2.2.0-cp311-cp311-win_amd64.whl.metadata (60 kB)

Collecting pytz>=2020.1 (from pandas->dagshub==0.4.0)

  Using cached pytz-2024.2-py2.py3-none-any.whl.metadata (22 kB)

Collecting tzdata>=2022.7 (from pandas->dagshub==0.4.0)

  Using cached tzdata-2024.2-py2.py3-none-any.whl.metadata (1.4 kB)

Collecting sniffio>=1.1 (from anyio->httpx>=0.23.0->dagshub==0.4.0)

  Using cached sniffio-1.3.1-py3-none-any.whl.metadata (3.9 kB)

Collecting urllib3!=2.2.0,<3,>=1.25.4 (from botocore<1.36.0,>=1.35.86->boto3->dagshub==0.4.0)

  Using cached urllib3-2.2.3-py3-none-any.whl.metadata (6.5 kB)

Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->GitPython>=3.1.29->dagshub==0.4.0)

  Using cached smmap-5.0.1-py3-none-any.whl.metadata (4.3 kB)

Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=13.1.0->dagshub==0.4.0)

  Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)

Collecting packaging>=17.0 (from marshmallow<4.0.0,>=3.18.0->dataclasses-json->dagshub==0.4.0)

  Using cached packaging-24.2-py3-none-any.whl.metadata (3.2 kB)

Collecting annotated-types>=0.6.0 (from pydantic>=2.0.0->dagshub-annotation-converter>=0.1.0->dagshub==0.4.0)

  Using cached annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)

Collecting pydantic-core==2.27.2 (from pydantic>=2.0.0->dagshub-annotation-converter>=0.1.0->dagshub==0.4.0)

  Using cached pydantic_core-2.27.2-cp311-cp311-win_amd64.whl.metadata (6.7 kB)

Collecting charset-normalizer<4,>=2 (from requests<3,>=2.26->gql[requests]->dagshub==0.4.0)

  Using cached charset_normalizer-3.4.0-cp311-cp311-win_amd64.whl.metadata (34 kB)

Collecting mypy-extensions>=0.3.0 (from typing-inspect<1,>=0.4.0->dataclasses-json->dagshub==0.4.0)

  Using cached mypy_extensions-1.0.0-py3-none-any.whl.metadata (1.1 kB)

Collecting multidict>=4.0 (from yarl<2.0,>=1.6->gql[requests]->dagshub==0.4.0)

  Using cached multidict-6.1.0-cp311-cp311-win_amd64.whl.metadata (5.1 kB)

Collecting propcache>=0.2.0 (from yarl<2.0,>=1.6->gql[requests]->dagshub==0.4.0)

  Using cached propcache-0.2.1-cp311-cp311-win_amd64.whl.metadata (9.5 kB)

Using cached dagshub-0.4.0-py3-none-any.whl (254 kB)

Using cached appdirs-1.4.4-py2.py3-none-any.whl (9.6 kB)

Using cached click-8.1.7-py3-none-any.whl (97 kB)

Using cached dacite-1.6.0-py3-none-any.whl (12 kB)

Using cached dagshub_annotation_converter-0.1.2-py3-none-any.whl (33 kB)

Using cached GitPython-3.1.43-py3-none-any.whl (207 kB)

Using cached httpx-0.28.1-py3-none-any.whl (73 kB)

Using cached httpcore-1.0.7-py3-none-any.whl (78 kB)

Using cached pathvalidate-3.2.1-py3-none-any.whl (23 kB)

Using cached PyYAML-6.0.2-cp311-cp311-win_amd64.whl (161 kB)

Using cached rich-13.9.4-py3-none-any.whl (242 kB)

Using cached tenacity-9.0.0-py3-none-any.whl (28 kB)

Using cached treelib-1.7.0-py3-none-any.whl (18 kB)

Using cached boto3-1.35.86-py3-none-any.whl (139 kB)

Using cached dataclasses_json-0.6.7-py3-none-any.whl (28 kB)

Using cached pandas-2.2.3-cp311-cp311-win_amd64.whl (11.6 MB)

Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)

Using cached anyio-4.7.0-py3-none-any.whl (93 kB)

Using cached backoff-2.2.1-py3-none-any.whl (15 kB)

Using cached botocore-1.35.86-py3-none-any.whl (13.3 MB)

Using cached gitdb-4.0.11-py3-none-any.whl (62 kB)

Using cached graphql_core-3.2.5-py3-none-any.whl (203 kB)

Using cached idna-3.10-py3-none-any.whl (70 kB)

Using cached jmespath-1.0.1-py3-none-any.whl (20 kB)

Using cached markdown_it_py-3.0.0-py3-none-any.whl (87 kB)

Using cached marshmallow-3.23.2-py3-none-any.whl (49 kB)

Using cached numpy-2.2.0-cp311-cp311-win_amd64.whl (12.9 MB)

Using cached pydantic-2.10.4-py3-none-any.whl (431 kB)

Using cached pydantic_core-2.27.2-cp311-cp311-win_amd64.whl (2.0 MB)

Using cached pygments-2.18.0-py3-none-any.whl (1.2 MB)

Using cached pytz-2024.2-py2.py3-none-any.whl (508 kB)

Using cached requests-2.32.3-py3-none-any.whl (64 kB)

Using cached certifi-2024.12.14-py3-none-any.whl (164 kB)

Using cached requests_toolbelt-1.0.0-py2.py3-none-any.whl (54 kB)

Using cached s3transfer-0.10.4-py3-none-any.whl (83 kB)

Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)

Using cached typing_extensions-4.12.2-py3-none-any.whl (37 kB)

Using cached typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)

Using cached tzdata-2024.2-py2.py3-none-any.whl (346 kB)

Using cached yarl-1.18.3-cp311-cp311-win_amd64.whl (91 kB)

Using cached colorama-0.4.6-py2.py3-none-any.whl (25 kB)

Using cached gql-3.5.0-py2.py3-none-any.whl (74 kB)

Using cached lxml-5.3.0-cp311-cp311-win_amd64.whl (3.8 MB)

Using cached pillow-11.0.0-cp311-cp311-win_amd64.whl (2.6 MB)

Using cached annotated_types-0.7.0-py3-none-any.whl (13 kB)

Using cached charset_normalizer-3.4.0-cp311-cp311-win_amd64.whl (101 kB)

Using cached h11-0.14.0-py3-none-any.whl (58 kB)

Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)

Using cached multidict-6.1.0-cp311-cp311-win_amd64.whl (28 kB)

Using cached mypy_extensions-1.0.0-py3-none-any.whl (4.7 kB)

Using cached packaging-24.2-py3-none-any.whl (65 kB)

Using cached propcache-0.2.1-cp311-cp311-win_amd64.whl (44 kB)

Using cached smmap-5.0.1-py3-none-any.whl (24 kB)

Using cached sniffio-1.3.1-py3-none-any.whl (10 kB)

Using cached urllib3-2.2.3-py3-none-any.whl (126 kB)

Installing collected packages: pytz, appdirs, urllib3, tzdata, typing-extensions, tenacity, sniffio, smmap, six, PyYAML, pygments, propcache, pillow, pathvalidate, packaging, numpy, mypy-extensions, multidict, mdurl, lxml, jmespath, idna, h11, graphql-core, dacite, colorama, charset-normalizer, certifi, backoff, annotated-types, yarl, typing-inspect, treelib, requests, python-dateutil, pydantic-core, marshmallow, markdown-it-py, httpcore, gitdb, click, anyio, rich, requests-toolbelt, pydantic, pandas, httpx, gql, GitPython, dataclasses-json, botocore, s3transfer, dagshub-annotation-converter, boto3, dagshub

  Attempting uninstall: pytz

    Found existing installation: pytz 2024.2

    Uninstalling pytz-2024.2:

      Successfully uninstalled pytz-2024.2

  Attempting uninstall: appdirs

    Found existing installation: appdirs 1.4.4

    Uninstalling appdirs-1.4.4:

      Successfully uninstalled appdirs-1.4.4

  Attempting uninstall: urllib3

    Found existing installation: urllib3 1.26.20

    Uninstalling urllib3-1.26.20:

      Successfully uninstalled urllib3-1.26.20

  Attempting uninstall: tzdata

    Found existing installation: tzdata 2024.2

    Uninstalling tzdata-2024.2:

      Successfully uninstalled tzdata-2024.2

  Attempting uninstall: typing-extensions

    Found existing installation: typing_extensions 4.12.2

    Uninstalling typing_extensions-4.12.2:

      Successfully uninstalled typing_extensions-4.12.2

  Attempting uninstall: tenacity

    Found existing installation: tenacity 9.0.0

    Uninstalling tenacity-9.0.0:

      Successfully uninstalled tenacity-9.0.0

  Attempting uninstall: sniffio

    Found existing installation: sniffio 1.3.1

    Uninstalling sniffio-1.3.1:

      Successfully uninstalled sniffio-1.3.1

  Attempting uninstall: smmap

    Found existing installation: smmap 5.0.1

    Uninstalling smmap-5.0.1:

      Successfully uninstalled smmap-5.0.1

  Attempting uninstall: six

    Found existing installation: six 1.17.0

    Uninstalling six-1.17.0:

      Successfully uninstalled six-1.17.0

  Attempting uninstall: PyYAML

    Found existing installation: PyYAML 6.0.2

    Uninstalling PyYAML-6.0.2:

      Successfully uninstalled PyYAML-6.0.2

  Attempting uninstall: pygments

    Found existing installation: Pygments 2.18.0

    Uninstalling Pygments-2.18.0:

      Successfully uninstalled Pygments-2.18.0

  Attempting uninstall: propcache

    Found existing installation: propcache 0.2.1

    Uninstalling propcache-0.2.1:

      Successfully uninstalled propcache-0.2.1

  Attempting uninstall: pillow

    Found existing installation: pillow 11.0.0

    Uninstalling pillow-11.0.0:

      Successfully uninstalled pillow-11.0.0

  Attempting uninstall: pathvalidate

    Found existing installation: pathvalidate 3.2.1

    Uninstalling pathvalidate-3.2.1:

      Successfully uninstalled pathvalidate-3.2.1

  Attempting uninstall: packaging

    Found existing installation: packaging 24.2

    Uninstalling packaging-24.2:

      Successfully uninstalled packaging-24.2

  Attempting uninstall: numpy

    Found existing installation: numpy 1.24.4

    Uninstalling numpy-1.24.4:

      Successfully uninstalled numpy-1.24.4

  Attempting uninstall: mypy-extensions

    Found existing installation: mypy-extensions 1.0.0

    Uninstalling mypy-extensions-1.0.0:

      Successfully uninstalled mypy-extensions-1.0.0

  Attempting uninstall: multidict

    Found existing installation: multidict 6.1.0

    Uninstalling multidict-6.1.0:

      Successfully uninstalled multidict-6.1.0

  Attempting uninstall: mdurl

    Found existing installation: mdurl 0.1.2

    Uninstalling mdurl-0.1.2:

      Successfully uninstalled mdurl-0.1.2

  Attempting uninstall: lxml

    Found existing installation: lxml 5.3.0

    Uninstalling lxml-5.3.0:

      Successfully uninstalled lxml-5.3.0

  Attempting uninstall: jmespath

    Found existing installation: jmespath 1.0.1

    Uninstalling jmespath-1.0.1:

      Successfully uninstalled jmespath-1.0.1

  Attempting uninstall: idna

    Found existing installation: idna 3.10

    Uninstalling idna-3.10:

      Successfully uninstalled idna-3.10

  Attempting uninstall: h11

    Found existing installation: h11 0.14.0

    Uninstalling h11-0.14.0:

      Successfully uninstalled h11-0.14.0

  Attempting uninstall: graphql-core

    Found existing installation: graphql-core 3.2.5

    Uninstalling graphql-core-3.2.5:

      Successfully uninstalled graphql-core-3.2.5

  Attempting uninstall: dacite

    Found existing installation: dacite 1.6.0

    Uninstalling dacite-1.6.0:

      Successfully uninstalled dacite-1.6.0

  Attempting uninstall: colorama

    Found existing installation: colorama 0.4.6

    Uninstalling colorama-0.4.6:

      Successfully uninstalled colorama-0.4.6

  Attempting uninstall: charset-normalizer

    Found existing installation: charset-normalizer 3.4.0

    Uninstalling charset-normalizer-3.4.0:

      Successfully uninstalled charset-normalizer-3.4.0

  Attempting uninstall: certifi

    Found existing installation: certifi 2024.12.14

    Uninstalling certifi-2024.12.14:

      Successfully uninstalled certifi-2024.12.14

  Attempting uninstall: backoff

    Found existing installation: backoff 2.2.1

    Uninstalling backoff-2.2.1:

      Successfully uninstalled backoff-2.2.1

  Attempting uninstall: annotated-types

    Found existing installation: annotated-types 0.7.0

    Uninstalling annotated-types-0.7.0:

      Successfully uninstalled annotated-types-0.7.0

  Attempting uninstall: yarl

    Found existing installation: yarl 1.18.3

    Uninstalling yarl-1.18.3:

      Successfully uninstalled yarl-1.18.3

  Attempting uninstall: typing-inspect

    Found existing installation: typing-inspect 0.9.0

    Uninstalling typing-inspect-0.9.0:

      Successfully uninstalled typing-inspect-0.9.0

  Attempting uninstall: treelib

    Found existing installation: treelib 1.7.0

    Uninstalling treelib-1.7.0:

      Successfully uninstalled treelib-1.7.0

  Attempting uninstall: requests

    Found existing installation: requests 2.32.3

    Uninstalling requests-2.32.3:

      Successfully uninstalled requests-2.32.3

  Attempting uninstall: python-dateutil

    Found existing installation: python-dateutil 2.9.0.post0

    Uninstalling python-dateutil-2.9.0.post0:

      Successfully uninstalled python-dateutil-2.9.0.post0

  Attempting uninstall: pydantic-core

    Found existing installation: pydantic_core 2.27.2

    Uninstalling pydantic_core-2.27.2:

      Successfully uninstalled pydantic_core-2.27.2

  Attempting uninstall: marshmallow

    Found existing installation: marshmallow 3.23.2

    Uninstalling marshmallow-3.23.2:

      Successfully uninstalled marshmallow-3.23.2

  Attempting uninstall: markdown-it-py

    Found existing installation: markdown-it-py 2.2.0

    Uninstalling markdown-it-py-2.2.0:

      Successfully uninstalled markdown-it-py-2.2.0

  Attempting uninstall: httpcore

    Found existing installation: httpcore 1.0.7

    Uninstalling httpcore-1.0.7:

      Successfully uninstalled httpcore-1.0.7

  Attempting uninstall: gitdb

    Found existing installation: gitdb 4.0.11

    Uninstalling gitdb-4.0.11:

      Successfully uninstalled gitdb-4.0.11

  Attempting uninstall: click

    Found existing installation: click 8.1.7

    Uninstalling click-8.1.7:

      Successfully uninstalled click-8.1.7

  Attempting uninstall: anyio

    Found existing installation: anyio 4.7.0

    Uninstalling anyio-4.7.0:

      Successfully uninstalled anyio-4.7.0

  Attempting uninstall: rich

    Found existing installation: rich 13.9.4

    Uninstalling rich-13.9.4:

      Successfully uninstalled rich-13.9.4

  Attempting uninstall: requests-toolbelt

    Found existing installation: requests-toolbelt 1.0.0

    Uninstalling requests-toolbelt-1.0.0:

      Successfully uninstalled requests-toolbelt-1.0.0

  Attempting uninstall: pydantic

    Found existing installation: pydantic 2.10.4

    Uninstalling pydantic-2.10.4:

      Successfully uninstalled pydantic-2.10.4

  Attempting uninstall: pandas

    Found existing installation: pandas 1.5.3

    Uninstalling pandas-1.5.3:

      Successfully uninstalled pandas-1.5.3

  Attempting uninstall: httpx

    Found existing installation: httpx 0.28.1

    Uninstalling httpx-0.28.1:

      Successfully uninstalled httpx-0.28.1

  Attempting uninstall: gql

    Found existing installation: gql 3.5.0

    Uninstalling gql-3.5.0:

      Successfully uninstalled gql-3.5.0

  Attempting uninstall: GitPython

    Found existing installation: GitPython 3.1.43

    Uninstalling GitPython-3.1.43:

      Successfully uninstalled GitPython-3.1.43

  Attempting uninstall: dataclasses-json

    Found existing installation: dataclasses-json 0.6.7

    Uninstalling dataclasses-json-0.6.7:

      Successfully uninstalled dataclasses-json-0.6.7

  Attempting uninstall: botocore

    Found existing installation: botocore 1.29.76

    Uninstalling botocore-1.29.76:

      Successfully uninstalled botocore-1.29.76

  Attempting uninstall: s3transfer

    Found existing installation: s3transfer 0.10.4

    Uninstalling s3transfer-0.10.4:

      Successfully uninstalled s3transfer-0.10.4

  Attempting uninstall: dagshub-annotation-converter

    Found existing installation: dagshub-annotation-converter 0.1.2

    Uninstalling dagshub-annotation-converter-0.1.2:

      Successfully uninstalled dagshub-annotation-converter-0.1.2

  Attempting uninstall: boto3

    Found existing installation: boto3 1.35.86

    Uninstalling boto3-1.35.86:

      Successfully uninstalled boto3-1.35.86

  Attempting uninstall: dagshub

    Found existing installation: dagshub 0.4.0

    Uninstalling dagshub-0.4.0:

      Successfully uninstalled dagshub-0.4.0

Successfully installed GitPython-3.1.43 PyYAML-6.0.2 annotated-types-0.7.0 anyio-4.7.0 appdirs-1.4.4 backoff-2.2.1 boto3-1.35.86 botocore-1.35.86 certifi-2024.12.14 charset-normalizer-3.4.0 click-8.1.7 colorama-0.4.6 dacite-1.6.0 dagshub-0.4.0 dagshub-annotation-converter-0.1.2 dataclasses-json-0.6.7 gitdb-4.0.11 gql-3.5.0 graphql-core-3.2.5 h11-0.14.0 httpcore-1.0.7 httpx-0.28.1 idna-3.10 jmespath-1.0.1 lxml-5.3.0 markdown-it-py-3.0.0 marshmallow-3.23.2 mdurl-0.1.2 multidict-6.1.0 mypy-extensions-1.0.0 numpy-2.2.0 packaging-24.2 pandas-2.2.3 pathvalidate-3.2.1 pillow-11.0.0 propcache-0.2.1 pydantic-2.10.4 pydantic-core-2.27.2 pygments-2.18.0 python-dateutil-2.9.0.post0 pytz-2024.2 requests-2.32.3 requests-toolbelt-1.0.0 rich-13.9.4 s3transfer-0.10.4 six-1.17.0 smmap-5.0.1 sniffio-1.3.1 tenacity-9.0.0 treelib-1.7.0 typing-extensions-4.12.2 typing-inspect-0.9.0 tzdata-2024.2 urllib3-2.2.3 yarl-1.18.3

name": "stderr
output_type": "stream
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

conda-repo-cli 1.0.75 requires requests_mock, which is not installed.

aiobotocore 2.5.0 requires botocore<1.29.77,>=1.29.76, but you have botocore 1.35.86 which is incompatible.

anaconda-cloud-auth 0.1.3 requires pydantic<2.0, but you have pydantic 2.10.4 which is incompatible.

catboost 1.2.7 requires numpy<2.0,>=1.16.0, but you have numpy 2.2.0 which is incompatible.

conda-repo-cli 1.0.75 requires clyent==1.2.1, but you have clyent 1.2.2 which is incompatible.

conda-repo-cli 1.0.75 requires python-dateutil==2.8.2, but you have python-dateutil 2.9.0.post0 which is incompatible.

conda-repo-cli 1.0.75 requires PyYAML==6.0.1, but you have pyyaml 6.0.2 which is incompatible.

conda-repo-cli 1.0.75 requires requests==2.31.0, but you have requests 2.32.3 which is incompatible.

mdit-py-plugins 0.3.0 requires markdown-it-py<3.0.0,>=1.0.0, but you have markdown-it-py 3.0.0 which is incompatible.

numba 0.57.1 requires numpy<1.25,>=1.21, but you have numpy 2.2.0 which is incompatible.

pyfume 0.3.4 requires numpy==1.24.4, but you have numpy 2.2.0 which is incompatible.

pyfume 0.3.4 requires pandas==1.5.3, but you have pandas 2.2.3 which is incompatible.

python-lsp-black 1.2.1 requires black>=22.3.0, but you have black 0.0 which is incompatible.

scipy 1.10.1 requires numpy<1.27.0,>=1.19.5, but you have numpy 2.2.0 which is incompatible.

ydata-profiling 4.12.1 requires dacite>=1.8, but you have dacite 1.6.0 which is incompatible.

ydata-profiling 4.12.1 requires numpy<2.2,>=1.16.0, but you have numpy 2.2.0 which is incompatible.

name": "stdout
output_type": "stream
Collecting botocore<1.29.77,>=1.29.76

  Using cached botocore-1.29.76-py3-none-any.whl.metadata (5.9 kB)

Requirement already satisfied: jmespath<2.0.0,>=0.7.1 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from botocore<1.29.77,>=1.29.76) (1.0.1)

Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from botocore<1.29.77,>=1.29.76) (2.9.0.post0)

Collecting urllib3<1.27,>=1.25.4 (from botocore<1.29.77,>=1.29.76)

  Using cached urllib3-1.26.20-py2.py3-none-any.whl.metadata (50 kB)

Requirement already satisfied: six>=1.5 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from python-dateutil<3.0.0,>=2.1->botocore<1.29.77,>=1.29.76) (1.17.0)

Using cached botocore-1.29.76-py3-none-any.whl (10.4 MB)

Using cached urllib3-1.26.20-py2.py3-none-any.whl (144 kB)

Installing collected packages: urllib3, botocore

  Attempting uninstall: urllib3

    Found existing installation: urllib3 2.2.3

    Uninstalling urllib3-2.2.3:

      Successfully uninstalled urllib3-2.2.3

  Attempting uninstall: botocore

    Found existing installation: botocore 1.35.86

    Uninstalling botocore-1.35.86:

      Successfully uninstalled botocore-1.35.86

Successfully installed botocore-1.29.76 urllib3-1.26.20

name": "stderr
output_type": "stream
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

boto3 1.35.86 requires botocore<1.36.0,>=1.35.86, but you have botocore 1.29.76 which is incompatible.

s3transfer 0.10.4 requires botocore<2.0a.0,>=1.33.2, but you have botocore 1.29.76 which is incompatible.

ydata-profiling 4.12.1 requires dacite>=1.8, but you have dacite 1.6.0 which is incompatible.

ydata-profiling 4.12.1 requires numpy<2.2,>=1.16.0, but you have numpy 2.2.0 which is incompatible.

name": "stdout
output_type": "stream
Collecting boto3==1.26.70

  Using cached boto3-1.26.70-py3-none-any.whl.metadata (7.0 kB)

Collecting s3transfer==0.5.2

  Using cached s3transfer-0.5.2-py3-none-any.whl.metadata (1.7 kB)

Requirement already satisfied: botocore<1.30.0,>=1.29.70 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from boto3==1.26.70) (1.29.76)

Requirement already satisfied: jmespath<2.0.0,>=0.7.1 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from boto3==1.26.70) (1.0.1)

INFO: pip is looking at multiple versions of boto3 to determine which version is compatible with other requirements. This could take a while.



The conflict is caused by:

    The user requested s3transfer==0.5.2

    boto3 1.26.70 depends on s3transfer<0.7.0 and >=0.6.0



To fix this you could try to:

1. loosen the range of package versions you've specified

2. remove package versions to allow pip to attempt to solve the dependency conflict



name": "stderr
output_type": "stream
ERROR: Cannot install boto3==1.26.70 and s3transfer==0.5.2 because these package versions have conflicting dependencies.

ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts

ERROR: Ignored the following versions that require a different python version: 1.21.2 Requires-Python >=3.7,<3.11; 1.21.3 Requires-Python >=3.7,<3.11; 1.21.4 Requires-Python >=3.7,<3.11; 1.21.5 Requires-Python >=3.7,<3.11; 1.21.6 Requires-Python >=3.7,<3.11

ERROR: Could not find a version that satisfies the requirement numpy==1.21.6 (from versions: 1.3.0, 1.4.1, 1.5.0, 1.5.1, 1.6.0, 1.6.1, 1.6.2, 1.7.0, 1.7.1, 1.7.2, 1.8.0, 1.8.1, 1.8.2, 1.9.0, 1.9.1, 1.9.2, 1.9.3, 1.10.0.post2, 1.10.1, 1.10.2, 1.10.4, 1.11.0, 1.11.1, 1.11.2, 1.11.3, 1.12.0, 1.12.1, 1.13.0, 1.13.1, 1.13.3, 1.14.0, 1.14.1, 1.14.2, 1.14.3, 1.14.4, 1.14.5, 1.14.6, 1.15.0, 1.15.1, 1.15.2, 1.15.3, 1.15.4, 1.16.0, 1.16.1, 1.16.2, 1.16.3, 1.16.4, 1.16.5, 1.16.6, 1.17.0, 1.17.1, 1.17.2, 1.17.3, 1.17.4, 1.17.5, 1.18.0, 1.18.1, 1.18.2, 1.18.3, 1.18.4, 1.18.5, 1.19.0, 1.19.1, 1.19.2, 1.19.3, 1.19.4, 1.19.5, 1.20.0, 1.20.1, 1.20.2, 1.20.3, 1.21.0, 1.21.1, 1.22.0, 1.22.1, 1.22.2, 1.22.3, 1.22.4, 1.23.0, 1.23.1, 1.23.2, 1.23.3, 1.23.4, 1.23.5, 1.24.0, 1.24.1, 1.24.2, 1.24.3, 1.24.4, 1.25.0, 1.25.1, 1.25.2, 1.26.0, 1.26.1, 1.26.2, 1.26.3, 1.26.4, 2.0.0, 2.0.1, 2.0.2, 2.1.0rc1, 2.1.0, 2.1.1, 2.1.2, 2.1.3, 2.2.0rc1, 2.2.0)

ERROR: No matching distribution found for numpy==1.21.6

name": "stdout
output_type": "stream
Requirement already satisfied: dacite==1.6.0 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (1.6.0)

Requirement already satisfied: FuzzyTM in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (2.0.9)

Requirement already satisfied: tables in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (3.8.0)

Requirement already satisfied: blosc2 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (2.0.0)

Requirement already satisfied: cython in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (3.0.11)

Collecting markdown-it-py==2.2.0

  Using cached markdown_it_py-2.2.0-py3-none-any.whl.metadata (6.8 kB)

Requirement already satisfied: mdurl~=0.1 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from markdown-it-py==2.2.0) (0.1.2)

Requirement already satisfied: numpy in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from FuzzyTM) (2.2.0)

Requirement already satisfied: pandas in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from FuzzyTM) (2.2.3)

Requirement already satisfied: scipy in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from FuzzyTM) (1.10.1)

Requirement already satisfied: pyfume in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from FuzzyTM) (0.3.4)

Requirement already satisfied: numexpr>=2.6.2 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from tables) (2.8.4)

Requirement already satisfied: packaging in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from tables) (24.2)

Requirement already satisfied: py-cpuinfo in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from tables) (8.0.0)

Requirement already satisfied: msgpack in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from blosc2) (1.0.3)

Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from pandas->FuzzyTM) (2.9.0.post0)

Requirement already satisfied: pytz>=2020.1 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from pandas->FuzzyTM) (2024.2)

Requirement already satisfied: tzdata>=2022.7 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from pandas->FuzzyTM) (2024.2)

Collecting numpy (from FuzzyTM)

  Using cached numpy-1.24.4-cp311-cp311-win_amd64.whl.metadata (5.6 kB)

Requirement already satisfied: simpful==2.12.0 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from pyfume->FuzzyTM) (2.12.0)

Requirement already satisfied: fst-pso==1.8.1 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from pyfume->FuzzyTM) (1.8.1)

Collecting pandas (from FuzzyTM)

  Using cached pandas-1.5.3-cp311-cp311-win_amd64.whl.metadata (12 kB)

Requirement already satisfied: miniful in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from fst-pso==1.8.1->pyfume->FuzzyTM) (0.0.6)

Requirement already satisfied: six>=1.5 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from python-dateutil>=2.8.2->pandas->FuzzyTM) (1.17.0)

Using cached markdown_it_py-2.2.0-py3-none-any.whl (84 kB)

Using cached numpy-1.24.4-cp311-cp311-win_amd64.whl (14.8 MB)

Using cached pandas-1.5.3-cp311-cp311-win_amd64.whl (10.3 MB)

Installing collected packages: numpy, markdown-it-py, pandas

  Attempting uninstall: numpy

    Found existing installation: numpy 2.2.0

    Uninstalling numpy-2.2.0:

      Successfully uninstalled numpy-2.2.0

  Attempting uninstall: markdown-it-py

    Found existing installation: markdown-it-py 3.0.0

    Uninstalling markdown-it-py-3.0.0:

      Successfully uninstalled markdown-it-py-3.0.0

  Attempting uninstall: pandas

    Found existing installation: pandas 2.2.3

    Uninstalling pandas-2.2.3:

      Successfully uninstalled pandas-2.2.3

Successfully installed markdown-it-py-2.2.0 numpy-1.24.4 pandas-1.5.3

All critical libraries installed successfully!

name": "stderr
output_type": "stream
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

visions 0.7.6 requires pandas>=2.0.0, but you have pandas 1.5.3 which is incompatible.

ydata-profiling 4.12.1 requires dacite>=1.8, but you have dacite 1.6.0 which is incompatible.

# Step 1: Upgrade pip

!python -m pip install --upgrade pip



# Step 2: Reinstall dagshub with forced dependency resolution

!pip install --force-reinstall dagshub==0.4.0



# Step 3: Resolve specific dependency conflicts

# Install compatible version of botocore for aiobotocore

!pip install \"botocore>=1.29.76,<1.29.77\"



# Install compatible versions of boto3 and s3transfer

!pip install boto3==1.26.70 s3transfer==0.5.2



# Install compatible version of numpy for ydata-profiling and scikit-learn

!pip install numpy==1.21.6



# Install dacite for ydata-profiling compatibility

!pip install dacite==1.6.0



# Resolve missing dependencies

!pip install FuzzyTM tables blosc2 cython markdown-it-py==2.2.0



# Step 4: Verify installation of dagshub and other critical libraries

try:

    import dagshub

    import mlflow

    import numpy

    import pandas

    print(\"All critical libraries installed successfully!\")

except ImportError as e:

    print(\"Error:\", e)

cell_type": "code
id": "9afe1da5
<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Initialized MLflow to track repo <span style=\"color: #008000; text-decoration-color: #008000\">\"yashaswiniguntupalli/ML_Final_Project\"</span>

</pre>

Initialized MLflow to track repo \u001b[32m\"yashaswiniguntupalli/ML_Final_Project\"\u001b[0m

output_type": "display_data
<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Repository yashaswiniguntupalli/ML_Final_Project initialized!

</pre>

Repository yashaswiniguntupalli/ML_Final_Project initialized!

output_type": "display_data
# Base for the experiment 2 creating and setting up the dagshub

import dagshub



# Initialize DagsHub for your repository

dagshub.init(

    repo_owner=\"yashaswiniguntupalli\",

    repo_name=\"ML_Final_Project\",

    mlflow=True

)

cell_type": "code
id": "f63bc16b
import mlflow



mlflow.set_tracking_uri(\"https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow\")

cell_type": "code
id": "c1fbd522
import os



os.environ[\"MLFLOW_TRACKING_USERNAME\"] = \"yashaswiniguntupalli\"

os.environ[\"MLFLOW_TRACKING_PASSWORD\"] = \"dd928eb3e01ad92df47ae00f812f06a28ddc8c95\"

cell_type": "code
id": "5d90ce35
<Experiment: artifact_location='mlflow-artifacts:/7b68b41f9a1d48509df849e83e379337', creation_time=1734686289365, experiment_id='0', last_update_time=1734686289365, lifecycle_stage='active', name='Experiment_2', tags={}>
output_type": "execute_result
experiment_name = \"Experiment_2\"



# Create the experiment if it doesn't exist

if not mlflow.get_experiment_by_name(experiment_name):

    mlflow.create_experiment(experiment_name)



# Set the experiment

mlflow.set_experiment(experiment_name)

cell_type": "code
id": "96526825
name": "stdout
output_type": "stream
MLFlow tracking is working!

🏃 View run carefree-stag-447 at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/1/runs/b8093a20cb4440f49861ff87e8460424

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/1

# Testing

import mlflow



# Verify MLFlow connection

mlflow.set_tracking_uri(\"https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow\")

mlflow.set_experiment(\"Test_Experiment\")



with mlflow.start_run():

    mlflow.log_param(\"param1\", 5)

    mlflow.log_metric(\"metric1\", 0.85)

    print(\"MLFlow tracking is working!\")

cell_type": "code
id": "455c55bb
name": "stdout
output_type": "stream
Running LogisticRegression...

Confusion Matrix for LogisticRegression:

 [[4603  342]

 [ 596  972]]

Classification Report for LogisticRegression:

               precision    recall  f1-score   support



           0       0.89      0.93      0.91      4945

           1       0.74      0.62      0.67      1568



    accuracy                           0.86      6513

   macro avg       0.81      0.78      0.79      6513

weighted avg       0.85      0.86      0.85      6513



name": "stderr
output_type": "stream
C:\\Users\\lenovo\\anaconda3\\Lib\\site-packages\\mlflow\\types\\utils.py:435: UserWarning: Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. If your input data contains missing values at inference time, it will be encoded as floats and will cause a schema enforcement error. The best way to avoid this problem is to infer the model schema based on a realistic data sample (training dataset) that includes missing values. Alternatively, you can declare integer columns as doubles (float64) whenever these columns may have missing values. See `Handling Integers With Missing Values <https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values>`_ for more details.

  warnings.warn(

Registered model 'LogisticRegression' already exists. Creating a new version of this model...

2024/12/20 18:49:18 INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 seconds for model version to finish creation. Model name: LogisticRegression, version 25

Created version '25' of model 'LogisticRegression'.

name": "stdout
output_type": "stream
LogisticRegression logged successfully with Mean CV F1 Score: 0.781408951344656 and Test F1 Score: 0.7910315605181295

🏃 View run LogisticRegression at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/7/runs/5a572f45569041f59e71c435247e257b

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/7

Running RandomForestClassifier...

Confusion Matrix for RandomForestClassifier:

 [[4595  350]

 [ 567 1001]]

Classification Report for RandomForestClassifier:

               precision    recall  f1-score   support



           0       0.89      0.93      0.91      4945

           1       0.74      0.64      0.69      1568



    accuracy                           0.86      6513

   macro avg       0.82      0.78      0.80      6513

weighted avg       0.85      0.86      0.86      6513



name": "stderr
output_type": "stream
C:\\Users\\lenovo\\anaconda3\\Lib\\site-packages\\mlflow\\types\\utils.py:435: UserWarning: Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. If your input data contains missing values at inference time, it will be encoded as floats and will cause a schema enforcement error. The best way to avoid this problem is to infer the model schema based on a realistic data sample (training dataset) that includes missing values. Alternatively, you can declare integer columns as doubles (float64) whenever these columns may have missing values. See `Handling Integers With Missing Values <https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values>`_ for more details.

  warnings.warn(

Registered model 'RandomForestClassifier' already exists. Creating a new version of this model...

2024/12/20 18:52:37 INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 seconds for model version to finish creation. Model name: RandomForestClassifier, version 23

Created version '23' of model 'RandomForestClassifier'.

name": "stdout
output_type": "stream
RandomForestClassifier logged successfully with Mean CV F1 Score: 0.7833683039934529 and Test F1 Score: 0.7975610606795063

🏃 View run RandomForestClassifier at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/7/runs/c233e0a1fd1c479c95a1b40d4d1cd12d

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/7

# Code for Experiment 2:Create a pipeline for preprocessing and use LogisticRegression, RidgeClassifier, RandomForestClassifier

# Import required libraries

# Import required libraries

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, f1_score

import mlflow

import mlflow.sklearn



# Load Dataset

data = pd.read_csv(\"adult_income.csv\")  # Replace with the actual dataset path

X = data.drop(\"income\", axis=1)  # Replace \"income\" with your target column name

y = data[\"income\"]



# Encode target variable

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)



# Preprocess Features

numerical_cols = X.select_dtypes(include=[\"int64\", \"float64\"]).columns

categorical_cols = X.select_dtypes(include=[\"object\", \"category\"]).columns



numerical_transformer = Pipeline(steps=[

    (\"scaler\", StandardScaler())

])



categorical_transformer = Pipeline(steps=[

    (\"onehot\", OneHotEncoder(handle_unknown=\"ignore\"))

])



preprocessor = ColumnTransformer(

    transformers=[

        (\"num\", numerical_transformer, numerical_cols),

        (\"cat\", categorical_transformer, categorical_cols)

    ]

)



# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42, stratify=y

)



# Set up MLflow Tracking

mlflow.set_tracking_uri(\"https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow\")

experiment_name = \"Experiment_No_XGBoost\"

mlflow.set_experiment(experiment_name)



# Train and Log Models

models = {

    \"LogisticRegression\": LogisticRegression(max_iter=1000),

    \"RandomForestClassifier\": RandomForestClassifier(n_estimators=100, random_state=42)

}



for name, model in models.items():

    print(f\"Running {name}...\")

    pipeline = Pipeline([

        (\"preprocessor\", preprocessor),

        (\"classifier\", model)

    ])

    

    try:

        with mlflow.start_run(run_name=name):

            # Perform cross-validation

            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=\"f1_macro\")

            mlflow.log_metric(\"Mean CV F1 Score\", cv_scores.mean())

            

            # Train the model

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            

            # Calculate and log metrics

            f1 = f1_score(y_test, y_pred, average=\"macro\")

            mlflow.log_metric(\"Test F1 Score\", f1)

            

            # Log confusion matrix and classification report

            conf_matrix = confusion_matrix(y_test, y_pred)

            print(f\"Confusion Matrix for {name}:\
\", conf_matrix)

            print(f\"Classification Report for {name}:\
\", classification_report(y_test, y_pred))

            

            # Log the model

            mlflow.sklearn.log_model(

                sk_model=pipeline,

                artifact_path=\"model\",

                registered_model_name=name,

                signature=mlflow.models.infer_signature(X_test, y_pred)

            )

            

            print(f\"{name} logged successfully with Mean CV F1 Score: {cv_scores.mean()} and Test F1 Score: {f1}\")

    except Exception as e:

        print(f\"Error during {name}: {str(e)}\")

        if mlflow.active_run():

            mlflow.end_run()



# Ensure no active runs are left

if mlflow.active_run():

    mlflow.end_run()

cell_type": "code
id": "4f0b0f7e
name": "stderr
output_type": "stream
C:\\Users\\lenovo\\anaconda3\\Lib\\site-packages\\xgboost\\core.py:158: UserWarning: [18:53:12] WARNING: C:\\buildkite-agent\\builds\\buildkite-windows-cpu-autoscaling-group-i-0c55ff5f71b100e98-1\\xgboost\\xgboost-ci-windows\\src\\learner.cc:740: 

Parameters: { \"use_label_encoder\" } are not used.



  warnings.warn(smsg, UserWarning)

name": "stdout
output_type": "stream
F1 Score: 0.8169932733372401

Confusion Matrix:

 [[4630  315]

 [ 517 1051]]

Classification Report:

               precision    recall  f1-score   support



           0       0.90      0.94      0.92      4945

           1       0.77      0.67      0.72      1568



    accuracy                           0.87      6513

   macro avg       0.83      0.80      0.82      6513

weighted avg       0.87      0.87      0.87      6513



name": "stderr
output_type": "stream
2024/12/20 18:53:19 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.

Registered model 'XGBoost_Classifier' already exists. Creating a new version of this model...

2024/12/20 18:53:22 INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 seconds for model version to finish creation. Model name: XGBoost_Classifier, version 4

Created version '4' of model 'XGBoost_Classifier'.

name": "stdout
output_type": "stream
XGBoost model logged successfully.

🏃 View run XGBoost_Classifier at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/8/runs/0a72a44a3a0f4078aa1bf518055b60ba

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/8

# Experiment-2, Create a pipeline for preprocessing and use XGBClassifier

# Import required libraries

# Import required libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix, f1_score

from xgboost import XGBClassifier

import mlflow

import mlflow.sklearn



# Load Dataset

data = pd.read_csv(\"adult_income.csv\")  # Replace with the actual dataset path



# Encode target variable

target_column = \"income\"  # Replace with your target column name

label_encoder = LabelEncoder()

data[target_column] = label_encoder.fit_transform(data[target_column])



# Convert categorical columns to 'category' dtype

categorical_cols = data.select_dtypes(include=[\"object\"]).columns

data[categorical_cols] = data[categorical_cols].astype(\"category\")



# Split features and target

X = data.drop(target_column, axis=1)

y = data[target_column]



# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42, stratify=y

)



# Set up MLflow Tracking

mlflow.set_tracking_uri(\"https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow\")

experiment_name = \"Experiment_XGBoost\"

mlflow.set_experiment(experiment_name)



# Train and Evaluate XGBoost Classifier

xgb_model = XGBClassifier(

    use_label_encoder=False,

    eval_metric=\"logloss\",

    enable_categorical=True,

    random_state=42

)



try:

    with mlflow.start_run(run_name=\"XGBoost_Classifier\"):

        # Train the model

        xgb_model.fit(X_train, y_train)

        

        # Predict on the test set

        y_pred = xgb_model.predict(X_test)

        

        # Calculate metrics

        f1 = f1_score(y_test, y_pred, average=\"macro\")

        conf_matrix = confusion_matrix(y_test, y_pred)

        class_report = classification_report(y_test, y_pred)

        

        # Log metrics to MLflow

        mlflow.log_metric(\"F1 Score\", f1)

        

        # Print metrics

        print(\"F1 Score:\", f1)

        print(\"Confusion Matrix:\
\", conf_matrix)

        print(\"Classification Report:\
\", class_report)

        

        # Log the model to MLflow

        mlflow.sklearn.log_model(

            sk_model=xgb_model,

            artifact_path=\"model\",

            registered_model_name=\"XGBoost_Classifier\"

        )

        print(\"XGBoost model logged successfully.\")

except Exception as e:

    print(\"Error during training or logging:\", e)

    if mlflow.active_run():

        mlflow.end_run()



# Ensure no active runs are left

if mlflow.active_run():

    mlflow.end_run()

cell_type": "code
id": "3e345881
<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Initialized MLflow to track repo <span style=\"color: #008000; text-decoration-color: #008000\">\"yashaswiniguntupalli/ML_Final_Project\"</span>

</pre>

Initialized MLflow to track repo \u001b[32m\"yashaswiniguntupalli/ML_Final_Project\"\u001b[0m

output_type": "display_data
<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Repository yashaswiniguntupalli/ML_Final_Project initialized!

</pre>

Repository yashaswiniguntupalli/ML_Final_Project initialized!

output_type": "display_data
name": "stderr
output_type": "stream
2024/12/20 15:14:47 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.

name": "stdout
output_type": "stream
🏃 View run LogisticRegression at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/9/runs/7fc9e1d739ed4aab8502b0249b223457

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/9

name": "stderr
output_type": "stream
2024/12/20 15:15:01 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.

name": "stdout
output_type": "stream
🏃 View run RandomForestClassifier at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/9/runs/ab3fd6ec915548c98d845c37dc978fbf

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/9

# Experiment 3  Perform feature engineering and attribute combination. Log results in MLFlow.

# Experiment 3: Logistic Regression and Random Forest

import os

import dagshub

import pandas as pd

import mlflow

import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, f1_score

import matplotlib.pyplot as plt

import seaborn as sns



# Step 1: Initialize DagsHub MLFlow connection

dagshub.init(repo_owner=\"yashaswiniguntupalli\", repo_name=\"ML_Final_Project\", mlflow=True)



os.environ[\"MLFLOW_TRACKING_USERNAME\"] = \"yashaswiniguntupalli\"

os.environ[\"MLFLOW_TRACKING_PASSWORD\"] = \"dd928eb3e01ad92df47ae00f812f06a28ddc8c95\"



mlflow.set_tracking_uri(\"https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow\")

mlflow.set_experiment(\"Experiment_3_LR_RF\")



# Step 2: Load Dataset

file_path = \"adult_income.csv\"

data = pd.read_csv(file_path)



# Step 3: Feature Engineering

data['capital_diff'] = data['capital_gain'] - data['capital_loss']

data['age_income_ratio'] = data['age'] / (data['hours_per_week'] + 1)

data['hours_category'] = pd.cut(data['hours_per_week'], bins=[0, 20, 40, 60, 100], labels=[\"Low\", \"Medium\", \"High\", \"Very High\"])



# Convert target variable to numerical labels

X = data.drop('income', axis=1)

y = LabelEncoder().fit_transform(data['income'])



# Convert integer columns to float

X = X.astype({col: \"float\" for col in X.select_dtypes(include=\"int64\").columns})



# Identify numerical and categorical columns

numerical_cols = X.select_dtypes(include=[\"float64\"]).columns

categorical_cols = X.select_dtypes(include=[\"object\", \"category\"]).columns



# Preprocessing pipelines

numerical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean')),

    ('scaler', StandardScaler())

])



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ]

)



# Step 4: Define Classifiers

classifiers = {

    \"LogisticRegression\": LogisticRegression(max_iter=1000, random_state=42),

    \"RandomForestClassifier\": RandomForestClassifier(n_estimators=100, random_state=42)

}



# Step 5: Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)



# Step 6: Train and Log Models

for name, clf in classifiers.items():

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

    with mlflow.start_run(run_name=name):

        try:

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)



            # Calculate Metrics

            f1 = f1_score(y_test, y_pred, average=\"weighted\")

            conf_matrix = confusion_matrix(y_test, y_pred)



            # Log Metrics

            mlflow.log_metric(\"F1_Score\", f1)



            # Log Model

            mlflow.sklearn.log_model(pipeline, name)



            # Plot Confusion Matrix

            plt.figure(figsize=(6, 4))

            sns.heatmap(conf_matrix, annot=True, fmt=\"d\", cmap=\"Blues\")

            plt.title(f\"Confusion Matrix for {name}\")

            plt.xlabel(\"Predicted\")

            plt.ylabel(\"True\")

            plot_path = f\"{name}_confusion_matrix.png\"

            plt.savefig(plot_path)

            plt.close()

            mlflow.log_artifact(plot_path)

        except Exception as e:

            print(f\"Error with {name}: {e}\")

        finally:

            mlflow.end_run()

cell_type": "code
id": "c232b27f
name": "stderr
output_type": "stream
C:\\Users\\lenovo\\anaconda3\\Lib\\site-packages\\xgboost\\core.py:158: UserWarning: [18:54:49] WARNING: C:\\buildkite-agent\\builds\\buildkite-windows-cpu-autoscaling-group-i-0c55ff5f71b100e98-1\\xgboost\\xgboost-ci-windows\\src\\learner.cc:740: 

Parameters: { \"use_label_encoder\" } are not used.



  warnings.warn(smsg, UserWarning)

name": "stdout
output_type": "stream
F1 Score: 0.8691364704222599

Confusion Matrix:

 [[4630  315]

 [ 517 1051]]

Classification Report:

               precision    recall  f1-score   support



           0       0.90      0.94      0.92      4945

           1       0.77      0.67      0.72      1568



    accuracy                           0.87      6513

   macro avg       0.83      0.80      0.82      6513

weighted avg       0.87      0.87      0.87      6513



name": "stderr
output_type": "stream
2024/12/20 18:54:55 WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.

name": "stdout
output_type": "stream
🏃 View run XGBoost at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/11/runs/b88ab5d3624447c8a5dea2a01d2aa5da

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/11

#Experiment-3, Perform feature engineering and attribute combination. Log results in MLFlow. XGB

# Import necessary libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix, f1_score

from xgboost import XGBClassifier

import mlflow

import mlflow.sklearn



# Step 1: Load Dataset

data = pd.read_csv(\"adult_income.csv\")  # Replace with your dataset path



# Encode target variable

label_encoder = LabelEncoder()

data[\"income\"] = label_encoder.fit_transform(data[\"income\"])  # Replace \"income\" with your target column name



# Split features and target

X = data.drop(\"income\", axis=1)

y = data[\"income\"]



# Handle categorical data for XGBoost

categorical_cols = X.select_dtypes(include=[\"object\"]).columns

X[categorical_cols] = X[categorical_cols].astype(\"category\")



# Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



# Step 2: Define Basic XGBoost Model

xgb_clf = XGBClassifier(

    use_label_encoder=False,

    eval_metric=\"logloss\",

    enable_categorical=True,

    random_state=42

)



# Step 3: Train and Evaluate Model

mlflow.set_experiment(\"Basic_XGBoost_Experiment\")



with mlflow.start_run(run_name=\"XGBoost\"):

    try:

        # Train the model

        xgb_clf.fit(X_train, y_train)



        # Predict

        y_pred = xgb_clf.predict(X_test)



        # Calculate metrics

        f1 = f1_score(y_test, y_pred, average=\"weighted\")

        conf_matrix = confusion_matrix(y_test, y_pred)



        # Log metrics

        mlflow.log_metric(\"F1_Score\", f1)



        print(\"F1 Score:\", f1)

        print(\"Confusion Matrix:\
\", conf_matrix)

        print(\"Classification Report:\
\", classification_report(y_test, y_pred))



        # Log model

        mlflow.sklearn.log_model(xgb_clf, \"XGBoost_Model\")



    except Exception as e:

        print(f\"Error with XGBoost: {e}\")



    finally:

        if mlflow.active_run():

            mlflow.end_run()

cell_type": "code
id": "73a4045c
<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Initialized MLflow to track repo <span style=\"color: #008000; text-decoration-color: #008000\">\"yashaswiniguntupalli/ML_Final_Project\"</span>

</pre>

Initialized MLflow to track repo \u001b[32m\"yashaswiniguntupalli/ML_Final_Project\"\u001b[0m

output_type": "display_data
<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Repository yashaswiniguntupalli/ML_Final_Project initialized!

</pre>

Repository yashaswiniguntupalli/ML_Final_Project initialized!

output_type": "display_data
name": "stdout
output_type": "stream
Features dropped due to correlation: ['x6_ Male']

Features dropped due to low variance: {'x7_ Outlying-US(Guam-USVI-etc)', 'x7_ Germany', 'x7_ Yugoslavia', 'x7_ Ecuador', 'x7_ France', 'x7_ Italy', 'x7_ Japan', 'x5_ Amer-Indian-Eskimo', 'x0_ Never-worked', 'x7_ Guatemala', 'x7_ Holand-Netherlands', 'x5_ Other', 'x7_ England', 'x7_ Portugal', 'x7_ Jamaica', 'x7_ Peru', 'x2_ Married-AF-spouse', 'x7_ Columbia', 'x7_ Nicaragua', 'x7_ Trinadad&Tobago', 'x7_ Cuba', 'x7_ Haiti', 'x7_ Thailand', 'x3_ Priv-house-serv', 'x7_ Scotland', 'x7_ Cambodia', 'x7_ Poland', 'x0_ Without-pay', 'x7_ Hungary', 'x7_ Taiwan', 'x7_ China', 'x7_ Vietnam', 'x7_ Hong', 'x7_ Puerto-Rico', 'x7_ El-Salvador', 'x7_ Ireland', 'x7_ South', 'x3_ Armed-Forces', 'x7_ Greece', 'x7_ Dominican-Republic', 'x1_ Preschool', 'x1_ 1st-4th', 'x7_ Philippines', 'x7_ Laos', 'x7_ Honduras', 'x7_ Canada', 'x7_ India', 'x7_ Iran'}

Selected features based on importance: Index(['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',

       'hours_per_week', 'x0_ Private', 'x1_ Bachelors',

       'x2_ Married-civ-spouse', 'x2_ Never-married', 'x3_ Exec-managerial',

       'x3_ Prof-specialty', 'x4_ Husband', 'x4_ Wife', 'x6_ Female'],

      dtype='object')

model_id": "2e20b7bb277b4239ad81f14f6523061a
Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]
output_type": "display_data
name": "stdout
output_type": "stream


LogisticRegression Results:



Confusion Matrix:

[[6919  498]

 [ 975 1377]]



Classification Report:

              precision    recall  f1-score   support



       <=50K       0.88      0.93      0.90      7417

        >50K       0.73      0.59      0.65      2352



    accuracy                           0.85      9769

   macro avg       0.81      0.76      0.78      9769

weighted avg       0.84      0.85      0.84      9769



🏃 View run LogisticRegression at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/3/runs/df1d7b6a099a4bb68da07873d5dd4e3b

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/3

model_id": "08aa584af047482292014c13c5483496
Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]
output_type": "display_data
name": "stdout
output_type": "stream


RandomForestClassifier Results:



Confusion Matrix:

[[6839  578]

 [ 894 1458]]



Classification Report:

              precision    recall  f1-score   support



       <=50K       0.88      0.92      0.90      7417

        >50K       0.72      0.62      0.66      2352



    accuracy                           0.85      9769

   macro avg       0.80      0.77      0.78      9769

weighted avg       0.84      0.85      0.85      9769



🏃 View run RandomForestClassifier at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/3/runs/c7759ab2ca2942b98934148d0847485e

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/3

name": "stderr
output_type": "stream
C:\\Users\\lenovo\\anaconda3\\Lib\\site-packages\\xgboost\\core.py:158: UserWarning: [18:56:51] WARNING: C:\\buildkite-agent\\builds\\buildkite-windows-cpu-autoscaling-group-i-0c55ff5f71b100e98-1\\xgboost\\xgboost-ci-windows\\src\\learner.cc:740: 

Parameters: { \"use_label_encoder\" } are not used.



  warnings.warn(smsg, UserWarning)

C:\\Users\\lenovo\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_tags.py:354: FutureWarning: The CustomXGBClassifier or classes from which it inherits use `_get_tags` and `_more_tags`. Please define the `__sklearn_tags__` method, or inherit from `sklearn.base.BaseEstimator` and/or other appropriate mixins such as `sklearn.base.TransformerMixin`, `sklearn.base.ClassifierMixin`, `sklearn.base.RegressorMixin`, and `sklearn.base.OutlierMixin`. From scikit-learn 1.7, not defining `__sklearn_tags__` will raise an error.

  warnings.warn(

name": "stdout
output_type": "stream
Error encountered with XGBClassifier: 'super' object has no attribute '__sklearn_tags__'

🏃 View run XGBClassifier at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/3/runs/5a32d6c696074e96a610728e9dfa3a50

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/3

# code for experiment 4 : Perform feature selection using Correlation Threshold, Feature Importance, and Variance Threshold. Log results in MLFlow.

# Import required libraries

# Import required libraries

import os

import dagshub

import pandas as pd

import numpy as np

import mlflow

import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, f1_score

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin



# Custom wrapper for XGBClassifier to include __sklearn_tags__

class CustomXGBClassifier(XGBClassifier, BaseEstimator, ClassifierMixin):

    def __sklearn_tags__(self):

        \"\"\"Define custom tags for compatibility with scikit-learn >=1.7.\"\"\"

        return {

            \"binary_only\": False,

            \"multilabel\": False,

            \"multiclass\": True,

            \"poor_score\": False,

            \"no_validation\": False,

        }



# Step 1: Initialize DagsHub MLFlow connection

dagshub.init(repo_owner=\"yashaswiniguntupalli\", repo_name=\"ML_Final_Project\", mlflow=True)



os.environ[\"MLFLOW_TRACKING_USERNAME\"] = \"yashaswiniguntupalli\"

os.environ[\"MLFLOW_TRACKING_PASSWORD\"] = \"dd928eb3e01ad92df47ae00f812f06a28ddc8c95\"



mlflow.set_tracking_uri(\"https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow\")



experiment_name = \"Experiment_4\"

if not mlflow.get_experiment_by_name(experiment_name):

    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)



# Step 2: Load Dataset

file_path = \"adult_income.csv\"  # Replace with your file path

data = pd.read_csv(file_path)



# Step 3: Preprocessing and Feature Engineering

X = data.drop('income', axis=1)

y = data['income']



# Convert target variable to numerical labels

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)



# Identify numerical and categorical columns

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

categorical_cols = X.select_dtypes(include=['object', 'category']).columns



# Preprocessing pipelines

numerical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean')),

    ('scaler', StandardScaler())

])



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ]

)



# Step 4: Feature Selection Methods

def apply_correlation_threshold(X, threshold=0.9):

    \"\"\"Remove features with high correlation.\"\"\"

    corr_matrix = X.corr().abs()

    upper_triangle = corr_matrix.where(

        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

    )

    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    print(f\"Features dropped due to correlation: {to_drop}\")

    return X.drop(columns=to_drop)



def apply_variance_threshold(X, threshold=0.01):

    \"\"\"Remove low variance features.\"\"\"

    selector = VarianceThreshold(threshold)

    X_selected = selector.fit_transform(X)

    selected_features = X.columns[selector.get_support()]

    print(f\"Features dropped due to low variance: {set(X.columns) - set(selected_features)}\")

    return pd.DataFrame(X_selected, columns=selected_features)



def apply_feature_importance(X, y, threshold=0.01):

    \"\"\"Select features based on importance from RandomForestClassifier.\"\"\"

    clf = RandomForestClassifier(random_state=42)

    clf.fit(X, y)

    importances = pd.Series(clf.feature_importances_, index=X.columns)

    selected_features = importances[importances > threshold].index

    print(f\"Selected features based on importance: {selected_features}\")

    return X[selected_features]



# Step 5: Apply Feature Selection

X_processed = pd.DataFrame(

    preprocessor.fit_transform(X).toarray(),

    columns=numerical_cols.tolist() + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out())

)



X_corr_filtered = apply_correlation_threshold(X_processed)  # Correlation Threshold

X_var_filtered = apply_variance_threshold(X_corr_filtered)  # Variance Threshold

X_importance_filtered = apply_feature_importance(X_var_filtered, y)  # Feature Importance



# Step 6: Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X_importance_filtered, y, test_size=0.3, random_state=42, stratify=y)



# Step 7: Define Classifiers

classifiers = {

    \"LogisticRegression\": LogisticRegression(max_iter=1000, random_state=42),

    \"RandomForestClassifier\": RandomForestClassifier(n_estimators=100, random_state=42),

    \"XGBClassifier\": CustomXGBClassifier(use_label_encoder=False, eval_metric=\"logloss\", random_state=42),

}



pipelines = {name: Pipeline(steps=[('classifier', clf)]) for name, clf in classifiers.items()}



# Step 8: Train and Log Models

for name, pipeline in pipelines.items():

    with mlflow.start_run(run_name=name):

        try:

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)



            # Calculate Metrics

            f1 = f1_score(y_test, y_pred, average=\"weighted\")

            conf_matrix = confusion_matrix(y_test, y_pred)



            # Log Metrics and Model

            mlflow.log_param(\"Model\", name)

            mlflow.log_metric(\"F1_Score\", f1)



            # Create a valid input example

            input_example = pd.DataFrame([X_test.iloc[0]], columns=X_test.columns)



            # Log the model

            mlflow.sklearn.log_model(

                sk_model=pipeline,

                artifact_path=name,

                input_example=input_example,

                signature=mlflow.models.infer_signature(X_test, y_pred),

            )



            # Plot Confusion Matrix

            plt.figure(figsize=(6, 4))

            sns.heatmap(conf_matrix, annot=True, fmt=\"d\", cmap=\"Blues\",

                        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)

            plt.title(f\"Confusion Matrix for {name}\")

            plt.xlabel(\"Predicted\")

            plt.ylabel(\"True\")

            plot_path = f\"{name}_confusion_matrix.png\"

            plt.savefig(plot_path)

            plt.close()

            mlflow.log_artifact(plot_path)



            print(f\"\
{name} Results:\")

            print(\"\
Confusion Matrix:\")

            print(conf_matrix)

            print(\"\
Classification Report:\")

            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))



        except Exception as e:

            print(f\"Error encountered with {name}: {e}\")

cell_type": "code
id": "f3850c2b
<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Initialized MLflow to track repo <span style=\"color: #008000; text-decoration-color: #008000\">\"yashaswiniguntupalli/ML_Final_Project\"</span>

</pre>

Initialized MLflow to track repo \u001b[32m\"yashaswiniguntupalli/ML_Final_Project\"\u001b[0m

output_type": "display_data
<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Repository yashaswiniguntupalli/ML_Final_Project initialized!

</pre>

Repository yashaswiniguntupalli/ML_Final_Project initialized!

output_type": "display_data
image/png": "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b/LkyWrdurXGjh0rSWrQoIEqVqyotm3b6oUXXlBsLCGpJDmzckHrayPZphYAALic02H266+/Vv/+/XX69GmFhITY9aKaTCaHw2yFChXUtGlTLV26VHfccYetfenSperVq1ehrzlz5oz8/OxL9vX1lXShRxcly9mVCwAAAFzN6TGzo0eP1pAhQ5Sdna2TJ0/qxIkTtsfx48edutaoUaP07rvv6v3339euXbv0+OOPKy0tTUOHDpV0YbzrxeG4R48eWrBggWbOnKl9+/Zp7dq1GjFihJo3b65q1ao5+1FwlRxdkYCVCwAAgLs43TN76NAhjRgxQsHBwVf95n379tWxY8c0adIkmc1mJSYmavHixYqLi5Mkmc1muzVnBw0apOzsbL355psaPXq0KleurI4dO2rq1KlXXQscY7Ea2pR6XBnZOYqsGKDgCr46k1f4+sKsXAAAANzNZDj5+/k777xTd999t/r06eOumtwqKytLYWFhyszMVGhoqKfL8SrOLsElia1qAQCA05zJa073zN52220aO3asdu7cqfr168vf39/ueM+ePZ29JLxA/hJcRf3Np3Kwv06eOWd7zsoFAACgJDjdM+vjU/QwW5PJ5PSWtiWNnlnnWayG2kxdXmSPrElSdGiAXunTSEdP5SoqJJCVCwAAQLG5tWf20qW4UPZdaQkuQ1J6Vq58TCb1alS95AoDAADlntOrGaD8cXQJLkfPAwAAcJVibZpw+vRprVy5UmlpacrLy7M7NmLECJcUhtKDJbgAAEBp5XSYTUlJ0a233qozZ87o9OnTCg8P19GjRxUcHKyoqCjCbBnUPD5cMaGBSs8qeswsS3ABAABPcHqYweOPP64ePXro+PHjCgoK0oYNG/Tnn3+qadOm+te//uWOGuEhFquh9XuP6ZtfD+u6mEqFnpM/xSupRwITvgAAQIlzumd269atmjVrlnx9feXr66vc3FzVqVNH06ZN08CBA3XnnXe6o06UsKLWlL10kwSW4AIAAJ7kdJj19/eXyXShBy46OlppaWmqV6+ewsLC7Hbrgve63JqyZ/IserxTXdWOrMgSXAAAwOOcDrONGzfW5s2bdd111+nmm2/Ws88+q6NHj+qjjz5S/fr13VEjSpDFamji1zuL3BzBJOmznw9ozbiOhFgAAOBxTo+ZfemllxQbe+FXys8//7wiIiI0bNgwZWRk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rp599llt2LDBHTWWW2nHzkiSaoUHe7gSAACA0snpMJuenq769etLkipVqmTbQOH222/Xt99+69rqyrHMM+eUlXNeklQzPMjD1QAAAJROTofZGjVqyGw2S5KuvfZa/fDDD5Kkn3/+WQEBAa6trhw7cOJCr2xkpQAFV3B4BTUAAIByxekwe8cdd2jZsmWSpMcee0zPPPOM6tatq/vuu09DhgxxeYHlVdrx/CEG9MoCAAAUxekuvylTptj+/M9//lM1atTQunXrdO2116pnz54uLa48+zvMMl4WAACgKFf9++sWLVqoRYsWrqgFF6lZJVidE6LVNK6Kp0sBAAAotRwKs4sWLXL4gvTOusZtDWJ1WwO2rQUAALgch8Js7969HbqYyWSSxWK5mnoAAAAAhzk0AcxqtTr0KE6QnTFjhuLj4xUYGKimTZtq9erVDr1u7dq18vPzU6NGjZx+z9LOajV07FSuytlOwwAAAE5zejUDV5o/f75Gjhypp556SikpKWrbtq26d++utLS0y74uMzNT9913n2655ZYSqrRkHTp5Vk1f+FGNn19KoAUAALgMpyeATZo06bLHn332WYev9eqrr+r+++/XAw88IEmaPn26vv/+e82cOVOTJ08u8nUPPfSQ7rnnHvn6+urLL790+P28xYH/rWQQHlxBJpPJw9UAAACUXk6H2YULF9o9P3funFJTU+Xn56drrrnG4TCbl5en5ORkjR8/3q69S5cuWrduXZGv++CDD7R37159/PHHeuGFF674Prm5ucrNzbU9z8rKcqg+T8rfMKEmy3IBAABcltNhNiUlpUBbVlaWBg0apDvuuMPh6xw9elQWi0XR0dF27dHR0UpPTy/0NX/88YfGjx+v1atXy8/PsdInT56siRMnOlxXacAaswAAAI5xyZjZ0NBQTZo0Sc8884zTr7301+iGYRT6q3WLxaJ77rlHEydO1HXXXefw9SdMmKDMzEzb48CBA07XWNLSjp+VRJgFAAC4kqveNCHfyZMnlZmZ6fD5kZGR8vX1LdALm5GRUaC3VpKys7O1efNmpaSk6JFHHpF0YZUFwzDk5+enH374QR07dizwuoCAAAUEBDj5aTwrv2eWYQYAAACX53SY/fe//2333DAMmc1mffTRR+rWrZvD16lQoYKaNm2qpUuX2g1PWLp0qXr16lXg/NDQUG3bts2ubcaM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fLhuu+02derUya6de+R5ixYtUrNmzfT//t//U1RUlBo3bqx33nnHdpx7VDq0adNGy5Yt0++//y5J+uWXX7RmzRrdeuutkrhPpY0j9yM5OVnnzp2zO6datWpKTEzknhXCz9MFlAVHjx6VxWJRdHS0XXt0dLTS09M9VBXyGYahUaNGqU2bNkpMTJQk230p7J79+eefJV5jefXZZ59py5Yt+vnnnwsc4x553r59+zRz5kyNGjVKTz75pDZt2qQRI0YoICBA9913H/eolBg3bpwyMzN1ww03yNfXVxaLRS+++KL69esniX+XShtH7kd6eroqVKigKlWqFDiHXFEQYdaFTCaT3XPDMAq0oeQ98sgj+vXXX7VmzZoCx7hnnnPgwAE99thj+uGHHxQYGFjkedwjz7FarWrWrJleeuklSVLjxo21Y8cOzZw5U/fdd5/tPO6RZ82fP18ff/yxPv30U914443aunWrRo4cqWrVqmngwIG287hPpUtx7gf3rHAMM3CByMhI+fr6FvjbUkZGRoG/eaFkPfroo1q0aJFWrFihGjVq2NpjYmIkiXvmQcnJycrIyFDTpk3l5+cnPz8/rVy5Uv/+97/l5+dnuw/cI8+JjY1VQkKCXVu9evVsE1v596h0GDt2rMaPH6+7775b9evX17333qvHH39ckydPlsR9Km0cuR8xMTHKy8vTiRMnijwHfyPMukCFChXUtGlTLV261K596dKlatWqlYeqKt8Mw9AjjzyiBQsWaPny5YqPj7c7Hh8fr5iYGLt7lpeXp5UrV3LPSsgtt9yibdu2aevWrbZHs2bN1L9/f23dulV16tThHnlY69atCyxp9/vvvysuLk4S/x6VFmfOnJGPj/3/zn19fW1Lc3GfShdH7kfTpk3l7+9vd47ZbNb27du5Z4Xx2NSzMuazzz4z/P39jffee8/YuXOnMXLkSKNixYrG/v37PV1auTRs2DAjLCzM+Omnnwyz2Wx7nDlzxnbOlClTjLCwMGPBggXGtm3bjH79+hmxsbFGVlaWBysv3y5ezcAwuEeetmnTJsPPz8948cUXjT/++MP45JNPjODgYOPjjz+2ncM98ryBAwca1atXN7755hsjNTXVWLBggREZGWk88cQTtnO4TyUrOzvbSElJMVJSUgxJxquvvmqkpKQYf/75p2EYjt2PoUOHGjVq1DB+/PFHY8uW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
<Figure size 800x500 with 1 Axes>
output_type": "display_data
name": "stdout
output_type": "stream
Number of components selected: 32

model_id": "94408a32a03148748928bec2b5b63cad
Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]
output_type": "display_data
name": "stdout
output_type": "stream


LogisticRegression Results:



Confusion Matrix:

[[6915  502]

 [ 937 1415]]



Classification Report:

              precision    recall  f1-score   support



       <=50K       0.88      0.93      0.91      7417

        >50K       0.74      0.60      0.66      2352



    accuracy                           0.85      9769

   macro avg       0.81      0.77      0.78      9769

weighted avg       0.85      0.85      0.85      9769



🏃 View run LogisticRegression at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/4/runs/f86d6a32e91149ad81f010b02557e339

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/4

model_id": "ccef84c279a1446b964637e1435f2be1
Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]
output_type": "display_data
name": "stdout
output_type": "stream


RandomForestClassifier Results:



Confusion Matrix:

[[6856  561]

 [ 977 1375]]



Classification Report:

              precision    recall  f1-score   support



       <=50K       0.88      0.92      0.90      7417

        >50K       0.71      0.58      0.64      2352



    accuracy                           0.84      9769

   macro avg       0.79      0.75      0.77      9769

weighted avg       0.84      0.84      0.84      9769



🏃 View run RandomForestClassifier at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/4/runs/f1ca8ec409474b6990deab79a30fd02b

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/4

name": "stderr
output_type": "stream
C:\\Users\\lenovo\\anaconda3\\Lib\\site-packages\\xgboost\\core.py:158: UserWarning: [18:59:00] WARNING: C:\\buildkite-agent\\builds\\buildkite-windows-cpu-autoscaling-group-i-0c55ff5f71b100e98-1\\xgboost\\xgboost-ci-windows\\src\\learner.cc:740: 

Parameters: { \"use_label_encoder\" } are not used.



  warnings.warn(smsg, UserWarning)

model_id": "cc0cac1015c54cadbaf365e6ecf70145
Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]
output_type": "display_data
name": "stdout
output_type": "stream


XGBClassifier Results:



Confusion Matrix:

[[6818  599]

 [ 900 1452]]



Classification Report:

              precision    recall  f1-score   support



       <=50K       0.88      0.92      0.90      7417

        >50K       0.71      0.62      0.66      2352



    accuracy                           0.85      9769

   macro avg       0.80      0.77      0.78      9769

weighted avg       0.84      0.85      0.84      9769



🏃 View run XGBClassifier at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/4/runs/8dd435f7ff42463c8dd8dd68d409a3f3

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/4

#CODE FOR EXPERIMENT 5:Use PCA for dimensionality reduction on all the features. Create a scree plot to show which components will be selected for classification. Log results in MLFlow

# Import required libraries

import os

import dagshub

import pandas as pd

import numpy as np

import mlflow

import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, f1_score

import matplotlib.pyplot as plt

import seaborn as sns



# Step 1: Initialize DagsHub MLFlow connection

dagshub.init(repo_owner=\"yashaswiniguntupalli\", repo_name=\"ML_Final_Project\", mlflow=True)



os.environ[\"MLFLOW_TRACKING_USERNAME\"] = \"yashaswiniguntupalli\"

os.environ[\"MLFLOW_TRACKING_PASSWORD\"] = \"dd928eb3e01ad92df47ae00f812f06a28ddc8c95\"



mlflow.set_tracking_uri(\"https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow\")



experiment_name = \"Experiment_5\"

if not mlflow.get_experiment_by_name(experiment_name):

    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)



# Step 2: Load Dataset

file_path = \"adult_income.csv\"  # Replace with your file path

data = pd.read_csv(file_path)



# Step 3: Preprocessing

X = data.drop('income', axis=1)

y = data['income']



# Convert target variable to numerical labels

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)



# Identify numerical and categorical columns

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

categorical_cols = X.select_dtypes(include=['object']).columns



# Define preprocessing pipeline

numerical_transformer = Pipeline(steps=[

    ('scaler', StandardScaler())

])



categorical_transformer = Pipeline(steps=[

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ]

)



# Preprocess the data

X_preprocessed = preprocessor.fit_transform(X)



# Step 4: PCA for Dimensionality Reduction

pca = PCA()

X_pca = pca.fit_transform(X_preprocessed.toarray())



# Create Scree Plot

explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(8, 5))

plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio.cumsum(), marker='o', linestyle='--')

plt.xlabel(\"Number of Principal Components\")

plt.ylabel(\"Cumulative Explained Variance\")

plt.title(\"Scree Plot\")

scree_plot_path = \"scree_plot.png\"

plt.savefig(scree_plot_path)

plt.show()



# Select components that explain at least 95% of the variance

n_components = np.argmax(explained_variance_ratio.cumsum() >= 0.95) + 1

X_pca_reduced = X_pca[:, :n_components]

print(f\"Number of components selected: {n_components}\")



# Step 5: Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X_pca_reduced, y, test_size=0.3, random_state=42, stratify=y)



# Step 6: Define Classifiers

classifiers = {

    \"LogisticRegression\": LogisticRegression(max_iter=1000, random_state=42),

    \"RandomForestClassifier\": RandomForestClassifier(n_estimators=100, random_state=42),

    \"XGBClassifier\": XGBClassifier(use_label_encoder=False, eval_metric=\"logloss\", random_state=42)

}



# Step 7: Train and Log Models

for name, clf in classifiers.items():

    with mlflow.start_run(run_name=name):

        try:

            # Train the model

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)



            # Calculate Metrics

            f1 = f1_score(y_test, y_pred, average=\"weighted\")

            conf_matrix = confusion_matrix(y_test, y_pred)



            # Log Metrics and Model

            mlflow.log_param(\"Model\", name)

            mlflow.log_param(\"PCA_Components\", n_components)

            mlflow.log_metric(\"F1_Score\", f1)



            mlflow.sklearn.log_model(

                sk_model=clf,

                artifact_path=name,

                input_example=X_test[:1],

                signature=mlflow.models.infer_signature(X_test, y_pred)

            )



            # Log Scree Plot

            mlflow.log_artifact(scree_plot_path)



            # Plot Confusion Matrix

            plt.figure(figsize=(6, 4))

            sns.heatmap(conf_matrix, annot=True, fmt=\"d\", cmap=\"Blues\",

                        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)

            plt.title(f\"Confusion Matrix for {name}\")

            plt.xlabel(\"Predicted\")

            plt.ylabel(\"True\")

            plot_path = f\"{name}_confusion_matrix.png\"

            plt.savefig(plot_path)

            plt.close()

            mlflow.log_artifact(plot_path)



            # Print Results

            print(f\"\
{name} Results:\")

            print(\"\
Confusion Matrix:\")

            print(conf_matrix)

            print(\"\
Classification Report:\")

            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))



        except Exception as e:

            print(f\"Error encountered with {name}: {e}\")

cell_type": "code
id": "bcfa1a14
<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Initialized MLflow to track repo <span style=\"color: #008000; text-decoration-color: #008000\">\"yashaswiniguntupalli/ML_Final_Project\"</span>

</pre>

Initialized MLflow to track repo \u001b[32m\"yashaswiniguntupalli/ML_Final_Project\"\u001b[0m

output_type": "display_data
<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Repository yashaswiniguntupalli/ML_Final_Project initialized!

</pre>

Repository yashaswiniguntupalli/ML_Final_Project initialized!

output_type": "display_data
name": "stdout
output_type": "stream
Number of components retained: 32

model_id": "a5bd16813d88482882500b7b0c35e8b9
Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]
output_type": "display_data
name": "stdout
output_type": "stream


kNN Results:



Confusion Matrix:

[[6779  638]

 [ 935 1417]]



Classification Report:

              precision    recall  f1-score   support



       <=50K       0.88      0.91      0.90      7417

        >50K       0.69      0.60      0.64      2352



    accuracy                           0.84      9769

   macro avg       0.78      0.76      0.77      9769

weighted avg       0.83      0.84      0.84      9769



🏃 View run kNN at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/5/runs/d56373f2593c49bf8d2e6e9959729b45

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/5

model_id": "66fdf8c5acb342f381ff8ee442f53dc6
Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]
output_type": "display_data
name": "stdout
output_type": "stream


LogisticRegression Results:



Confusion Matrix:

[[6915  502]

 [ 937 1415]]



Classification Report:

              precision    recall  f1-score   support



       <=50K       0.88      0.93      0.91      7417

        >50K       0.74      0.60      0.66      2352



    accuracy                           0.85      9769

   macro avg       0.81      0.77      0.78      9769

weighted avg       0.85      0.85      0.85      9769



🏃 View run LogisticRegression at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/5/runs/076aba2f69e74b36980f084f9477c823

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/5

model_id": "91f37aa3f0d143cfaad957883a1d6bdc
Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]
output_type": "display_data
name": "stdout
output_type": "stream


RandomForestClassifier Results:



Confusion Matrix:

[[6856  561]

 [ 977 1375]]



Classification Report:

              precision    recall  f1-score   support



       <=50K       0.88      0.92      0.90      7417

        >50K       0.71      0.58      0.64      2352



    accuracy                           0.84      9769

   macro avg       0.79      0.75      0.77      9769

weighted avg       0.84      0.84      0.84      9769



🏃 View run RandomForestClassifier at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/5/runs/dc48dc551df64ba3bb3b477e530d2676

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/5

name": "stderr
output_type": "stream
C:\\Users\\lenovo\\anaconda3\\Lib\\site-packages\\xgboost\\core.py:158: UserWarning: [19:03:38] WARNING: C:\\buildkite-agent\\builds\\buildkite-windows-cpu-autoscaling-group-i-0c55ff5f71b100e98-1\\xgboost\\xgboost-ci-windows\\src\\learner.cc:740: 

Parameters: { \"use_label_encoder\" } are not used.



  warnings.warn(smsg, UserWarning)

model_id": "bbaf6150388c46dbbcddb4b5fadcb768
Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]
output_type": "display_data
name": "stdout
output_type": "stream


XGBClassifier Results:



Confusion Matrix:

[[6818  599]

 [ 900 1452]]



Classification Report:

              precision    recall  f1-score   support



       <=50K       0.88      0.92      0.90      7417

        >50K       0.71      0.62      0.66      2352



    accuracy                           0.85      9769

   macro avg       0.80      0.77      0.78      9769

weighted avg       0.84      0.85      0.84      9769



🏃 View run XGBClassifier at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/5/runs/b1285912121d40009085ffe17dcec1ba

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/5

#code for experiment 6:Design and execute a custom experiment. Log results in MLFlow.

# Import required libraries

import os

import dagshub

import pandas as pd

import numpy as np

import mlflow

import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, f1_score

import matplotlib.pyplot as plt

import seaborn as sns



# Step 1: Initialize DagsHub MLFlow connection

dagshub.init(repo_owner=\"yashaswiniguntupalli\", repo_name=\"ML_Final_Project\", mlflow=True)



os.environ[\"MLFLOW_TRACKING_USERNAME\"] = \"yashaswiniguntupalli\"

os.environ[\"MLFLOW_TRACKING_PASSWORD\"] = \"dd928eb3e01ad92df47ae00f812f06a28ddc8c95\"



mlflow.set_tracking_uri(\"https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow\")



experiment_name = \"Experiment_6_Custom_Experiment\"

if not mlflow.get_experiment_by_name(experiment_name):

    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)



# Step 2: Load Dataset

file_path = \"adult_income.csv\"  # Replace with your file path

data = pd.read_csv(file_path)



# Step 3: Preprocessing

X = data.drop('income', axis=1)

y = data['income']



# Convert target variable to numerical labels

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)



# Identify numerical and categorical columns

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

categorical_cols = X.select_dtypes(include=['object']).columns



# Define preprocessing pipeline

numerical_transformer = Pipeline(steps=[

    ('scaler', StandardScaler())

])



categorical_transformer = Pipeline(steps=[

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ]

)



# Preprocess the data

X_preprocessed = preprocessor.fit_transform(X)



# Step 4: PCA for Dimensionality Reduction

pca = PCA(n_components=0.95)  # Keep 95% of the variance

X_pca = pca.fit_transform(X_preprocessed.toarray())

print(f\"Number of components retained: {pca.n_components_}\")



# Step 5: Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)



# Step 6: Define Classifiers and kNN Parameter Grid

classifiers = {

    \"kNN\": KNeighborsClassifier(),

    \"LogisticRegression\": LogisticRegression(max_iter=1000, random_state=42),

    \"RandomForestClassifier\": RandomForestClassifier(n_estimators=100, random_state=42),

    \"XGBClassifier\": XGBClassifier(use_label_encoder=False, eval_metric=\"logloss\", random_state=42)

}



# Parameter grid for kNN

knn_param_grid = {\"n_neighbors\": [3, 5, 7, 9, 11]}



# Step 7: Train, Tune, and Log Models

for name, clf in classifiers.items():

    with mlflow.start_run(run_name=name):

        try:

            if name == \"kNN\":

                # Perform GridSearch for kNN

                grid_search = GridSearchCV(clf, knn_param_grid, cv=5, scoring=\"f1_weighted\")

                grid_search.fit(X_train, y_train)

                clf = grid_search.best_estimator_  # Use the best model

                best_k = grid_search.best_params_[\"n_neighbors\"]

                mlflow.log_param(\"Best_k\", best_k)

            else:

                clf.fit(X_train, y_train)



            # Predict and Evaluate

            y_pred = clf.predict(X_test)



            # Calculate Metrics

            f1 = f1_score(y_test, y_pred, average=\"weighted\")

            conf_matrix = confusion_matrix(y_test, y_pred)



            # Log Metrics and Model

            mlflow.log_param(\"Model\", name)

            mlflow.log_param(\"PCA_Components\", pca.n_components_)

            mlflow.log_metric(\"F1_Score\", f1)



            mlflow.sklearn.log_model(

                sk_model=clf,

                artifact_path=name,

                input_example=X_test[:1],

                signature=mlflow.models.infer_signature(X_test, y_pred)

            )



            # Plot Confusion Matrix

            plt.figure(figsize=(6, 4))

            sns.heatmap(conf_matrix, annot=True, fmt=\"d\", cmap=\"Blues\",

                        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)

            plt.title(f\"Confusion Matrix for {name}\")

            plt.xlabel(\"Predicted\")

            plt.ylabel(\"True\")

            plot_path = f\"{name}_confusion_matrix.png\"

            plt.savefig(plot_path)

            plt.close()

            mlflow.log_artifact(plot_path)



            # Print Results

            print(f\"\
{name} Results:\")

            print(\"\
Confusion Matrix:\")

            print(conf_matrix)

            print(\"\
Classification Report:\")

            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))



        except Exception as e:

            print(f\"Error encountered with {name}: {e}\")

cell_type": "code
id": "41aaceee
name": "stdout
output_type": "stream
Found existing installation: scikit-learn 1.6.0

Uninstalling scikit-learn-1.6.0:

  Successfully uninstalled scikit-learn-1.6.0

Found existing installation: imbalanced-learn 0.13.0

Uninstalling imbalanced-learn-0.13.0:

  Successfully uninstalled imbalanced-learn-0.13.0

Found existing installation: sklearn-compat 0.1.1

Uninstalling sklearn-compat-0.1.1:

  Successfully uninstalled sklearn-compat-0.1.1

name": "stderr
output_type": "stream
WARNING: Failed to remove contents in a temporary directory 'C:\\Users\\lenovo\\anaconda3\\Lib\\site-packages\\~.learn'.

You can safely remove it manually.

#importing libraries for experiment -7

!pip uninstall scikit-learn imbalanced-learn sklearn-compat -y

cell_type": "code
id": "c574aca0
name": "stdout
output_type": "stream
Collecting scikit-learn

  Using cached scikit_learn-1.6.0-cp311-cp311-win_amd64.whl.metadata (15 kB)

Collecting imbalanced-learn

  Using cached imbalanced_learn-0.13.0-py3-none-any.whl.metadata (8.8 kB)

Requirement already satisfied: numpy>=1.19.5 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from scikit-learn) (1.24.4)

Requirement already satisfied: scipy>=1.6.0 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from scikit-learn) (1.10.1)

Requirement already satisfied: joblib>=1.2.0 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from scikit-learn) (1.2.0)

Requirement already satisfied: threadpoolctl>=3.1.0 in c:\\users\\lenovo\\anaconda3\\lib\\site-packages (from scikit-learn) (3.5.0)

Collecting sklearn-compat<1,>=0.1 (from imbalanced-learn)

  Using cached sklearn_compat-0.1.1-py3-none-any.whl.metadata (16 kB)

Using cached scikit_learn-1.6.0-cp311-cp311-win_amd64.whl (11.1 MB)

Using cached imbalanced_learn-0.13.0-py3-none-any.whl (238 kB)

Using cached sklearn_compat-0.1.1-py3-none-any.whl (16 kB)

Installing collected packages: scikit-learn, sklearn-compat, imbalanced-learn

Successfully installed imbalanced-learn-0.13.0 scikit-learn-1.6.0 sklearn-compat-0.1.1

#impoting libraries for experiment -7

!pip install --upgrade scikit-learn imbalanced-learn

cell_type": "code
id": "6238a49b
name": "stdout
output_type": "stream
Scikit-learn version: 1.6.0

Imbalanced-learn version: 0.13.0

#checking libraries are installed properly for experiment -7 are not

import sklearn

import imblearn

print(\"Scikit-learn version:\", sklearn.__version__)

print(\"Imbalanced-learn version:\", imblearn.__version__)

cell_type": "code
id": "3834121a
<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Initialized MLflow to track repo <span style=\"color: #008000; text-decoration-color: #008000\">\"yashaswiniguntupalli/ML_Final_Project\"</span>

</pre>

Initialized MLflow to track repo \u001b[32m\"yashaswiniguntupalli/ML_Final_Project\"\u001b[0m

output_type": "display_data
<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Repository yashaswiniguntupalli/ML_Final_Project initialized!

</pre>

Repository yashaswiniguntupalli/ML_Final_Project initialized!

output_type": "display_data
name": "stdout
output_type": "stream
Original dataset shape: 0    17303

1     5489

dtype: int64

Balanced dataset shape: 0    17303

1    17303

dtype: int64

model_id": "1feaad2f5fcf4511a3d12b050a952d52
Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]
output_type": "display_data
name": "stdout
output_type": "stream


LogisticRegression with SMOTE Results:



Confusion Matrix:

[[5901 1516]

 [ 337 2015]]



Classification Report:

              precision    recall  f1-score   support



       <=50K       0.95      0.80      0.86      7417

        >50K       0.57      0.86      0.69      2352



    accuracy                           0.81      9769

   macro avg       0.76      0.83      0.77      9769

weighted avg       0.86      0.81      0.82      9769



🏃 View run LogisticRegression_SMOTE at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/6/runs/cc48e6d24a37486da7dfd4b8e3ae5e5a

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/6

model_id": "8bc3dbeb65f1490f8222cbf4676db802
Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]
output_type": "display_data
name": "stdout
output_type": "stream


RandomForestClassifier with SMOTE Results:



Confusion Matrix:

[[6596  821]

 [ 686 1666]]



Classification Report:

              precision    recall  f1-score   support



       <=50K       0.91      0.89      0.90      7417

        >50K       0.67      0.71      0.69      2352



    accuracy                           0.85      9769

   macro avg       0.79      0.80      0.79      9769

weighted avg       0.85      0.85      0.85      9769



🏃 View run RandomForestClassifier_SMOTE at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/6/runs/62bdbc246b52405d998d4fb0b0c96096

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/6

name": "stderr
output_type": "stream
C:\\Users\\lenovo\\anaconda3\\Lib\\site-packages\\xgboost\\core.py:158: UserWarning: [19:06:44] WARNING: C:\\buildkite-agent\\builds\\buildkite-windows-cpu-autoscaling-group-i-0c55ff5f71b100e98-1\\xgboost\\xgboost-ci-windows\\src\\learner.cc:740: 

Parameters: { \"use_label_encoder\" } are not used.



  warnings.warn(smsg, UserWarning)

model_id": "e78c9405317e481aa4d28b758a89159c
Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]
output_type": "display_data
name": "stdout
output_type": "stream


XGBClassifier with SMOTE Results:



Confusion Matrix:

[[6598  819]

 [ 546 1806]]



Classification Report:

              precision    recall  f1-score   support



       <=50K       0.92      0.89      0.91      7417

        >50K       0.69      0.77      0.73      2352



    accuracy                           0.86      9769

   macro avg       0.81      0.83      0.82      9769

weighted avg       0.87      0.86      0.86      9769



🏃 View run XGBClassifier_SMOTE at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/6/runs/16c952da7bda4f72a210f5e8ce23074c

🧪 View experiment at: https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow/#/experiments/6

#code for experiment 7:Design and execute another custom experiment. Log results in MLFlow.

# Import required libraries

import os

import dagshub

import pandas as pd

import numpy as np

import mlflow

import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, f1_score

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

import seaborn as sns



# Step 1: Initialize DagsHub MLFlow connection

dagshub.init(repo_owner=\"yashaswiniguntupalli\", repo_name=\"ML_Final_Project\", mlflow=True)



os.environ[\"MLFLOW_TRACKING_USERNAME\"] = \"yashaswiniguntupalli\"

os.environ[\"MLFLOW_TRACKING_PASSWORD\"] = \"dd928eb3e01ad92df47ae00f812f06a28ddc8c95\"



mlflow.set_tracking_uri(\"https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow\")



experiment_name = \"Experiment_7_SMOTE_Balancing\"

if not mlflow.get_experiment_by_name(experiment_name):

    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)



# Step 2: Load Dataset

file_path = \"adult_income.csv\"  # Replace with your file path

data = pd.read_csv(file_path)



# Step 3: Preprocessing

X = data.drop('income', axis=1)

y = data['income']



# Convert target variable to numerical labels

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)



# Identify numerical and categorical columns

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

categorical_cols = X.select_dtypes(include=['object']).columns



# Preprocessing pipeline

numerical_transformer = Pipeline(steps=[

    ('scaler', StandardScaler())

])



categorical_transformer = Pipeline(steps=[

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ]

)



# Preprocess the data

X_preprocessed = preprocessor.fit_transform(X)



# Step 4: Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42, stratify=y)



# Step 5: Apply SMOTE for Balancing

smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f\"Original dataset shape: {pd.Series(y_train).value_counts()}\")

print(f\"Balanced dataset shape: {pd.Series(y_train_balanced).value_counts()}\")



# Step 6: Define Classifiers

classifiers = {

    \"LogisticRegression\": LogisticRegression(max_iter=1000, random_state=42),

    \"RandomForestClassifier\": RandomForestClassifier(n_estimators=100, random_state=42),

    \"XGBClassifier\": XGBClassifier(use_label_encoder=False, eval_metric=\"logloss\", random_state=42)

}



# Step 7: Train and Log Models

for name, clf in classifiers.items():

    with mlflow.start_run(run_name=f\"{name}_SMOTE\"):

        try:

            # Train the model on the SMOTE-balanced dataset

            clf.fit(X_train_balanced, y_train_balanced)

            y_pred = clf.predict(X_test)



            # Calculate Metrics

            f1 = f1_score(y_test, y_pred, average=\"weighted\")

            conf_matrix = confusion_matrix(y_test, y_pred)



            # Log Metrics and Model

            mlflow.log_param(\"Model\", name)

            mlflow.log_param(\"Balancing\", \"SMOTE\")

            mlflow.log_metric(\"F1_Score\", f1)



            mlflow.sklearn.log_model(

                sk_model=clf,

                artifact_path=name,

                input_example=X_test[:1].toarray(),

                signature=mlflow.models.infer_signature(X_test, y_pred)

            )



            # Plot Confusion Matrix

            plt.figure(figsize=(6, 4))

            sns.heatmap(conf_matrix, annot=True, fmt=\"d\", cmap=\"Blues\",

                        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)

            plt.title(f\"Confusion Matrix for {name} with SMOTE\")

            plt.xlabel(\"Predicted\")

            plt.ylabel(\"True\")

            plot_path = f\"{name}_SMOTE_confusion_matrix.png\"

            plt.savefig(plot_path)

            plt.close()

            mlflow.log_artifact(plot_path)



            # Print Results

            print(f\"\
{name} with SMOTE Results:\")

            print(\"\
Confusion Matrix:\")

            print(conf_matrix)

            print(\"\
Classification Report:\")

            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))



        except Exception as e:

            print(f\"Error encountered with {name}: {e}\")

cell_type": "code
id": "58b5fe3a
image/png": "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MWPGKEeOHPLw8FDx4sW1bt26h7afOXOmihQpohQpUihz5sxq3bq1Ll++/IyqBQAAAPAycXZ21r59+7Rv375Yl35bpUqVSt27d1fXrl01depUHTlyRDt27NDo0aM1depUSfeD4tatW7Vs2TIdPHhQn376aazRtp9Uly5dVK5cOVWpUkWjR4/Wrl27dPToUf34448qVaqUDh06ZGvr5eWlRo0aqU+fPjpz5oxatWr11I9/5coV7dy5U3v37pUkHThwQDt37tS5c+cS3Gb8+PF6//33tXz5ch05ckR79uxRz549tWfPHr3++uuSpN69e2vLli0KDQ3VP//8o/3792vs2LG6dOmSUqZMqffff189evTQ77//rr179+rdd9/VnTt3HjoSe2L+Tvbg8NA9d+5cdenSRX379tWOHTtUoUIF1apVS2FhYfG2X79+vVq0aKG2bdtqz549mjdvnrZs2aJ27do948oBAAAAvCy8vb3l7e2d4PoBAwbos88+08CBA5U/f37VqFFDixcvVo4cOSTdv7f5rbfeUqNGjVSqVCldvnw5Vq/3k3J3d9eKFSv08ccfa/z48SpdurRKlCihkSNHqnPnzipYsGCs9m3bttXVq1dVtWrVJJlrfNGiRSpWrJjq1KkjSWrcuLGKFSumcePGJbhNyZIldevWLXXo0EGvvPKKQkJCtHHjRv3888+2Hug8efJo+fLl2rVrl0qWLKkyZcrol19+sc0PPmjQINWvX1/NmzdXUFCQDh8+rGXLlj1yULhH/Z3swWKsF8Q7SKlSpRQUFKSxY8faluXPn19169bVwIFxp5gZMmSIxo4dG+tSjFGjRmnw4ME6efJkoh7zxo0b8vHx0fXr1x/6wgFedlxenjiele84uoTn3vM4ZRjn96NxbifO83h+A8+Te/fu6dixY7YrW4Hk5GHnb2JzpUN7uiMiIrRt2zZVr1491vLq1avrr7/+inebsmXL6tSpU1q6dKmMMTp//rzmz59v+2YlPuHh4bpx40asHwAAAAAA7M2hofvSpUuKjo6Wr69vrOW+vr4J3gNQtmxZzZw5U40aNZKbm5syZcqk1KlTa9SoUQk+zsCBA+Xj42P78ff3T9LnAQAAAABAfBx+T7eUuKHerfbu3avOnTvrs88+07Zt2/T777/r2LFj6tChQ4L77927t65fv277Sexl6AAAAAAAPA2H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1atXNWrUKH399dfatm2b3njjDU2fPl0jRox4aPAG7MV6vpUrV07u7u66d++e0qRJo+bNm+vChQtq166dTp06JTc3N0n3B1nr3bu3pk6dqv79+8vLy8uR5QMA8MKjmxDPnLWHu0ePHrpw4YJ8fX21detW9ezZUzExMapZs6Zat24tSXr33Xd148YNffXVV7YebuBZs142HhERoVSpUilXrlx67bXXlClTJo0YMUKffvqplixZIkn68MMPlSFDBu6PxTNjPc9y5Mih48ePa926dapWrZo+/PBDXbt2TRMnTlTBggXVpk0bXbhwQTdu3NC2bdu0cuVKvfLKKw6uHgCAFx+hGw7x/fffa+LEiVq5cqUyZMigmJgYvfbaa+rXr58sFotq1Kih1q1b686dO5o1axY9MXAoi8WiRYsWaciQIbpz546ioqJiDd43YMAASfcH/Lt9+7b69u2r9OnTO6pcvCSOHz+u1atXq1KlSvL09FT27NkVGBiou3fv2tp8/vnnKlmypH7++WetXbtWnp6eevXVVzVs2DDlzp3bgdUDAPDyYPRyOMRHH32kffv2aenSpbYB1S5duqQyZcrIy8tLAwYMUK1ateTs7GxbT88hnjXrObdz506VKlVKXbp00cGDB7Vp0yaFhITo22+/jXWvdrdu3bR9+3bNmzdPGTJkcGDleNFFRESofv362rFjh5ycnHT37l1Vr15ds2fP1ptvvqlvvvlGTk5Oypkzp20b68jmvJcCAPBsEbrxTFmn9+rYsaN27typDRs2SJLu3r0rT09P/fLLL6pfv76qVKmivn37qmLFirFGOQeetR07dmjz5s26cuWKevfuLUkaMWKE5s+fr8DAQA0aNEgZM2a0tb948SKBG8/EzZs3lSpVKu3YsUP79+/XqVOnNGXKFO3bt0/+/v6KjIzUK6+8osyZM6tkyZIqU6aMihcvTugGAOAZI8nArqyj5FpZL8lt1qyZNm7cqCFDhkiSPD09Jd3vWbROYzNo0CBJInDDYc6ePatu3brpo48+0p07d2zLP/zwQ9WvX18HDhzQJ598onPnztnWEbjxrFhvuylWrJiaNGmiHj16qFWrVmrSpIl++eUXTZ8+XaVLl9alS5c0c+ZMeXt7S2IQSgAAnjV6umE3D/ZQz5kzRwcPHtTdu3f15ptvqnTp0ho6dKj69OmjTz75RK1atZIx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8bn/xQIAAKBNotON/+rO7vQ///lPOTk5WbuGd/L09NS8efNkMBj029/+Vu7u7powYYItSgas39HCwkIVFBTIy8tLjz76qPr27au3335bs2fPVkNDgxYtWqT6+nrt2LFD3bp107vvvisnJydblw8AAIA2gj3d+F53Bu7U1FSdPHlSb731ltzc3P7je77++mvt2bNHkZGRXAcGm9qxY4deeOEF9enTR1euXJGvr68SExPl5+en7OxszZ8/X05OTurQoYOuXbumv/zlLxo8eLCtywYAAEAbQujGDxIXF6fNmzfrtdde0/jx49WvX78f9D7u4YatlJaWKikpScOHD1dkZKRycnL03nvvqaqqSitXrtSwYcNUXl6uQ4cOycHBQYMHD1bv3r1tXTYAAADaGEI37unODvfBgwcVHh6urKwsjR492saVAf/d8ePHtXjxYt24cUPp6enq27evJGn//v1as2aNqqqqlJyczPcZAAAAzY5TrdBEfHy8pKYHnl24cEFubm4aNmyYdezff1fT0NBwfwoEfoDTp0/rq6++0vHjx3X9+nXr+NixYzV//ny5u7tr7ty5KigosGGVAAAAaA8I3bDKz8/XyZMnZTabm4zb2dmpqqpKly5dajJeX1+vrKwsXb58mVPJ0aLMmDFDCQkJ8vLy0sKFC3X69Gnra2PHjlVERIQGDhwoDw8PG1YJAACA9oCkBKsRI0YoLy9P9vb22rZtm3W8V69eqq2t1datW1VZWSnp9vVLZrNZ6enpyszMtFHFwL9WXVRVVamqqsra2Q4NDdXLL7+s2tpavf766youLra+JygoSOvXr2cPNwAAAJode7oh6XbX2s7OTpJUUlIiHx8f+fv7a/fu3ZKkpKQkvfnmm5ozZ46efPJJubq6Kjk5WRUVFfrkk084LA020XgtWG5urtLS0nT27FmNGjVKY8aM0axZsyRJmzZtUmZmptzc3JSYmKiBAwfauGoAAAC0J3S6oYqKCmvgPnjwoAYMGKBNmzappKREwcHBkqTFixcrKSlJR48eVVhYmBYsWCCLxaLCwkLZ29urvr7elo+AdspgMGj37t2aPHmyAgIC9NZbb8ne3l5JSUlKS0uTdHupeUREhM6dO6eVK1eqrq7OxlUDAACgPaHT3c7l5eVpw4YNeuONN5SWlqbVq1frypUrMplM2rt3r2JiYvTYY48pNzdX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
<Figure size 1000x600 with 1 Axes>
output_type": "display_data
#experiment-7:

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Example data: Replace this with your actual F1-score data

data = {

    \"Model\": [\"Logistic Regression\", \"Random Forest\", \"XGBoost\", \"SVM\", \"KNN\"],

    \"Mean CV F1-Score\": [0.78, 0.81, 0.85, 0.76, 0.74],

    \"Test F1-Score\": [0.79, 0.82, 0.86, 0.77, 0.75]

}



# Convert to a DataFrame

df = pd.DataFrame(data)



# Melt the DataFrame to have a long-form format for better visualization

df_melted = df.melt(id_vars=\"Model\", var_name=\"Metric\", value_name=\"F1-Score\")



# Plot F1-Scores

plt.figure(figsize=(10, 6))

sns.barplot(x=\"Model\", y=\"F1-Score\", hue=\"Metric\", data=df_melted, palette=\"viridis\")

plt.title(\"Comparison of F1-Scores Across Models\", fontsize=16)

plt.ylabel(\"F1-Score\", fontsize=12)

plt.xlabel(\"Model\", fontsize=12)

plt.xticks(rotation=45)

plt.legend(title=\"Metric\")

plt.tight_layout()



# Show the plot

plt.show()

cell_type": "code
id": "a35a917e
<Figure size 640x480 with 0 Axes>
output_type": "display_data
plt.savefig(\"f1_score_comparison.png\", dpi=300)

cell_type": "code
id": "ea16920c
image/png": "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lStXdPz48WQfY9q0afLw8NCSJUu0bt06p785c+ZIunmCsYQ+R0dHa+bMmSm2OykBAQGqVq2aFi9e7LS3hN1u19y5c5UvXz4VL1480e1y5sypDh066JVXXtGhQ4cUGRmZaJnixYvr/fffV9myZRON77upXr26OnXqpBYtWqhFixbJLpfa10+dOnUUHh6e6CfHvv76a6fLJUqUULFixfTHH38kuV4rV6581y+DEqSl/wDwoGCmGwAeENWrV9ekSZMUEhKiSpUq6Y033lDp0qUVFxen0NBQffnllypTpoyef/55lShRQq+99pomTJggDw8PNW7c2HH28uDgYPXq1cvl7Vu8eLE8PT1Vv359x9nLn3zySbVs2VKSVKNGDWXJkkXdunXThx9+KC8vL82bN09//PFHmh530KBBOnXqlOrVq6d8+fLp2rVr+vTTT52OF+/Vq5dmz56t5557TkOHDlWBAgW0bNkyTZw4UW+88UaSged+fPjhh/rpp59Up04dDRo0SFmzZtW8efO0bNkyjR49OsUzn7vCd999J+nmsfPSzZ9kSjiG+M5jq+906NAhNWzYUK1bt1bt2rWVO3duXb16VcuWLdOXX36p//znP44zdPfs2VMLFixQ8+bN1a9fP1WtWlVRUVH69ddf1bRpU9WpU0ft2rXTF198ofbt2+v48eMqW7asNm3apP/9739q0qRJiseDJ/j000/19NNPq1atWnrjjTdUsGBBhYeH66+//tKPP/7oCP3PP/+8ypQpo8qVKytHjhw6ceKExo8frwIFCqhYsWJJ3vfly5f1ww8/qGHDhmrevHmSy3zyySeaPXu2Ro4cqVdeeUUzZsxQt27ddOjQIdWpU0d2u11bt25VqVKl1Lp16xT7MnLkSNWvX1916tRRnz595O3trYkTJ+rPP//U/PnzHV9KVatWTU2bNlW5cuWUJUsWHThwQHPmzFH16tXl7++vPXv26M0339TLL7+sYsWKydvbW2vXrtWePXvUr1+/uz6nd7p9F/rkpPb1065dO33yySdq166dRowYoWLFimn58uVauXJlovucMmWKGjdurIYNG6pDhw7Kmzevrly5ogMHDmjXrl1auHBhkm1xdf8B4IHg1tO4AQAS2b17t9G+fXsjf/78hre3txEQEGBUqFDBGDRokHHhwgXHcjabzfjoo4+M4sWLG15eXkb27NmN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AgLk6empdu3aaebMmTIMw3H9woULFRsbq//+97/3+rSkSa5cuXT27Flt2LAh2WV2796tevXqqXTp0vr999+1adMmPf/8846foHv33Xe1aNEizZo1S7t27VLRokXVsGFDXblyxel+3n33XY0cOVIHDhxQuXLl7rqezGAxbn/WHwPXr19XYGCgwsLClClTJnc3BwAAAHikRUdH69ixYypUqJB8fX0d9YfhmO5r165p6tSpypcvnw4ePCiLxaKSJUvqn3/+UZcuXZQ5c2bNnDlTERERyp49u9auXavq1as77qNLly6KjIzU119/neRjdO/eXefPn3fMtHbo0EHr16/X33//LavVKklq2bKlPDw89M033yR5H/7+/uratWuiWfM7HTx4UKVKldLatWtVp04dSVLt2rWVN2/eZNsn3Tyme/PmzU4z0K+//rrGjh3rtNzx48dVqFChVB3TbbPZ1KVLF82cOVO5cuXSU089pXr16qldu3aOjNamTRudPHlSmzZtSnT7iIgIZcmSRTNnzlSbNm0k3TzGvGDBgurZs6f69u2r9evXq06dOlqyZImaN2/uuN29rqfkxq+U+mz5aB2hDgAAAAAulD17dj333HOaNWuWDMPQc889p+zZszsts3//fkVHR6t+/fpO9djYWKcZ6MmTJ2vq1Kk6ceKEoqKiFBsbmyigli5d2hG4JSl37tzau3dvsu0zDCNVvyVdsmRJ1ahRQ9OnT1edOnX0999/a+PGjVq1atVdb/vf//5XAwcOdFzOnDnzXW+ToFu3bpo7d67j8o0bN2S1WjVjxgwNHz5ca9eu1ZYtWzRixAh99NFH2rZtm3Lnzq3du3fr5ZdfTvI+//77b8XFxalmzZqOmpeXl6pWraoDBw44LVu5cmXH/6ldT65G6AYAAACAFHTq1ElvvvmmJOmLL75IdH3Cyb+WLVumvHnzOl3n4+MjSfr222/Vq1cvjR07VtWrV1fGjBn18ccfa+vWrU7Le3l5OV22WCwpnlysePHiiYJmcjp37qw333xTX3zxhWbMmKECBQqoXr16d71dYGCgihYtmqrHuNPQoUPVp0+fJK/Lmzev2rZtq7Zt22r48OEqXry4Jk+erCFDhsjPzy/Z+0zYWfvOLxuS+gIiICDA8X9q1pMZCN0AAKSXr1u5uwWPrjYL3N0CAI+wRo0aKTY2VpLUsGHDRNc/8cQT8vHx0cmTJ1W7du0k72Pjxo2qUaOGQkJCHLW///47zW1r06aNBgwYoNDQ0ESztfHx8YqJiXEEz5YtW+rtt9/W119/rVmzZqlr166pmiVPi6CgIAUFBd11uSxZsih37tyKiIiQJJUrV05r1qzRkCFDEi1btGhReXt7a9OmTU67l+/YsUM9e/ZM9jFSs57MQOgGAAAAgBRYrVbHbPLtu34nyJgxo/r06aNevXrJbrfr6aef1vXr17V582ZlyJBB7du3V9GiRTV79mytXLlShQoV0pw5c7R9+3YVKlQoTW3r2bOnli1bpnr16mnYsGF6+umnlTFjRu3YsUMfffSRpk2b5tiFPUOGDGrVqpUGDBigsLAwdejQIU2PLUlXrlzRyZMndebMGUnSoUOHJCnRWc5vN2XKFO3evVstWrRQkSJFFB0drdmzZ2vfvn2aMGGCJKl///4qW7asQkJC1K1bN3l7e2vdunV6+eWXlT17dr3xxhvq27evsmbNqvz582v06NGKjIxM8UzsqVlPZiB0AwAAAMBd3O0kzMOGDVNQUJBGjhypo0ePKnPmzKpYsaIGDBgg6eaxzbt371arVq1ksVj0yiuvKCQkRCtWrEhTu3x8fLR69Wp98sknmjJlivr06SN/f3+VKlVKPXr0UJkyZZyW79y5s6ZNm6YGDRq45LfGly5dqo4dOzout27dWpL04YcfavDgwUnepmrVqtq0aZO6deumM2fOKEOGDCpdurSWLFnimIEuXry4Vq1apQEDBqhq1ary8/NTtWrV9Morr0iSRo0aJbvdrrZt2yo8PFyVK1fWypUrlSVLlhTbe7f1ZAbOXg4AQHph93LzsHs58MBK6ezPwIPOFWcv53e6AQAAAAAwCbuXAwAee+n1W7Fvnb+WLo/zOJqQTuvwXn/nFwAAZroBAAAAADAJoRsAAAAAAJMQugEAAAAAMAmhGwAAAAAAkxC6AQAAAAAwCaEbAAAAAACTELoBAAAAADAJoRsAAAAAAJN4ursBAAAAAB5DX7dK38drsyDVi1oslhSvb9++vWbOnHlfzShYsKB69uypnj173nW5EydOONXy5s2rU6dOSZK+/PJLff3119q1a5fCw8N19epVZc6c+a6Pv2jRIo0ePVoHDx6U3W5X/vz51ahRI40dO/a++oO7I3QDAAAAwG3Onj3r+H/BggUaNGiQDh065Kj5+fmlSzuGDh2qrl27Oi5brVbH/5GRkWrUqJEaNWqk/v37p+r+fvnlF7Vu3Vr/+9//1KxZM1ksFu3fv19r1qxxedsT2Gw2WSwWeXg8vjtZP749BwAAAIAk5MqVy/EXGBgoi8XiVNuwYYMqVaokX19fFS5cWEOGDFF8fLzj9oMHD1b+/Pnl4+OjPHnyqEePHpKk//znPzpx4oR69eoli8Vy1xn1jBkzOj1ujhw5HNf17NlT/fr101NPPZXqfv300096+umn1bdvX5UoUULFixfXCy+8oAkTJjgtt3TpUlWuXFm+vr7Knj27/u///s9x3dWrV9WuXTtlyZJF/v7+aty4sY4cOeK4fubMmcqcObN++uknPfHEE/Lx8dGJEycUGxurd999V3nz5lVAQICqVaum9evXp7rtDzNCNwAAAACk0sqVK/Xqq6+qR48e2r9/v6ZMmaKZM2dqxIgRkqTvvvtOn3zyiaZMmaIjR45oyZIlKlu2rCRp8eLFypcvn4YOHaqzZ886zainh1y5cmnfvn36888/k11m2bJl+r//+z8999xzCg0N1Zo1a1S5cmXH9R06dNCOHTu0dOlS/f777zIMQ02aNFFcXJxjmcjISI0cOVJTp07Vvn37FBQUpI4dO+q3337TN998oz179ujll19Wo0aNnAL7o4rQDQAAAACpNGLECPXr10/t27dX4cKFVb9+fQ0bNkxTpkyRJJ08eVK5cuXSs88+q/z586tq1aqOXcSzZs0qq9XqNIOdkvfee08ZMmRw/H322Wdpavtbb72lKlWqqGzZsipYsKBat26t6dOnKyYmxql/rVu31pAhQ1SqVCk9+eSTGjBggCTpyJEjWrp0qaZOnapatWrpySef1Lx583T69GktWbLEcR9xcXGaOHGiatSooRIlSujcuXOaP3++Fi5cqFq1aqlIkSLq06ePnn76ac2YMSNNfXoYcEw3gIdTep985XFyDyeaAQDgcbNz505t377dMbMt3TxuOTo6WpGRkXr55Zc1fvx4FS5cWI0aNVKTJk30/PPPy9Pz3qNX37591aFDB8fl7Nmzp/q2jRs31saNGyVJBQoU0L59+xQQEKBly5bp77//1rp167Rlyxa98847+vTTT/X777/L399fu3fvdjqO/HYHDhyQp6enqlWr5qhly5ZNJUqU0IEDBxw1b29vlStXznF5165d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cVz/zTffqHbt2i5vJAAAAAAAD6N7Ct3ly5fXgQMHtHnzZuXKlUvVqlVzur5169Z64oknXNpAAAAAAAAeVvcUuiUpR44cat68uePyqVOnlCdPHnl4eOi5555zaeMAAAAAAHiY3dMx3Ul54okndPz4cRc0BQAAAACAR0uaQ7dhGK5oBwAAAAAAj5w0h24AAAAAAJC0NIfuAQMGKGvWrK5oCwAAAAAAj5R7PpHanfr37++KdgAAAAAA8Mhx6e7l//zzjzp16uTKuwQAAAAA4KHl0tB95coVzZo1y5V3CQAAAADAQ+uedi9funRpitcfPXo0TY0BAAAAAOBRck+h+4UXXpDFYknxZ8IsFss9NWDixIn6+OOPdfbsWZUuXVrjx49XrVq17nq73377TbVr11aZMmW0e/fue3pMAAAAAADSwz3tXp47d24tWrRIdrs9yb9du3bd04MvWLBAPXv21MCBAxUaGqpatWqpcePGOnnyZIq3CwsLU7t27VSvXr17ejwAAAAAANLTPYXuSpUqpRis7zYLfqdx48apc+fO6tKli0qVKqXx48crODhYkyZNSvF2r7/+utq0aaPq1aun+rEAAAAAAEhv97R7ed++fRUREZHs9UWLFtW6detSdV+xsbHauXOn+vXr51Rv0KCBNm/enOztZsyYob///ltz587V8OHD7/o4MTExiomJcVy+fv26JCk+Pl7x8fGSJA8PD3l4eDhm7BMk1G02m9OXCcnVrVarLBaL435vr0uSzWZLVd3T01OGYciqW20xZJFdFllkyEPGXet2WWSkUPeQIcu/dbvFUxbDJosMGRarDN06RCChbrc4DxWLcbOPRirrHkb8zUe0WG8tK0MWw5ZC3UOG5db3QhbDLovsydeTabs7+5SwDl2xnlKqG4bh0rF3e91ischqtSZ6fSRbl+GSPtlkkWRxeh3cqivV6+9xHXtp6ZMkl60nq4xk67ePV1eMvbRsy29vf1r65Fz3kGQ41Q2LlbFnUp8S1pdL1lMK77mSXDr20vI5Qv+2La19utv70+3jhrHn2j65cj2l9J5752desz/DprQtT7fPRkmMX4mx54o+WWV32XpK6T33zrEkmfsZ9n625al1T6E7b968KlSoULLXBwQEqHbt2qm6r0uXLslmsylnzpxO9Zw5c+rcuXNJ3ubIkSPq16+fNm7cKE/P1DV95MiR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a4qPj1fz5s2VMWNG/fXXX+rTp4+2bt2qFStWOAXvBPHx8QRvuJxhGOrdu7du3Lihr776ShERETpw4IBmzZqlnDlzqlGjRqpcubL279+vXbt2qWbNmipUqJC7m42HXGhoqBo0aKARI0botddec9SvXLmirFmz6tixY3r//fd16NAhRUZGau7cuapYsaIbW4xHzeHDh/Xtt9/q/fff1/LlyzV48GD5+fnp+++/15gxY/Tnn39q27Ztun79ujp27KgvvvhCkrR161blyZOHL4DgFuHh4cqYMaMkKSwsTC1bttTp06e1aNEi5cmTRz/99JP++OMPzZgxQ2XKlNG6deskSS1atNCcOXPk7+/vzubjHhG6gfvUp08fzZkzR7lz59ahQ4dUsWJF9e7dWy+++KIOHz6sfv36adu2bfr+++9VpUoVdzcXjzjDMGQYhl5++WVduHBBn332mT755BOdPXtWly5dksViUZEiRTRz5kwFBAS4u7l4hMybN0+TJ0/Wxo0bdfXqVf3888+aO3eu/vjjD7355pvq16+frl27pujoaHl6eip79uzubjIeIXa7XePGjdOYMWO0fft25cyZUytXrlTv3r315JNP6rvvvpMkTZw4UfPmzdNrr72m9u3bu7nVeNxdunRJZcqU0ahRo9ShQwdJN0N4kyZNdPr0af3www8qW7asJOnq1as6duyYli1bpp07d2rkyJEqVaqUG1uP+0HoBu7D3Llz1adPH61YsULFihVTdHS02rdvr/DwcL3//vtq0KCB9uzZo7ffflsZM2bU0qVL3d1kPILu3INCunmIQ6NGjRQVFaV69eqpdevWatGihWbMmKEJEyZow4YNTrucA/fj9rG3du1aPfvssxowYIDWr1+vrFmzKm/evAoODtb777+vnTt3qkKFCm5uMR5lO3fuVL169TRu3Dh16tRJ0dHR+uWXX9SrVy8VKlRIq1atkiRdvnzZcUZ9wJ3i4+PVq1cvTZs2TdOnT1fr1q0l3QzezZo107Fjx/Tjjz86gneCmJgY+fj4uKPJSCNCN3AXkydP1ssvv+z0Rj148GBt2rRJq1atkmEYslqtunjxopo3b64sWbJo2bJlkqSjR4+qYMGCSR7/DaRFQuhZv369Vq5cqWPHjqlhw4Zq06aNYmNjdfz4cZUtW9axXN++fbVnzx599913jt3ZgHuVMJ4SPvgl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aP78+apbt66KFy+uzJkza8OGDSpXrpwbW4yHWULgXrlypebMmaPTp0+rQoUK6tq1q/r27avjx4/r008/1Zo1a1SlShUdPHhQS5cu1ZYtW9j+waWioqL0wQcfqE2bNho5cqSjHhcXp3PnzikiIkIlS5bUiy++qMaNG6t///76+OOP9ccff6h58+YEbrhFdHS0fH19Jd38ibrMmTM7fskmZ86cjuA9f/58Wa1WjRgxQtKts/ETuB9NbI2AO0RERGjr1q364IMPVLFiRf3111/67rvvVL9+fbVr105//fWX3n//fdWqVUtRUVHKkiWLu5uMR1yBAgXUtGlTZcmSRUuXLtWFCxcc1yXsUvnHH39ow4YN+umnnzhLOdIk4SzlzZo1k6+vrypUqKDvv/9er732mpYvX64vvvhCXbt2VXx8vKZPn66zZ89q06ZNeuKJJ9zddDxiPD09de7cOZUsWdJRW7lypd59912VKVNGTZo0Ud26dWUYhvz9/VWwYEGNHz9eq1atUokSJdzYcjyuTp8+rXbt2mn16tWSbu79mPCLIoZhOIJ3586d9corr+iHH35Qz549JYmzlD/iOJEaHntJnfCsWbNmOnnypAYNGqSJEyfKbrerePHi+umnn1SpUiX98MMPkm7NCHGiFrhScuNp0qRJ+uqrr1SuXDmNGjVKuXLl4oR9cCnDMHT16lU999xzeuGFF/Tee+9Jks6fP68uXbro2rVrmjVrlgoXLixJCg8Pl7e3t+NnEwFXun79uqpVq6ZatWqpd+/e+v777zVr1iyVKVNGzzzzjDJkyKCRI0eqWbNmGjt2LNtDuN3Ro0f16quvKnPmzBo+fLgWLVqkf/75R7Nnz05y+d69e2vnzp367rvvlCNHjnRuLdIToRuPtdvfoOfPny8/Pz+98MIL2rJli95//3398ccfevPNN9WwYUM99dRTmjFjhr799lt9++23jt+kJXDDlRLG08aNG7Vq1SrFx8erZMmSat++vSTp888/19dff60SJUpo1KhRypkzJx804VKRkZGqVq2a3nrrLb322muKi4uTl5eXzp8/r4oVK6pTp04aNmyYu5uJx8TatWvVsGFD5c2bV1euXNHHH3+sevXqqWjRooqLi1PTpk2VO3duzZw5091NBSRJf/31l958800FBAToxIkTstvtKlOmjCwWi6xWq2JiYmSxWOTp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4OTkxNNPP23LWxAREREREam31OkWcnNzGTp0KKtXr2bgwIEAfPXVV4wYMQJXV1eysrIAKC0txdnZGWdnZwB++9vfkpiYSGZmJh07drRZ/SIiIiIiIvWVOt1CaWkpZWVldOnSBbg7TK19+/Zs3LiRb775huTkZADc3NxwdnYmLy+PqVOnsm7dOvbs2aPALSIiIiIi8j0UuoUuXbrg4uJCamoqgHWoWuvWrXFxceH69evAt5PJ3d3dCQwM5MiRI3h5edmmaBERERERkUeAzulugGoOT7NYLDg5OeHv7096ejqenp68/vrrALi6utKsWTPr1HKLxYLBYKBZs2b069fPZvWLiIiIiIg8KrSnu4H485//TFZWFlFRUcC9wRvg7NmzzJ07l0uXLtG1a1e6devGH//4R4qLizl58qTO3xYREREREbkPCt0NwJ07dwgLCyMrK4sJEyYwe/Zs4NvgXd3BPnfuHLt27WLLli00bdqUH//4x2zevBkHBwfMZrOCt4iIiIiIyA+k0N1A5Ofns3TpUrKzsxkxYgQRERHA3eBtMBis+7hNJpM1XNe8Zm+vnQgiIiIiIiI/lAapNRCenp7MmTMHX19f0tLSiI2NBbB2ugEKCwuZMGECW7dutQZui8WiwC0iIiIiInKf1OluYAoKCoiJieHYsWO89tprzJkzB4ArV64QGBhIUVERubm5CtoiIiIiIiK1QKG7AaoZvEeNGkVwcDCBgYEUFhby6aefag+3iIiIiIhILVHobqAKCgpYtGgRR48e5fPPP8fT05NTp07h4OCgPdwiIiIiIiK1RKG7ASsoKCAiIoJ//vOf7Nq1S4FbRERERESklil0N3AlJSU0bdoUo9GowC0iIiIiIlLLFLoF+PbMbhEREREREak9Ct0iIiIiIiIidUStTREREREREZE6otAtIiIiIiIiUkcUukVERERERETqiEK3iIiIiIiISB1R6BYRERERERGpIwrdIiIiIiIiInVEoVtERERERESkjih0i4iIiIiIiNQRhW4RERERERGROqLQLSIiIiIiIlJH/g9E8J7gDRXTjwAAAABJRU5ErkJggg==
<Figure size 1000x600 with 1 Axes>
output_type": "display_data
name": "stdout
output_type": "stream
Best Model:

Model               XGBoost

Mean CV F1-Score       0.85

Test F1-Score          0.86

Name: 2, dtype: object

#Determining the best model

import matplotlib.pyplot as plt

import pandas as pd



# Example F1-score data for models

data = {

    \"Model\": [\"Logistic Regression\", \"Random Forest\", \"XGBoost\", \"SVM\", \"KNN\"],

    \"Mean CV F1-Score\": [0.78, 0.81, 0.85, 0.76, 0.74],

    \"Test F1-Score\": [0.79, 0.82, 0.86, 0.77, 0.75]

}



# Convert to DataFrame

df = pd.DataFrame(data)



# Plot Mean CV F1-Score and Test F1-Score

plt.figure(figsize=(10, 6))

plt.bar(df[\"Model\"], df[\"Mean CV F1-Score\"], alpha=0.7, label=\"Mean CV F1-Score\")

plt.bar(df[\"Model\"], df[\"Test F1-Score\"], alpha=0.7, label=\"Test F1-Score\", width=0.5)

plt.ylabel(\"F1-Score\")

plt.title(\"Comparison of F1-Scores Across Models\")

plt.legend()

plt.xticks(rotation=45)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

plt.show()



# Determine the best model based on Test F1-Score

best_model_row = df.loc[df[\"Test F1-Score\"].idxmax()]

print(\"Best Model:\")

print(best_model_row)

cell_type": "code
id": "5b723881
['best_model.pkl']
output_type": "execute_result
#saving the best model using joblib and saving the pikle file - weights

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier

import joblib



# Load the dataset

data = pd.read_csv(\"adult_income.csv\")  # Replace with the actual dataset file



# Define feature columns and target column

X = data.drop(\"income\", axis=1)  # Replace 'income' with the actual target column name

y = data[\"income\"]              # Replace 'income' with the actual target column name



# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Preprocessing for numeric and categorical columns

numeric_features = ['age', 'fnlwgt', 'education_num', 'hours_per_week', 'capital_gain', 'capital_loss']

categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']



numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(handle_unknown='ignore')



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)

    ]

)



# Define the model pipeline

model_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('classifier', RandomForestClassifier())

])



# Train the model

model_pipeline.fit(X_train, y_train)



# Save the model

joblib.dump(model_pipeline, 'best_model.pkl')

cell_type": "code
id": "3a0d4df7
# Saved the best model pk file
cell_type": "code
id": "852e2cdc
# @app.post(\"/predict\")

# def predict(input_data: ModelInput):

#     try:

#         # Convert input to DataFrame

#         data = pd.DataFrame([input_data.dict()])

        

#         # Apply preprocessing if not part of the model

#         # Example: encode categorical variables

#         # data = preprocess_function(data)



#         # Make prediction

#         prediction = model.predict(data)

#         return {\"prediction\": int(prediction[0])}

#     except Exception as e:

#         print(\"Error during prediction: \", e)

#         return {\"error\": str(e)}

cell_type": "code
id": "52b886dc
# @app.post(\"/predict\")

# def predict(input_data: ModelInput):

#     try:

#         data = pd.DataFrame([input_data.dict()])



#         # Reorder columns to match the training data

#         data = data[[\"age\", \"workclass\", \"fnlwgt\", \"education\", \"education_num\",

#                      \"marital_status\", \"occupation\", \"relationship\", \"race\",

#                      \"sex\", \"capital_gain\", \"capital_loss\", \"hours_per_week\", \"native_country\"]]



#         prediction = model.predict(data)

#         return {\"prediction\": int(prediction[0])}

#     except Exception as e:

#         print(\"Error: \", e)

#         return {\"error\": str(e)}

cell_type": "code
id": "d21c9e94
# # verifying the feature output 

# @app.post(\"/predict\")

# def predict(input_data: ModelInput):

#     try:

#         data = pd.DataFrame([input_data.dict()])



#         # Reorder columns to match the training data

#         data = data[[\"age\", \"workclass\", \"fnlwgt\", \"education\", \"education_num\",

#                      \"marital_status\", \"occupation\", \"relationship\", \"race\",

#                      \"sex\", \"capital_gain\", \"capital_loss\", \"hours_per_week\", \"native_country\"]]



#         prediction = model.predict(data)

#         return {\"prediction\": int(prediction[0])}

#     except Exception as e:

#         print(\"Error: \", e)

#         return {\"error\": str(e)}

cell_type": "code
id": "caf1936b
name": "stdout
output_type": "stream
Model trained and saved successfully!

name": "stderr
output_type": "stream
INFO:     Started server process [5160]

INFO:     Waiting for application startup.

INFO:     Application startup complete.

INFO:     Uvicorn running on http://127.0.0.1:8003 (Press CTRL+C to quit)

name": "stdout
output_type": "stream
INFO:     127.0.0.1:50782 - \"GET / HTTP/1.1\" 200 OK

INFO:     127.0.0.1:50782 - \"GET /favicon.ico HTTP/1.1\" 404 Not Found

INFO:     127.0.0.1:50788 - \"GET /predict HTTP/1.1\" 405 Method Not Allowed

INFO:     127.0.0.1:50844 - \"GET /docs HTTP/1.1\" 200 OK

INFO:     127.0.0.1:50844 - \"GET /openapi.json HTTP/1.1\" 200 OK

INFO:     127.0.0.1:50904 - \"POST /predict HTTP/1.1\" 200 OK

name": "stderr
output_type": "stream
C:\\Users\\lenovo\\AppData\\Local\\Temp\\ipykernel_5160\\3324803211.py:106: PydanticDeprecatedSince20: The `dict` method is deprecated; use `model_dump` instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/

  data = pd.DataFrame([input_data.dict()])

# Required Libraries

# creating the fast API and running the model to serve the model and got the output sucessfully

import joblib

import pandas as pd

from fastapi import FastAPI

from pydantic import BaseModel

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder



# File paths for saved model and label encoder

MODEL_PATH = \"best_model.pkl\"

LABEL_ENCODER_PATH = \"label_encoder.pkl\"



# Initialize FastAPI

app = FastAPI()



# Input Data Model

class InputData(BaseModel):

    age: int

    workclass: str

    fnlwgt: int

    education: str

    education_num: int

    marital_status: str

    occupation: str

    relationship: str

    race: str

    sex: str

    capital_gain: int

    capital_loss: int

    hours_per_week: int

    native_country: str



# Function to train the model

def train_model():

    # Load dataset

    data = pd.read_csv(\"adult_income.csv\")  # Ensure you have the correct dataset in the same directory

    

    # Define feature columns and target

    feature_columns = [

        \"age\", \"workclass\", \"fnlwgt\", \"education\", \"education_num\",

        \"marital_status\", \"occupation\", \"relationship\", \"race\", \"sex\",

        \"capital_gain\", \"capital_loss\", \"hours_per_week\", \"native_country\"

    ]

    target_column = \"income\"

    

    X = data[feature_columns]

    y = data[target_column]



    # Label encode the target column

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y)



    # Save label encoder

    joblib.dump(label_encoder, LABEL_ENCODER_PATH)



    # Preprocessing for numerical and categorical features

    numeric_features = [\"age\", \"fnlwgt\", \"education_num\", \"capital_gain\", \"capital_loss\", \"hours_per_week\"]

    numeric_transformer = StandardScaler()



    categorical_features = [

        \"workclass\", \"education\", \"marital_status\", \"occupation\",

        \"relationship\", \"race\", \"sex\", \"native_country\"

    ]

    categorical_transformer = OneHotEncoder(handle_unknown=\"ignore\")



    preprocessor = ColumnTransformer(

        transformers=[

            (\"num\", numeric_transformer, numeric_features),

            (\"cat\", categorical_transformer, categorical_features),

        ]

    )



    # Define the model pipeline

    pipeline = Pipeline(steps=[(\"preprocessor\", preprocessor), (\"classifier\", RandomForestClassifier())])



    # Split the data into train and test sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    # Train the pipeline

    pipeline.fit(X_train, y_train)



    # Save the trained model

    joblib.dump(pipeline, MODEL_PATH)

    print(\"Model trained and saved successfully!\")



# Uncomment this line to train the model and create the necessary files

train_model()



# Load the model and label encoder

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(LABEL_ENCODER_PATH)



@app.get(\"/\")

def read_root():

    return {\"message\": \"Welcome to the Income Prediction API. Use /predict to make predictions.\"}



@app.post(\"/predict\")

def predict(input_data: InputData):

    try:

        # Convert input data to DataFrame

        data = pd.DataFrame([input_data.dict()])



        # Make prediction

        prediction = model.predict(data)

        predicted_label = label_encoder.inverse_transform(prediction)[0]



        return {\"prediction\": predicted_label}

    except Exception as e:

        return {\"error\": str(e)}



# For running in Jupyter Notebook

if __name__ == \"__main__\":

    import nest_asyncio

    import uvicorn



    nest_asyncio.apply()

    uvicorn.run(app, host=\"127.0.0.1\", port=8003)

cell_type": "code
id": "9c95fdf3
# #code running it on anoother note book - got an output

# import requests



# # URL of the running FastAPI server

# url = \"http://127.0.0.1:8003/predict\"



# # Input data for prediction (modify values as needed)

# input_data = {

#     \"age\": 35,

#     \"workclass\": \"Private\",

#     \"fnlwgt\": 215646,

#     \"education\": \"Bachelors\",

#     \"education_num\": 13,

#     \"marital_status\": \"Married-civ-spouse\",

#     \"occupation\": \"Exec-managerial\",

#     \"relationship\": \"Husband\",

#     \"race\": \"White\",

#     \"sex\": \"Male\",

#     \"capital_gain\": 0,

#     \"capital_loss\": 0,

#     \"hours_per_week\": 40,

#     \"native_country\": \"United-States\"

# }



# # Send POST request

# response = requests.post(url, json=input_data)



# # Print the response

# print(\"Status Code:\", response.status_code)

# print(\"Response Body:\", response.json())



cell_type": "code
id": "f43c00ab
display_name": "Python 3 (ipykernel)
language": "python
name": "python3
name": "ipython
file_extension": ".py
mimetype": "text/x-python
name": "python
nbconvert_exporter": "python
pygments_lexer": "ipython3
version": "3.11.5