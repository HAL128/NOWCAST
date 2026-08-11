# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# %%
us_monthly = pd.read_csv('../data/output/df_monthly_us.csv')
us_monthly_new = pd.read_csv('../data/output/df_monthly_us_new.csv')



# %%

df = pd.DataFrame()

df['date'] = us_monthly['date']

# Calculate job endings
df['current_month'] = us_monthly['indeed_job_postings_index_NSA']
df['previous_month'] = df['current_month'].shift(1)
df['new_postings'] = us_monthly_new['indeed_job_postings_index_NSA']
df['job_endings'] = df['new_postings'] - (df['current_month'] - df['previous_month'])

# Display results
print("Job Endings (positive values indicate job reductions):")
print(df[['date', 'job_endings']].dropna())

# %%
# Set graph style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Plot job endings trend
plt.plot(df['date'], df['job_endings'], 
         linewidth=2, color='#1f77b4', label='Job Endings')

# Graph settings
plt.title('Trend of Job Endings', fontsize=14, pad=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Job Endings', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.legend()

# Display and save graph
plt.tight_layout()
plt.savefig('../data/output/job_endings_trend.png')
plt.show()

# %%



