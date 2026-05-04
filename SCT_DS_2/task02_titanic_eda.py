import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor':'#1a1a2e','axes.facecolor':'#16213e',
    'axes.labelcolor':'white','xtick.color':'white',
    'ytick.color':'white','text.color':'white',
    'grid.color':'#2a2a4a','axes.titlesize':13,
    'axes.titleweight':'bold','axes.titlecolor':'white'
})
COLORS = ['#5DCAA5','#EF9F27','#378ADD','#D4537E','#7F77DD']

df = pd.read_csv('train.csv')
df.columns = df.columns.str.strip()
print(f"Loaded: {df.shape[0]} rows")

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)
df.drop(columns=['Name','Ticket','PassengerId'], inplace=True)
df['Pclass'] = df['Pclass'].astype(str)
df['AgeGroup'] = pd.cut(df['Age'],
    bins=[0,12,18,35,60,100],
    labels=['Child','Teen','Adult','Middle','Senior'])
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Survived_Label'] = df['Survived'].map({0:'Died',1:'Survived'})
print(f"Cleaned! Survival Rate: {df['Survived'].mean()*100:.1f}%")

# CHART 1 - Overview Dashboard
fig = plt.figure(figsize=(18,12))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('Titanic EDA — Overview Dashboard',
             fontsize=18, fontweight='bold', color='white', y=0.98)
gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0,0])
counts = df['Survived_Label'].value_counts()
b1 = ax1.bar(counts.index, counts.values,
             color=['#5DCAA5','#EF4444'], edgecolor='white', width=0.5)
for b,v in zip(b1,counts.values):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+5,
             str(v), ha='center', fontsize=12, fontweight='bold', color='white')
ax1.set_title('Overall Survival Count')
ax1.set_ylabel('Passengers')

ax2 = fig.add_subplot(gs[0,1])
g2 = df.groupby('Sex')['Survived'].mean()*100
b2 = ax2.bar(g2.index, g2.values,
             color=['#378ADD','#D4537E'], edgecolor='white', width=0.4)
for b,v in zip(b2,g2.values):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+1,
             f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold', color='white')
ax2.set_title('Survival Rate by Gender')
ax2.set_ylim(0,100)

ax3 = fig.add_subplot(gs[0,2])
g3 = df.groupby('Pclass')['Survived'].mean()*100
b3 = ax3.bar(['1st','2nd','3rd'], g3.values,
             color=COLORS[:3], edgecolor='white', width=0.5)
for b,v in zip(b3,g3.values):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+1,
             f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold', color='white')
ax3.set_title('Survival by Class')
ax3.set_ylim(0,100)

ax4 = fig.add_subplot(gs[1,0])
ax4.hist(df[df['Survived']==0]['Age'], bins=25, alpha=0.7,
         color='#EF4444', label='Died', edgecolor='white')
ax4.hist(df[df['Survived']==1]['Age'], bins=25, alpha=0.7,
         color='#5DCAA5', label='Survived', edgecolor='white')
ax4.set_title('Age Distribution')
ax4.set_xlabel('Age')
ax4.legend(facecolor='#1a1a2e', labelcolor='white')

ax5 = fig.add_subplot(gs[1,1])
g5 = df.groupby('AgeGroup', observed=True)['Survived'].mean()*100
b5 = ax5.bar(g5.index, g5.values, color=COLORS, edgecolor='white')
for b,v in zip(b5,g5.values):
    ax5.text(b.get_x()+b.get_width()/2, b.get_height()+1,
             f'{v:.0f}%', ha='center', fontsize=10, fontweight='bold', color='white')
ax5.set_title('Survival by Age Group')
ax5.set_ylim(0,100)

ax6 = fig.add_subplot(gs[1,2])
df['Port'] = df['Embarked'].map({'S':'Southampton','C':'Cherbourg','Q':'Queenstown'})
g6 = df.groupby('Port')['Survived'].mean()*100
b6 = ax6.bar(g6.index, g6.values, color=COLORS[:3], edgecolor='white', width=0.4)
for b,v in zip(b6,g6.values):
    ax6.text(b.get_x()+b.get_width()/2, b.get_height()+1,
             f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold', color='white')
ax6.set_title('Survival by Port')
ax6.set_ylim(0,100)

plt.savefig('chart1_overview_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
print("Saved chart1!")
plt.show()

# CHART 2 - Heatmap
fig2, ax = plt.subplots(figsize=(10,7))
fig2.patch.set_facecolor('#1a1a2e')
num_df = df[['Survived','Age','SibSp','Parch','Fare','FamilySize']].copy()
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, ax=ax, linewidths=0.5,
            annot_kws={'size':12,'weight':'bold'})
ax.set_title('Correlation Heatmap', pad=15)
plt.tight_layout()
plt.savefig('chart2_correlation_heatmap.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
print("Saved chart2!")
plt.show()

# CHART 3 - Advanced
fig3, axes = plt.subplots(1, 3, figsize=(18,6))
fig3.patch.set_facecolor('#1a1a2e')
fig3.suptitle('Advanced Relationships', fontsize=16,
              fontweight='bold', color='white')

cg = df.groupby(['Pclass','Sex'])['Survived'].mean().unstack()*100
x = np.arange(len(cg.index))
bm = axes[0].bar(x-0.175, cg['male'],   0.35, label='Male',
                 color='#378ADD', edgecolor='white')
bf = axes[0].bar(x+0.175, cg['female'], 0.35, label='Female',
                 color='#D4537E', edgecolor='white')
for b in list(bm)+list(bf):
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+1,
                 f'{b.get_height():.0f}%', ha='center',
                 fontsize=9, fontweight='bold', color='white')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['1st','2nd','3rd'])
axes[0].set_title('Class and Gender')
axes[0].set_ylim(0,110)
axes[0].legend(facecolor='#1a1a2e', labelcolor='white')

axes[1].hist(df[df['Survived']==0]['Fare'], bins=30, alpha=0.7,
             color='#EF4444', label='Died', edgecolor='white')
axes[1].hist(df[df['Survived']==1]['Fare'], bins=30, alpha=0.7,
             color='#5DCAA5', label='Survived', edgecolor='white')
axes[1].set_title('Fare Distribution')
axes[1].set_xlim(0,300)
axes[1].legend(facecolor='#1a1a2e', labelcolor='white')

fs = df.groupby('FamilySize')['Survived'].mean()*100
axes[2].bar(fs.index, fs.values, color=COLORS*3, edgecolor='white')
for xv,v in zip(fs.index,fs.values):
    axes[2].text(xv, v+1, f'{v:.0f}%', ha='center',
                 fontsize=9, fontweight='bold', color='white')
axes[2].set_title('Family Size Survival')
axes[2].set_xticks(fs.index)
axes[2].set_ylim(0,110)

plt.tight_layout()
plt.savefig('chart3_advanced_relationships.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
print("Saved chart3!")
plt.show()

print("\nAll 3 charts done!")
print(f"Female survival: {df[df['Sex']=='female']['Survived'].mean()*100:.1f}%")
print(f"Male survival:   {df[df['Sex']=='male']['Survived'].mean()*100:.1f}%")