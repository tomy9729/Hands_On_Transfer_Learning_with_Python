# -*- coding: utf-8 -*-
"""
파이썬을 통해 배우는 딥러닝 전이학습
1장 머신러닝의 기초 원리
탐색적 데이터 분석
왕좌의 게임 데이터 
"""

import numpy as np
import pandas as pd
from collections import Counter

#시각화
import seaborn as sns
import matplotlib.pyplot as plt

#params 세팅
params = {'legend.fontsize' : 'x-large',
          'figure.figsize' : (30,10),
          'axes.labelsize' : 'x-large',
          'axes.titlesize' : 'x-large',
          'xtick.labelsize' : 'x-large',
          'ytick.labelsize' : 'x-large'}

sns.set_style('whitegrid')
sns.set_context('talk')

plt.rcParams.update(params)

battles_df = pd.read_csv('battles.csv')

battles_df

#여러 해에 걸친 전투의 분포
sns.countplot(y='year', data=battles_df)
plt.title('Battle Distribution over Years')
plt.show()

#지역별 전투
sns.countplot(x = 'region', data=battles_df)
plt.title('Battles by Regions')
plt.show()

sns.countplot(x = 'defender_king', data=battles_df)
plt.title('defender_king')
plt.show()

#왕별로 공격한 전투
attacker_king = battles_df.attacker_king.value_counts()
attacker_king.name = ''  # y-axis-label을 off로 해도 된다.
attacker_king.plot.pie(figsize=(6,6),autopct = '%.2f')

#왕별로 승리한 횟수
attack_winners = battles_df[battles_df.\
                            attacker_outcome=='win']\
                                ['attacker_king'].\
                                value_counts().\
                                reset_index()

attack_winners.rename(
    columns={'index':'king',
             'attacker_king':'wins'},
             inplace = True)

attack_winners.loc[:,'win_type'] = 'attack'

defend_winners = battles_df[battles_df.\
                            attacker_outcome == 'loss']\
                            ['defender_king'].\
                            value_counts().\
                            reset_index()

defend_winners.rename(
    columns = {'index':'king',
               'defender_king' : 'wins'},
               inplace=True)

defend_winners.loc[:,'win_type'] = 'defend'

sns.barplot(x="king",
            y="wins",
            hue = "win_type",
            data=pd.concat([attack_winners,
                            defend_winners]))
plt.title('King and Their Wins')
plt.ylabel('wins')
plt.xlabel('king')
plt.show()

# 대립하는 왕들
temp_df = battles_df.dropna(
    subset = ["attacker_king","defender_king"])[
              ["attacker_king","defender_king"]
              ]
archenemy_df = pd.DataFrame(
    list(Counter(
        [tuple(set(king_pair))
        for king_pair in temp_df.values
         if len(set(king_pair))>1]).\
          items()),
        columns=['king_pair',
                 'battle_count'])
archenemy_df['versus_text'] = archenemy_df.\
                                apply(
                                    lambda row : 
                                    '{} Vs {}'.format(row['king_pair'][0], row['king_pair'][1]),axis=1)
archenemy_df.sort_values('battle_count',
                         inplace = True,
                         ascending = False)

archenemy_df[['versus_text','battle_count']].set_index('versus_text',inplace = True)

sns.barplot(data = archenemy_df,
            x = 'versus_text',
            y = 'battle_count')

plt.xticks(rotation = 45)
plt.xlabel('Arechenemies')
plt.ylabel('Number of Battles')
plt.title('Archenemies')
plt.show()

#전쟁 종류
sns.countplot(y='battle_type',data=battles_df)
plt.title('Battle Type Distribution')
plt.show()

sns.countplot(y='attacker_outcome',data=battles_df)
plt.title('Attack Win/Loss Distribution')
plt.show()

battles_df['attacker_house_count'] = (4 - battles_df[['attacker_1', 
                                                'attacker_2', 
                                                'attacker_3', 
                                                'attacker_4']].\
                                        isnull().sum(axis = 1))

battles_df['defender_house_count'] = (4 - battles_df[['defender_1',
                                                'defender_2', 
                                                'defender_3', 
                                                'defender_4']].\
                                        isnull().sum(axis = 1))

battles_df['total_involved_count'] = battles_df.apply(lambda row: \
                                      row['attacker_house_count'] + \
                                      row['defender_house_count'],
                                                      axis=1)
battles_df['bubble_text'] = battles_df.apply(lambda row: \
          '{} had {} house(s) attacking {} house(s) '.\
          format(row['name'],
                 row['attacker_house_count'],
                 row['defender_house_count']),
                 axis=1)

#공격할 때와 방어할 때 동맹 수
house_balance = battles_df[
        battles_df.attacker_house_count != \
        battles_df.defender_house_count][['name',
                                       'attacker_house_count',
                                       'defender_house_count']].\
        set_index('name')
house_balance.plot(kind='bar')

#공격할 때와 수비할 때의 군대 규모
army_size_df = battles_df.dropna(subset=['total_involved_count',
                          'attacker_size',
                          'defender_size',
                         'bubble_text'])
army_size_df.plot(kind='scatter', x='defender_size',y='attacker_size',
                  s=army_size_df['total_involved_count']*150)

