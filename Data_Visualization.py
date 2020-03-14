#!/usr/bin/env python
# coding: utf-8

# # data Visualisation

# In[1]:


pip install pywaffle


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from pywaffle import Waffle
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[3]:


canada=pd.read_excel('Canada.xlsx',sheet_name="Canada by Citizenship",skiprows=range(20), skipfooter=2)
#skip row and footer to skip inreleveant data


# In[4]:


canada


# In[5]:


canada.info()


# In[6]:


canada.head()


# In[7]:


canada.tail()


# In[8]:


canada['AreaName'].value_counts()


# In[9]:


canada['OdName'].value_counts()


# In[10]:


# rename columns
canada.rename(columns={'OdName':'Country','AreaName':'Continent','RegName':'Region'},inplace=True)


# In[11]:


canada.head()


# # set index as country

# In[12]:


canada.set_index('Country',inplace=True)


# In[13]:


canada.drop(columns=['Type','Coverage','AREA','REG','DEV'],inplace=True,axis=1)


# # add new column with sum of all row

# In[14]:


canada['Total']=canada.sum(axis=1)
canada.head()


# # find null values

# In[15]:


canada.isnull().sum()


# In[ ]:





# # call columns 1980,1981,1982,1983,1984,1985

# In[16]:


canada[[1980,1981,1982,1983,1984,1985]]


# # find data of America Samoa

# In[17]:


canada.loc['American Samoa']


# # find data of algeria country

# In[18]:


canada.loc['Algeria']


# # find data of japan country from 1980 to 1984

# In[19]:


canada.loc['Japan', [1980,1981,1982,1983,1984]]


# # if country is equal to asia than true else false

# In[20]:


canada['Continent']=='Asia'
   


# # Draw a line plot for Immigration from Haiti from 2005-2013

# In[21]:


haiti_plot=canada.loc['Haiti', [2005,2006,2007,2008,2009,2010,2011,2012,2013]]


# In[22]:


haiti_plot.plot()


# # Plot line plot for eartquake

# In[23]:


haiti_plot.plot()
plt.text(2010,5000,'2010 Earthquake')
plt.show()


# # plot line graph for immigration India,China from 1980 to 2013

# In[24]:


india=canada.loc['India',[1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013]]
china=canada.loc['China',[1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013]]


# In[25]:


india.plot()
china.plot()
plt.xlabel('Year')
plt.ylabel('Immigration')


# In[26]:


years=[1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013]
ci=canada.loc[['India','China'],years]
ci.head()
ci.plot()


# # plot line graph for immigrants India,China from 1980 to 2013

# In[27]:


india=canada.loc['India',[1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013]]
china=canada.loc['China',[1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013]]
india.plot()
china.plot()
plt.xlabel('Year')
plt.ylabel('Immigrants')


# In[28]:


Chiindia=canada.loc[['India','China'],years]
Chiindia.head()
Chiindia.plot()


# In[29]:


Chiindia.transpose().head()


# # Compare the trend of top 5 countries

# In[30]:


top5=canada['Total'].nlargest(5)


# In[31]:


top5.plot.bar()


# In[32]:


canada.sort_values(by='Total',ascending=False,axis=0)
top=canada.head()
top=top[years].transpose()
top.plot(kind='line')


# # plot line graph for Immigrants for Top 5 countries

# In[33]:


top5=canada['Total'].nlargest(5)
top5.plot()
plt.xticks(rotation='vertical')


# # plot area plot for Immigration Trend of Top 5 Countries

# In[34]:


canada.sort_values(by='Total',ascending=False,axis=0)
top=canada.head()
top=top[years].transpose()
top.plot(kind='area',alpha=0.4,figsize=(20,10),stacked=False)


# # Use the scripting layer to create a stacked area plot of the 5 countries that contributed the least to immigration to Canada from 1980 to 2013. Use a transparency value of 0.45.

# In[35]:


canada.sort_values(by='Total',axis=0,)
top=canada.head()
top=top[years].transpose()
top.plot(kind='area',alpha=0.45,figsize=(20,10))


# # 9.Use the artist layer to create an unstacked area plot of the 5 countries that contributed the least to immigration to Canada from 1980 to 2013. Use a transparency value of 0.55.

# In[36]:


canada.sort_values(by='Total',axis=0,)
top=canada.head()
top=top[years].transpose()
top.plot(kind='area',stacked=False,alpha=0.55,figsize=(20,10))


# # What is the frequency distribution of the number (population) of new immigrants from the various countries to Canada in 2013?

# In[37]:


mi=canada[2013].min()
ma=canada[2013].max()


# In[38]:


canada[2013].plot.hist()


# In[39]:


canada[2013].head()


# # What is the immigration distribution for Denmark, Norway, and Sweden for years 1980 - 2013?

# In[40]:


g=canada.loc[['Denmark','Norway','Sweden'],[1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013]]
g=g.transpose()
g.plot(kind='hist',alpha=0.5)
plt.xlabel('Year')
plt.ylabel('Immigrants')
plt.show()


# # 12.Use the scripting layer to display the immigration distribution for Greece, Albania, and Bulgaria for years 1980 - 2013? Use an overlapping plot with 15 bins and a transparency value of 0.35.

# In[41]:


count,bin_edges=np.histogram(g,15)#for 15 bins
g=canada.loc[['Greece','Albania','Bulgaria'],[1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013]]
g=g.transpose()
g.plot(kind='hist',bins=15,figsize=(20,10),xticks=bin_edges,color=['deeppink','orange','midnightblue'])#for 15 bins
plt.xlabel('Year')
plt.ylabel('Immigrants')
plt.show()


# In[42]:


for name, hex in matplotlib.colors.cnames.items():
    print(name,hex)


# # compare the number of iceland immigrants to canada from year 1980-2013

# In[43]:


ice=canada.loc[['Iceland'],[1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013]]
ice=ice.transpose()
ice.plot(kind='bar',figsize=(20,10),color=['deeppink'])#for 15 bins
plt.annotate('',
            xy=(32, 70),
            xytext=(28, 20),
            xycoords='data',
            arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='blue')
            )
plt.annotate('2008-2011 Financial crisis',
             xy=(28,30),
             rotation=72.5,
             va='bottom',
             ha='left'
            )

plt.show()


# # Using the scripting layer and the df dataset, create a horizontal bar plot showing the total number of immigrants to Canada from the top 15 countries, for the period 1980 - 2013. Label each country with the total immigrant count.

# In[44]:


top15=canada['Total'].nlargest(15)
top15.plot(kind='barh',figsize=(12,12))
plt.title('The total number of immigrants to Canada from the top 15 countries, for the period 1980 - 2013')
count=0
for val in top15:
    plt.annotate(val,xy=(val+10000,count), color='black')
    count=count+1
plt.show()    


# # Box plot of Japanese Immigrants from 1980 - 2013

# In[45]:


japan=canada.loc['Japan',years]


# In[46]:


japan.plot(kind='box')


# # 16. Box Plots of Immigrants from China and India (1980 - 2013)

# In[47]:


indchi=canada.loc[['India','China'],years]
indchi=indchi[years].transpose()
indchi.plot(kind='box')


# In[48]:


fig=plt.figure()
indchi=canada.loc[['India','China'],years]
indchi=indchi[years].transpose()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)
indchi.plot(kind='box',vert=False,figsize=(20,10),ax=ax0)
indchi.plot(kind='line',ax=ax1,figsize=(20,10))


# # plot scatter for Total Immigration to Canada from 1980 - 2013

# In[49]:


tot=pd.DataFrame(canada[years].sum(axis=0))
tot.reset_index(inplace=True)
tot.columns=['years','total']

tot.years=list(map(int,tot.years))
tot.head()


# In[50]:


tot.plot(kind='scatter',x='years',y='total',color='red')


# # plot line graph Total Immigration to Canada from 1980 - 2013

# In[51]:


tot['total'].plot(kind='line', figsize=(20,10))

plt.show()


# # Create a scatter plot of the total immigration from Denmark, Norway, and Sweden to Canada from 1980 to 2013?

# In[52]:


g=canada.loc[['Denmark','Norway','Sweden'],years]
g=g.transpose()
g.head()


# # plot pie for top 5 continent

# In[53]:


continent=canada.set_index('Continent')


# In[54]:


continent=continent.groupby('Continent',axis=0,).sum()
continent.head()


# In[55]:


continent['Total'].plot(kind ='pie',
                       figsize=(20,10),
                       autopct='%1.1f%%',
                         colors = ['gold', 'yellowgreen', 'lightcoral', 'deeppink','deepskyblue','crimson'],
                        explode = (0, 0, 0, 0.1,0,0.1),
                       #startangle=90,
                       shadow=True)


# In[56]:


trans=canada[years]


# In[57]:


trans=trans.transpose()


# In[58]:


trans.head()
trans.index=map(int,trans.index)
trans.index.name='years'
trans.reset_index(inplace=True)


# # normalize Brazil & Argentina data

# In[59]:


norm_brazil=(trans['Brazil']-trans['Brazil'].min())/(trans['Brazil'].max()-trans['Brazil'].min())


# In[60]:


trans['Brazil']


# In[61]:


norm_argentina=(trans['Argentina']-trans['Argentina'].min())/(trans['Argentina'].max()-trans['Argentina'].min())


# In[62]:


trans['Argentina']


# In[63]:


trans.plot(kind='scatter',x='years',y='Brazil',color='red')
trans.plot(kind='scatter',x='years',y='Argentina',color='green')


# # plot scatter for Immigration from Brazil and Argentina from 1980 - 2013

# In[64]:


fig=plt.figure()

ax0=trans.plot(kind='scatter',
          x='years',
          y='Brazil',
          alpha=0.5,
          figsize=(20,10),     
          color='red',
          s=norm_brazil*2000 +10,
          xlim=(1975,2015))

ax1=trans.plot(kind='scatter',
          x='years',
          y='Argentina',
          alpha=0.5,
          color='blue',
          s=norm_argentina*2000 +10,
          ax=ax0)
ax0.legend(['Brazil','Argentina'],loc='upper left', fontsize=25)

plt.title('Immigration from Brazil and Argentina from 1980 - 2013',fontsize=25)
plt.show()


# # create waffle 

# In[65]:


dataframe= {'crime_type':['felony','misdemenor','violaton'],
           'number_of_cases':[54,12,8]}
df=pd.DataFrame(dataframe)
df


# In[66]:


fig=plt.figure(FigureClass=Waffle,
              rows= 8,
              values=df.number_of_cases,
              labels=list(df.crime_type),
              legend={'loc':'upper left', 'bbox_to_anchor':(1.1,1)})


# In[67]:


dataframe= {'district':['District 12','District 23','District 32','District 45','District 65','District 67','District 33'],
           'party':['Republican','Republican','Republican','Republican','Democratic','Democratic','Democratic']}
dis=pd.DataFrame(dataframe)
dis


# In[68]:


count=dis.party.value_counts()
count


# In[69]:


fig=plt.figure(FigureClass=Waffle,
              rows=2,
              values=count,
              labels=list(count.index)#for legend
              )


# # create waffle chart for ['Denmark', 'Norway', 'Sweden']

# In[70]:


waffle=canada.loc[['Denmark','Norway','Sweden'],'Total']
waffle


# In[71]:


# STEP 1
show=canada.loc[['Denmark','Norway','Sweden'],:]
total=sum(show['Total'])
category_proportion=[(float(value)/total)for value in show['Total']]
category_proportion


# In[72]:


# STEP 2
width=40
height=10
total_tiles= width*height
print('total number of tiles', total_tiles)


# In[73]:


# Step 3
tiles_per_category=[round(proportion*total_tiles)for proportion in category_proportion]
tiles_per_category


# In[74]:


#initialise the waffle chart as an empty matrix
waffle_chart=np.zeros((height,width))

#define indices to loop waffle chart
category_index=0
tile_index=0

#populate the waffle chart
for col in range(width):
    for row in range(height):
        tile_index = tile_index + 1
        
        #if the number of tiles popilated for the current category is equal to
        if tile_index>sum(tiles_per_category[0:category_index]):
            
            # proceed to the next category
            category_index = category_index + 1
            
        #set the class values to an integerwhich increase with class
        waffle_chart[row,col] = category_index


# In[75]:


#initiate anew figure object
fig=plt.figure()

#use matshow to display the waffle chart
colormap=plt.cm.coolwarm
plt.matshow(waffle_chart,cmap=colormap)
plt.colorbar()
plt.show()


# In[76]:


#step 6
#initiate anew figure object
fig=plt.figure()

#use matshow to display the waffle chart
colormap=plt.cm.coolwarm
plt.matshow(waffle_chart,cmap=colormap)
plt.colorbar()


#get the axis
ax= plt.gca()

#set minor ticks
ax.set_xticks(np.arange(-.5,(width),1),minor=True)
ax.set_yticks(np.arange(-.5,(height),1),minor=True)

#add gridlines based on minor ticks
ax.grid(which='minor',color='w',linestyle='-',linewidth=2)

plt.xticks([])
plt.yticks([])


#compute cumlative sum of individual category to match color scheme between
values_cumsum = np.cumsum(show['Total'])
total_values=values_cumsum[len(values_cumsum)-1]

#create legend
legend_handles=[]
for i, category in enumerate(show.index.values):
    label_str=category +'('+ str(show['Total'][i])+')'
    color_val= colormap(float(values_cumsum[i])/total_values)
    legend_handles.append(mpatches.Patch(color=color_val,label=label_str)) #import matplotlib.patches as mpatches
    
#add legend
plt.legend(handles=legend_handles,
          loc='lower center',
          ncol=len(show.index.values),
          bbox_to_anchor=(0.,-0.2,0.95,0.1))


# In[81]:


pip install wordcloud


# In[82]:


from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS


# In[85]:


figure=plt.figure(figsize=(10,7))
text=canada.index[0:100]
wordcloud=WordCloud(background_color='White').generate(str(text))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:




