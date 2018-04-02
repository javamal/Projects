import requests
import re
import matplotlib
import scipy.stats
import os
import numpy as np
import pandas as pd

#update 4.1.2018 - following code needs update. google no longer provides historical tables in the finance url

def source(ticker,market,start,end,page):
    url = "https://www.google.com/finance/historical?q="+market+"%3A"+ticker+"&startdate="+start[0]+"+"+start[1]+"%2C+"+start[2]+"&enddate="+end[0]+"+"+end[1]+"%2C+"+end[2]+"&start="+page
    print(url)
    source = requests.get(url)
    try:
        source.raise_for_status()
    except:
        return("download failed")
    default = "C:\\Users\\Turtle\\Desktop\\machine learning"
    create = open(os.path.join(default,"sourcecode.txt"),"wb") #binary write
    
    for i in source:
        create.write(i) 
    create.close()
    create = open(os.path.join(default,"sourcecode.txt"),"r")
    text=create.read()
    create.close()
    return(text)

def price_table(sys,market,start,end):
    table = pd.DataFrame([])
    i = 0 #page counter
    while True:
        source_text = source(sys,market,start,end,str(i*30))       
        date = pd.DataFrame(re.findall("<td class=\"lm\">(.+)\n",source_text))
        addition = np.array(re.findall("<td class=\"rgt\">(.+)\n",source_text))
        if len(date) == 0 and len(addition) == 0:
            break        
        feature = pd.DataFrame(np.reshape(addition,(int(len(addition)/4),4)))            
        df = pd.concat([date,feature],axis = 1) #add date and feature column bind
        table = pd.concat([table,df],axis = 0).reset_index(drop = True) #for each page, row bind
                           
        i = i + 1
    if len(table) == 0:
        return("error")
    else:
        table.columns = ["Date","Open","High","Low","Close"]
        return(table)
    
def delta(df,attribute,up = True): #up = True determines the direction of time series
    for i in range(len(df)):
        df.ix[i,attribute] = re.sub(",","",df.ix[i,attribute])
    df = df.convert_objects(convert_numeric = True)        
    top = df.ix[0:len(df)-2,:].reset_index(drop = True) 
    #array addition, deduction, etc. are based on designated index, not actual index within array
    down = df.ix[1:(len(df)),:].reset_index(drop = True)
    if up == True:
        dif = top.ix[:,attribute]-down.ix[:,attribute]
        change = dif/down.ix[:,attribute]
    else:
        dif = -top.ix[:,attribute]+down.ix[:,attribute]
        change = dif/top.ix[:,attribute]    
    return(change)

def beta(sys,market,comp_tick,comp_market,start,end):
    #creating percent change arrays
    a=price_table(sys,market,start,end)
    b=price_table(comp_tick,comp_market,start,end)
    if type(a)==str or type(a)==str:
        return("Error - Check for a)Ticker-"+sys+", b)Listing period, c)Listed market-"+market)
    else:
        comp=a.ix[:,["Date","Close"]]
        sp=b.ix[:,["Date","Close"]]  
        comp_delta=delta(comp,"Close")
        sp_delta=delta(sp,"Close")
        result=pd.concat([comp_delta,sp_delta],axis=1)
        result.columns=[sys,comp]
        return(result)
        
def beta_plot(sys,market,comp_tick,comp_market,start,end):
    #ticker1, ticker1 market, ticker2, ticker2 market, start, end
    df = beta(sys,market,comp_tick,comp_market,start,end)
    if df.ix[:,0].isnull().any() == True or df.ix[:,1].isnull().any() == True:
        #df.isnull().any() or df.isnull().all()
        return("Missing Date on Ticker Price")    
    if type(df) == str:
        return("Error - Check for a)Ticker-" + sys + ", b)Listing period, c)Listed market-"+market)
    else:
        x = df.ix[:,0]
        y = df.ix[:,1]
        reg = scipy.stats.linregress(x,y)
        coef = reg[0]
        intercept = reg[1]
        fit = coef * x + intercept
        matplotlib.pyplot.plot(x,y,"rs",x,fit,"b--")
        matplotlib.pyplot.xlabel(sys)
        matplotlib.pyplot.ylabel(comp_tick)
        matplotlib.pyplot.title('Beta = '+str(coef))
        #matplotlib.pyplot.text(60, .025, r'$\mu=100,\ \sigma=15$')
        matplotlib.pyplot.grid(True)
        #matplotlib.pyplot.show()

price_table("GOOG","NASDAQ",["Oct","29","2014"],["Oct","29","2015"])
beta_plot("GOOG","NASDAQ",".INX","INDEXSP",["Jul","29","2014"],["Oct","29","2015"])
