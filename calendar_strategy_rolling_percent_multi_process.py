# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 09:36:17 2021

@author: Viola
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

import mysql.connector
import os
import matplotlib.pyplot as plt
import sys
import multiprocessing
import statsmodels.api as sm


#set save path
file_path=r'Z:\金融工程\WYF\option_calendar_strategy'
save_path=os.path.join(file_path,"backtest_results_rolling_percent_htm")
if os.path.exists(save_path)== False:
    os.mkdir(save_path)


signal_record_path_call=os.path.join(save_path,"signal_record","call")
if os.path.exists(signal_record_path_call)== False:
    os.makedirs( signal_record_path_call)

signal_record_path_put=os.path.join(save_path,"signal_record","put")
if os.path.exists(signal_record_path_put)== False:
    os.makedirs( signal_record_path_put)

signal_boundary_call=os.path.join(save_path,"boundary_record","call")
if os.path.exists(signal_boundary_call)== False:
    os.makedirs(signal_boundary_call)
    
signal_boundary_put=os.path.join(save_path,"boundary_record","put")
if os.path.exists(signal_boundary_put)== False:
    os.makedirs(signal_boundary_put)

signal_trade_call=os.path.join(save_path,"trade_record","call")
if os.path.exists(signal_trade_call)== False:
    os.makedirs(signal_trade_call)
    
signal_trade_put=os.path.join(save_path,"trade_record","put")
if os.path.exists(signal_trade_put)== False:
    os.makedirs(signal_trade_put)
    
rst_call=os.path.join(save_path,"call")
if os.path.exists(rst_call)== False:
    os.makedirs(rst_call)

rst_put=os.path.join(save_path,"put")
if os.path.exists(rst_put)== False:
    os.makedirs(rst_put)

rst_call_ret=os.path.join(save_path,"call","df_ret")
if os.path.exists(rst_call_ret)== False:
    os.makedirs(rst_call_ret)

rst_put_ret=os.path.join(save_path,"put","df_ret")
if os.path.exists(rst_put_ret)== False:
    os.makedirs(rst_put_ret)



def get_data(sql):
    
    db= mysql.connector.connect(
        host="10.10.101.24",      # address
        user="read",    # user name
        passwd="read" )   # password
    
    db_cursor=db.cursor()
    db_cursor.execute(sql)
    data=db_cursor.fetchall()
    db.close()
    df_data=pd.DataFrame(data,columns=[x[0] for x in  db_cursor.description])
    return df_data



def get_current_near_atm_option(x,near_sort=0,method=None):
     
    l_expiry=np.sort(x["Expiration"].unique())
    min_expiry=l_expiry[near_sort]
    data_daily_near=x[x["Expiration"]==min_expiry]
    data_daily_near["criterion"]= abs(data_daily_near["Delta"])-0.5
    
    def find_near_atm():
        min_diff=np.min(abs(data_daily_near["criterion"]))
        df_atm=data_daily_near[data_daily_near["criterion"]==min_diff]
        if len(df_atm)==0:
            df_atm=data_daily_near[data_daily_near["criterion"]==-min_diff]
            
        if len(df_atm) !=1:
            df_atm=df_atm.drop_duplicates(subset="Delta")
        return df_atm

    if method is None:
    #find delta near 0.5,then get the close price
        df_atm=find_near_atm()
        df_atm=df_atm.iloc[:,:-1]

    elif method=="avg":
    #find delta below and above 0.5, the close price is avg
        df_up=data_daily_near[data_daily_near["criterion"]>=0]
        df_down=data_daily_near[data_daily_near["criterion"]<0]
        
        if len(df_up)!=0 and len(df_down)!=0:
            df_up_min=df_up[df_up["criterion"]==np.min(df_up["criterion"])]
            df_down_max=df_down[df_down["criterion"]==np.max(df_down["criterion"])]
            avg_price=np.mean([df_up_min["PX"].values[0],df_down_max["PX"].values[0]])
            df_up_min.loc[:,"PX"]=avg_price
            df_atm=df_up_min.iloc[:,:4]
        else:
            df_atm=find_near_atm().iloc[:,:-1]

    else:
        raise TypeError("Method not support")
     
    return df_atm


class calender_strategy():
     
    def __init__(self,iv_ratio,roll_iv_ratio_percent,\
                 upper_bound,close_bound,close_method):
         
        #input data
        self.iv_ratio=iv_ratio
        self.roll_iv_ratio_percent=roll_iv_ratio_percent
        self.trade_date=self.iv_ratio.index
        
        #signal setting
        self.upper_bound=upper_bound
        self.close_bound=close_bound
        self.close_method=close_method
        
        #record info
    
        #record when open a new position
        self.close_bound_hist=pd.DataFrame(columns=["close"])  
        self.status_record=pd.DataFrame(columns=["signal"])
        self.df_signal_symbol=pd.DataFrame(columns=["signal"])
        self.df_hold_position=pd.DataFrame(columns=["position"])
        self.df_boundary=pd.DataFrame()


    
    def _get_close_criterion(self,df_roll_percent):
        
        if len(self.close_bound_hist) !=0:
            close_hist=self.close_bound_hist.iloc[-1]["close"]
        else:
            close_hist=np.nan
        
        close_now=df_roll_percent[self.close_bound]

        #find close position criterion
        if self.close_method=="hist_percent":
            close_criterion= close_hist
                 
        elif self.close_method=="rolling_percent":
            close_criterion= close_now

        elif self.close_method=="comb_min":
            close_criterion= np.min([close_hist,close_now])
               
        elif self.close_method=="comb_max":
            close_criterion= np.max([close_hist,close_now])
        else:
            raise TypeError("Request method not support")
        return close_criterion

    def trade_signal(self,df_iv_ratio,df_roll_percent):
        pass
        
    def print_signal_status(self,date):
        
        signal=self.df_signal_symbol.iloc[-1]["signal"]
        position=self.df_hold_position.iloc[-1]["position"]
        #print signal & position
        print("Date{}:signal is {}, position is {}".format(date,signal,position))
   
        
    def run_signal(self):
        
        for date in self.trade_date:
            df_iv_daily=self.iv_ratio.loc[date]
            df_roll_percent_daily=self.roll_iv_ratio_percent.loc[date]
            self.trade_signal(df_iv_daily,df_roll_percent_daily)
            self.print_signal_status(date)


class sell_near_strategy(calender_strategy):
    
    def __init__(self,iv_ratio,roll_iv_ratio_percent,iv_ratio_type,current_contract,\
                 upper_bound,close_bound,close_method,d2e_min_open,d2e_min_close):
        
        super(sell_near_strategy,self).__init__(iv_ratio,roll_iv_ratio_percent,\
                 upper_bound,close_bound,close_method)
        
        self.iv_ratio_type=iv_ratio_type
            
        #get current contract info
        self.current_contract=current_contract
        
        #if d2e< min cannot open trade
        self.d2e_min_open=d2e_min_open
        #if d2e< min must close
        self.d2e_min_close=d2e_min_close

    
    def trade_signal(self,df_iv_ratio,df_roll_percent):
        
        date=df_iv_ratio.name
        iv_ratio=df_iv_ratio[self.iv_ratio_type]
        d2e=self.current_contract[self.current_contract["DATE"]==date]["D2E"].values[0]
        roll_upper=df_roll_percent[self.upper_bound]
        close_criterion=self._get_close_criterion(df_roll_percent)
        
        df_boundary_sub=pd.DataFrame(np.array([roll_upper,close_criterion]).reshape(1,2),\
                                    index=[date],columns=["upper_bound","close_bound"])
                
        self.df_boundary=self.df_boundary.append(df_boundary_sub)

        if np.isnan(roll_upper):
            self.status_record.loc[date]=[np.nan]
            self.df_signal_symbol.loc[date]=[np.nan]
            self.df_hold_position.loc[date]=[0]
        
        else:
            #check status
            if iv_ratio>=roll_upper:
                 self.status_record.loc[date]=[1]
                  
            else:
                self.status_record.loc[date]=[0]
                

            pre_hold_position=self.df_hold_position.iloc[-1]["position"]
            pre_status=self.status_record.iloc[-2]["signal"]
            current_status=self.status_record.iloc[-1]["signal"]

            #check whether there is position first
            #has position before
            if pre_hold_position !=0:
                
                if d2e <=self.d2e_min_close:
                    self.df_signal_symbol.loc[date]=[0]
                    self.df_hold_position.loc[date]=[0]

                else:
                    #check whether need to close position
                    if  iv_ratio<= close_criterion:
                        self.df_signal_symbol.loc[date]=[0]
                        self.df_hold_position.loc[date]=[0]
                    else:
                        self.df_signal_symbol.loc[date]=[np.nan]
                        self.df_hold_position.loc[date]= [pre_hold_position]

            #no position before
            else:
                #check whether can enter a position
                #generate positive signal

                #signal=1 (0/np.nan to 1)
                if ((pre_status == 0 and current_status==1) or (np.isnan(pre_status) and current_status==1)):
                    if d2e>= self.d2e_min_open:
                        self.df_signal_symbol.loc[date]=[1]
                        self.df_hold_position.loc[date]=[1]
                        self.close_bound_hist.loc[date]=[df_roll_percent[self.close_bound]]
                    else:
                       self.df_signal_symbol.loc[date]=[np.nan]
                       self.df_hold_position.loc[date]=[pre_hold_position]

                else:
                    self.df_signal_symbol.loc[date]=[np.nan]
                    self.df_hold_position.loc[date]=[pre_hold_position]


def get_trade(df_signal,df_option_current,df_option_next,greek_type):
       
    df_trade_info=pd.DataFrame()
    ratio_temp=[np.nan,np.nan]
    code_temp=[np.nan,np.nan]
    for date,df_signal in df_signal.iterrows():
        
        # position=df_hold_position.loc[date]["position"]
        signal=df_signal["signal"]
        
        #because option minite database misses data of '2019-12-25'
        try:
            opt_cut=df_option_current[df_option_current["DATE"]==date][["DATE","CODE",greek_type]]
            opt_nxt=df_option_next[df_option_next["DATE"]==date][["DATE","CODE",greek_type]]
            greek_cut,greek_nxt=opt_cut[greek_type].values[0],opt_nxt[greek_type].values[0]
            code_cut,code_nxt=opt_cut["CODE"].values[0],opt_nxt["CODE"].values[0]
        except:
            continue
        opt_cut1=opt_cut.iloc[:,:2]
        opt_nxt1=opt_nxt.iloc[:,:2]
        
        if np.isnan(signal):
             opt_cut1["ratio"]= 0
             opt_nxt1["ratio"]= 0
            # if position==0:
            #     opt_cut1["ratio"]=0
            #     opt_nxt1["ratio"]=0                        
            # else:
            #     opt_cut1["ratio"]=ratio_temp[0]
            #     opt_nxt1["ratio"]=ratio_temp[1]              
                                                     
        elif signal==0:
             opt_cut1["ratio"]= -ratio_temp[0]
             opt_nxt1["ratio"]= -ratio_temp[1]
             opt_cut1["CODE"]=code_temp[0]
             opt_nxt1["CODE"]=code_temp[1]

        else:
            if signal==1:
                cut_ratio=-(greek_nxt)/(greek_cut+greek_nxt)
                nxt_ratio=(greek_cut)/(greek_cut+greek_nxt) 
                opt_cut1["ratio"]=cut_ratio
                opt_nxt1["ratio"]=nxt_ratio
                ratio_temp[0]=cut_ratio
                ratio_temp[1]=nxt_ratio
                code_temp[0]=code_cut
                code_temp[1]=code_nxt
                
            else:
                cut_ratio=(greek_nxt)/(greek_cut+greek_nxt)
                nxt_ratio=-(greek_cut)/(greek_cut+greek_nxt) 
                opt_cut1["ratio"]=cut_ratio
                opt_nxt1["ratio"]=nxt_ratio
                ratio_temp[0]=cut_ratio
                ratio_temp[1]=nxt_ratio
                code_temp[0]=code_cut
                code_temp[1]=code_nxt
  
        df_trade=pd.concat([opt_cut1,opt_nxt1])
        df_trade_info=df_trade_info.append(df_trade)
        
    df_trade_info.columns=["date","code","ratio"]
    df_trade_info["date"]=pd.to_datetime(df_trade_info["date"])
    #df_trade_info=df_trade_info[df_trade_info["ratio"]!=0]
    df_trade_info.reset_index(drop=True,inplace=True)
    return df_trade_info

 #input ret series
class perform_metrics():

    def _nav(self,ret):
        nav=np.cumprod(1+ret)
        return nav
    
    def max_drawdown(self,ret):  
        nav=self._nav(ret)
        l_mdd=(nav/nav.cummax()-1).cummin()
        mdd=l_mdd.iloc[-1]
        return mdd
    
    def sharpe_ratio(self,ret):
        sharpe=np.sqrt(250.0)* np.mean(ret) / ((np.std(ret)**2)**(1/2))
        return sharpe
    
    def ret_yearly(self,ret):
        nav=self._nav(ret)
        nav=nav.reset_index()
        nav["year"]=[i.year for i in nav["date"]]
        df_ret_yearly=pd.DataFrame(nav.groupby(["year"])["ret"].apply(lambda x:x.iloc[-1]/x.iloc[0]-1))
        return df_ret_yearly
       
    # def annualized_ret(self,ret):
    #     ret=np.sum(ret)*250.0/(len(ret))
    #     return ret
    def annualized_ret(self,ret):
        nav=self._nav(ret)
        nav=nav.reset_index()
        nav["year"]=[i.year for i in nav["date"]]
        year=len(nav["year"].unique())
        ret=(nav.iloc[-1,1]/1)**(1/year)-1
        return ret
    
    def plot_nav(self,ret,title,save_path): 
        mdd=np.round(self.max_drawdown(ret),4)
        sharpe=np.round(self.sharpe_ratio(ret),4)
        annual_ret= np.round(self.annualized_ret(ret),4)
        nav=self._nav(ret)
        plt.figure(figsize=(15,10))
        plt.plot(nav)
        plt.xlabel("year")
        plt.ylabel("cummulative return")
        plt.title("cummulative return\nParam:{}\nShape:{}\nMax_Drawdown:{}\nAnnual_return:{}".format(title,sharpe,mdd,annual_ret))
        plt.grid()
        plt.savefig(save_path)


#backtest

path=r'Z:\金融工程\OPT\Backtesting'
sys.path.append(path)
import backtest_pnl

def get_backtest_rst(start_time,df_trade_info):

    opt, und = backtest_pnl.hist_data('510050','SSE',start_time)
    hedger = und.copy()
    opt=opt[opt["date"]<=df_trade_info["date"].iloc[-1]]
    und=und[und["date"]<=df_trade_info["date"].iloc[-1]]
    hedger=hedger[hedger["date"]<=df_trade_info["date"].iloc[-1]]  
    df_trade_info=df_trade_info[df_trade_info["date"]> pd.to_datetime(opt["date"].iloc[0])]
    df_trade_info.reset_index(drop=True,inplace=True)
    df_pos, df_trade, df_pnl = backtest_pnl.get_pnl('50',opt, und,hedger,1,df_trade_info,path)
    
    
    df_pnl1= df_pnl[['pnl', 'holdingpnl', 'tradingpnl', 'deltapnl',
   'gammapnl', 'thetapnl', 'vegapnl', 'residualpnl']].groupby(df_pnl['date']).sum()
    df_pos["abs_rmb_pos"]=np.abs(df_pos["rmb_pos"])
    df_pos1=df_pos.groupby(["date"]).sum()
    return df_pos1,df_trade,df_pnl1

def get_ret(df_pos,df_pnl,df_trade_info,margin_ratio=1):
    #find open position trade day
    trade_day=df_trade_info[df_trade_info["ratio"]!=0]
    trade_day=trade_day.drop_duplicates(subset="date")
    trade_day["ratio"]=np.abs(trade_day["ratio"])
    trade_day=trade_day.drop_duplicates(subset="ratio")
    trade_day["flag"]=1
    trade_day=trade_day[["date","flag"]]
    trade_day=trade_day[trade_day["date"]>=df_pos.index[0]]
    df_pos2=df_pos.reset_index()[["date","margin"]]
    df_pos2["margin"]=np.round(df_pos2["margin"],2)
    df_pos2=df_pos2[df_pos2["margin"]!=0]
    
    df_pos2=df_pos2.merge(trade_day,on="date",how="outer")
    df_pos2["margin_init"]=df_pos2.apply(lambda x: x[1] if x[2]==1 else np.nan,axis=1)
    df_pos2["margin_init"]=df_pos2["margin_init"].fillna(method="ffill")
    df_pos2["margin_init"]=df_pos2["margin_init"]/margin_ratio
    if len(df_pos2)!=len(df_pnl):
        df_pos2=df_pos2.iloc[:len(df_pnl)]
    df_pnl["ret"]=df_pnl["pnl"].values/df_pos2["margin_init"].values
    ret=df_pnl["ret"]
    return ret




'''
Get data
'''

option_sql= "select * from `option`.`option_description` where Underlying = '510050.SH' "
option_data=get_data(option_sql)
 
call_data=option_data[option_data["CPSF"]=="C"][["DATE","CODE","PX","Expiration","D2E","Delta","Gamma","Vega","Theta"]].reset_index(drop=True)
put_data=option_data[option_data["CPSF"]=="P"][["DATE","CODE","PX","Expiration","D2E","Delta","Gamma","Vega","Theta"]].reset_index(drop=True)
l_date=list(call_data["DATE"].unique())
 
#get ATM data current month
call_atm=call_data.groupby( by="DATE",group_keys=False).apply(lambda x:get_current_near_atm_option(x,0)).reset_index(drop=True)
put_atm=put_data.groupby( by="DATE",group_keys=False).apply(lambda x:get_current_near_atm_option(x,0)).reset_index(drop=True)
 
#get ATM data current month
call_atm_next=call_data.groupby( by="DATE",group_keys=False).apply(lambda x:get_current_near_atm_option(x,near_sort=1)).reset_index(drop=True)
put_atm_next=put_data.groupby( by="DATE",group_keys=False).apply(lambda x:get_current_near_atm_option(x,near_sort=1)).reset_index(drop=True)
 
#miniute iv ratio
df_call_iv=pd.read_csv(os.path.join(file_path,"call_iv_ratio.csv"))
df_put_iv=pd.read_csv(os.path.join(file_path,"put_iv_ratio.csv"))
 
df_call_iv["date"]=pd.to_datetime(df_call_iv["date"])
df_put_iv["date"]=pd.to_datetime(df_put_iv["date"])
df_call_iv.set_index("date",inplace=True)
df_put_iv.set_index("date",inplace=True)

df_call_iv["iv_ratio"].fillna(method="bfill",inplace=True)
df_put_iv["iv_ratio"].fillna(method="bfill",inplace=True)

df_call_iv["log_iv_ratio"].fillna(method="bfill",inplace=True)
df_put_iv["log_iv_ratio"].fillna(method="bfill",inplace=True)

#can change to 14:50:00
df_call_iv_daily=df_call_iv[df_call_iv["time"]=="0 days 15:00:00"]
df_put_iv_daily=df_put_iv[df_put_iv["time"]=="0 days 15:00:00"]


#daily rolling iv percentile
file_path_percent_data="Z:\金融工程\WYF\option_calendar_strategy\percentile_data_natural_day"
iv_percent_file=os.listdir(file_path_percent_data)
dic_roll_iv_percent=dict(zip([i.split(".")[0] for i in iv_percent_file],[pd.read_csv(os.path.join(file_path_percent_data,i),index_col=0,parse_dates=True) for i in iv_percent_file]))


def get_trade_info(upper,close,roll_month,method,greek):
    
    global call_atm,put_atm,call_atm_next,put_atm_next,df_call_iv_daily,df_put_iv_daily,dic_roll_iv_percent
    
    l_key=dic_roll_iv_percent.keys()
    roll_name_call=[i for i in l_key if "call" in i and roll_month in i and "log" in i][0]
    roll_name_put=[i for i in l_key if "put" in i and roll_month in i and "log" in i][0]

    strategy_call=sell_near_strategy(df_call_iv_daily,dic_roll_iv_percent[roll_name_call],"log_iv_ratio",call_atm,\
                                     upper,close,method,10,0)
    
    strategy_call.run_signal()
    
    df_signal_call=strategy_call.df_signal_symbol
    df_signal_call_open=df_signal_call[df_signal_call["signal"]==1].index
    df_signal_call_close=df_signal_call[df_signal_call["signal"]==0].index
    signal_call=pd.DataFrame([df_signal_call_open,df_signal_call_close]).T
    signal_call.columns=["open","close"]
    
    boundary_call=strategy_call.df_boundary

    df_trade_info_call=get_trade(df_signal_call,call_atm,call_atm_next,greek)

    strategy_put=sell_near_strategy(df_put_iv_daily,dic_roll_iv_percent[roll_name_put],"log_iv_ratio",put_atm,\
                                    upper,close,method,10,0)
    strategy_put.run_signal()

    df_signal_put=strategy_put.df_signal_symbol
    df_signal_put_open=df_signal_put[df_signal_put["signal"]==1].index
    df_signal_put_close=df_signal_put[df_signal_put["signal"]==0].index
    signal_put=pd.DataFrame([df_signal_put_open,df_signal_put_close]).T
    signal_put.columns=["open","close"]
    
    boundary_put=strategy_put.df_boundary

    df_trade_info_put=get_trade(df_signal_put,put_atm,put_atm_next,greek)

    
    save_path_call_trade=os.path.join(signal_trade_call,"trade_upper_sigma{}_close_sigma{}_roll_month{}_method{}_greek{}.csv".format(upper,close,roll_month,method,greek))
    save_path_put_trade=os.path.join(signal_trade_put,"trade_upper_sigma{}_close_sigma{}_roll_month{}_method{}_greek{}.csv".format(upper,close,roll_month,method,greek))

    
    save_path_call_signal=os.path.join(signal_record_path_call,"signal_upper_sigma{}_close_sigma{}_roll_month{}_method{}_greek{}.csv".format(upper,close,roll_month,method,greek))
    save_path_put_signal=os.path.join(signal_record_path_put,"signal_upper_sigma{}_close_sigma{}_roll_month{}_method{}_greek{}.csv".format(upper,close,roll_month,method,greek))

    save_path_call_boundary=os.path.join(signal_boundary_call,"boundary_upper_sigma{}_close_sigma{}_roll_month{}_method{}_greek{}.csv".format(upper,close,roll_month,method,greek))
    save_path_put_boundary=os.path.join(signal_boundary_put,"boundary_upper_sigma{}_close_sigma{}_roll_month{}_method{}_greek{}.csv".format(upper,close,roll_month,method,greek))
   
    df_trade_info_call.to_csv(save_path_call_trade)
    df_trade_info_put.to_csv(save_path_put_trade)
   
    signal_call.to_csv(save_path_call_signal)
    signal_put.to_csv(save_path_put_signal)
    
    boundary_call.to_csv(save_path_call_boundary)
    boundary_put.to_csv(save_path_put_boundary)
    print("complete param: upper:{},close:{},roll_month:{},method:{},greek:{}".format(upper,close,roll_month,method,greek))


    
def run_backtest_rst(df_trade_info_file,start_time="2017-01-01"):
    
        perform=perform_metrics()
        file_name=df_trade_info_file[:-4]
        param=file_name[6:]
        
        df_trade_info_call=pd.read_csv(os.path.join(signal_trade_call,df_trade_info_file))
        df_trade_info_put=pd.read_csv(os.path.join(signal_trade_put,df_trade_info_file))
        
        df_trade_info_call["date"]=pd.to_datetime(df_trade_info_call["date"])
        df_trade_info_put["date"]=pd.to_datetime(df_trade_info_put["date"])

        df_pos_call,df_trade_call,df_pnl_call=get_backtest_rst(start_time,df_trade_info_call)
        df_ret_call=get_ret(df_pos_call,df_pnl_call,df_trade_info_call)
        dic_rst_call={"df_pos":df_pos_call,"df_trade":df_trade_call,"df_pnl":df_pnl_call,"df_ret":df_ret_call}
        
        df_pos_put,df_trade_put,df_pnl_put=get_backtest_rst(start_time,df_trade_info_put)
        df_ret_put=get_ret(df_pos_put,df_pnl_put,df_trade_info_put)
        dic_rst_put={"df_pos":df_pos_put,"df_trade":df_trade_put,"df_pnl":df_pnl_put,"df_ret":df_ret_put}

        #get performance analysis & save result
        save_path_call=os.path.join(rst_call,file_name+".png")
        perform.plot_nav(df_ret_call,file_name,save_path_call)
        
        save_path_put=os.path.join(rst_put,file_name+".png")
        perform.plot_nav(df_ret_put,file_name,save_path_put)

        
        perform_ret_call_yearly=perform.ret_yearly(df_ret_call)
        perform_ret_put_yearly=perform.ret_yearly(df_ret_put)
        
        perform_ret_call=pd.DataFrame([dict(zip(["max_drawdown","sharpe","annual_ret"],[perform.max_drawdown(df_ret_call),\
                                              perform.sharpe_ratio(df_ret_call),perform.annualized_ret(df_ret_call)]))])
        
        perform_ret_put=pd.DataFrame([dict(zip(["max_drawdown","sharpe","annual_ret"],[perform.max_drawdown(df_ret_put),\
                                              perform.sharpe_ratio(df_ret_put),perform.annualized_ret(df_ret_put)]))])
        
        print("complete analyzing performance {}".format(file_name))
        
        return param,(dic_rst_call,dic_rst_put),(perform_ret_call_yearly,perform_ret_put_yearly),(perform_ret_call,perform_ret_put)






if __name__== "__main__":
    
    #use multiprocssing to get trade info

    l_upper_bound=["85percent"]
    l_close_bound=["50percent"]
    l_rolling=["6month"]
    l_close_method=["rolling_percent"]
    l_greek=["Vega"]
    
    
    l_pool_arg=[]
    
    for upper in l_upper_bound:
        for close in l_close_bound:
            for roll_month in l_rolling:
                for method in l_close_method:
                    for greek in l_greek:
                        arg=(upper,close,roll_month,method,greek)
                        l_pool_arg.append(arg)

    
    pool1 = multiprocessing.Pool(2)
    with pool1 as p1:
        p1.starmap(get_trade_info,l_pool_arg)
        

    l_trade_info_file=[(i,) for i in os.listdir(signal_trade_call)]
    
    
    pool2= multiprocessing.Pool(2)
    with pool2 as p2:
        l_rst=p2.starmap(run_backtest_rst,l_trade_info_file)
        
    
    df_trade_success_call=pd.DataFrame()
    df_metric_call=pd.DataFrame()

    df_trade_success_put=pd.DataFrame()
    df_metric_put=pd.DataFrame()

    def trade_success(prama,df_ret_call,df_ret_put,start_time="2017-01-01"):
        
        read_path_call_signal=os.path.join(signal_record_path_call,"signal_"+prama+".csv")
        read_path_put_signal=os.path.join(signal_record_path_put,"signal_"+prama+".csv")

        df_signal_call=pd.read_csv(read_path_call_signal,usecols=[1,2],parse_dates=[0,1])
        df_signal_put=pd.read_csv( read_path_put_signal,usecols=[1,2],parse_dates=[0,1])
        
        start_time=pd.to_datetime("2017-01-01")
                
        def get_if_success(df_trade_time,df_ret):
            open_day= df_trade_time["open"]
            close_day= df_trade_time["close"]
            success=np.cumprod(1+df_ret[(df_ret.index>=open_day) & (df_ret.index<=close_day)]).iloc[-1]>1
            return success
            
        df_signal_call=df_signal_call[df_signal_call["open"]>=start_time]
        df_signal_put=df_signal_put[df_signal_put["open"]>=start_time]
        
        df_signal_call.reset_index(inplace=True)
        df_signal_put.reset_index(inplace=True)

        
        total_call=len(df_signal_call)
        total_put=len(df_signal_call)
        
        count_call=0
        count_put=0
        for i in range(len(df_signal_call)):
            df_call_sub=df_signal_call.iloc[i]
            try:
                success=get_if_success(df_call_sub,df_ret_call)
                if success:
                    count_call=count_call+1

            except:
                total_call=total_call-1
                
            
        for i in range(len(df_signal_put)):
            df_put_sub=df_signal_put.iloc[i]
            try:
                success=get_if_success(df_put_sub,df_ret_put)
                if success:
                    count_put=count_put+1
            except:
                total_put=total_put-1
              
                
        df_success_call=pd.DataFrame([count_call/total_call],index=[prama])
        df_success_put=pd.DataFrame([count_put/total_put],index=[prama])
        
        return df_success_call,df_success_put

    for rst in l_rst:
        param_name=rst[0]
        df_ret_call,df_ret_put=rst[1][0]["df_ret"],rst[1][1]["df_ret"]
        success_rst=trade_success(param_name,df_ret_call,df_ret_put)
        df_call_success,df_put_success=success_rst[0],success_rst[1]
        df_trade_success_call=df_trade_success_call.append(df_call_success)
        df_trade_success_put=df_trade_success_put.append(df_put_success)
        
        df_ret_call.to_csv(os.path.join(rst_call_ret,param_name+".csv"))
        df_ret_put.to_csv(os.path.join(rst_put_ret,param_name+".csv"))
        df_perform_call,df_perform_put=rst[3][0],rst[3][1]
        df_perform_call.index=[param_name]
        df_perform_put.index=[param_name]
        df_metric_call=df_metric_call.append(df_perform_call)
        df_metric_put=df_metric_put.append(df_perform_put)
        
        
    df_trade_success_call.to_csv(os.path.join(rst_call,"success_ratio.csv"))
    df_trade_success_put.to_csv(os.path.join(rst_put,"success_ratio.csv"))
    df_metric_call.to_csv(os.path.join(rst_call,"perform.csv"))
    df_metric_put.to_csv(os.path.join(rst_put,"perform.csv"))