# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:14:03 2021

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
save_path=os.path.join(file_path,"backtest_results_ou_process_median_log_check")
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
     
    def __init__(self,iv_ratio_min,iv_ratio):
         
        self.iv_ratio_min=iv_ratio_min
        #input daily data
        self.iv_ratio=iv_ratio
        self.trade_date=self.iv_ratio.index
        
        #record when open a new position
        self.close_bound_hist=pd.DataFrame(columns=["close"])  
        self.status_record=pd.DataFrame(columns=["signal"])
        self.df_signal_symbol=pd.DataFrame(columns=["signal"])
        self.df_hold_position=pd.DataFrame(columns=["position"])

    
    def trade_signal(self,df_iv_ratio):
        pass
        
    def print_signal_status(self,date):
        
        signal=self.df_signal_symbol.iloc[-1]["signal"]
        position=self.df_hold_position.iloc[-1]["position"]
        #print signal & position
        print("Date{}:signal is {}, position is {}".format(date,signal,position))
   
        
    def run_signal(self):
    
        for date in self.trade_date:
            df_iv_daily=self.iv_ratio.loc[date]
            self.trade_signal(df_iv_daily)
            self.print_signal_status(date)


class sell_near_strategy(calender_strategy):
    
    def __init__(self,iv_ratio_min,iv_ratio,iv_ratio_type,current_contract,\
                upper_sigma_multiple,close_sigma_multiple,look_back_period,\
                use_mean,use_mirror_mapping,\
                d2e_min_open,d2e_min_close):
        
        super(sell_near_strategy,self).__init__(iv_ratio_min,iv_ratio)
        
        self.iv_ratio_type=iv_ratio_type
        
        #get boundary setting parameter
        self.use_mean=use_mean
        self.use_mirror_mapping=use_mirror_mapping
        self.upper_sigma_multiple=upper_sigma_multiple
        self.close_sigma_multiple=close_sigma_multiple
        self.look_back_period=look_back_period
        
        #get boundary
        self.df_boundary=self. _get_daily_boundary()
    
        #get current contract info
        self.current_contract=current_contract
        
        #if d2e< min cannot open trade
        self.d2e_min_open=d2e_min_open
        #if d2e< min must close
        self.d2e_min_close=d2e_min_close
    
    def _get_daily_boundary(self):
        
        df_boundary=pd.DataFrame()
        
        def reg(x,y):
            X = sm.add_constant(x)
            model_ols=sm.OLS(y,X).fit()
            const,slope=model_ols.params[0],model_ols.params[1]
            std_resid=np.std(model_ols.resid)
            return const,slope,std_resid

        for date in self.trade_date:
            start_date=pd.to_datetime(date)-relativedelta(months=self.look_back_period)
            if start_date < np.min(self.trade_date):
                df_boundary_sub=pd.DataFrame(np.array([np.nan]*2).reshape(1,2),\
                                    index=[date],columns=["upper_bound","close_bound"])
            else:
                df_sub=self.iv_ratio_min[(self.iv_ratio_min.index>=start_date) & (self.iv_ratio_min.index<=date)]
                
                const,slope,std_resid=reg(df_sub[self.iv_ratio_type].shift(1).dropna().values,df_sub[self.iv_ratio_type].iloc[1:].values)
                
                mu=const/(1-slope)
                sigma=std_resid*np.sqrt((2*-np.log(slope))/(1-slope**2))
                
                upper_bound=mu+self.upper_sigma_multiple*sigma
                close_bound=mu-self.close_sigma_multiple*sigma
                df_boundary_sub=pd.DataFrame(np.array([upper_bound,close_bound]).reshape(1,2),\
                                    index=[date],columns=["upper_bound","close_bound"])
            df_boundary=df_boundary.append(df_boundary_sub)
            print("complete find {} boundary".format(date))
        return df_boundary
            
            
    def trade_signal(self,df_iv_ratio):
        
        date=df_iv_ratio.name
        iv_ratio=df_iv_ratio[self.iv_ratio_type]
        
        #because option minite database misses data of '2019-12-25'
        try:
            d2e=self.current_contract[self.current_contract["DATE"]==date]["D2E"].values[0]
        except:
            d2e=0
    
        close_criterion=self.df_boundary.loc[date]["close_bound"]
        roll_upper=self.df_boundary.loc[date]["upper_bound"]
        
        if np.isnan(roll_upper):
            self.status_record.loc[date]=[np.nan]
            self.df_signal_symbol.loc[date]=[np.nan]
            self.df_hold_position.loc[date]=[0]
        
        else:
            #check status
            if iv_ratio>=roll_upper:
                 self.status_record.loc[date]=[1]
                  
            elif iv_ratio<=close_criterion:
                 self.status_record.loc[date]=[-1]
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
                    # 1 to -1
                    if  current_status==-1:
                        self.df_signal_symbol.loc[date]=[0]
                        self.df_hold_position.loc[date]=[0]
                    else:
                        self.df_signal_symbol.loc[date]=[np.nan]
                        self.df_hold_position.loc[date]= [pre_hold_position]

            #no position before
            else:
                #check whether can enter a position
                #generate positive signal
    
                #signal=1 (-1/np.nan/0 to 1)
                if (pre_status == 0 and current_status==1) or (np.isnan(pre_status) and current_status==1)or \
                   (pre_status == -1 and current_status==1):
                    if d2e>= self.d2e_min_open:
                        self.df_signal_symbol.loc[date]=[1]
                        self.df_hold_position.loc[date]=[1]
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




def get_trade_info(upper,close,month,greek):
    
    global call_atm,put_atm,call_atm_next,put_atm_next,df_call_iv,df_put_iv,df_call_iv_daily,df_put_iv_daily

    if close <0 and abs(close)>=upper:
        pass
    else:
        
        strategy_call=sell_near_strategy(df_call_iv,df_call_iv_daily,"log_iv_ratio",call_atm,\
                                         upper,close,month,False,True,10,5)
        strategy_call.run_signal()
        
        df_signal_call=strategy_call.df_signal_symbol
        df_signal_call_open=df_signal_call[df_signal_call["signal"]==1].index
        df_signal_call_close=df_signal_call[df_signal_call["signal"]==0].index
        signal_call=pd.DataFrame([df_signal_call_open,df_signal_call_close]).T
        signal_call.columns=["open","close"]
        
        boundary_call=strategy_call.df_boundary

        df_trade_info_call=get_trade(df_signal_call,call_atm,call_atm_next,greek)

        strategy_put=sell_near_strategy(df_put_iv,df_put_iv_daily,"log_iv_ratio",put_atm,\
                                        upper,close,month,False,True,10,5)
        strategy_put.run_signal()

        df_signal_put=strategy_put.df_signal_symbol
        df_signal_put_open=df_signal_put[df_signal_put["signal"]==1].index
        df_signal_put_close=df_signal_put[df_signal_put["signal"]==0].index
        signal_put=pd.DataFrame([df_signal_put_open,df_signal_put_close]).T
        signal_put.columns=["open","close"]
        
        boundary_put=strategy_put.df_boundary

        df_trade_info_put=get_trade(df_signal_put,put_atm,put_atm_next,greek)

        
        save_path_call_trade=os.path.join(signal_trade_call,"trade_upper_sigma{}_close_sigma{}_roll_month{}_greek{}.csv".format(np.round(upper,1),np.round(close,1),month,greek))
        save_path_put_trade=os.path.join(signal_trade_put,"trade_upper_sigma{}_close_sigma{}_roll_month{}_greek{}.csv".format(np.round(upper,1),np.round(close,1),month,greek))

        
        save_path_call_signal=os.path.join(signal_record_path_call,"signal_upper_sigma{}_close_sigma{}_roll_month{}_greek{}.csv".format(np.round(upper,1),np.round(close,1),month,greek))
        save_path_put_signal=os.path.join(signal_record_path_put,"signal_upper_sigma{}_close_sigma{}_roll_month{}_greek{}.csv".format(np.round(upper,1),np.round(close,1),month,greek))

        save_path_call_boundary=os.path.join(signal_boundary_call,"boundary_upper_sigma{}_close_sigma{}_roll_month{}_greek{}.csv".format(np.round(upper,1),np.round(close,1),month,greek))
        save_path_put_boundary=os.path.join(signal_boundary_put,"boundary_upper_sigma{}_close_sigma{}_roll_month{}_greek{}.csv".format(np.round(upper,1),np.round(close,1),month,greek))
       
        df_trade_info_call.to_csv(save_path_call_trade)
        df_trade_info_put.to_csv(save_path_put_trade)
       
        signal_call.to_csv(save_path_call_signal)
        signal_put.to_csv(save_path_put_signal)
        
        boundary_call.to_csv(save_path_call_boundary)
        boundary_put.to_csv(save_path_put_boundary)
        print("complete param: upper:{},close:{},roll_month:{},greek:{}".format(np.round(upper,1),np.round(close,1),month,greek))

    
def run_backtest_rst(df_trade_info_file,start_time="2017-01-01"):
    
        perform=perform_metrics()
        file_name=df_trade_info_file[:-4]
        
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
        
        return (dic_rst_call,dic_rst_put),(perform_ret_call_yearly,perform_ret_put_yearly),(perform_ret_call,perform_ret_put)






if __name__== "__main__":
    
    #use multiprocssing to get trade info
    l_upper=np.linspace(1,2,11)
    l_close=np.linspace(0,0.3,4)
    l_month=[1,3,6,12]
    l_greek=["Vega"]

    # l_upper=[1.6]
    # l_close=[0]
    # l_month=[3]
    # l_greek=["Vega"]
    
    l_pool_arg=[]
    
    for upper in l_upper:
        for close in l_close:
            for month in l_month:
                for greek in l_greek:
                    arg=(upper,close,month,greek,)
                    l_pool_arg.append(arg)
                    
    pool1 = multiprocessing.Pool()               
    with pool1 as p1:
        p1.starmap(get_trade_info,l_pool_arg)
        

    l_trade_info_file=[(i,) for i in os.listdir(signal_trade_call)]
    
    
    pool2= multiprocessing.Pool()               
    with pool2 as p2:
        rst=p2.starmap(run_backtest_rst,l_trade_info_file)
        
    
    df_trade_success_call=pd.DataFrame()
    df_metric_call=pd.DataFrame()

    df_trade_success_put=pd.DataFrame()
    df_metric_put=pd.DataFrame()

    def trade_success(prama_tuple,df_ret_call,df_ret_put,start_time="2017-01-01"):
        
        read_path_call_signal=os.path.join(signal_record_path_call,"signal_upper_sigma{}_close_sigma{}_roll_month{}_greek{}.csv".format(np.round(prama_tuple[0],1),np.round(prama_tuple[1],1),prama_tuple[2],prama_tuple[3]))
        read_path_put_signal=os.path.join(signal_record_path_put,"signal_upper_sigma{}_close_sigma{}_roll_month{}_greek{}.csv".format(np.round(prama_tuple[0],1),np.round(prama_tuple[1],1),prama_tuple[2],prama_tuple[3]))

        df_signal_call=pd.read_csv(read_path_call_signal,usecols=[1,2],parse_dates=[0,1])
        df_signal_put=pd.read_csv( read_path_put_signal,usecols=[1,2],parse_dates=[0,1])
        
        start_time=pd.to_datetime("2017-01-01")
        
        index_name="upper_sigma{}_close_sigma{}_roll_month{}_greek{}".format(np.round(prama_tuple[0],1),np.round(prama_tuple[1],1),prama_tuple[2],prama_tuple[3])
        
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
            except:
                total_call=total_call-1
                
            if success:
                count_call=count_call+1
            
        for i in range(len(df_signal_put)):
            df_put_sub=df_signal_put.iloc[i]
            try:
                success=get_if_success(df_put_sub,df_ret_put)
            except:
                total_put=total_put-1
              
            if success:
                count_put=count_put+1
                
        df_success_call=pd.DataFrame([count_call/total_call],index=[index_name])
        df_success_put=pd.DataFrame([count_put/total_put],index=[index_name])
        
        return df_success_call,df_success_put

    for idx,arg in enumerate(l_pool_arg):
        param_name="upper_sigma{}_close_sigma{}_roll_month{}_greek{}".format(np.round(arg[0],1),np.round(arg[1],1),arg[2],arg[3])
        arg_rst=rst[idx]
        df_ret_call,df_ret_put=arg_rst[0][0]["df_ret"],arg_rst[0][1]["df_ret"]
        success_rst=trade_success(arg,df_ret_call,df_ret_put)
        df_call_success,df_put_success=success_rst[0],success_rst[1]
        df_trade_success_call=df_trade_success_call.append(df_call_success)
        df_trade_success_put=df_trade_success_put.append(df_put_success)
        
        df_ret_call.to_csv(os.path.join(rst_call_ret,param_name+".csv"))
        df_ret_put.to_csv(os.path.join(rst_put_ret,param_name+".csv"))
        df_perform_call,df_perform_put=arg_rst[2][0],arg_rst[2][1]
        df_perform_call.index=[param_name]
        df_perform_put.index=[param_name]
        df_metric_call=df_metric_call.append(df_perform_call)
        df_metric_put=df_metric_put.append(df_perform_put)
        
        
    df_trade_success_call.to_csv(os.path.join(rst_call,"success_ratio.csv"))
    df_trade_success_put.to_csv(os.path.join(rst_put,"success_ratio.csv"))
    df_metric_call.to_csv(os.path.join(rst_call,"perform.csv"))
    df_metric_put.to_csv(os.path.join(rst_put,"perform.csv"))
    

    

    
    
    l_upper1=np.linspace(1, 2, 11)
    l_close1=np.linspace(0,0.3,4)
    l_month1=[1,3,6,12]
    l_greek1=["Vega"]

    l_upper = np.linspace(0.6, 0.95, 8)
    l_close = np.linspace(0, 0.3, 4)
    l_month = [12, 18, 21, 24]
    l_greek = ["Vega"]
    # l_upper=[1.3,1.7]
    # l_close=[0]
    # l_month=[6]
    # l_greek=["Vega"]
    
    l_pool_arg=[]
    
    for upper in l_upper:
        for close in l_close:
            for month in l_month:
                for greek in l_greek:
                    arg=(upper,close,month,greek,)
                    l_pool_arg.append(arg)
                    
    for upper in l_upper1:
        for close in l_close1:
            for month in l_month1:
                for greek in l_greek1:
                    arg=(upper,close,month,greek,)
                    l_pool_arg.append(arg)
    

    
    
    

        
        
        

        
    
                    
                    
    
    
    
    


    
    
    


    

    
    
    
    
    
    
    

