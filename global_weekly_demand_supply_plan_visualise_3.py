# global_weekly_demand_supply_plan_visualise.py

# 処理内容
# 入力ファイル: 拠点node別サプライチェーン需給tree
#               複数年別、1月-12月の需要数
#               

# 処理        : iso_year+iso_weekをkeyにして、需要数を月間から週間に変換する

#               前処理で、各月の日数と月間販売数から、月毎の日平均値を求める
#               年月日からISO weekを判定し、
#               月間販売数の日平均値をISO weekの変数に加算、週間販売数を計算

#               ***** pointは「年月日からiso_year+iso_weekへの変換処理」 *****
#               dt = datetime.date(year, month, day) 
#               iso_year, iso_week, _ = dt.isocalendar()

#               for nodeのループ下で、
#               YM_key_list.append(key)  ## keyをappendして
#               pos = len( YW_key_list ) ## YM_key_listの長さを位置にして
#               YW_value_list( pos ) += average_daily_value ## 値を+=加算

# 出力リスト  : node別 複数年のweekの需要 S_week



# *********************************
# start of code
# *********************************
import pandas as pd
import csv

import math
import numpy as np

import datetime
import calendar

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objs as go
import plotly.offline as offline 

from copy import deepcopy

import itertools



# ***********************************
# def
# ***********************************
def visualise_psi_label(node_I_psi, node_name):
    
    #データの定義
    x, y, z = [], [], []
    
    for i in range(len(node_I_psi)):
    
        for j in range(len(node_I_psi[i])):
    
            #node_idx = node_name.index('JPN')
    
            node_label = node_name[i] # 修正
    
            for k in range(len(node_I_psi[i][j])):
                x.append(j)
                y.append(node_label)
                z.append(k)
    
    text = []
    
    for i in range(len(node_I_psi)):
    
        for j in range(len(node_I_psi[i])):
    
            for k in range(len(node_I_psi[i][j])):
    
                text.append(node_I_psi[i][j][k])
    
    
    # y軸のラベルを設定
    y_axis = dict(
        tickvals=node_name,
        ticktext=node_name
    )
    
    #3D散布図の作成
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        text=text,
        marker=dict(
            size=5,
            color=z,
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    #レイアウトの設定
    fig.update_layout(
        title="Node Connections",
        scene=dict(
            xaxis_title="Week",
            yaxis_title="Location",
            zaxis_title="Lot ID"
        ),
        width=800,
        height=800,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )
    
    #グラフの表示
    fig.show()



# 前処理として、年月の月間販売数の一日当たりの平均値を計算する
def calc_average_sales(monthly_sales, year):

    month_daily_average = [0]*12

    for i, month_qty in enumerate(monthly_sales):

        month = i + 1

        days_in_month = calendar.monthrange(year, month)[1]

        month_daily_average[i] = monthly_sales[i] / days_in_month

    return month_daily_average




# ある年の月次販売数量を年月から年ISO週に変換する
def calc_weekly_sales(node, monthly_sales, year,year_month_daily_average, sales_by_iso_year, yyyyww_value, yyyyww_key):

    weekly_sales = [0] * 53    

 
    for i, month_qty in enumerate( monthly_sales ):

        # 開始月とリストの要素番号を整合
        month = i + 1

        # 月の日数を調べる
        days_in_month = calendar.monthrange(year, month )[1]

        # 月次販売の日平均
        avg_daily_sales = year_month_daily_average[year][ i ] # i=month-1

        # 月の日毎の処理
        for day in range(1, days_in_month + 1):
        # その年の"年月日"を発生

            ## iso_week_noの確認 年月日でcheck その日がiso weekで第何週か
            #iso_week = datetime.date(year,month, day).isocalendar()[1]

            # ****************************
            # year month dayからiso_year, iso_weekに変換
            # ****************************
            dt = datetime.date(year, month, day)

            iso_year, iso_week, _ = dt.isocalendar()


            # 辞書に入れる場合 
            sales_by_iso_year[iso_year][iso_week-1] += avg_daily_sales


            # リストに入れる場合
            node_year_week_str = f"{node}{iso_year}{iso_week:02d}"


            if node_year_week_str not in yyyyww_key:

                yyyyww_key.append(node_year_week_str)

            pos = len(yyyyww_key) - 1

            yyyyww_value[pos]  += avg_daily_sales

    return sales_by_iso_year[year]


# *******************************************************
# trans S from monthly to weekly
# *******************************************************
def trans_month2week(input_file, outputfile):


# IN:      'S_month_data.csv'
# PROCESS: nodeとyearを読み取る yearはstart-1年に"0"セットしてLT_shiftに備える
# OUT:     'S_iso_week_data.csv'

# *********************************
# read monthly S
# *********************************

    # csvファイルの読み込み
    df = pd.read_csv( input_file ) # IN:      'S_month_data.csv'



#    #@230513 capacity setting
#    # *********************************
#    # mother plant capacity parameter
#    # *********************************
#
#    demand_supply_ratio = 1.2  # demand_supply_ratio = ttl_supply / ttl_demand


    # *********************************
    # initial setting of total demand and supply
    # *********************************

    # total_demandは、各行のm1からm12までの列の合計値 

    df_capa = pd.read_csv( input_file )
 
    df_capa['total_demand'] = df_capa.iloc[:, 3:].sum(axis=1)

    # yearでグループ化して、月次需要数の総和を計算
    df_capa_year = df_capa.groupby(['year'],as_index=False).sum()


    ## 結果を表示
    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.width', None)
    #pd.set_option('display.max_colwidth', -1)

    #print('df_capa', df_capa)
    #print('df_capa_year', df_capa_year)


    # リストに変換
    month_data_list = df.values.tolist()

    # node_nameをユニークなキーとしたリストを作成する
    node_list = df['node_name'].unique().tolist()
    
    
    # *********************************
    # write csv file header [prod-A,node_name,year.w0,w1,w2,w3,,,w51,w52,w53]
    # *********************************
    
    file_name_out = outputfile # OUT:     'S_iso_week_data.csv'
    
    with open(file_name_out, mode='w', newline='') as f:
    
        writer = csv.writer(f)
    
        writer.writerow(['product_name', 'node_name', 'year', 
'w1' ,'w2' ,'w3' ,'w4' ,'w5' ,'w6' ,'w7' ,'w8' ,'w9' ,'w10','w11','w12','w13',
'w14','w15','w16','w17','w18','w19','w20','w21','w22','w23','w24','w25','w26',
'w27','w28','w29','w30','w31','w32','w33','w34','w35','w36','w37','w38','w39',
'w40','w41','w42','w43','w44','w45','w46','w47','w48','w49','w50','w51','w52',
'w53'])
    
    
# *********************************
# plan initial setting
# *********************************

# node別に、中期計画の3ヵ年、5ヵ年をiso_year+iso_week連番で並べたもの
# node_lined_iso_week = { node-A+year+week: [iso_year+iso_week1,2,3,,,,,],   } 
# 例えば、2024W00, 2024W01, 2024W02,,, ,,,2028W51,2028W52,2028W53という5年間分
    
    node_lined_iso_week = {} 
    
    node_yyyyww_value = []
    node_yyyyww_key   = []
    

    for node in node_list:
    
        #print('node',node)
    
        df_node = df[df['node_name'] == node]
    
        #print('df_node',df_node)
    
        # リストに変換
        node_data_list = df_node.values.tolist()
    
        #
        # getting start_year and end_year
        #
        start_year = node_data_min = df_node['year'].min()
        end_year   = node_data_max = df_node['year'].max()
    
        #print('max min',node_data_max, node_data_min)
    
    
        # S_month辞書の初期セット
        monthly_sales_data    = {}
    
    
    # *********************************
    # plan initial setting
    # *********************************
    
        plan_year_st = start_year                  # 2024  # plan開始年
    
        plan_range = end_year - start_year + 1     # 5     # 5ヵ年計画分のS計画
    
        plan_year_end = plan_year_st + plan_range
    
    




#
# an image of data "df_node"
#
#product_name	node_name	year	m1	m2	m3	m4	m5	m6	m7	m8	m9	m10	m11	m12
#prod-A	CAN	2024	0	0	0	0	0	0	0	0	0	0	0	0
#prod-A	CAN	2025	0	0	0	0	0	0	0	0	0	0	0	0
#prod-A	CAN	2026	0	0	0	0	0	0	0	0	0	0	0	0
#prod-A	CAN	2027	0	0	0	0	0	0	0	0	0	0	0	0
#prod-A	CAN	2028	0	0	0	0	0	0	0	0	0	0	0	0
#prod-A	CAN_D	2024	122	146	183	158	171	195	219	243	231	207	195	219
#prod-A	CAN_D	2025	122	146	183	158	171	195	219	243	231	207	195	219


    # *********************************
    # by node    node_yyyyww = [ node-a, yyyy01, yyyy02,,,, ]
    # *********************************
    
        yyyyww_value = [0]*53*plan_range  # 5ヵ年plan_range=5
    
        yyyyww_key   = []
    
    
    
        for data in node_data_list:
    
            # node別　3年～5年　月次需要予測値
    
            #print('data',data)
    
    
            
            # 辞書形式{year: S_week_list, }でデータ定義する
            sales_by_iso_year = {}
            
    
# 前後年付きの辞書 53週を初期セット 
# **********************************
# 空リストの初期設定
# start and end setting from S_month data # 月次Sのデータからmin&max 
# **********************************
            
            #前年の52週が発生する可能性あり # 計画の前後の-1年 +1年を見る
            work_year = plan_year_st - 1 
    
            for i in range(plan_range+2):   # 計画の前後の-1年 +1年を見る
            
                year_sales = [0] * 53 # 53週分の要素を初期セット
            
                # 年の辞書に週次Sをセット
                sales_by_iso_year[work_year] = year_sales 
            
                work_year += 1
            
    # *****************************************
    # initial setting end
    # *****************************************
    
    # *****************************************
    # start process
    # *****************************************
    
            # ********************************
            # generate weekly S from monthly S
            # ********************************
            
            # S_monthのcsv fileを読んでS_month_listを生成する
            # pandasでcsvからリストにして、node_nameをキーに順にM2W変換
            
            # ****************** year ****** Smonth_list ******
            monthly_sales_data[ data[2] ] = data[3:] 
            
            # data[0] = prod-A
            # data[1] = node_name
            # data[2] = year
    
            #print('monthly_sales_data',monthly_sales_data)


        # **************************************
        # 年月毎の販売数量の日平均を計算する
        # **************************************
        year_month_daily_average = {}
        
        #print('plan_year_st_st',plan_year_st)
        #print('plan_year_end',plan_year_end)
    
        for y in range(plan_year_st,plan_year_end):
        
            year_month_daily_average[y] = calc_average_sales(monthly_sales_data[    y], y)
    
    
        # 販売数量を年月から年ISO週に変換する
        for y in range(plan_year_st,plan_year_end):
        
            ##print('input monthly sales by year', y, monthly_sales_data[y])
    
            sales_by_iso_year[y] = calc_weekly_sales(node, monthly_sales_data[y], y, year_month_daily_average, sales_by_iso_year, yyyyww_value, yyyyww_key)
    
    
        work_yyyyww_value = [node] + yyyyww_value
        work_yyyyww_key   = [node] + yyyyww_key
    
        node_yyyyww_value.append( work_yyyyww_value )
        node_yyyyww_key.append( work_yyyyww_key )
    
    
        # 複数年のiso週毎の販売数を出力する
        for y in range(plan_year_st,plan_year_end):
        
            #for i in range(53):
            #
            #    #print('year week sales_by_iso_year',y,i+1,sales_by_iso_year[y]    [i])
    
            rowX = ['product-X'] + [node] + [y] + sales_by_iso_year[y]
            ##print('rowX',rowX)
    
            with open(file_name_out, mode='a', newline='') as f:
    
                writer = csv.writer(f)
    
                writer.writerow(rowX)
    
    
# **********************
# リスト形式のS出力
# **********************

#for i, node_key in enumerate(node_yyyyww_key):
#
#    print( i )
#    print( node_key[0] )
#    print( node_key[1:] )
#
#    node_val = node_yyyyww_value[ i ]
#
#    print( node_val[0] )
#    print( node_val[1:] )
#
#
#for node_val in node_yyyyww_value:
#    print( node_val )

#['SHA_N', 22.580645161290324, 22.580645161290324, 22.580645161290324, 22.580645161290324, 26.22914349276974, 28.96551724137931, 28.96551724137931, 28.96551724137931, 31.067853170189103, 33.87096774193549, 33.87096774193549, 33.87096774193549, 33.87096774193549, 30.33333333333333, 30.33333333333333, 30.33333333333333, 30.33333333333333, 31.247311827956988, 31.612903225806452,

#    print( node_val[0] )
#    print( node_val[1:] )


    # **********************
    # リスト形式のkey='node'+'yyyyww'出力
    # **********************
    #print('node_yyyyww_key',node_yyyyww_key)

    return node_yyyyww_value, node_yyyyww_key, plan_range, df_capa_year

# *********************
# END of week data generation 
# node_yyyyww_value と node_yyyyww_keyに複数年の週次データがある
# *********************



# *******************************************************
# lot by lot PSI
# *******************************************************
def makeS(S_week, lot_size): # Sの値をlot単位に変換

    return [math.ceil(num / lot_size) for num in S_week]



def setS(psi_list, node_name, Slot, yyyyww_list ):

    #print('Slot',Slot)
    #print('yyyyww_list',yyyyww_list)

    for w, (lots, yyyyww) in enumerate(zip(Slot, yyyyww_list)):

        step_list = []

        for i in range(lots):

            lot_id = str(yyyyww) + str(i)
            #lot_id = node_name + str(yyyyww) + str(i)
            #lot_id = node_name + str(year) + str(w) + str(i)

            #print('str(yyyyww)',str(yyyyww))
            #print('lot_id',lot_id)

            step_list.append(lot_id)

        # week 0="S"
        psi_list[w][0] = step_list

        #print('step_list',step_list)

    return psi_list


# checking constraint to inactive week , that is "Long Vacation"
#@230520
def check_lv_week_bw(const_lst, check_week):

    #print('const_lst',const_lst)

    num = check_week

    if const_lst == []:

        #print('test const_lst',const_lst)

        pass

    else:

        while num in const_lst:

            num -= 1

    return num


def check_lv_week_fw(const_lst, check_week):

    #print('const_lst',const_lst)

    num = check_week

    if const_lst == []:

        #print('test const_lst',const_lst)

        pass

    else:

        while num in const_lst:

            num += 1

    return num



def calcPS2I(psiS2P):

    plan_len = len(psiS2P)

    for w in range(1, plan_len): # starting_I = 0 = w-1 / ending_I = 53
    #for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53

        s   = psiS2P[w][0]
        co  = psiS2P[w][1]

        i0  = psiS2P[w-1][2]
        i1  = psiS2P[w][2]

        p   = psiS2P[w][3]

        # *********************
        # # I(n-1)+P(n)-S(n)
        # *********************

        #print('i0',i0)
        #print('p',p)


        work = i0 + p  


        #print('work',work)
        #print('s',s)


        #@230321 TOBT memo ここで、期末の在庫、S出荷=売上を操作している
        # S出荷=売上を明示的にlogにして、売上として記録し表示する処理
        # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する

        # モノがお金に代わる瞬間

        diff_list = [x for x in work if x not in s] # I(n-1)+P(n)-S(n)

        psiS2P[w][2] = i1 = diff_list

    return psiS2P



def shiftS2P_LV(psiS, safety_stock_week, lv_week): # LV:long vacations

    #print('lv_week',lv_week)

    ss = safety_stock_week

    plan_len = len( psiS ) - 1 # -1 for week list position

    #print('plan_len & psiS', plan_len , psiS)


    for w in range(plan_len, ss, -1): # backward planningで需要を降順でシフト

#my_list = [1, 2, 3, 4, 5]
#for i in range(2, len(my_list)):
#    my_list[i] = my_list[i-1] + my_list[i-2] 

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        eta_plan = w - ss # ss:safty stock

        #print('eta_plan = w - ss', eta_plan, w, ss)

        eta_shift = check_lv_week_bw(lv_week, eta_plan) #ETA:Eatimate Time Arrival

        ##print('w psiS[w][0] ', w, psiS[w][0])
        ##print('eta_plan psiS[eta_plan][3] ', eta_plan, psiS[eta_plan][3])

        # リスト追加 extend 
        # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする
        psiS[eta_shift][3].extend( psiS[w][0] )  # P made by shifting S with 

        #print('psiS[eta_shift][3] appended',eta_shift, psiS[eta_plan][3])

    return psiS



####self.psi_list4supply = shiftP2S_LV(self.psi_list4supply, safety_stock_week, lv_week)


def shiftP2S_LV(psiP, safety_stock_week, lv_week): # LV:long vacations
#def shiftS2P_LV(psiS, safety_stock_week, lv_week): # LV:long vacations

    #print('lv_week',lv_week)

    ss = safety_stock_week

    plan_len = len( psiP ) - 1 # -1 for week list position
    #plan_len = len( psiS ) - 1 # -1 for week list position

    #print('plan_len & psiS', plan_len , psiS)


    for w in range(plan_len - 1): # foreward planningで確定Pを確定Sにシフト
    #for w in range(plan_len): # foreward planningで確定Pを確定Sにシフト

    #for w in range(plan_len, ss, -1): # backward planningで需要を降順でシフト

#my_list = [1, 2, 3, 4, 5]
#for i in range(2, len(my_list)):
#    my_list[i] = my_list[i-1] + my_list[i-2] 

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        etd_plan = w + ss # ss:safty stock
        #eta_plan = w - ss # ss:safty stock

        #print('eta_plan = w - ss', eta_plan, w, ss)

        etd_shift = check_lv_week_fw(lv_week, etd_plan) #ETD:Eatimate TimeDept.
        #eta_shift = check_lv_week(lv_week, eta_plan)#ETA:Eatimate Time Arrival

        print('w psiP[w][0] ', w, psiP[w][0])
        print('etd_shift psiP[etd_shift][0] ', etd_shift, psiP[etd_shift][0])

        # リスト追加 extend 
        # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする
        psiP[etd_shift][0].extend( psiP[w][3] )  # P made by shifting S with 

        #psiS[eta_shift][3].extend( psiS[w][0] )  # P made by shifting S with 

        #print('psiS[eta_shift][3] appended',eta_shift, psiS[eta_plan][3])

    return psiP
    #return psiS






























# **************************************
# 可視化トライアル
# **************************************

# node dictの在庫Iを可視化
def show_node_I4bullwhip_color(node_I4bullwhip):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # x, y, z軸のデータを作成
    x = np.arange(len(node_I4bullwhip['HAM_N']))

    n = len(node_I4bullwhip.keys())
    y = np.arange(n)

    X, Y = np.meshgrid(x, y)

    z = list(node_I4bullwhip.keys())

    Z = np.zeros((n, len(x)))

    # node_I4bullwhipのデータをZに格納
    for i, node in enumerate(z):
        Z[i,:] = node_I4bullwhip[node]

    # 3次元の棒グラフを描画
    dx = dy = 1.2 # 0.8
    dz = Z
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(n):
        ax.bar3d(X[i], Y[i], np.zeros_like(dz[i]), dx, dy, dz[i], color=colors[i % len(colors)], alpha=0.8)

    # 軸ラベルを設定
    ax.set_xlabel('Week')
    ax.set_ylabel('Node')
    ax.set_zlabel('Inventory')

    # y軸の目盛りをnode名に設定
    ax.set_yticks(y)
    ax.set_yticklabels(z)

    plt.show()



def show_psi_3D_graph_node(node):

    node_name = node.name

    #node_name = psi_list[0][0][0][:-7]
    #node_name = psiS2P[0][0][0][:-7]

    #print('node_name',node_name)

    psi_list = node.psi_list

    # 二次元マトリクスのサイズを定義する
    x_size = len(psi_list)
    y_size = len(psi_list[0])
    
    #x_size = len(psiS2P)
    #y_size = len(psiS2P[0])
    
    # x軸とy軸のグリッドを生成する
    x, y = np.meshgrid(range(x_size), range(y_size))
    

    # y軸の値に応じたカラーマップを作成
    color_map = plt.cm.get_cmap('cool')


    # z軸の値をリストから取得する
    z = []
    
    for i in range(x_size):
        row = []
        for j in range(y_size):

            row.append(len(psi_list[i][j]))
            #row.append(len(psiS2P[i][j]))

        z.append(row)
    
    #print('z',z)

    ravel_z = np.ravel(z)

    #print('ravel_z',ravel_z)

    norm = plt.Normalize(0,3)
    #norm = plt.Normalize(0,dz.max())


    # 3Dグラフを作成する
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    
    
    #print('x',x)
    #print('y',y)
    
    z_like = np.zeros_like(z)
    #print('z_like',z_like)
    
    #print("dx",0.05)
    #print("dy",0.05)
    
    #print('dz',z)
    
    
    # ********************
    # x/yの逆転
    # ********************
    original_matrix = z
    
    inverted_matrix = []

    for i in range(len(original_matrix[0])):
        inverted_row = []
        for row in original_matrix:
            inverted_row.append(row[i])
        inverted_matrix.append(inverted_row)
    
    z_inv = inverted_matrix
    
    
    
    #print(inverted_matrix)
    
    #print("liner X",np.ravel(x))
    #print("liner Y",np.ravel(y))
    #print("liner z",np.ravel(z))
    #print("liner z",np.ravel(z_inv))
    
    #print("z[0][1]",z[0][1])
    
    
    #colors = plt.cm.terrain_r(norm(z_inv))
    #colors = plt.cm.terrain_r(norm(dz))


    # ********************
    # 4色での色分け
    # ********************

    # 色分け用のデータ
    color_data = [1, 2, 3, 4]

    # 色は固定
    # colorsのリストは、S/CO/I/Pに対応する
    #colors = ['cyan', 'blue', 'red', 'gold']
    #colors = ['cyan', 'blue', 'maroon', 'gold']
    colors = ['cyan', 'blue', 'brown', 'gold']

    y_list = np.ravel(y)

    #print('y',y)
    #print('y_list',y_list)

    #print('colors',colors)

    c_map = []
    
    for index in y_list:

        c_map.append(colors[index])

    #print('c_map',c_map)


    # ********************
    # bar3D
    # ********************

    
    ax.bar3d(np.ravel(x), np.ravel(y), np.ravel(np.zeros_like(z)),0.05,0.05,np.ravel(z_inv), color=c_map )

    ax.set_title(node_name, fontsize='16') # タイトル

    plt.show()



def make_node_psi_dict( node_yyyyww_value, node_yyyyww_key, nodes ):

    for i, node_val in enumerate(node_yyyyww_value):
    
        node_name = node_val[0]
        S_week    = node_val[1:]
    
        #print('node_name',node_name)
        #print('S_week',S_week)

        node = nodes[node_name]   # node_nameからnodeインスタンスを取得

        # node.lot_sizeを使う
        lot_size = node.lot_size # Node()からセット
    

        # makeSでSlotを生成
        Slot = makeS(S_week, lot_size)

        # nodeに対応するpsi_listを生成する
        psi_list = [[[] for j in range(4)] for w in range( len(S_week) )] 


        node_key = node_yyyyww_key[i]

        ####node_name = node_key[0] # node_valと同じ

        yyyyww_list   = node_key[1:]

        # lot_listのリスト化
        psiS = setS(psi_list, node_name, Slot, yyyyww_list )

        node_psi_dict[node_name] = psiS #初期セットSを渡す。本来はleaf_nodeのみ

    return node_psi_dict




























# 1. データを読む　データ長を調べる
# 2. class Nodeでtreeを作る
# 3. psi_listを作る node_psi_dicを作る
# 4. class Nodeのself.psi_listに、node_psi_dicのpsi_listをpointor接続する
# 5. Nodeのtree構造は、node間のサーチ・巡回用。psi操作は、接続元のpsiで処理

# 長さ、len(S_week) の値を 入力ファイルS_monthのyear_start/end/rangeから
#


# Slot_value = makeS 値をlot_sizeで割る

# setS   は、値をlot_idとlot_list構造に変換 Slot_value2id_list
#            lot_id = str(yyyyww) + str(i)
#
#    Slotのvalueを、Slot_id =  yyyyww + i のリスト構造に変換する
#
#    psiS = setS(psi_list, node_name, Slot, yyyyww_list )

# psiSをdict[node: psi_list]で外に持つか?、class Nodeのself.spi_listに持つか?
# 外にある辞書型node_psiを指すだけの方が「軽くなる」ハズ。
# self.psi_listで、Nodeインスタンスのspi_listに直接データを置くと「重くなる」?
# load/saveは別途。

# ***************************************
# mother plant/self.nodeの確定Sから子nodeを分離
# ***************************************
def extract_node_conf(req_plan_node, S_confirmed_plan):

    node_list = list(itertools.chain.from_iterable(req_plan_node))

    extracted_list = []
    extracted_list.extend(S_confirmed_plan)

    # フラットなリストに展開する
    flattened_list = [item for sublist in extracted_list for item in sublist]

    # node_listとextracted_listを比較して要素の追加と削除を行う
    extracted_list = [[item for item in sublist if item in node_list] for sublist in extracted_list]

    return extracted_list



def separated_node_plan(node_req_plans, S_confirmed_plan):

    shipping_plans = []

    for req_plan in node_req_plans:

        shipping_plan = extract_node_conf(req_plan, S_confirmed_plan)

        shipping_plans.append(shipping_plan)                    

    return shipping_plans





# **********************************
# create tree
# **********************************
class Node:

    def __init__(self, name):
        self.name = name
        self.children = []

        # application attribute # nodeをインスタンスした後、初期値セット
        self.psi_list = None
        self.psi_list4supply = None

        self.safety_stock_week = 0
        #self.safety_stock_week = 2

        #self.lv_week = []

        self.lot_size             = 1 # defalt set

        # leadtimeとsafety_stock_weekは、ここでは同じ
        self.leadtime             = 1 # defalt set

        self.long_vacation_weeks  = []



    def add_child(self, child):
        self.children.append(child)



    def set_attributes(self, row):
        self.lot_size            = int( row[3] )
        self.leadtime            = int( row[4] )
        self.long_vacation_weeks = eval( row[5] )



    def set_psi_list(self, psi_list):

        self.psi_list = psi_list


    # supply_plan
    def set_psi_list4supply(self, psi_list):

        self.psi_list4supply = psi_list



    def get_set_childrenP2S2psi(self, plan_range):
    #def get_set_childrenS2psi(self, plan_range):

        ## 子node Pの収集、LT offsetによるS移動 リストの加算extend

        self.psi_list = [[[] for j in range(4)] for w in range(53*plan_range)]

        #print('self.name', self.name)
        #print('self.psi_list', self.psi_list)

        for child in self.children:

            #print('child.name', child.name)
            #print('child.psi_list', child.psi_list)

            for w in range( 53*plan_range ): 
            #for w in range( 53*5 ): 

                #print('child.psi_list[w][3]', w, child.psi_list[w][3])

                self.psi_list[w][0].extend(child.psi_list[w][3]) #setting P2S

                #print('self.psi_list[w][0]', w, self.psi_list[w][0])

        #print('self.psi_list', self.name,self.psi_list)








#                                
# node_confirmed_plans   = feedback_confirmedS2childrenP(node_req_plans, S_confirmed_plan)

# node_confirmed_plans: supply_planの子nodeの確定したP_confirm
#    = feedback_confirmedS2childrenP(
# node_req_plans: demand_planの子nodeの要求したS_request
# , S_confirmed_plan: supply_planの親nodeの確定したS_confirm
# )


    # 入出データは、いずれもtree構造の中でrootからself.で操作できる
    # 必要なのはw期間plan_range


    def feedback_confirmedS2childrenP(self, plan_range):
    #def get_set_childrenP2S2psi(self, plan_range):
    #def get_set_childrenS2psi(self, plan_range):

        ### 子node Pの収集、LT offsetによるS移動 リストの加算extend
        #
        #self.psi_list = [[[] for j in range(4)] for w in range(53*plan_range)]
        #
        ##print('self.name', self.name)
        ##print('self.psi_list', self.psi_list)


#def feedback_confirmedS2childrenP(node_req_plans, confirmed_plan):
##def calculate_shipping_plan(node_req_plans, confirmed_plan):
#
        node_req_plans       = []
        node_confirmed_plans = []

        self_confirmed_plan =  [[] for _ in range(53*plan_range)]
        #mother_confirmed_plan =  [[] for _ in range(53*plan_range)]

        # ************************************
        # setting mother_confirmed_plan
        # ************************************
        for w in range( 53*plan_range ): 
        #for w in range( 53*5 ): 

            # ここは再帰しているので、確定計画をcalc_PISでtree展開しておく???
            #
            # 親node自身のsupply_planのpsi_list[w][0]がconfirmed_S
            self_confirmed_plan[w].extend(self.psi_list4supply[w][0]) 


            ## mother plant root_nodeのsupply_planのpsi_list[w][0]がconfirmed_S
            #mother_confirmed_plan[w].extend(child.psi_list4supply[w][0]) 



        # ************************************
        # setting node_req_plans 各nodeからの要求S(=P)
        # ************************************
        # 子nodeのdemand_planのpsi_list[w][3]のPがS_requestに相当する
        # すべての子nodesから、S_reqをappendしてnode_req_plansを作る
        for child in self.children:

            #print('child.name', child.name)
            #print('child.psi_list', child.psi_list)

            child_S_req = [[] for _ in range(53*plan_range)]

            for w in range( 53*plan_range ): 
            #for w in range( 53*5 ): 

                #print('child.psi_list[w][3]', w, child.psi_list[w][3])

                child_S_req[w].extend(child.psi_list[w][3]) #setting P2S

            node_req_plans.append( child_S_req )


# node_req_plans      子nodeのP=S要求計画planのリストplans
# self_confirmed_plan 自nodeの供給計画の確定S


## 出荷先ごとの出荷計画を求める
#node_req_plans = [req_plan_node_1, req_plan_node_2, req_plan_node_3]



        # ***************************
        # node 分離
        # ***************************
        node_confirmed_plans = []

        node_confirmed_plans = separated_node_plan(node_req_plans, self_confirmed_plan)
        #shipping_plans = separated_node_plan(node_req_plans, self_confirmed_plan)
        #shipping_plans = separated_node_plan(node_req_plans, S_confirmed_plan)
        print('node_confirmed_plans',node_confirmed_plans)











## *************************************
## solve    mother plant leveling
## *************************************
#        node_confirmed_plans = []
#
#        for req_plan in node_req_plans:
#
#            print('len(node_req_plans)',len(node_req_plans),'230521 node_req_plans', node_req_plans)
#            print('req_plan',req_plan)
#
#
#            node_confirmed_plan = [[] for _ in range(len(node_req_plans))]
#        
#            for week, confirmed_demands in enumerate(self_confirmed_plan):
#            #for week, confirmed_demands in enumerate(mother_confirmed_plan):
#
#                print('230521 week',week)
#
#                for demand in confirmed_demands:
#
#                    for i, req_demand in enumerate(req_plan):
#
#                        if demand in req_demand:
#
#                            print('xxxx week',week)
#                             
#
#                            node_confirmed_plan[week].append(demand)
#
#                            break
#        
#            node_confirmed_plans.append(node_confirmed_plan)
#            #node_confirmed_plans.append(shipping_plan)
#
#        print('node_confirmed_plans',node_confirmed_plans)
#        #return node_confirmed_plans
#
#






        for i, child in enumerate(self.children):

            print('child.name', child.name)
            print('child.psi_list4supply', child.psi_list4supply)

            #child_S_req = [[] for _ in range(53*plan_range)]

            for w in range( 53*plan_range ): 
            #for w in range( 53*5 ): 

                print('child.psi_list[w][3]', w, child.psi_list[w][3])

                # 子nodeのsupply_planのPにmother_plantの確定Sをセット
                child.psi_list4supply[w][3] = [] # clearing list

                # i番目の子nodeの確定Sをsupply_planのPにextendでlot_idをcopy
                child.psi_list4supply[w][3].extend(node_confirmed_plans[i][w]) 

                #child_S_req[w].extend(child.psi_list[w][3]) #setting P2S

            #node_req_plans.append( child_S_req )

            # ココまででsupply planの子nodeにPがセットされたことになる。




        # *******************************************
        # supply_plan上で、PfixをSfixにPISでLT offsetする
        # *******************************************

        # ここでは再帰せずに、親子の一世代分のみP2Sを操作してS_fixを生成する
        #@230521

#### ココに直接、書く

#        calcP2S()


#    def calcP2S(self):

# **************************
# Safety Stock as LT shift
# **************************
        # leadtimeとsafety_stock_weekは、ここでは同じ
        safety_stock_week = self.leadtime

# **************************
# long vacation weeks 
# **************************
        lv_week           = self.long_vacation_weeks

        # P to S の計算処理
        self.psi_list4supply = shiftP2S_LV(self.psi_list4supply, safety_stock_week, lv_week)

        ## S to P の計算処理
        #self.psi_list = shiftS2P_LV(self.psi_list, safety_stock_week, lv_week)

#        pass



















#        for child in self.children:
#
#            #print('child.name', child.name)
#            #print('child.psi_list', child.psi_list)
#
#            for w in range( 53*plan_range ): 
#            #for w in range( 53*5 ): 
#
#                #print('child.psi_list[w][3]', w, child.psi_list[w][3])
#
#                self.psi_list[w][0].extend(child.psi_list[w][3]) #setting P2S
#
#                #print('self.psi_list[w][0]', w, self.psi_list[w][0])
#
#        #print('self.psi_list', self.name,self.psi_list)




#def feedback_confirmedS2childrenP(node_req_plans, confirmed_plan):
#def calculate_shipping_plan(node_req_plans, confirmed_plan):
#
#    node_confirmed_plans = []
#
#    for req_plan in node_req_plans:
#        shipping_plan = [[] for _ in range(len(req_plan))]
#        
#        for week, confirmed_demands in enumerate(confirmed_plan):
#
#            for demand in confirmed_demands:
#
#                for i, req_demand in enumerate(req_plan):
#
#                    if demand in req_demand:
#
#                        shipping_plan[week].append(demand)
#
#                        break
#        
#        node_confirmed_plans.append(shipping_plan)
#
#    return node_confirmed_plans





    def calcPS2I(self):
    
        #psiS2P = self.psi_list # copyせずに、直接さわる
    
        plan_len = len(self.psi_list)

        for w in range(1,plan_len): # starting_I = 0 = w-1 / ending_I =plan_len
        #for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53
    

            s   = self.psi_list[w][0]
            co  = self.psi_list[w][1]
    
            i0  = self.psi_list[w-1][2]
            i1  = self.psi_list[w][2]
    
            p   = self.psi_list[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************
    
            #print('i0',i0)
            #print('p',p)
    
    
            work = i0 + p  # 前週在庫と当週着荷分 availables
    
    
            #print('work',work)
            #print('s',s)
    
    
            #@230321 TOBE memo ここで、期末の在庫、S出荷=売上を操作している
            # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
            # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する
    
            # モノがお金に代わる瞬間
    
            diff_list = [x for x in work if x not in s] # I(n-1)+P(n)-S(n)
    
            self.psi_list[w][2] = i1 = diff_list
    
        #return psiS2P  # returnしなくて良い。self.psi_listを直接、操作。



    def calcPS2I4supply(self):
    
        #psiS2P = self.psi_list # copyせずに、直接さわる
    
        print('self.name',self.name)
        print('self.psi_list4supply',self.psi_list4supply)

        plan_len = 53*plan_range 
        #plan_len = len(self.psi_list4supply)

        for w in range(1,plan_len): # starting_I = 0 = w-1 / ending_I =plan_len
        #for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53
    

            s   = self.psi_list4supply[w][0]
            co  = self.psi_list4supply[w][1]
    
            i0  = self.psi_list4supply[w-1][2]
            i1  = self.psi_list4supply[w][2]
    
            p   = self.psi_list4supply[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************
    
            #print('i0',i0)
            #print('p',p)
    
    
            work = i0 + p  # 前週在庫と当週着荷分 availables
    
    
            #print('work',work)
            #print('s',s)
    
    
            #@230321 TOBE memo ここで、期末の在庫、S出荷=売上を操作している
            # S出荷=売上を明示的にlogにして、売上として記録し、表示する処理
            # 出荷されたS=売上、在庫I、未出荷COの集合を正しく表現する
    
            # モノがお金に代わる瞬間
    
            diff_list = [x for x in work if x not in s] # I(n-1)+P(n)-S(n)
    
            self.psi_list4supply[w][2] = i1 = diff_list
    
        #return psiS2P  # returnしなくて良い。self.psi_listを直接、操作。




    def calcS2P(self):

# **************************
# Safety Stock as LT shift
# **************************
        # leadtimeとsafety_stock_weekは、ここでは同じ
        safety_stock_week = self.leadtime

# **************************
# long vacation weeks 
# **************************
        lv_week           = self.long_vacation_weeks

        # S to P の計算処理
        self.psi_list = shiftS2P_LV(self.psi_list, safety_stock_week, lv_week)

        pass







def create_tree(csv_file):

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)

        next(reader)  # ヘッダー行をスキップ

        # nodeインスタンスの辞書を作り、親子の定義に使う
        nodes = {row[2]: Node(row[2]) for row in reader}

        #print('nodes',nodes)

        f.seek(0)  # ファイルを先頭に戻す

        next(reader)  # ヘッダー行をスキップ

        next(reader)  # root行をスキップ 

        for row in reader:

            print('row',row)

            parent = nodes[row[0]]

            child = nodes[row[1]]

            parent.add_child(child)

            child.set_attributes(row) #子ノードにアプリケーション属性をセット

    return nodes           # すべてのインスタンス・ポインタを返して使う
    #return nodes['JPN']   # "JPN"のインスタンス・ポインタ



def set_psi_lists(node, node_psi_dict):
    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []: # 子nodeがないleaf nodeの場合

        #print('leaf', node.name )

        node.set_psi_list(node_psi_dict.get(node.name))

    else:

        #print('no leaf', node.name )

        node.get_set_childrenP2S2psi(plan_range)
        #node.get_set_childrenS2psi(plan_range)

    for child in node.children:

        set_psi_lists(child, node_psi_dict)



def set_psi_lists_postorder(node, node_psi_dict):

    for child in node.children:

        set_psi_lists_postorder(child, node_psi_dict)

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []: # 子nodeがないleaf nodeの場合
        #print('leaf', node.name )

        # 辞書のgetメソッドでキーから値を取得。キーが存在しない場合はNone
        node.set_psi_list(node_psi_dict.get(node.name)) 

        # shifting S2P
        node.calcS2P()  # backward plan with postordering 


    else: 

        #print('no leaf', node.name )

        # gathering S and Setting S
        node.get_set_childrenP2S2psi(plan_range)
        #node.get_set_childrenS2psi(plan_range)

        # shifting S2P
        node.calcS2P()  # backward plan with postordering 




def make_psi4supply(node,node_psi_dict):

    print('stting 4supply node.name',node.name)

    node_psi_dict[node.name] = node.psi_list # 新しいdictにpsiをセット

    for child in node.children:

        make_psi4supply(child, node_psi_dict)

    return node_psi_dict



def set_psi_lists4supply(node, node_psi_dict):

    print('stting 4supply node.name',node.name)

    node.set_psi_list4supply(node_psi_dict.get(node.name))

    for child in node.children:

        set_psi_lists4supply(child, node_psi_dict)





# 確定Pのセット
def feedback_psi_lists(node, node_psi_dict):
    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。
    if node.children == []: # 子nodeがないleaf nodeの場合

        #print('leaf', node.name )

        #node.set_psi_list(node_psi_dict.get(node.name))

        pass

    else:

        #print('no leaf', node.name )


#node_confirmed_plans = feedback_confirmedS2childrenP(node_req_plans, S_confirmed_plan)

        # returnせずに子nodeのpsiのPに返す child.psi_list[w][3]に直接セット
        #feedback_confirmedS2childrenP(node_req_plans, S_confirmed_plan)

        node.feedback_confirmedS2childrenP(plan_range)

        #node.get_set_childrenP2S2psi(plan_range)
        #node.get_set_childrenS2psi(plan_range)

    for child in node.children:

        feedback_psi_lists(child, node_psi_dict)








def get_all_psi(node):

    node_all_psi[node.name] = node.psi_list

    node_search.append(node)

    for child in node.children:

        get_all_psi(child)

    ####return( node_I4bullwhip )





def get_all_psi4supply(node):

    node_all_psi[node.name] = node.psi_list4supply


    node_search.append(node)

    for child in node.children:

        get_all_psi4supply(child)



def set_all_I4bullwhip(node):

    node_search.append(node)


    for child in node.children:

        set_all_I4bullwhip(child)


    # node辞書に時系列set
    #node.set_I4bullwhip()

    I_hi_len = [] #在庫の高さ=リストの長さ

    for w in range( len( node.psi_list ) ):

        I_hi_len.append( len(node.psi_list[w][2]) ) 


    node_I4bullwhip[node.name] = I_hi_len

    return( node_I4bullwhip )



def calc_all_psi2i(node):

    node_search.append(node)

    node.calcPS2I()

    for child in node.children:

        calc_all_psi2i(child)



def calc_all_psi2i4supply(node):

    node_search.append(node)

    node.calcPS2I4supply()

    for child in node.children:

        calc_all_psi2i4supply(child)




def calc_all_psi2i_postorder(node):

    for child in node.children:

        calc_all_psi2i_postorder(child)

    node_search.append(node)

    node.calcPS2I()  # backward plan with postordering 



def calc_all_psiS2P_postorder(node):

    for child in node.children:

        calc_all_psiS2P_postorder(child)

    node_search.append(node)

    node.calcS2P()  # backward plan with postordering 

























# ***************************
# tree definition initialise
# ***************************

node_search = []


node_I4bullwhip = {}

#nodes = {}

nodes_outbound = {}
nodes_inbound  = {}

#csv_file = 'supply_chain_tree_attributes_test_small3.csv'
#csv_file = 'supply_chain_tree_attributes_test_small.csv'

#outbound_tree_file = 'supply_chain_tree_outbound_attributes_sssmall.csv'
outbound_tree_file = 'supply_chain_tree_outbound_attributes.csv'

inbound_tree_file  = 'supply_chain_tree_inbound_attributes.csv'


# M2Wで生成したpsi_listをnodeをキーとするdictに変換する
# node_psi_dictは生成済み


# ***************************
# create tree
# ***************************


#nodesですべてのnodeインスタンスを取得
nodes_outbound = create_tree(outbound_tree_file)  
root_node_outbound = nodes_outbound['JPN_OUT']

nodes_inbound = create_tree(inbound_tree_file)  
root_node_inbound = nodes_inbound['JPN']


# ***************************
# trans_month2week
# ***************************
#in_file    = "S_month_data_small3_2zero.csv"
#in_file    = "S_month_data_small3_2.csv"
#in_file    = "S_month_data_small3_1.csv"
#in_file    = "S_month_data_small3.csv"
#in_file    = "S_month_data_small2.csv"
#in_file    = "S_month_data_small.csv"

in_file    = "S_month_data.csv"

#in_file    = "S_month_data_prev_year.csv"
#in_file    = "S_month_data_prev_year_sssmall.csv"

out_file   = "S_iso_week_data.csv"


plan_range = 1   #### 計画期間=1年

node_yyyyww_value, node_yyyyww_key, plan_range, df_capa_year =trans_month2week(in_file, out_file)


# an image of data
#
#for node_val in node_yyyyww_value:
#    print( node_val )
#
##['SHA_N', 22.580645161290324, 22.580645161290324, 22.580645161290324, 22.580645161290324, 26.22914349276974, 28.96551724137931, 28.96551724137931, 28.96551724137931, 31.067853170189103, 33.87096774193549, 33.87096774193549, 33.87096774193549, 33.87096774193549, 30.33333333333333, 30.33333333333333, 30.33333333333333, 30.33333333333333, 31.247311827956988, 31.612903225806452,


#node_yyyyww_key [['CAN', 'CAN202401', 'CAN202402', 'CAN202403', 'CAN202404', 'CAN202405', 'CAN202406', 'CAN202407', 'CAN202408', 'CAN202409', 'CAN202410', 'CAN202411', 'CAN202412', 'CAN202413', 'CAN202414', 'CAN202415', 'CAN202416', 'CAN202417', 'CAN202418', 'CAN202419', 


# ********************************
# make_node_psi_dict
# ********************************

# 1. treeeを生成して、nodes[node_name]辞書で、各nodeのinstanceを操作する

# 2. 週次S yyyywwの値valueを月次Sから変換、
#    週次のlotの数Slotとlot_keyを生成、

# 3. ロット単位=lot_idとするリストSlot_id_listを生成しながらpsi_list生成

# 4. node_psi_dict=[node1: psi_list1,,,]を生成し、treeのnode.psi_listに接続する

S_week = []

# initial 
node_psi_dict = {} # node_psi辞書

node_outbound_demand_psi_dict = {} # node_psi辞書4demand plan

node_outbound_supply_psi_dict = {} # node_psi辞書4supply plan

node_inbound_psi_dict = {} # node_psi辞書



#make_node_psi_dictを作る
node_outbound_demand_psi_dict = make_node_psi_dict( node_yyyyww_value, node_yyyyww_key, nodes_outbound )

print('node_outbound_demand_psi_dict',node_outbound_demand_psi_dict)


#
# def make_node_outbound_supply_psi_dict(): => treeを手繰って、all nodes生成??
#
#def make_node_psi_dict( node_yyyyww_value, node_yyyyww_key, nodes ):
#
#    for i, node_val in enumerate(node_yyyyww_value):
#    
#        node_name = node_val[0]
#        S_week    = node_val[1:]
#    
#        #print('node_name',node_name)
#        #print('S_week',S_week)
#
#        node = nodes[node_name]   # node_nameからnodeインスタンスを取得
#
#        # node.lot_sizeを使う
#        lot_size = node.lot_size # Node()からセット
#    
#
#        # makeSでSlotを生成
#        Slot = makeS(S_week, lot_size)
#
#        # nodeに対応するpsi_listを生成する
#        psi_list = [[[] for j in range(4)] for w in range( len(S_week) )] 






# ***************************************
# PSI main process
# ***************************************


#@230511 stop yyyywwの中にJPN_OUTがあるが、そもそもinbound側にこのyyyywwは不要
#node_inbound_psi_dict = make_node_psi_dict( node_yyyyww_value, node_yyyyww_key, nodes_inbound )


# ***************************************
# set_psi_lists_postorder
# ***************************************

# Sをnode.psi_listにset
set_psi_lists_postorder(root_node_outbound, node_outbound_demand_psi_dict) 

#print('root_node.psi_list',root_node.psi_list)

# STOP
#show_psi_3D_graph_node(root_node)


# ***************************************
# calc_all_psi2i
# ***************************************
node_search = []

calc_all_psi2i(root_node_outbound) # SP2I計算はpreorderingでForeward Planningする

# STOP
#show_psi_3D_graph_node(root_node)


#for n in node_search:
#
#    print('node_search node.name',n.name)







# ***************************************
# optimise Supply Chain 最適化ではなく単純な平準化
# ***************************************

# 全ロット一直線にリスト化する感覚

# ***************************************
# optimise root　
# 平準化するために週当たりの平均の生産出荷ロット数でslice
# この時、当初セットされた時間を遅れない事、先行生産する方向に前倒しする
# 1. 早いもの順で、ロットのリスト位置で、allocate
# 2. profit maxで allocate
# ***************************************


# ***************************************
# input&set root/JPN_OUT 2 root/JPN optimizer 
# ***************************************

#@230513 capacity setting
# *********************************
# mother plant capacity parameter
# *********************************

# def calc_capacity(root_node_outbound, demand_supply_ratio, df_capa_year):

demand_supply_ratio = 3  # demand_supply_ratio = ttl_supply / ttl_demand


# ********************
# common_plan_unit_lot_size
# OR
# lot_size on root( = mother plant )
# ********************
plant_lot_size = 0





# STOP  mother plantのlot_size定義を取るのはやめて、
# common plant unitとして一つのlot_sizeを使う

common_plan_unit_lot_size = 100 #24 #50 # 100  # 100   # 3 , 10, etc

plant_lot_size     = common_plan_unit_lot_size

#plant_lot_size     = root_node_outbound.lot_size

print('plant_lot_size',plant_lot_size)


# ********************
# 辞書 year key: total_demand
# ********************




#vol_w = 0
#lot_w = 0
#
#for i, row in df_capa.iterrows():
#
#    vol_w += row['total_demand']
#    lot_w += row['total_demand'] // plant_lot_size
#
#    plant_capa_vol[row['year']] = vol_w
#    plant_capa_lot[row['year']] = lot_w


# STOP
#df_capa_year['total_lots'] = df_capa_year['total_demand'] //  plant_lot_size


# 切り捨ては、a//b
# 切り上げは、(a+b-1)//b

plant_capa_vol = {}
plant_capa_lot = {}

week_vol       = 0


for i, row in df_capa_year.iterrows():

    print('row',row)

    plant_capa_vol[row['year']] = row['total_demand']

    #plant_capa_lot[row['year']] = (row['total_demand']+plant_lot_size -1)// plant_lot_size # 切り上げ


    # apply "demand_supply_ratio"

    week_vol =  row['total_demand'] * demand_supply_ratio  // 52

    #week_vol = ( row['total_demand'] * demand_supply_ratio  + 52 - 1 ) // 52

    print('week_vol',week_vol)



    plant_capa_lot[row['year']]=(week_vol + plant_lot_size -1) //plant_lot_size

    #plant_capa_lot[row['year']] = ((row['total_demand']+52-1 // 52)+plant_lot_size-1) // plant_lot_size
    #plant_capa_lot[row['year']] = row['total_demand'] // plant_lot_size


#plant_capa_dic = {}
#for i, row in df_capa.iterrows():
#    plant_capa_dic[row['year']] = row['total_demand']




print('plant_capa_vol',plant_capa_vol)
print('plant_capa_lot',plant_capa_lot)





# **********************
# ISO weekが年によって52と53があるので、year_stとyear_endも持ち歩く?? 誤差??
# ここでは、誤差として、53*plan_rangeの年別53週のaverage_capaとして定義
# **********************

# 53*plan_range
#

year_st      = 2020
year_end     = 2021

year_st      = df_capa_year['year'].min()
year_end     = df_capa_year['year'].max()

week_capa    = []
week_capa_w  = []

for year in range(year_st, year_end+1):  # 5_years 

    week_capa_w = [ plant_capa_lot[year] ] * 53
    #week_capa_w = [ (plant_capa_lot[year] + 53 - 1) // 53 ] * 53   # 切り上げ

    week_capa += week_capa_w


print('week_capa',week_capa)

leveling_S_in  = []

leveling_S_in     = root_node_outbound.psi_list


for w in range( 53 * plan_range ):

    #print('capa week limit and S_lots', w, week_capa[w] )
    #print('capa week limit and S_lots', w, leveling_S_in[w][0] )



    #print('capa week len(S_lots)', w, week_capa[w], len(leveling_S_in[w][0]) )
    print('capa week limit and S_lots', w, week_capa[w], leveling_S_in[w][0] )





# *****************************
# mother plan leveling    setting initial data
# *****************************

#a sample data setting

week_no = 53 * plan_range
#week_no = 12

S_confirm = 15

S_lots = []
S_lots_list = []



for w in range( 53 * plan_range ):

    S_lots_list.append( leveling_S_in[w][0] )
    #S_lots[w]  = leveling_S_in[w][0] 

#S_lots_list    = leveling_S_in[:][0] # for wで回してセットする


#S_lots_list    = data['S_lots']
print('S_lots_list',S_lots_list)


prod_capa_limit = week_capa

#prod_capa_limit = data['prod_capa_limit']
print('prod_capa_limit',prod_capa_limit)


S_confirm_list = [[] for i in range(53 * plan_range)]   # [[],[],,,,[]]

#S_confirm_list = [[] for i in range(week_no+1)]   # [[],[],,,,[]]
print('S_confirm_list',S_confirm_list)





# **********************************
# 多次元リストの要素数をcount
# **********************************
def multi_len(l):
    count = 0
    if isinstance(l, list):
        for v in l:
            count += multi_len(v)
        return count
    else:
        return 1



# a way of leveling
#
#      supply           demand
# ***********************************
# *                *                *
# * carry_over_out *                *
# *                *   S_lot        *
# *** capa_ceil ****   get_S_lot    *
# *                *                *
# *  S_confirmed   *                *
# *                *                *
# *                ******************
# *                *  carry_over_in *
# ***********************************

#
# carry_over_out = ( carry_over_in + S_lot ) - capa 
#



def leveling_operation(carry_over_in, S_lot, capa_ceil ):

    demand_side = []

    demand_side.extend(carry_over_in )

    demand_side.extend( S_lot )

    if len( demand_side ) <= capa_ceil:

        S_confirmed = demand_side

        carry_over_out = []                       # 繰り越し無し


    else:

        S_confirmed    = demand_side[:capa_ceil]  # 能力内を確定する

        carry_over_out = demand_side[capa_ceil:]  # 能力を超えた分を繰り越す

    return S_confirmed, carry_over_out




def confirm_S(S_lots_list, prod_capa_limit):

    carry_over_in = []

    week_no = 53 * plan_range - 1

    for w in range(week_no, -1, -1):  # 6,5,4,3,2,1,0

        print('w',w)

        S_lot       = S_lots_list[w]
        capa_ceil   = prod_capa_limit[w]

        S_confirmed, carry_over_out = leveling_operation(carry_over_in, S_lot, capa_ceil )

        carry_over_in = carry_over_out

        S_confirm_list[w] = S_confirmed


    return S_confirm_list








##
## test dataを本番のpsiに替える
##
#
## 出荷先の要求計画
#req_plan_node_1 = [["L1","L2","L3"],[]]
#req_plan_node_2 = [["La","Lb","Lc"],[]]
#req_plan_node_3 = [["Lx","Ly","Lz"],[]]
#
## 工場で平準化され確定した出荷計画
#S_confirmed_plan = [["L1","L2","L3","La","Lb"],["Lc","Lx","Ly","Lz"]]
#
## 出荷先ごとの出荷計画を求める
#node_req_plans = [req_plan_node_1, req_plan_node_2, req_plan_node_3]
#
#node_confirmed_plans = calculate_shipping_plan(node_req_plans, S_confirmed_plan)
#
## 結果の出力
#for i, node in enumerate(node_req_plans):
#    print(f"Node {i+1}:")
#    print("Request Plan:", node)
#    print("Shipping Plan:", node_confirmed_plans[i])
#    print()









# ******************
# initial setting
# ******************

####carry_over_in = []

capa_ceil = 10

# S_confirm_list = [[] for i in range(week_no+1)]   # [[],[],,,,[]]


S_confirm_list = confirm_S(S_lots_list, prod_capa_limit)


print(f"S_confirm={S_confirm_list}, S_lots={S_lots_list}")


# **********************************
# 多次元リストの要素数をcountして、confirm処理の前後の要素数を比較check
# **********************************
S_lots_list_element    = multi_len(S_lots_list)

S_confirm_list_element = multi_len(S_confirm_list)

print('S_lots_list_element compare', S_lots_list_element)
print('S_confirm_list_element compare', S_confirm_list_element)



# ***************************************
# initial setting Outbound Supply Plan
# ***************************************

# make_psi_dict4outbound_supply_plan()
# アウトバウンドの供給計画用のpsi辞書を作る

# connect_psi2Node( rootnode, outbound_supply_plan_dic )
# psi辞書をNodeのpsi_list4outbound_supply_planに接続する

# make_supply_plan
# 前提: 生産平準化levelingされた確定S_fixedをもとに供給計画を生成する。

# アウトバウンドで、mother_plantの確定Sをもとに、
# rootnodeから、pre_orderingで処理する

# buffer在庫拠点: buffer_stock_pointの指定を各nodeに設定する。

# mother plantの確定S:fixedから子node P:fixedを生成した後、、

# BSP:buffer stcok pointの手前までのP:fixed_I2S計算
# その1:確定Pを「そのまま流す」計算PI2Sを適用する
#       P:fixed_I2S:fixedを計算、供給計画生成#

# BSP:buffer stock point以降のP:fixed_S:latest_2I計算
# その2:確定Pと最遅需要計画SでPS2Iからbuffer在庫生成

# 在庫を可視化する


# ***************************************
# output&set root/JPN optimizer 2 out&in 
# ***************************************

# ***************************************
# output&set root/JPN optimizer 2 out&in 
# ***************************************








# 需要計画が生成された結果を、供給計画の初期データとしてセット
#
# ***************************************
# psi_dictのdeepcopy
# ***************************************
#node_outbound_demand_psi_dict = {} # node_psi辞書4demand plan
#node_outbound_supply_psi_dict = {} # node_psi辞書4supply plan






print('node_outbound_demand_psi_dict',node_outbound_demand_psi_dict)



#
# demand_planが生成されたタイミングで、
# supply用のnode_psi_dictをdemand用のself.psi_listを辿って生成する。
# transport nodeが生成される。
#
node_outbound_supply_psi_dict = {} # node_outbound_supply_psi_dictの初期セット

node_outbound_supply_psi_dict = make_psi4supply(root_node_outbound,node_outbound_supply_psi_dict) 



#
# node_outbound_demand_psi_dictでは、末端市場のleafnodeのみセット
# deepcopyでは不十分
#
#node_outbound_supply_psi_dict = deepcopy(node_outbound_demand_psi_dict) 

print('node_outbound_supply_psi_dict',node_outbound_supply_psi_dict)


#
# root_nodeのS psi_list[w][0]に、levelingされた確定出荷S_confirm_listをセット
# "JPN-OUT"
#
for w in range( 53 * plan_range ):

    node_name = root_node_outbound.name #Nodeからnode_nameを取出す

    print('node_name',node_name)

    node_outbound_supply_psi_dict[node_name][w][0] = S_confirm_list[w]


print('node_outbound_supply_psi_dict',node_outbound_supply_psi_dict)


# supply_plan用のnode_psi_dictをtree構造のNodeに接続する
#set_dict2tree()  

# Sをnode.psi_listにset  # psi_listをclass Nodeに接続

print('connect node_outbound_supply_psi_dict',node_outbound_supply_psi_dict)

set_psi_lists4supply(root_node_outbound, node_outbound_supply_psi_dict) 


#
# 接続して直ぐに、mother_plantの確定Sをtreeのすべての子ノードにfeedbackする
#

print('feedback node_outbound_supply_psi_dict',node_outbound_supply_psi_dict)

feedback_psi_lists(root_node_outbound, node_outbound_supply_psi_dict)
#feedback_psi_lists(node, node_psi_dict):



# S_confirm_list: mother planの出荷計画を平準化、確定した出荷計画を
# children_node.P_request: すべての子nodeの出荷要求数のリストと比較して、
# children_node.P_confirmed: それぞれの子nodeの出荷確定数を生成する



#@230521
#node_confirmed_plans = feedback_confirmedS2childrenP(node_req_plans, S_confirmed_plan)




#その他の関数naming
# compare_confirmedS2childrenP()
# feedback_levelingS2childrenP()



#
#treeのrootnodeからサーチして、psi計算する
#

# buffer_stockまで、確定S=確定P
#calc_SI2P()

#calc_all_psi2i(root_node_outbound) 
# SP2I計算はpreorderingでForeward Planningす
























#print('leveling_S_in',leveling_S_in)

# def leveling_S(in, out, plant_capa):


# ***************************************
# optimise root or leveling production
# ***************************************

# rootのpsi_list[0] Sの需要を平準化するために前倒しする

#
#leveling_production(root_node_outbound, )
#


# ***************************************
# output&set root/JPN optimizer 2 out&in 
# ***************************************










# ***************************************
# INBOUND
# ***************************************

# ***************************************
# initial set root/JPN_OUT 2 root/JPN
# ***************************************

# ***************************************
# set_parent_P2S2psi()  self.psi_listのPを 子nodeの空psiの同週のSにextend
# ***************************************












# *********************************
# visualise with Axes3D
# *********************************

# STOP
#node_I4bullwhip = set_all_I4bullwhip(root_node)
#
#show_node_I4bullwhip_color(node_I4bullwhip)



# *********************************
# node_all_psiからIを抽出してnode_I_list生成してvisualise
# *********************************
node_all_psi = {} # node:psi辞書に抽出

get_all_psi(root_node_outbound)

#get_all_psi4supply(root_node_outbound)


#print('node_all_psi', node_all_psi )


# X
week_len = len(node_yyyyww_key)

# Y
node_w_list = list( node_all_psi.keys() )

node_len = len(node_w_list)

node_I_list = [[]*i for i in range(node_len)]





#print('node_I_list',node_I_list)

for node_name, psi_list in node_all_psi.items():

    node_index = node_w_list.index(node_name)

    #supply_inventory_list = [[]*i for i in range( 53 * plan_range )]
    supply_inventory_list = [[]*i for i in range(len(psi_list))]

    for week in range(len(psi_list)):

        step_lots = psi_list[week][2]

        #print('step_lots',step_lots)

        supply_inventory_list[week] = step_lots




    node_I_list[node_index] = supply_inventory_list 


print('Y:node_len node_I_list ',week_len, node_I_list)

#visualise_psi_label(node_I_list, node_w_list)







# ***************************************
# calc_all_psi2i
# ***************************************
node_search = []

# SP2I4supplyの計算はsupply planのpsiをpreorderingでForeward Planningする
calc_all_psi2i4supply(root_node_outbound) 








# *********************************
# node_all_psiからIを抽出してnode_I_list生成してvisualise
# *********************************
node_all_psi = {}

#get_all_psi(root_node_outbound)
get_all_psi4supply(root_node_outbound)


print('node_all_psi', node_all_psi )


# X
week_len = 53 * plan_range + 1
#week_len = len(node_yyyyww_key[0]) # node数が入る所を[0]で数える・・・

print('X:week_len  53 * plan_range ', week_len, 53 * plan_range)
#print('X:week_len  len(node_yyyyww_key[0]) ', week_len,node_yyyyww_key[0])


# Y
node_w_list = list( node_all_psi.keys() )
node_len = len(node_w_list)

print('Y:node_len  node_w_list', node_len, node_w_list)



#print('node_w_list', node_w_list)

#print('X;week_len  Y:node_len',week_len,node_len)


node_I_list = [[]*i for i in range(node_len)]

#print('node_I_list',node_I_list)





# lot_stepの「値=長さ」の入れ物 x軸=week y軸=node
#
#week_len = len(node_yyyyww_key)ではなく 53 * plan_range でmaxに広げておく
#node_len = len(node_w_list)

I_lot_step_week_node =  [ [None]*week_len  for _ in range( node_len )]










for node_name, psi_list in node_all_psi.items():

    node_index = node_w_list.index(node_name)

    supply_inventory_list = [[]*i for i in range( 53 * plan_range )]
    #supply_inventory_list = [[]*i for i in range(len(psi_list))]

    for week in range(53 * plan_range):
    #for week in range(len(psi_list)):

        step_lots = psi_list[week][2]

        print('step_lots',step_lots)




        week_pos = week
        node_pos = node_w_list.index( node_name )

        print(f'[{week_pos}][{node_pos}]')

        I_lot_step_week_node[node_pos][week_pos] = len( step_lots )
        #I_lot_step_week_node[week_pos][node_pos] = len( step_lots )








        supply_inventory_list[week] = step_lots

    node_I_list[node_index] = supply_inventory_list 

#visualise_psi_label(node_I_list, node_w_list)




I_visual_df = pd.DataFrame(I_lot_step_week_node, index=node_w_list)

data = [
    go.Bar(x=I_visual_df.index, y=I_visual_df[0])
]

layout = go.Layout(
    title='Inventory Bullwip animation Global Supply Chain',
    xaxis={'title': 'Location node'},
    yaxis={'title': 'Lot-ID count', 'showgrid': False},
    font={'size': 10},
    width=800,
    height=400,
    showlegend=False
)

frames = []
for week in I_visual_df.columns:
    frame_data = [go.Bar(x=I_visual_df.index, y=I_visual_df[week])]
    frame_layout = go.Layout(
        annotations=[
            go.layout.Annotation(
                x=0.95,
                y=1,
                xref='paper',
                yref='paper',
                text=f"Week number: {week}",
                showarrow=False,
                font={'size': 14}
            )
        ]
    )
    frame = go.Frame(data=frame_data, layout=frame_layout)
    frames.append(frame)

fig = go.Figure(data=data, layout=layout, frames=frames)

offline.plot(fig)
