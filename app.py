from flask import Flask, render_template,url_for,request
from flask import render_template
import pickle
import pandas as pd
import numpy as np

import dtale
#import matplotlib.pyplot as plt # for plotting graphs
#import seaborn as sns # for plotting graphs
#import datetime as dt 
#import joblib

app=Flask(__name__)
@app.route('/')
def home():
    #return 'hai'
    return render_template("home.htm")
    
@app.route('/predict', methods = ['POST','GET'])
def predict():
    
    pickled_model = pickle.load(open('model1.pkl', 'rb'))
    #pickled_model.predict(rfm.iloc[:,4:7])
    rfm_p = pd.DataFrame()
    rfm_p['no'] = pd.DataFrame(pickled_model.labels_)
    
    # Open the file in binary mode
    with open('rfm.pkl', 'rb') as drfm:
          
        # Call load method to deserialze
        drfm = pickle.load(drfm)        
    print(drfm)

    from datetime import date
    from datetime import datetime, timedelta
    import datetime
    present =datetime.datetime.now()
    #(present-d.date()).days
    if request.method == 'POST': 
        rn = request.form['rn']
        ltd = request.form['ltd']
        fr = request.form['fr']
        gr = request.form['gr']
        print(rn,ltd,fr,gr)
        #rn='retailerID2'
        
        #if ((drfm['Retailer_name'].str.lower()).eq(r.lower())).any() == False:
            #   print("no")
    
        # d = '01-04-2018'
        ##########################################################################
        if ((drfm['Retailer_name'].str.lower()).eq(rn.lower())).any() == False :
            d= datetime.datetime.strptime(ltd,'%Y-%m-%d')
            print('d-',d)
            #eeee=present.date()-d.date()
            #eeee.days
            p=present.date()
            #p=datetime.datetime.strptime("p",'%Y-%m-%d')
            
            #ltd=date.fromisoformat(ltd)
            #ltd= ltd.fromisoformat()
            #df['DateTime'] = pd.to_datetime(df['DateTime'])
                            
            #ltd = ltd.strftime("%d/%m/%Y")
            print('ltd - ',ltd)  
            ltd=present.date()-d.date()
            ltd= (ltd.days)
            print(ltd)
            ip = pd.DataFrame([[rn,ltd,fr,gr]])
            ip.columns=['Retailer_name','Recency','Frequency','Monetary']
            ip['Recency'] = ip['Recency'].astype(int)
            ip['Frequency'] = ip['Frequency'].astype(int)
            ip['Monetary'] = ip['Monetary'].astype(float)
            print(ip)
            #   print(drfm)
            ip.info()
            #ip=pd.DataFrame([['bb',1352 ,116,20505]])
           # drfm.loc[len(drfm)] = ip(ignore_index=True)
            drfm = pd.concat([drfm, ip], ignore_index = True)
            drfm.reset_index()
            #drfm.loc[214]
            #drfm.append([ip],ignore_index=True) 
            print(drfm)
            #drfm.info    
            #    }
            
    drfm['r_quartile'] = pd.qcut(drfm['Recency'].rank(method='first'), 4, ['1','2','3','4'])
    drfm['f_quartile'] = pd.qcut(drfm['Frequency'], 4, ['4','3','2','1'])
    drfm['m_quartile'] = pd.qcut(drfm['Monetary'], 4, ['4','3','2','1'])
    drfm.head()
    drfm.tail()
    drfm['RFM_Score'] = drfm.r_quartile.astype(str)+ drfm.f_quartile.astype(str) + drfm.m_quartile.astype(str)

    #Step 4: Segmenting with K-Means. Identify the optimal k.
    #sns.heatmap(drfm.iloc[:,:].corr())
        
#    from sklearn.cluster import KMeans
    #from sklearn.preprocessing import MinMaxScaler
    #scaler= MinMaxScaler()
    #drfm_nor = pd.DataFrame(scaler.fit_transform(drfm.iloc[:,1:4]))
        
    #rfm_nor.columns=['Recency','Frequency','Monetary']
    #rfm_nor['Retailer_name']=drfm.Retailer_name
    #rfm_nor.describe()
    #sse = []
    #for i in range(0,10):
    #    kmeans = KMeans(n_clusters=i+1,random_state=1231).fit(rfm_nor)
    #    sse.append(kmeans.inertia_)
    #r=list(range(1,11))
    #plt.plot((list(range(1,11))) , sse , "ro-"); plt.xlabel("No. of clusters"); plt.ylabel("Tot_within_ss")
        
    # OR
        
    #sns.pointplot(x = list(range(1,11)),y=sse)
    # Based on the above scree plot choosen 4 clusters #
    drfm_c=drfm.copy(deep=True)
    
    #pred=pickled_model.predict(drfm.iloc[:,4:7])
#    model1= KMeans(n_clusters = 4).fit(drfm.iloc[:,4:7])
    drfm_c['cluster_no'] = (pd.DataFrame(pickled_model.predict(drfm.iloc[:,4:7]))).astype(int)
    w= drfm_c.copy(deep=True)

    w['cluster_no'] = w['cluster_no'].replace([3,2,1,0],['HNI','PRIVILEGE','GOLD','SILVER'])
       
    
    #w.head()
    #dtale.show(drfm_c).open_browser()   
    #dtale.show(w).open_browser()   
    clust_no = pd.Series(drfm_c[(drfm_c['Retailer_name'] == rn)].cluster_no)
    re = (drfm_c[(drfm_c['Retailer_name'] == rn)].r_quartile).astype(int)
    fr = (drfm_c[(drfm_c['Retailer_name'] == rn)].f_quartile).astype(int)
    mo = (drfm_c[(drfm_c['Retailer_name'] == rn)].m_quartile).astype(int)
    rec=(100/re.iloc[0]).astype(int)
    frq=(100/fr.iloc[0]).astype(int)
    mon=(100/mo.iloc[0]).astype(int)
    
    
#    plt.xlabel("Courses offered")   
 #   plt.ylabel("No. of students enrolled")
  #  plt.title("Students enrolled in different courses")
   # plt.show()
    print(clust_no)
    type(clust_no)
    my_pred=clust_no.replace([2,0,3,1],['HNI','PRIVILEGE','GOLD','SILVER']).to_string(index=False)
    my_pred

    #my_pred = clust_no.replace([0,1,2,3],['HNI','PRIVILEGE','GOLD','SILVER'])
    #my_pred
    return render_template('result.htm',prediction=my_pred,rn=rn,re=rec,fr=frq,mo=mon)      
#pip install yellowbrick
#from yellowbrick.cluster import SilhouetteVisualizer
#from yellowbrick.datasets import load_nfl
        
#SilhouetteVisualizer(KMeans(4,random_state=42), colors='yellowbrick').fit(rfm_nor)
#w= rfm_c.copy(deep=False)
#SilhouetteVisualizer(KMeans(4,random_state=42), colors='yellowbrick').fit(rfm_nor)
            
        
if __name__== '__main__' :
    app.run(debug=True,use_reloader=False)
            
        
        
