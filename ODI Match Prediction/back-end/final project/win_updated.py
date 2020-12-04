#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)


def win_pred_func(Team1,Team2,Ground):
    #Team1=Team1
    #Team2=Team2
    #Ground = Ground


    """""""""        
        newarray = x_test[['id']].values
        newarray2 = newarray.ravel()
        newdf = pd.DataFrame({'Winner':predictions, 'id':newarray2})
        sns.factorplot(x="id", y="Winner", data=newdf)
        plt.yticks(np.arange(0, 14, 1))
        plt.xticks(rotation='vertical')
    plt.show()
    """""""""
        
    matches = pd.read_csv("odi.xls")
    matches['Winner'].fillna('Draw', inplace=True)
    matches['Ground'].fillna('Dubai',inplace=True)
    matches['Margin'].fillna(int(0),inplace=True)
    matches.replace(['Australia', 'England', 'New Zealand', 'East Africa', 'Sri Lanka',
           'Pakistan', 'India', 'West Indies', 'Canada', 'Bangladesh',
           'South Africa', 'Zimbabwe', 'U.A.E.', 'Netherlands', 'Kenya',
           'Scotland', 'Namibia', 'Hong Kong', 'Asia XI', 'Africa XI',
           'Bermuda', 'Ireland', 'Afghanistan', 'P.N.G.','ICC World XI','U.S.A.'],['Aus','Eng','NZ','EAFR','SL','PAK','IND','WI','CAN','BAN','RSA','ZIM',
           'UAE','NETH','KEN','SCT','NAM','HK','ASXI','AFXI','BER','IRL','AFG','PNG','ICCXI','USA'],inplace = True)
    encode = {  'Team 1':{'Aus':1,'Eng':2,'NZ':3,'EAFR':4,'SL':5,'PAK':6,'IND':7,'WI':8,'CAN':9,'BAN':10,'RSA':11,'ZIM':12,
                          'UAE':13,'NETH':14,'KEN':15,'SCT':16,'NAM':17,'HK':18,'ASXI':19,'AFXI':20,'BER':21,'IRL':22,'AFG':23,
                          'PNG':24,'ICCXI':25,'U.S.A.':26},
                'Team 2':{'Aus':1,'Eng':2,'NZ':3,'EAFR':4,'SL':5,'PAK':6,'IND':7,'WI':8,'CAN':9,'BAN':10,'RSA':11,'ZIM':12,
                          'UAE':13,'NETH':14,'KEN':15,'SCT':16,'NAM':17,'HK':18,'ASXI':19,'AFXI':20,'BER':21,'IRL':22,'AFG':23,
                          'PNG':24,'ICCXI':25,'USA':26},
                'Winner':{'Aus':1,'Eng':2,'NZ':3,'EAFR':4,'SL':5,'PAK':6,'IND':7,'WI':8,'CAN':9,'BAN':10,'RSA':11,'ZIM':12,
                          'UAE':13,'NETH':14,'KEN':15,'SCT':16,'NAM':17,'HK':18,'ASXI':19,'AFXI':20,'BER':21,'IRL':22,'AFG':23,
                          'PNG':24,'ICCXI':25,'USA':26,'no result':27,'tied':28},
                'Ground':{'Melbourne':1, 'Manchester':2, "Lord's":3, 'Birmingham':4, 'Christchurch':5,
                          'Swansea':6, 'Leeds':7, 'The Oval':8, 'Dunedin':9, 'Nottingham':10,
                          'Wellington':11, 'Adelaide':12, 'Auckland':13, 'Scarborough':14, 'Sialkot':15,
                          'Albion':16, 'Sahiwal':17, 'Lahore':18, "St John's":19, 'Castries':20, 'Quetta':21,
                          'Sydney':22, 'Brisbane':23, 'Karachi':24, 'Perth':25, 'Kingstown':26, 'Hamilton':27,
                          'Ahmedabad':28, 'Jalandhar':29, 'Cuttack':30, 'Colombo (SSC)':31, 'Amritsar':32,
                          'Delhi':33, 'Hyderabad (Sind)':34, 'Bengaluru':35, 'Gujranwala':36, 'Multan':37,
                          'Port of Spain':38, 'Napier':39, "St George's":40, 'Colombo (PSS)':41,
                          'Taunton':42, 'Leicester':43, 'Bristol':44, 'Worcester':45, 'Southampton':46,
                          'Derby':47, 'Tunbridge Wells':48, 'Chelmsford':49, 'Hyderabad (Deccan)':50,
                          'Jaipur':51, 'Srinagar':52, 'Vadodara':53, 'Indore':54, 'Jamshedpur':55,
                          'Guwahati':56, 'Moratuwa':57, 'Sharjah':58, 'Kingston':59, 'New Delhi':60,
                          'Thiruvananthapuram':61, 'Peshawar':62, 'Faisalabad':63, 'Pune':64, 'Hobart':65,
                          'Nagpur':66, 'Chandigarh':67, 'Bridgetown':68, 'Rawalpindi':69, 'Launceston':70,
                          'Kandy':71, 'Colombo (RPS)':72, 'Rajkot':73, 'Kanpur':74, 'Mumbai':75,
                          'Devonport':76, 'Kolkata':77, 'Chennai':78, 'Faridabad':79, 'Gwalior':80,
                          'Georgetown':81, 'Dhaka':82, 'Chittagong':83, 'Visakhapatnam':84,
                          'Mumbai (BS)':85, 'Margao':86, 'Lucknow':87, 'Sargodha':88, 'New Plymouth':89,
                          'Mackay':90, 'Ballarat':91, 'Canberra':92, 'Berri':93, 'Albury':94, 'Harare':95,
                          'Bulawayo':96, 'Cape Town':97, 'Port Elizabeth':98, 'Centurion':99,
                          'Johannesburg':100, 'Bloemfontein':101, 'Durban':102, 'East London':103, 'Patna':104,
                          'Mohali':105, 'Singapore':106, 'Toronto':107, 'Nairobi (Gym)':108,
                          'Nairobi (Club)':109, 'Nairobi (Aga)':110, 'Paarl':111, 'Benoni':112, 'Kochi':113,
                          'Kimberley':114, 'Sheikhupura':115, 'Taupo':116, 'Hove':117, 'Canterbury':118,
                          'Northampton':119, 'Cardiff':120, 'Chester-le-Street':121, 'Dublin':122,
                          'Edinburgh':123, 'Amstelveen':124, 'Galle':125, 'Melbourne (Docklands)':126,
                          'Potchefstroom':127, 'Jodhpur':128, 'Dambulla':129, 'Nairobi':130, 'Gros Islet':131,
                          'Tangier':132, 'Vijayawada':133, 'Kwekwe':134, 'Queenstown':135,
                          'Pietermaritzburg':136, 'Cairns':137, 'Darwin':138, 'Bogra':139, 'Khulna':140,
                          'Fatullah':141, 'Abu Dhabi':142, 'Basseterre':143, 'Belfast':144, 'Ayr':145,
                          'Kuala Lumpur':146, 'Mombasa':147, 'Nairobi (Jaff)':148, 'Nairobi (Ruaraka)':149,
                          'North Sound':150, 'Providence':151, 'Glasgow':152, 'Rotterdam':153,
                          'King City (NW)':154, 'Aberdeen':155, 'Dubai (DSC)':156, 'Roseau':157, 'The Hague':158,
                          'Schiedam':159, 'Hambantota':160, 'Pallekele':161, 'Whangarei':162, 'Ranchi':163,
                          'Dharamsala':164, 'ICCA Dubai':165, 'Dublin (Malahide)':166, 'Nelson':167,
                          'Lincoln':168, 'Mount Maunganui':169, 'Townsville':170, 'Mong Kok':171,
                          'Greater Noida':172, 'Port Moresby':173}
            }
    matches.replace(encode, inplace =True)
    matches = matches[['Team 1','Team 2','Winner','Ground','win_by_runs','win_by_wickets']]
    df = pd.DataFrame(matches)
    x = df[['Team 1','Team 2','Ground']]
    y = df[['Winner']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
   ## clf1 = svm.SVC()
    clf2 = tree.DecisionTreeClassifier()
    bagg_clf=BaggingClassifier(base_estimator=clf2,n_estimators=100,max_samples=1.0,bootstrap=True,n_jobs=-1,random_state=47)
    
   ## clf3 = RandomForestClassifier()
   ## clf11 = MLPClassifier()
    #classification_model(clf1, X_train, X_test, y_train, y_test, "SVC")
    #classification_model(clf2, X_train, X_test, y_train, y_test, "DecisionTreeClassifier")
    #classification_model(clf3, X_train, X_test, y_train, y_test, "RandomForestClassifier")
    #classification_model(clf11, X_train, X_test, y_train, y_test, "MLPClassifier")

 ##   ask = ["India","Australia","Melbourne",]
    bagg_clf.fit(X_train,y_train)
    predictions = bagg_clf.predict([[Team1,Team2,Ground]])
    #print(predictions)
    #model.fit(x_train,y_train)
    y_test_pred = bagg_clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_test_pred)
    names= [None] * len(predictions)
    j = 0
    print("===============name================")


    for i in predictions:
        if i == 1:
          names[j]="Australia"
          j=j+1
        elif i == 2:
          names[j] = "England"
          j=j+1
        elif i == 3:
          names[j] = "New Zealand"
          j=j+1
        elif i == 4:
          names[j] = "East Africa"
          j=j+1
        elif i == 5:
          names[j] = "Sri Lanka"
          j=j+1
        elif i == 6:
          names[j] = "Pakistan"
          j=j+1
        elif i == 7:
          names[j] = "India"
          j=j+1
        elif i == 8:
          names[j] = "West Indies"
          j=j+1
        elif i == 9:
          names[j] = "Canada"
          j=j+1
        elif i == 10:
          names[j] = "Bangladesh"
          j=j+1
        elif i == 11:
          names[j] = "South Africa"
          j=j+1
        elif i == 12:
          names[j] = "Zimbabwe"
          j=j+1
        elif i == 13:
          names[j] = "U.A.E"
          j=j+1
        elif i == 14:
          names[j] = "Netherlands"
          j=j+1
        elif i == 15:
          names[j] = "Kenya"
          j=j+1
        elif i == 16:
          names[j] = "Scotland"
          j=j+1
        elif i == 17:
          names[j] = "Namibia"
          j=j+1
        elif i == 18:
          names[j] = "Hong Kong"
          j=j+1
        elif i == 19:
          names[j] = "Asia XI"
          j=j+1
        elif i == 20:
          names[j] = "Africa XI"
          j=j+1
        elif i == 21:
          names[j] = "Bermuda"
          j=j+1
        elif i == 22:
          names[j] = "Ireland"
          j=j+1
        elif i == 23:
          names[j] = "Afghanistan"
          j=j+1
        elif i == 24:
          names[j] = "P.N.G"
          j=j+1
    #print('Accuracy : %s' % '{0:.3%}'.format(accuracy), name)
    print("winning team name and accuracy",names,accuracy*100)
    #print(names)

    def result():
        return names


# win_pred_func(1,2,)

# In[82]:


win_pred_func(11,10,103)


# In[36]:





# In[37]:





# In[69]:


import pandas as pd

pd.read_csv("odi.xls")


# In[ ]:





# In[ ]:





# In[ ]:




