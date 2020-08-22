
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn.metrics import r2_score
import math
import statistics
import datetime
import re
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, model_selection, tree, preprocessing, metrics, linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
import statsmodels.api as sm

print("******************************************")
print("preprocessing")

# Q1 change boolean variable to 1 or 0
def change_boolean(s):
    if s == 'true':
        return 1
    else:
        return 0
    return

#  change empty rows
def fill(s) :
    if s=="":
        return 0.0
    return float(s)


# preprocessing, getting one union dataframe from all 8 pickles
def make_union_df(pickles,light, worst ,other_var_bool,amount_applied,penalty_amount,other_var_numeric):
    prob = {}

    list_group = [light, worst, amount_applied, penalty_amount]
    list_group_name = ['sum_light', 'sum_worst', 'sum_amount_applied', 'sum_penalty_amount']

    list_df = []
    for pickle in pickles:
        data = pd.read_pickle(pickle)
        df = pd.DataFrame(data)

        # change true to 1, false to 0
        for col in light + worst + other_var_bool:
            df[col] = df[col].apply(change_boolean)
        # change if row="" to 0
        for col in amount_applied + penalty_amount + other_var_numeric:
            df[col] = df[col].apply(fill)
        # covert the timestamp columns to datetime obj
        df['time:timestamp'] = df['time:timestamp'].apply(time2date)
        # adding 3 more columns
        for i in range(len(list_group)):
            df[list_group_name[i]] = df[list_group[i]].sum(axis=1)

     #     check the percentage of the penalty apps to all for each pickle
        penalty_apps = df[df['sum_penalty_amount'] > 0].shape[0]
        prob[pickle] = [penalty_apps / (df.shape[0]),'events: '+str(df.shape[0]), 'application :'+ str(df['tr_application'].value_counts().size),'applicants : '+str(df['tr_applicant'].value_counts().size)]
        list_df.append(df)
        # printing for statistics section
    print(prob)
    return pd.concat(list_df, ignore_index=True)


print("******************************************")
print("Q1 Functions")

# Q1 convert timestamp to datetime obj, this form make run time more faster
def time2date(dstr):
    pp = re.compile(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})')
    return datetime.datetime(*map(int, pp.match(dstr).groups()))

# Q1 finding the duration of each application
def diff_days(d):
    d=d.sort_values()
    d_list=d.tolist()
    return (d_list[-1]-d_list[0]).days

# Q1 Function that runs the requested algorithm and returns model and prediction value of the testing and probs of roc curve
def fit_model(model_requested, X_train, y_train, X_test,y_test):
    model = model_requested.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    acc = round(model_requested.score(X_test, y_test) * 100, 4)
    return model,test_pred,probs,acc

# Q1 print_logistic regression_summary
def creat_logistic_summary(X_train,y_train):
    X_train_log = sm.add_constant(X_train)
    mod = sm.Logit(y_train.astype(float), X_train_log.astype(float))
    logit_res = mod.fit()
    print(logit_res.summary())

# Q1 get confusion matrix
def confusion_matrix_df(y_test, y_pred):
    d_f=pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        columns=['Predicted Not Rejected', 'Predicted Rejected'],
        index=['Actual Not Rejected', 'Actual Rejected']
    )
    return d_f

# Q1 Function that plot roc curve
def make_roc_curve(models, probs, colors):
    plt.title('Roc Curve')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    def plot_roc_curves(y_test, prob, model):
        fpr, tpr, threshold = metrics.roc_curve(y_test, prob)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, 'b', label=model + ' AUC = %0.2f' % roc_auc, color=colors[i])
        plt.legend(loc='lower right')
    for i, model in list(enumerate(models)):
        plot_roc_curves(y_test, probs[i], models[i])
    plt.show()

print("******************************************")
print("Q2  part A Functions")
# Q2 a simple masking
def masking_by_columns(df,columns,condion):
    mask = df[columns] == condion
    return df[mask]

# Q2 part A calculation average and ignoring the zero
def mean_without_zero(col):
    without_zero = col.copy()
    without_zero = without_zero.replace(0, np.NaN)
    return np.round(without_zero.mean(),4)

# Q2 part A calculation of confidance_interval alpha=0.05 (1-(alpha/2))=0.975
def confidance_interval(a):
    sample_mean = statistics.mean(a)
    z_critical = stats.norm.ppf(q=0.975)  # Get the z-critical value*
    pop_stdev = statistics.stdev(a)  # Get the population standard deviation
    margin_of_error = z_critical * (pop_stdev / math.sqrt(len(a)))
    confidence_interval = (sample_mean - margin_of_error,
                           sample_mean + margin_of_error)
    return np.round(confidence_interval,4)

print("******************************************")
print("Q2  part B Functions")


# Q2 part B correlation_matrix discribe the data frame
def plot_correlation_matrix(df):
    correlation_Coeficient =df.corr()
    f, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(correlation_Coeficient,linewidths=2.0, ax=ax , annot=True)
    ax.set_title('Correlation Mtrix')
    plt.show()

# Q2 part B histogram discribe the data frame
def plot_histogram(df):
    df.hist(bins=50, figsize=(30,20));
    plt.show()

# Q2 part B multiple varivable linear regression
def linear_regression(x_train,x_test,y_train,y_test):
    lm = LinearRegression()
    lm = lm.fit(x_train,y_train)
    coefficients = pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(np.transpose(lm.coef_))], axis = 1)
    y_pred = lm.predict(x_test)
    y_error = y_test - y_pred
    print(r2_score(y_test,y_pred))
    X_train = sm.add_constant(x_train) ## adding an intercept (beta_0) to our model
    X_test = sm.add_constant(x_test)
    lm2 = sm.OLS(y_train,X_train).fit()
    print(lm2.summary())

# Q2 part B single linear regression
def single_regression(X,y):
    plt.scatter(X,y)
    plt.xlabel("quantity of worst and light")
    plt.ylabel("sum_penalty_amount")
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X) # make the predictions by the model
    # Print out the statistics
    print(model.summary())
    # plot red line for the regression
    plt.plot(X,predictions, "r")
    plt.show()

print("******************************************")
print("Q3 Functions")


# Q3 exponential_smoothing to get prediction of 2018 year
def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

# Q3 plot pie chart - the load of each department
def plot_pie_chart(dep_df):
    list_departments = dep_df.index.tolist()
    list_events = dep_df.tolist()
    list_normal = []
    for i in range(len(list_events)):
        list_normal.append(list_events[i] / sum(list_events))
    # Pie Chart
    labels = list_departments
    sizes = list_normal
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.01, 0.01, 0, 0)  # explode 1st 2nd slice
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title('Precentage of the events per department')
    plt.show()

#Q3 plot forecast of 2018 by the value of the exponential smoothing
def plot_forecast_2018_events(df_year):
    dep_e7, dep_4e, dep_6b, dep_d4 = [], [], [], []
    dep_df = [dep_e7, dep_4e, dep_6b, dep_d4]
    dep_name = ['e7', '4e', '6b', 'd4']
    for i in range(len(dep_df)):
        dep_df[i] = masking_by_columns(df_year, 'tr_department', dep_name[i])
    y_e7, y_4e, y_6b, y_d4 = [], [], [], []
    y = [y_e7, y_4e, y_6b, y_d4]
    exp_e7, exp_4e, exp_6b, exp_d4 = [], [], [], []
    exp = [exp_e7, exp_4e, exp_6b, exp_d4]

    for i in range(len(y)):
        y[i] = dep_df[i]['eventid'].tolist()
        exp[i] = exponential_smoothing(y[i], 0.25)
        y[i].append(exp[i][-1])

    x_years = dep_df[0]['tr_year'].tolist()
    x_years.append("2018 Forecast")

    colors = ['r', 'g', 'b', 'y']
    for i in range(len(y)):
        plt.plot(x_years, y[i], colors[i], label=dep_name[i])
    plt.legend()
    plt.gcf().canvas.set_window_title('Events per departments Through The Years')
    plt.xlabel('Year')
    plt.ylabel('Number Of Events')
    plt.title('Events Through The Years')
    plt.show()

# Q3 getting the distribution of the events per resource and simultaneity get pareto
def pareto_chart_plot(x_data1, y_data1, y_pareto, dep_name):
    fig, ax1 = plt.subplots()
    plt.xticks(rotation=90)
    ax1.bar(x_data1, y_data1, label='data 1')
    ax1.set_xlabel('org:resource')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Number of Events', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    y_pareto = np.cumsum(y_pareto)
    ax2.plot(x_data1, y_pareto, '-r')
    ax2.set_ylabel('Pareto - Cumulative % of events', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_title('Event amount in Department :' + dep_name)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    ## Introduction:
    pickles = ['control_summary_df.pickle', 'Department_control_parcels_df.pickle', 'Entitlement_application_df.pickle',
               'Geo_parcel_document_df.pickle', 'Inspection_df.pickle', 'Parcel_document_df.pickle',
               'Payment_application_df.pickle','Reference_alignment_df.pickle']


    ## columns we used in the questions
    id_app = ['tr_applicant', 'tr_application']
    light = ['tr_penalty_ABP', 'tr_penalty_AGP', 'tr_penalty_AJLP', 'tr_penalty_AUVP',
             'tr_penalty_AVBP', 'tr_penalty_AVGP', 'tr_penalty_AVJLP', 'tr_penalty_AVUVP', 'tr_penalty_B2',
             'tr_penalty_C4','tr_penalty_C9', 'tr_penalty_CC', 'tr_penalty_GP1', 'tr_penalty_JLP1', 'tr_penalty_JLP2',
             'tr_penalty_JLP5','tr_penalty_JLP6', 'tr_penalty_JLP7']
    worst = ['tr_penalty_B3', 'tr_penalty_B4', 'tr_penalty_B5', 'tr_penalty_B6','tr_penalty_B16', 'tr_penalty_BGK', 'tr_penalty_C16',
             'tr_penalty_JLP3', 'tr_penalty_V5', 'tr_penalty_BGP','tr_penalty_BGKV', 'tr_penalty_B5F']
    amount_applied = ["tr_amount_applied0", "tr_amount_applied1", "tr_amount_applied2", "tr_amount_applied3"]
    penalty_amount = ['tr_penalty_amount0', 'tr_penalty_amount1', 'tr_penalty_amount2', 'tr_penalty_amount3']

    # other var for regression in Q2
    other_var_numeric = ['tr_area', 'tr_cross_compliance', 'tr_number_parcels']
    other_var_bool = ['tr_redistribution', 'tr_rejected', 'tr_selected_risk', 'tr_small farmer', 'tr_young farmer']
    dates_name = ['tr_year', 'time:timestamp']

    df_all_documents = make_union_df(pickles, light, worst, other_var_bool, amount_applied, penalty_amount, other_var_numeric)

    ##  place the code used to generate the tables, diagrams of the introduction

    ## Question 1:
    ## the code used to generate the tables, diagrams, models of question 1

    print("Q1")
    columns_q1_name = ['tr_applicant', 'tr_application', 'time:timestamp', 'tr_rejected', 'tr_area','tr_cross_compliance']

    df_all_documents_Q1=df_all_documents[columns_q1_name].copy()

    # finding the duration of each application
    df_all_documents_Q1.rename(index=str, columns={'time:timestamp': 'duration'}, inplace=True)
    group_by_app = df_all_documents_Q1.groupby(
        by=['tr_applicant', 'tr_application', 'tr_rejected', 'tr_area','tr_cross_compliance'])['duration'].apply(diff_days)
    group_by_app = group_by_app.reset_index()


     # describe for getting statistical analysis
    print(group_by_app.describe());

    # we identified a problem with the ratio of the rejected= true and rejected=false
    print("rejected :", group_by_app['tr_rejected'].value_counts())
    sns.countplot(x='tr_rejected', data=group_by_app, palette='hls')
    plt.show()

    # calc the average of the duration rejected or accepted application
    group_by_app_true=group_by_app[group_by_app['tr_rejected'] == 1]
    group_by_app_false=group_by_app[group_by_app['tr_rejected'] == 0]

    # the duration average of rejected= true and rejected=false
    print("the duration average of rejected=true:",mean_without_zero(group_by_app_true['duration']))
    print("the duration average of rejected=false:",mean_without_zero(group_by_app_false['duration']))

    print("The column does not reflect, so the rejected requests and requests that are not rejected should be balanced. "
        "We will do this at a ratio of about 20:80.")

    # balance the rejected =true
    group_by_app_false = group_by_app_false.head(2160)

    # geting data frame that include ratio of 20:80
    frames = [group_by_app_true, group_by_app_true, group_by_app_false]
    group_by_app = pd.concat(frames)


    print(group_by_app['tr_rejected'].value_counts())

    x = group_by_app.iloc[:, 3:]
    y = group_by_app['tr_rejected']

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> models  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # deviding to testing and training
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=25)

    print(">>>>>>>>>>>>>>>>>>>>>>>>LogisticRegression<<<<<<<<<<<<<<<<<<<<<<<<<")
    # LogisticRegression model
    model, y_pred_log_reg, probs_log_reg,score = fit_model(LogisticRegression(C=1e50), X_train, y_train, X_test,y_test)
    print(classification_report(y_test, y_pred_log_reg))
    print(confusion_matrix_df(y_test, y_pred_log_reg))
    print( "mean accuracy on the given test data: " ,score,'%')
    print(model)
    print('coefficient:', np.round(model.coef_,3))
    print('intercept:', np.round(model.intercept_,3))
    creat_logistic_summary(X_train, y_train)

    print(">>>>>>>>>>>>>>>>>>>>>>>>RandomForest<<<<<<<<<<<<<<<<<<<<<<<<<")
    rfc = RandomForestClassifier(n_estimators=10)
    model, y_pred_rfc, probs_rfc,score = fit_model(rfc, X_train, y_train, X_test,y_test)
    print(classification_report(y_test, y_pred_rfc))
    print(confusion_matrix_df(y_test, y_pred_rfc))
    print( "mean accuracy on the given test data: " ,score,'%')

    # we are not include the Decision Tree in the report but we put here its code
    # print(">>>>>>>>>>>>>>>>>>>>>>>>DecisionTreeClassifier<<<<<<<<<<<<<<<<<<<<<<<<<")
    # # DecisionTreeClassifier model
    # model, y_pred_decision_tree, probs_decision_tree,score = fit_model(DecisionTreeClassifier(), X_train, y_train, X_test,y_test)
    # print(classification_report(y_test, y_pred_decision_tree))
    # print(confusion_matrix_df(y_test, y_pred_decision_tree))

    # # get pic of decision tree
    # export_graphviz(model, out_file='tree2.dot', feature_names=x.columns)

    print(">>>>>>>>>>>>>>>>>>>>>>>>plot<<<<<<<<<<<<<<<<<<<<<<<<<")
    # making one plot with the roc curve
    models = ['Logistic Regression', 'Random Forest']
    probs_model = [probs_log_reg, probs_rfc]
    color = ['green', 'magenta']
    make_roc_curve(models, probs_model, color)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10, 10))

    ## Question 2:
    ## place the code used to generate the tables, diagrams, models of question 2
    print("Q2 part A")

    # extracting only columns we need
    df_all_documents_Q2a = df_all_documents[['sum_penalty_amount', 'tr_applicant', 'tr_application', 'tr_year']].copy()

    # creating 2 df that's contains applicants that applied for 2016 and 2017 only
    # the pickles that there is no 2017 : Department_control_parcels_df
    # the pickles that there is no 2016 : Parcel_document_df
    df_2016 = pd.DataFrame()
    df_2017 = pd.DataFrame()
    df_years = [df_2016, df_2017]
    years = ['2016', '2017']
    for i in range(len(df_years)):
        df_years[i] = masking_by_columns(df_all_documents_Q2a, 'tr_year', years[i])
        df_years[i] = df_years[i][['tr_applicant', 'tr_application', 'sum_penalty_amount']]
        # to get only unique application (no duplicate)
        df_years[i] = df_years[i].groupby(by=['tr_applicant', 'tr_application']).first()
        df_years[i] = df_years[i].reset_index()
        # do sum_of_penalty_amount to each applicant
        df_years[i] = df_years[i].groupby(by=['tr_applicant'])['sum_penalty_amount'].sum()
        df_years[i] = df_years[i].reset_index()



    # make union data frame of 2016 2017
    marge_2016_2017 = pd.merge(df_years[0], df_years[1], on=['tr_applicant'])
    marge_2016_2017 = pd.DataFrame(marge_2016_2017)

    marge_2016_2017.rename(index=str, columns={'sum_penalty_amount_x': 'sum_penalty_amount_2016',
                                               'sum_penalty_amount_y': 'sum_penalty_amount_2017'}, inplace=True)
    marge_2016_2017['diff_2016-2017'] = marge_2016_2017['sum_penalty_amount_2016'] - marge_2016_2017['sum_penalty_amount_2017']

    # including only records without zeros for including persons without  to calc good avg
    mask3 = (marge_2016_2017['sum_penalty_amount_2016'] == 0) & (marge_2016_2017['sum_penalty_amount_2017'] == 0)
    marge_2016_2017 = marge_2016_2017[~mask3]

    # describe for getting statistical analysis
    print(marge_2016_2017.describe())

    print("average of panelty per applicant in 2016: ",
          mean_without_zero(marge_2016_2017['sum_penalty_amount_2016']))
    print("average of panelty per applicant in 2017: ",
          mean_without_zero(marge_2016_2017['sum_penalty_amount_2017']))

    a = marge_2016_2017['diff_2016-2017'].tolist()
    # Confidence interval to the diff column
    print("Confidence interval:", confidance_interval(a))
    # coreletion between 2016 to 2017
    print("correlation: \n ", np.corrcoef(marge_2016_2017['sum_penalty_amount_2016'].tolist(),
                                          marge_2016_2017['sum_penalty_amount_2017'].tolist()))

    print("********************************************")
    print("Q2 part B")
    # Q2 columns
    list_group_name = ['sum_light', 'sum_worst', 'sum_amount_applied', 'sum_penalty_amount']

    columns_name = id_app + light + worst + amount_applied + penalty_amount + other_var_numeric + other_var_bool+dates_name+list_group_name

    df_all_documents_Q2b = df_all_documents[columns_name].copy()
    # To avoid duplicates
    df_all_documents_Q2b = df_all_documents_Q2b.groupby(by=['tr_applicant', 'tr_application']).first()
    df_all_documents_Q2b = df_all_documents_Q2b.reset_index()

    # generate 3 dataframe
    df_penalty = pd.DataFrame()
    df_adding_var_bool = pd.DataFrame()
    df_adding_var_numeric = pd.DataFrame()

    all_df = [df_penalty, df_adding_var_bool, df_adding_var_numeric]
    list_var = [list_group_name, other_var_bool, other_var_numeric]
    lis_func = [np.sum, np.sum, np.mean]

    # for penalty and amount aplied and for boolean var we need sum
    # for numeric we need mean because we need mean of area, num of parcel
    for i in range(len(list_var)):
        all_df[i] = df_all_documents_Q2b.groupby(by=['tr_applicant'])[list_var[i]].apply(lis_func[i])
        all_df[i] = all_df[i].reset_index()

    # create union df
    marge_var_bool_to_penalty_df = pd.merge(all_df[0], all_df[1], on=['tr_applicant'])
    marge_bool_numeric_penalty_df = pd.merge(marge_var_bool_to_penalty_df, all_df[2], on=['tr_applicant'])

    # describe for getting statistical analysis
    print(marge_bool_numeric_penalty_df.describe())

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> correlation matrix <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # calculate the correlation matrix for union df
    plot_correlation_matrix(marge_bool_numeric_penalty_df)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> histogram matrix <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # histogram for union df
    plot_histogram(marge_bool_numeric_penalty_df)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>multiple variable regression<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # get only db with people got penalty
    marge_bool_numeric_penalty_df = marge_bool_numeric_penalty_df[marge_bool_numeric_penalty_df['sum_penalty_amount'] > 0]

    # Linear_Regression after extracting multiculinary variable
    X = marge_bool_numeric_penalty_df[
        ['sum_light', 'sum_worst', 'sum_amount_applied', 'tr_redistribution', 'tr_rejected',
         'tr_selected_risk', 'tr_small farmer', 'tr_young farmer', 'tr_cross_compliance',
         'tr_number_parcels']]

    y = marge_bool_numeric_penalty_df['sum_penalty_amount']
    # deviding the data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    # linear regression
    linear_regression(x_train, x_test, y_train, y_test)

    # get regression with significant variable only
    X = marge_bool_numeric_penalty_df[
        ['sum_light', 'sum_worst', 'sum_amount_applied', 'tr_young farmer', 'tr_selected_risk', 'tr_cross_compliance',
         'tr_number_parcels']]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    linear_regression(x_train, x_test, y_train, y_test)

    ## Question 3:
    ## place the code used to generate the tables, diagrams, models of question 3
    columns_q3 = ['tr_department', 'org:resource', 'eventid', 'tr_year']

    df_all_documents_Q3 = df_all_documents[columns_q3]
    groupItem = df_all_documents_Q3.groupby(by=['tr_department', 'org:resource', 'tr_year'])['eventid'].count()
    groupItem = groupItem.reset_index()

    # describe for getting statistical analysis
    print(groupItem.describe())

    #calc the number of events for each department
    dep_group=groupItem.groupby(by=['tr_department'])['eventid'].sum().sort_values(ascending=False)
    #printing num of application in each department
    print(dep_group)

    # plot pie chart
    plot_pie_chart(dep_group)

    # exploring with pareto the 5 main resources
    final_group_by= groupItem.groupby(by=['tr_department','org:resource'])['eventid'].sum()
    final_group_by=final_group_by.reset_index()

    # how much resource each department
    num_res_each_dep={}
    all_dep_name=['e7','4e','6b','d4']
    df_e7=pd.DataFrame()
    df_4e=pd.DataFrame()
    df_6b=pd.DataFrame()
    df_d4=pd.DataFrame()
    df_all_dep=[df_e7,df_4e,df_6b,df_d4]

    for i in range(len(all_dep_name)):
        df_all_dep[i]=masking_by_columns(final_group_by, 'tr_department', all_dep_name[i]).sort_values(by='eventid', ascending=False)
        num_res_each_dep[all_dep_name[i]]=df_all_dep[i]['org:resource'].value_counts().size
    print("number of resource:",num_res_each_dep)

    dep_n=['e7','4e']
    dep_data=[df_all_dep[0],df_all_dep[1]]
    for i in range(len(dep_n)):
        # pareto
        pareto_chart_plot(dep_data[i]['org:resource'], dep_data[i]['eventid'], dep_data[i]['eventid'] / dep_group['e7'], dep_n[i])

        # exploring the 5 main resources in the 2 most loaded departments
        num_resources = dep_data[i]['org:resource'].value_counts().size

        top_5= dep_data[i].head(5).copy()
        top_5['ratio_from_the_applications'] = (top_5['eventid'] / dep_group[dep_n[i]]) * 100
        print("5 resources is: " + str(round((5 / num_resources),4) * 100) + '% from all the resources and they loaded ' + str(round(sum(top_5['ratio_from_the_applications'].values),4)) + "%")

    #plot number of events each departments by years
    final_group_by_year= groupItem.groupby(by=['tr_department','tr_year'])['eventid'].sum()
    final_group_by_year=final_group_by_year.reset_index()
    plot_forecast_2018_events(final_group_by_year)
