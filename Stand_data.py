from sklearn.preprocessing import StandardScaler
import joblib
import datetime

def standardize_data(data, save = False):
    scaler = StandardScaler()

    standardized_data = scaler.fit_transform(data)
    
    if save == True:
        joblib.dump(scaler, '/mnt/c/Ben_workspace/PythonCode/ICat/Model/scaler_detaG_1205.pkl')
        print('scaler.pkl is saved')

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print('Data normalization has been completed                     ', current_time)


    return standardized_data