import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
from lazypredict.Supervised import LazyClassifier

data = pd.read_csv("diabetes.csv")
# data = ProfileReport(data,title = "Profiling Report")  
# data.to_file("data_diabetes.html")

class Diabete:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    # chia dữ liệu train , test
    def split_data(self):
        x_train , x_test, y_train, y_test=train_test_split(self.x,self.y,test_size=0.2,random_state=42)
        return x_train, x_test, y_train, y_test
    
    # chọn xem dùng model nào sau khi chia tệp dữ liệu xong
    def select_model(self):
        x_train , x_test, y_train, y_test = self.split_data()
        select = LazyClassifier(verbose=0,ignore_warnings=True,custom_metric=None)
        model,predictions = select.fit(x_train, x_test, y_train, y_test)
        print(pd.DataFrame(model).sort_values(by="Accuracy",ascending=False))

    # chọn QuadraticDiscriminantAnalysis vì đứng đầu kết quả về recall:
    def use_model(self):
        x_train , x_test, y_train, y_test = self.split_data()
        param = {
            "priors": [None, [0.5, 0.5], [0.25, 0.75]],  # Tối ưu priors (tỉ lệ của các lớp)
            "reg_param": [0.0, 0.1, 0.5, 1.0],  # Tối ưu tham số regularization
        }
        gsv = GridSearchCV(
            estimator=QuadraticDiscriminantAnalysis(),
            param_grid=param,
            scoring="recall",  # Hoặc có thể dùng recall, f1, v.v.
            cv=4,
            verbose=2,
        )

        # Fit GridSearchCV để tìm các tham số tốt nhất
        gsv.fit(x_train, y_train)

        # In ra các kết quả tối ưu
        print("Best Estimator: ", gsv.best_estimator_)
        print("Best Score: ", gsv.best_score_)
        print("Best Params: ", gsv.best_params_)
        print("Best Index: ", gsv.best_index_)
        model = QuadraticDiscriminantAnalysis(priors=[0.25, 0.75],reg_param=0.1)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        print(classification_report(y_test,y_pred))

# Dùng OOP
def main():
    target = "Outcome"
    x = data.drop(target,axis=1)
    y = data[target]  
    dia = Diabete(x,y) 
    dia.split_data()
    dia.select_model()
    dia.use_model()

main()


