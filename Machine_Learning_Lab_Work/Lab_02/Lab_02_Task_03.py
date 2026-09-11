from Lab_02_Task_02 import mse_multi,r2_multi,rmse_multi
from Lab_02_Task_01 import mse, r2 ,rmse
import pandas as pd
data={"Model":["Linear Regression","Multi Linear Regression"],
      "MSE":[mse,mse_multi],
      "R Square":[r2,r2_multi],
      "RMSE":[rmse,rmse_multi]
      }

df=pd.DataFrame(data)
print(df)
