from sklearn.linear_model import LinearRegression

X = [ [500],[1000],[1500],[2000]]
y = [150000,250000,350000,450000] 

model = LinearRegression()

model.fit(X,y)

print(model.predict([[1200]]))