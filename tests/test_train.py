# import numpy as np
# from src.train import train_model


# def test_train_model_simple():
#     # dataset lineal simple y = 2*x
#     X = np.array([[1.0], [2.0], [3.0], [4.0]])
#     y = np.array([2.0, 4.0, 6.0, 8.0])
#     model = train_model(X, y)
#     pred = model.predict([[5.0]])[0]
#     assert round(pred, 6) == 10.0
