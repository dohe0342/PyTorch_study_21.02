import numpy as np
import time

N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6

elapsed_time = time.time()

for t in range(500):
    #순전파 단계 : 예측값 y를 계산합니다.
    h = np.dot(x, w1) #첫 번째 레이어, (64,1000) x (1000,100)
    h_relu = np.maximum(h, 0) #첫 번째 레이어 relu
    y_pred = np.dot(h_relu, w2) #두 번째 레이어 (64,100) x (100,10)

    #손실(loss)을 계산하고 출력합니다.
    loss = np.square(y_pred - y).sum() #loss는 mse(mean square error)
    #print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = np.dot(h_relu.T, grad_y_pred)
    grad_h_relu = np.dot(grad_y_pred, w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    #가중치 갱신
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

elapsed_time = time.time() - elapsed_time
print('elapsed time : ', 1000*elapsed_time, ' ms')