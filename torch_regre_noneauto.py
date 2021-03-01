import torch
import time

if __name__ == '__main__':
    dtype = torch.float
    device = torch.device('cpu')

    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)

    learning_rate = 1e-6

    elapsed_time = time.time()

    for t in range(500):
        h = torch.mm(x, w1)
        h_relu = h.clamp(min=0)
        y_pred = torch.mm(h_relu, w2)

        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss)

        #loss = (y_pred - y).pow(2).sum()
        #loss.backward()

        #with torch.no_grad():
        #    w1 -= learning_rate * w1.grad
        #    w2 -= learning_rate * w2.grad

        #w1.grad.zero()
        #w2.grad.zero()

        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h<0] = 0
        grad_w1 = x.t().mm(grad_h)
    
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

    elapsed_time = time.time() - elapsed_time
    print('elapsed time : ', 1000*elapsed_time, ' ms')