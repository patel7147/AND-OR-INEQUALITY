import numpy as np
X = [0.5,2.5]
Y = [0.2,0.9]


def f(w,b,x):
    return 1.0/(1.0 + np.exp(-(w*x + b)))

def error(w,b):
    err =0.0
    for x,y in zip(X,Y):
        fx = f(w,b,x)
        err +=0.5 * (fx - y)**2 
    return err

def grad_w(w,b,x,y):
    fx = f(w,b,x)
    return (fx - y) * x * (1-fx) * (fx)

def grad_b(w,b,x,y):
    fx = f(w,b,x)
    return (fx - y) * (1-fx) * (fx)

def do_gradient_descent():
    w,b,eta,max_epoch = 0.9,0.01,0.09,2200
    print(f"Weight {w}, Bias {b}, Learning_rate {eta}, Epoch {max_epoch}")
    print("-------------")
    for i in range(max_epoch):
        dw=0
        db=0
        for x,y in zip(X,Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        w = w - (eta * dw)
        b = b - (eta * db)
        
    print("Final Weight :",w)
    print("-------------")
    print("Final Bias : ",b)
    print("-------------")
    err = error(w,b)
    print("Error",err)

do_gradient_descent()
