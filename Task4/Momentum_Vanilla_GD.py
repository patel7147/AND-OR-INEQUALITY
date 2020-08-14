import time
import numpy as np
X =[0.5,2.5]
Y =[0.2,0.9]

def f(w,b,x):
  return 1.0/(1.0 + np.exp(-(w*x+b)))

def error(w,b):
  err= 0.0
  for x,y in zip(X,Y):
    fx = f(w,b,x)
    err += 0.5 * (fx-y)**2
  return err


def grad_b(w,b,x,y):
  fx = f(w,b,x)
  return (fx-y)*(fx)*(1-fx)

def grad_w(w,b,x,y):
  fx = f(w,b,x)
  return (fx-y)*(fx)*(1-fx)*x

def do_momentum_gradient():
    w,b,eta=-4,-4,20
    prev_v_w,prev_v_b,gamma=0,0,0.1
    for i in range(1000):
        dw,db=0,0
        for x,y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
     
        v_w=gamma*prev_v_w+eta*dw
        v_b=gamma*prev_v_b+eta*db
        w=w-v_w
        b=b-v_b
        prev_v_w=v_w
        prev_v_b= v_b
    
    print("Momentum-Gradient-Error=",error(w, b))    
        
def do_gradient_descent():
  w,b,eta,max_epoch = -4,-4,20,1000  
  print(f"Weight {w}, Bias {b}, Learning-rate {eta}, Epoch {max_epoch}")              
  for i in range(max_epoch):
    dw=0
    db=0
    for x,y in zip(X,Y):
      dw += grad_w(w,b,x,y)
      db += grad_b(w,b,x,y)
    w = w - eta *dw
    b = b - eta * db
  print("Vanilla-Gradient-Error=",error(w,b)) 
 
times=list()

stop_time0 = time.process_time()
do_gradient_descent()
stop_time1 = time.process_time()

times.append(stop_time1-stop_time0)
print("time=",times)

print("---------------")


stop_time2 = time.process_time()
do_momentum_gradient()
stop_time3 = time.process_time()


times.append(stop_time3-stop_time2)
print("time=",times)
