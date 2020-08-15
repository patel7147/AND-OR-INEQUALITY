import time
import numpy as np
X =[0.5,2.5]
Y =[0.2,0.9]
epoch=700

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

def do_nesterov_gradient():
    w,b,eta=-2,-2,15
    prev_v_w,prev_v_b,gamma=0,0,0.1
    for i in range(epoch):
        dw,db=0,0
        v_w=gamma*prev_v_w
        v_b=gamma* prev_v_b
        for x,y in zip(X, Y):
            dw += grad_w(w-v_w,b-v_b,x, y)
            db += grad_b(w-v_w,b-v_b,x, y)
        
        v_w=gamma*prev_v_w+eta*dw
        v_b=gamma*prev_v_b+eta*db
        w=w-v_w
        b=b- v_b
        prev_v_w=v_w
        prev_v_b=v_b
        
    print("Nestrov-Gradient-Error=",error(w, b)) 

def do_momentum_gradient():
    w,b,eta=-2,-2,15
    prev_v_w,prev_v_b,gamma=0,0,0.1
    for i in range(epoch):
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
  w,b,eta =-2,-2,15
  print(f"Weight {w}, Bias {b}, Learning-rate {eta}, Epoch {epoch}")   
  print("----------------------------")           
  for i in range(epoch):
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
print("time=",times[0])

print("---------------")


stop_time2 = time.process_time()
do_momentum_gradient()
stop_time3 = time.process_time()

times.append(stop_time3-stop_time2)
print("time=",times[1])
print("---------------")
stop_time4 = time.process_time()
do_nesterov_gradient()
stop_time5 = time.process_time()

times.append(stop_time4-stop_time5)
print("time=",times[2])
print("---------------")
print("Time of 3 Gradient =",times)
