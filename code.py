import math as M
import matplotlib.pyplot as plt
import numpy as np
#finding roots using bisection method
#limitation only find one root between a and b 
# let f(x) = x^^3 - x -1 
# def f(x):
#     return x**3 - x- 1
# def f(x):
#     return x**n - 5

# def f(x):
#     return x*M.sin(x) + M.cos(x) 
# n = int(input("Enter the value of n :"))
# a = int(input("Enter the value of a :"))
# b = int(input("Enter the value of b :"))
# def bisection(a,b):
#     e = 0.0001
#     i =1
#     condition = True
#     while condition:
#         c = (a+b)/2
#         if f(a)*f(c)<0:
#             b=c
#         else:
#             a = c
#         i+=1
#         condition = abs(f(c)) > e
#     print(f"root is {c} and the value of the function is {round(f(c))}")
# if f(a)*f(b)>0:
#     print("in the given interval roots doesnt exists")
# else:
#     bisection(a,b)
    
    
#newton rephson method
# f = lambda x: x**3 - x - 1
# g = lambda x: 3*x**2 - 1
# def newton_raphson(f,g,x0):
#     e=0.0001
#     if abs(f(x0)) < e:
#         return x0
#     else:
#         return newton_raphson(f,g,(x0 - f(x0)/g(x0)))
# x0 = float(input("Enter the initial guess x0: "))
# result = newton_raphson(f,g,x0)
# print(f"root {result} is  value of function is {f(result)} ")


#secant method
#our relation is xn = xn-1 - f(xn-1) * (xn-1 - xn-2)/(f(xn-1) - f(xn-2))
# x_0 = float(input("Enter the first guess :"))
# x_1 = float(input("Enter the second guess :"))
# e = 0.0001
# def f(x):
#     return x**3 - x -1 

# def secant_method(f,x_0,x_1):
#     if abs(f(x_0)) < e:
#         return x_0
#     elif abs(f(x_1)) <e:
#         return x_1
#     else:
#         return secant_method(f,x_1 , x_1-(f(x_1)*(x_1 -x_0)/(f(x_1)-f(x_0))))
# root = secant_method(f,x_0,x_1)
# print(f"root is {root} and value of function is {f(root)}")


#least square fit line

# x = [1,2,3,4,5,6,7]
# y =[21,43,22,54,65,72,88]
# xy =[]
# m = len(x)
# for i in range(m):
#     xy.append(x[i]*y[i])
# x2 = [i**2 for i in x]
# a1= (m*sum(xy) - sum(x)*sum(y))/(m*sum(x2) - (sum(x))**2)
# a2=(sum(y) - a1*sum(x))/m
# y_pred = []
# for i in range(m):
#     y_pred.append(a1*x[i] + a2)
# print(f"slope is {a1}")
# print(f"intercept is {a2}")
# plt.scatter(x,y)
# plt.plot((min(x),max(x)),(min(y_pred),max(y_pred)))
# plt.show()

#least squ fit 2 (Non linear curves)
# p =[i/2 for i in range(1,7)]
# v =[1.62,1.00,0.75,0.62,0.52,0.46]

# #logarithmic transformation 
# X =[M.log(i) for i in p ]
# Y =[M.log(i) for i in v ]

# #calculate XY and X^2
# XY = [X[i]*Y[i] for i in range(len(X))]
# X2= [i**2 for i in X]

# #calculate coefficient a1 and a0
# n =len(X)
# a1 = (n* sum(XY) -sum(X) *sum(Y))/(n*sum(X2) - sum(X)**2)
# a0= (sum(Y) - a1*sum(X))/n

# #create a smooth curve 

# p_smooth = np.linspace(min(p),max(p),100) # 100 points between min and max value of p

# X_smooth = [M.log(i) for i in p_smooth]
# v_pred_smooth = [M.exp(a0 + a1* (i)) for i in X_smooth]
# print(f"slope is {a1}")
# print(f"intercept is {M.exp(a0)}")
# #plotting 
# plt.scatter(p,v)
# plt.plot(p_smooth,v_pred_smooth)
# plt.show()


#plotting functions using series 
#sin function  = 1- x^3/3! + x^5/5! - x^7/7!. . . . . 
# x = [i * M.pi/180 for i in range(361)]
# n =int(input("Enter the value of n :"))
# def sin_expansion(x_values,n):
#     y_values=[]
#     x_value=[]
#     for x in x_values:
#         sum=0
#         for i in range(n):
#             sum+=((-1)**i)*(x**(2*i+1))/M.factorial(2*i+1)
#         y_values.append(sum)
#         x_value.append(x)
#     return y_values , x_value
# result, x_value = sin_expansion(x,n)
# sin_plot = [M.sin(i) for i in x]
# plt.scatter(x_value,result,color='blue')
# plt.plot(x,sin_plot,color='red')
# plt.show()

# def cos_expansion(x_values,n):
#     y_values =[]
#     x_value=[]
#     for x in x_values:
#         sum =0
#         for i in range(n):
#             sum+= (-1)**i*(x**(2*i))/(M.factorial(2*i))
#         y_values.append(sum)
#         x_value.append(x)
#     return y_values,x_value
# result, x_value = cos_expansion(x,n)
# cos_plot = [M.cos(i) for i in x]
# plt.scatter(x_value,result,color='blue')
# plt.plot(x,cos_plot,color='red')
# plt.show()  
# x = [i*0.1 for i in range(-9,10)]
# def log_expansion(x_values,n):
#     y_values=[]
#     x_value=[]
#     for x in x_values:
#         sum=0
#         for i in range(1,n+1):
#             sum+= ((-1)**(i+1))*(x**i)/i
#         y_values.append(sum)
#         x_value.append(x)
#     return y_values,x_value
# result, x_value = log_expansion(x,n)
# log_plot = [M.log(1+i) for i in x]
# plt.scatter(x_value,result,color='blue')
# plt.plot(x,log_plot,color='red')
# plt.show()  

#recurrence relation
# def legendre_poly(x,n):
#     p_0= 1
#     p_1 = x
#     if n==0:
#         return 1
#     elif n==1:
#         return x
#     else:
        
#         p_prev1 = p_0
#         p_prev2 = p_1
#         for i in range(2,n+1):
#             p_current = ((2*i -1)*x*p_prev1 -(i-1)*p_prev2)/i
#             p_prev2 ,p_prev1 = p_prev1 , p_current
#         return p_current
# x_values = np.linspace(-1,1,500)
# for n in range(6):
#     y_val = [legendre_poly(x,n) for x in x_values]
#     plt.plot(x_values ,y_val ,label = f"n ={n}")
# plt.title("legendre polynomial")
# plt.legend()
# plt.grid()
# plt.show()

#simpson integration 
# def simpson_integration(f,a,b,n):
#     if n%2 ==1:
#         print("Number of subintervals must be even")
#         return
    
#     h = (b - a)/n
#     x=a 
#     integral = f(x) + f(b)
#     for i in range(1,n):
#         x+=h
#         if i %2 ==0:
#             integral += 2 * f(x)
#         else:
#             integral += 4* f(x)
            
#     integral*= h/3
#     return integral
# def function(x):
#     return x**2 
# a = float(input("Enter the lower limit :"))
# b = float(input("Enter the upper limit :"))
# n = int(input("Enter the number of iteration :"))
# result = simpson_integration(function,a,b,n)
# print(f"The value of integral is {result}")

#differential equation
def euler_method(f,x0,y0,h,n):
    x = [x0]
    y= [y0]
    for i in range(n):
        y_next = y[-1] + h * f(x[-1],y[-1])
        x_next = x[-1] +h
        x.append(x_next)
        y.append(y_next)
    return x,y
def equation(x,y):
    return y - x

x0=0
y0=2
h=0.01
n=20

x,y = euler_method(equation,x0,y0,h,n)
print(f"y(0.1)= {y[1]}")
print(f"y(0.2)= {y[2]}")

#rk-4 method
def runge_kutta_method(f, x0, y0, h, n):
    x = [x0]
    y = [y0]
    
    for i in range(n):
        k1 = h * f(x[-1], y[-1])
        k2 = h * f(x[-1] + h / 2, y[-1] + k1 / 2)
        k3 = h * f(x[-1] + h / 2, y[-1] + k2 / 2)
        k4 = h * f(x[-1] + h, y[-1] + k3)
        
        y_next = y[-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x_next = x[-1] + h
        
        x.append(x_next)
        y.append(y_next)
    
    return x, y
def equation(x, y):
    return y -x
x0 = 0
y0 = 2
h = 0.01
n = 20
x,y = runge_kutta_method(equation,x0,y0,h,n)
print(f"y(0.1) ={y[1]}")
print(f"y(0.2) ={y[2]}")


#rk method 
def runge_kutta_2nd_order(f, x0, y0, h, n):
    x = [x0]
    y = [y0]
    
    for i in range(n):
        k1 = h * f(x[-1], y[-1])
        k2 = h * f(x[-1] + h, y[-1] + k1)
        
        y_next = y[-1] + (k1 + k2) / 2
        x_next = x[-1] + h
        
        x.append(x_next)
        y.append(y_next)
    
    return x, y

def equation(x, y):
    return y - x


x0 = 0
y0 = 2
h = 0.01
n = 20

x, y = runge_kutta_2nd_order(equation, x0, y0, h, n)

print(f"y(0.1) = {y[1]}")
print(f"y(0.2) = {y[2]}")


#weighted least square fit
x=[203,58,210,202,198,158,165,201,157,131,166,160,186,125,218,146]
y=[495,173,479,504,510,416,393,442,317,311,400,337,423,344,533,344]
sigma_y=[21,15,27,14,30,16,14,25,52,16,34,31,42,26,16,22]
xy_sigma=[]
weight_squ=[]
x_2=[]
y_sigma2=[]
x_sigma2=[]
x2_sigma2=[]
for i in range(len(x)):
    xy_sigma.append(x[i]*y[i]/sigma_y[i]**2)
    weight_squ.append(1/sigma_y[i]**2)
    x_2.append(x[i]**2)
    x_sigma2.append(x[i]*weight_squ[i])
    y_sigma2.append(y[i]*weight_squ[i])
    x2_sigma2.append(x_2[i]*weight_squ[i])
a1=(sum(weight_squ)*sum(xy_sigma) - sum(x_sigma2)*sum(y_sigma2))/(sum(weight_squ)*sum(x2_sigma2) - (sum(x_sigma2))**2)
a0= (sum(y_sigma2)-a1*(sum(x_sigma2)))/sum(weight_squ)
print(f"slope is {a1}")
print(f"intercept is {a0}")
y_pred=[]
for i in range(len(x)):
    y_pred.append(a0 + x[i]*a1)
ao_error = M.sqrt(sum(x2_sigma2) / (sum(weight_squ) * (sum(weight_squ) * sum(x2_sigma2) - sum(x_sigma2)**2)))
a1_error = M.sqrt(sum(weight_squ) / (sum(weight_squ) * sum(x2_sigma2) - sum(x_sigma2)**2))

print(f"error in a0 is {ao_error}")
print(f"error in a1 is {a1_error}")
plt.scatter(x,y)
plt.xlabel("x_label")
plt.ylabel("y_label")
plt.plot(x,y_pred)
plt.show()
