# Comparing Optimization Algorithms for Function Minimization

## **📌 Project Overview**
This project implements and visualizes **multiple gradient-based optimization algorithms** used for function minimization. The goal is to analyze how different **learning rates, momentum strategies, and adaptive optimization methods** influence convergence speed and stability.

### **🚀 Key Features**
✅ **Gradient-Based Optimization Methods**  
✅ **Comparison of Learning Rate Schedules** (Constant vs. Power-Based)  
✅ **Momentum-Based Optimization (Nesterov, Momentum, RMSprop)**  
✅ **Adaptive Learning Optimizers (AdaGrad, Adam)**  
✅ **3D Visualization of Optimization Paths**  

---

## **📌 Objective Function**
The function being optimized:
\[
f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (2x_1 + x_2^2 - 7)^2
\]

Its **gradient**:
\[
\nabla f(x_1, x_2) = \left( 4x_1^3 + 4x_1x_2 - 36x_1 + 4x_2^2 - 28, \quad 4x_2^3 + 2x_2 + 8x_1x_2 \right)
\]

---

## **📌 Implemented Optimization Algorithms**
### **1️⃣ Gradient Descent**
```python
def gradient_descent(self, x1, x2, t1, t2, func_grad, rate, t):
    gr_x1, gr_x2 = func_grad(x1, x2)
    return x1 - self.lr(rate, t) * gr_x1, x2 - self.lr(rate, t) * gr_x2, 0, 0
