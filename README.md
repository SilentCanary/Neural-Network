# Neural-Network

# 🧠 MNIST Neural Network in C++ (from scratch)

This is a basic neural network built **entirely from scratch in C++** to recognize handwritten digits using the **MNIST dataset**.

It uses:

- 🧮 Eigen library for matrix operations
- 3 layers:  
  - Input layer: 784 neurons  
  - Hidden layers: 128 → 64 (with ReLU)  
  - Output layer: 10 neurons (with Softmax)
- 📉 Cross-entropy loss
- 🔁 Mini-batch gradient descent
- ✅ Accuracy and confusion matrix evaluation

---

## 🔧 How to Use

1. Run the preprocessing script to generate CSV files:
   ```bash
   python preprocessing.py
   ```
