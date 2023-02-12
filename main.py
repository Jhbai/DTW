from toolkit.DTWProcess import PatternRecognition 
from toolkit.DataGenerator import Generator
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    #PAT = np.random.normal(size = (500, 1))
    #SEQ = np.random.normal(size = (100000, 1))
    PAT, SEQ = Generator(1000, 128000, 0.015, 0.4).make()
    model = PatternRecognition(PAT, 100)
    model._fit(SEQ)
    # 樣態
    plt.figure(figsize = (24, 6))
    plt.subplot(3, 1, 1)
    plt.plot(PAT, label = 'PATTERN', color = 'blue')
    plt.grid()
    plt.legend()
    # 預測結果
    plt.subplot(3,1,2)
    plt.plot(SEQ, label = 'sequences', color = 'green', alpha = 0.4)
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    result = model.Results[0]
    print(result)
    R0, R1 = result[0], result[1]
    plt.plot([i for i in range(R0, R1)], SEQ[R0:R1],color = 'green', label = 'prediction')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('Prediction.png')

