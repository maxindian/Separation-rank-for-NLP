import numpy as np
import matplotlib.pyplot as plt
#import scipy.stats as stats


font_size = 15
title_size = font_size +2
label_size = font_size -0.8
x=[100, 150, 200, 250, 300]

plt.subplot(231)
# trec
y1=[91.32, 91.54, 91.89, 92.07, 92.09]
y2=[91.55, 91.79, 91.97, 91.98, 92.24]
y3= [90.87,91.70, 91.88, 91.90, 92.13]
plt.plot(x,y3,color= "#b2dac7",marker="^",label='3rd layer LSTM',markersize=font_size)
plt.plot(x,y2,color="#f39894",marker='o',label='2nd layer LSTM',markersize=font_size)
plt.plot(x,y1,color= "#228fbd", marker="s",label='1st layer LSTM',markersize=font_size)


plt.title('Named Entity Recognition (NER)',fontsize=title_size)
plt.xlabel('the number of hidden units R',size=font_size)
plt.ylabel('F1',size=font_size)
plt.legend(fontsize=label_size,loc='lower right')

plt.subplot(234)
y4=[0.619, 0.629, 0.641, 0.645, 0.648]
y5=[0.640, 0.642, 0.645, 0.649, 0.655]
y6=[0.661, 0.667, 0.670, 0.673, 0.675]
plt.plot(x,y6,color= "#b2dac7",marker="^",label='3rd layer LSTM', markersize=font_size)
plt.plot(x,y5,color="#f39894",marker='o',label='2nd layer LSTM', markersize=font_size)
plt.plot(x,y4,color= "#228fbd", marker="s",label='1st layer LSTM', markersize=font_size)

plt.title('Word Sense Disambiguation (WSD)',fontsize=title_size)
plt.xlabel('the number of hidden units R',size=font_size)
plt.ylabel('F1',size=font_size)
plt.legend(fontsize=label_size,loc='lower right')

plt.subplot(233)
y4=[0.868, 0.88, 0.885, 0.89, 0.895]
y5=[0.87,0.887,0.893,0.905,0.913]
y6=[0.878,0.885,0.89,0.91,0.92]
plt.plot(x,y6,color= "#b2dac7",marker="^",label='3rd layer LSTM', markersize=font_size)
plt.plot(x,y5,color="#f39894",marker='o',label='2nd layer LSTM', markersize=font_size)
plt.plot(x,y4,color= "#228fbd", marker="s",label='1st layer LSTM', markersize=font_size)

plt.title('Constituency Parsing',fontsize=title_size)
plt.xlabel('the number of hidden units R',size=font_size)
plt.ylabel('F1',size=font_size)
plt.legend(fontsize=label_size,loc='lower right')

plt.subplot(232)
y_1=[0.739, 0.935, 0.951, 0.953, 0.955]
y_2=[0.731, 0.889, 0.953, 0.951, 0.943]
y_3=[0.811, 0.943, 0.945, 0.951,0.954]
plt.plot(x,y_3,color= "#b2dac7",marker="^",label='3rd layer LSTM', markersize=font_size)
plt.plot(x,y_2,color="#f39894",marker='o',label='2nd layer LSTM', markersize=font_size)
plt.plot(x,y_1,color= "#228fbd", marker="s",label='1st layer LSTM', markersize=font_size)
plt.title('POS tagging',fontsize=title_size)
plt.xlabel('the number of hidden units R',size=font_size)
plt.ylabel('F1',size=font_size)
plt.legend(fontsize=label_size,loc='lower right')

plt.subplot(235)
y_4=[0.786,0.801,0.841,0.846,0.859]
y_5=[0.864,0.874,0.883,0.882,0.887]
y_6=[0.895,0.893,0.894,0.896,0.901]
plt.plot(x,y_6,color= "#b2dac7",marker="^",label='3rd layer LSTM', markersize=font_size)
plt.plot(x,y_5,color="#f39894",marker='o',label='2nd layer LSTM', markersize=font_size)
plt.plot(x,y_4,color= "#228fbd", marker="s",label='1st layer LSTM', markersize=font_size)
plt.title('Semantic classifier',fontsize=title_size)
plt.xlabel('the number of hidden units R',size=font_size)
plt.ylabel('F1',size=font_size)
plt.legend(fontsize=label_size,loc='lower right')

plt.subplot(236)
y_4=[0.38, 0.41, 0.42, 0.435, 0.5]
y_5=[0.45, 0.46, 0.48, 0.56, 0.61]
y_6=[0.51, 0.54, 0.60, 0.58, 0.63]
plt.plot(x,y_6,color= "#b2dac7",marker="^",label='3rd layer LSTM', markersize=font_size)
plt.plot(x,y_5,color="#f39894",marker='o',label='2nd layer LSTM', markersize=font_size)
plt.plot(x,y_4,color= "#228fbd", marker="s",label='1st layer LSTM', markersize=font_size)
plt.title('Coreference Resolution',fontsize=title_size)
plt.xlabel('the number of hidden units R',size=font_size)
plt.ylabel('F1',size=font_size)
plt.legend(fontsize=label_size,loc='lower right')

plt.show()





