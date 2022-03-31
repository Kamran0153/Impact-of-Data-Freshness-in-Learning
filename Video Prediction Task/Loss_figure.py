# -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


csv_reader_psnr = csv.reader(open("psnr.csv"), delimiter='\t')
csv_reader_mse = csv.reader(open("mse_test_256.csv"), delimiter='\t')
csv_reader_ssim = csv.reader(open("ssim.csv"), delimiter='\t')
x_psnr=[]
y_psnr=[]
x_mse=[]
y_mse=[]
x_ssim=[]
y_ssim=[]
csv_reader_avg=[]

csv_reader_psnr=list(csv_reader_psnr)
csv_reader_mse =list(csv_reader_mse)
csv_reader_ssim=list(csv_reader_ssim)


for i in range(1,len(csv_reader_mse[0])-1):
    mse_sum=0
    for j in range(1,len(csv_reader_mse)):
        csv_reader_mse[j][i] = float(csv_reader_mse[j][i])
        mse_sum=csv_reader_mse[j][i]+mse_sum
    csv_reader_avg.append(mse_sum/(len(csv_reader_mse)-1))
    
y_mse = csv_reader_avg
print(y_mse)







for i in range(1,len(csv_reader_psnr[0])-1):
    x_psnr.append(csv_reader_psnr[0][i])
for i in range(1,len(csv_reader_psnr[0])-1):
    y_psnr.append(csv_reader_psnr[6][i])
'''
for i in range(1,len(csv_reader_mse [0])-1):
    x_mse.append(csv_reader_mse [0][i])
'''
x_mse = range(0,16)
y_mse.insert(0,0)
for i in range(1,len(csv_reader_ssim[0])-1):
    x_ssim.append(csv_reader_ssim[0][i])
for i in range(1,len(csv_reader_ssim[0])-1):
    y_ssim.append(csv_reader_ssim[6][i])
    

plt.figure(figsize=(8, 6))
#plt.subplot(132)
plt.title('MSE')        #
plt.xlabel('Frame Index')            
plt.ylabel('Average MSE')
plt.xlim(0, 15)
plt.plot(x_mse, y_mse, 'g', label='SAVP')
x_major_locator=MultipleLocator(1)
plt.gca().xaxis.set_major_locator(x_major_locator)
plt.show()
'''

plt.subplot(131)
plt.title('PSNR')        #
plt.xlabel('Frame Index')            
plt.ylabel('average PSNR')
plt.plot(x_psnr, y_psnr, 'r', label='SAVP')



plt.subplot(133)
plt.title('SSIM')        #
plt.xlabel('Frame Index')            
plt.ylabel('Average SSIM')
plt.plot(x_ssim, y_ssim, 'b', label='SAVP')
'''



